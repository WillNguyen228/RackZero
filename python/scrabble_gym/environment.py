"""Gymnasium turns backed exclusively by the Java rules and scoring engine."""

from collections import Counter
from dataclasses import dataclass
from itertools import product
import os
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

import gymnasium as gym
from gymnasium import spaces
import jpype
import numpy as np


ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ?"
SIZE = 15
RACK_SIZE = 7
DIRECTIONS = ((0, 1), (1, 0))
JAVA_PACKAGE = "com.masterchiefproject.scrabble."


@dataclass(frozen=True)
class Move:
    """Complete main word; blank indices refer only to newly placed tiles."""

    kind: str = "pass"
    row: int = 0
    col: int = 0
    direction: int = 0
    word: str = ""
    blanks: tuple[int, ...] = ()
    exchange: str = ""

    def to_action(self) -> dict:
        word = np.zeros(SIZE, dtype=np.int64)
        blanks = np.zeros(SIZE, dtype=np.int8)
        if len(self.word) > SIZE or any(c not in ALPHABET[:-1] for c in self.word):
            raise ValueError("word must contain at most 15 uppercase A-Z letters")
        if any(not isinstance(i, int) or not 0 <= i < len(self.word) for i in self.blanks):
            raise ValueError("Invalid blank positions")
        if len(set(self.blanks)) != len(self.blanks):
            raise ValueError("Duplicate blank positions")
        if any(c not in ALPHABET for c in self.exchange):
            raise ValueError("exchange must contain uppercase A-Z or ?")
        for i, letter in enumerate(self.word):
            word[i] = ALPHABET.index(letter) + 1
        blanks[list(self.blanks)] = 1
        if self.kind not in ("place", "exchange", "pass"):
            raise ValueError(f"Unknown move kind: {self.kind!r}")
        return {
            "kind": ("place", "exchange", "pass").index(self.kind),
            "row": self.row, "col": self.col, "direction": self.direction,
            "word": word, "blanks": blanks,
            "exchange": np.array([self.exchange.count(c) for c in ALPHABET], dtype=np.int64),
        }


@dataclass(frozen=True)
class ScoredMove:
    move: Move
    score: int


def _jar_path(path: str | Path | None) -> Path:
    if path is not None:
        result = Path(path)
    elif os.environ.get("SCRABBLE_JAR"):
        result = Path(os.environ["SCRABBLE_JAR"])
    else:
        target = Path(__file__).resolve().parents[2] / "target"
        result = target / "scrabble-gym.jar"
        if not result.is_file():
            result = target / "scrabble-game-2.1.0.jar"
    if not result.is_file():
        raise FileNotFoundError(
            f"Engine JAR not found: {result}. Run python -m scrabble_gym.build "
            "or mvn package, or supply jar_path / SCRABBLE_JAR."
        )
    return result.resolve()


def _parse_words(text: str, source: str) -> frozenset[str]:
    words = set()
    for number, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith("#"):
            continue
        for token in line.split():
            word = token.upper()
            if not token.isascii() or not token.isalpha() or not 2 <= len(word) <= SIZE:
                raise ValueError(f"{source}:{number}: invalid word {token!r}")
            words.add(word)
    if not words:
        raise ValueError(f"{source}: dictionary must not be empty")
    return frozenset(words)


class ScrabbleEnv(gym.Env):
    """One step controls one player; the returned observation is the next player's."""

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self, word_list: str | Path | None = None, *,
        words: Iterable[str] | None = None, max_turns: int = 200,
        render_mode: str | None = None, jar_path: str | Path | None = None,
    ):
        super().__init__()
        if not isinstance(max_turns, int) or isinstance(max_turns, bool) or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")
        if word_list is not None and words is not None:
            raise ValueError("Specify words or word_list, not both")
        jar = _jar_path(jar_path)
        if words is not None:
            self.words = frozenset(words)
        elif word_list is not None:
            self.words = _parse_words(Path(word_list).read_text(encoding="utf-8"), str(word_list))
        else:
            with ZipFile(jar) as archive:
                # The demo dictionary also contains one-letter and overlength entries.
                self.words = frozenset(
                    w for w in archive.read("dictionary/words.txt").decode("utf-8").split()
                    if w.isascii() and w.isalpha() and w.isupper() and 2 <= len(w) <= SIZE
                )
        if not self.words or any(
            not isinstance(w, str) or not w.isascii() or not w.isalpha()
            or w != w.upper() or not 2 <= len(w) <= SIZE for w in self.words
        ):
            raise ValueError("words must contain uppercase ASCII words of length 2..15")
        jpype.addClassPath(str(jar))
        if not jpype.isJVMStarted():
            jpype.startJVM()
        self._Game = jpype.JClass(JAVA_PACKAGE + "game.ScrabbleGame")
        self._Placement = jpype.JClass(JAVA_PACKAGE + "game.Placement")
        self._Request = jpype.JClass(JAVA_PACKAGE + "game.PlacementRequest")
        self._Position = jpype.JClass(JAVA_PACKAGE + "game.Position")
        self._InvalidMove = jpype.JClass(JAVA_PACKAGE + "game.InvalidMoveException")
        self._List = jpype.JClass("java.util.ArrayList")
        self._Integer = jpype.JClass("java.lang.Integer")
        word_set = self.words
        self._validator = jpype.JProxy(
            JAVA_PACKAGE + "dictionary.WordValidator",
            dict(isValid=lambda word: str(word) in word_set),
        )
        self._trie: dict = {}
        for word in sorted(self.words):
            node = self._trie
            for letter in word:
                node = node.setdefault(letter, {})
            node[""] = {}
        self.max_turns = max_turns
        self.render_mode = render_mode
        self.action_space = spaces.Dict({
            "kind": spaces.Discrete(3),
            "row": spaces.Discrete(SIZE), "col": spaces.Discrete(SIZE),
            "direction": spaces.Discrete(2),
            "word": spaces.MultiDiscrete(np.full(SIZE, 27)),
            "blanks": spaces.MultiBinary(SIZE),
            "exchange": spaces.MultiDiscrete(np.full(27, RACK_SIZE + 1)),
        })
        self.observation_space = spaces.Dict({
            "board": spaces.Box(0, 26, (SIZE, SIZE), dtype=np.int8),
            "board_blanks": spaces.MultiBinary((SIZE, SIZE)),
            "rack": spaces.MultiDiscrete(np.full(27, RACK_SIZE + 1)),
            "scores": spaces.Box(-10000, 10000, (2,), dtype=np.int32),
            "player": spaces.Discrete(2),
            "opponent_rack_size": spaces.Discrete(8),
            "bag_size": spaces.Discrete(101),
            "scoreless_turns": spaces.Discrete(7),
            "turn": spaces.Discrete(max_turns + 1),
        })
        self._game = None
        self._moves: tuple[ScoredMove, ...] | None = None
        self._done = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if options:
            raise ValueError("No reset options are supported")
        super().reset(seed=seed)
        java_seed = int(self.np_random.integers(0, np.iinfo(np.int64).max))
        names = self._List()
        names.add("Player 0")
        names.add("Player 1")
        self._game = self._Game(names, self._validator, jpype.JLong(java_seed))
        self._player = 0
        self._turn = 0
        self._done = False
        self._moves = None
        return self._observation(), self._info()

    def _active(self) -> None:
        if self._game is None:
            raise RuntimeError("Call reset() before using the environment")
        if self._done:
            raise RuntimeError("Episode has ended; call reset()")

    def _rack(self, player: int) -> list:
        return list(self._game.players().get(player).rack())

    def _scores(self) -> np.ndarray:
        return np.array([p.score() for p in self._game.players()], dtype=np.int32)

    def _observation(self) -> dict:
        board = np.zeros((SIZE, SIZE), dtype=np.int8)
        blanks = np.zeros((SIZE, SIZE), dtype=np.int8)
        for r in range(SIZE):
            for c in range(SIZE):
                tile = self._game.board().tileAt(r, c)
                if tile is not None:
                    board[r, c] = ALPHABET.index(str(tile.letter())) + 1
                    blanks[r, c] = tile.blank()
        rack = Counter(str(t.letter()) for t in self._rack(self._player))
        return {
            "board": board, "board_blanks": blanks,
            "rack": np.array([rack[c] for c in ALPHABET], dtype=np.int64),
            "scores": self._scores(), "player": self._player,
            "opponent_rack_size": len(self._rack(1 - self._player)),
            "bag_size": int(self._game.bagRemaining()),
            "scoreless_turns": int(self._game.scorelessTurns()), "turn": self._turn,
        }

    def _info(self) -> dict:
        return {"current_player": self._player, "scores": self._scores(),
                "dictionary_size": len(self.words)}

    def _decode(self, action: dict | Move) -> Move:
        if isinstance(action, Move):
            action = action.to_action()
        if not self.action_space.contains(action):
            raise ValueError("Action does not belong to action_space")
        kind = int(action["kind"])
        if kind == 2:
            return Move()
        if kind == 1:
            return Move("exchange", exchange="".join(
                c * int(n) for c, n in zip(ALPHABET, action["exchange"])
            ))
        codes = list(action["word"])
        length = codes.index(0) if 0 in codes else SIZE
        if any(codes[length:]) or any(action["blanks"][length:]):
            raise ValueError("Word must use trailing zero padding only")
        return Move(
            "place", int(action["row"]), int(action["col"]), int(action["direction"]),
            "".join(ALPHABET[int(code) - 1] for code in codes[:length]),
            tuple(i for i in range(length) if action["blanks"][i]),
        )

    @staticmethod
    def _inside(row: int, col: int) -> bool:
        return 0 <= row < SIZE and 0 <= col < SIZE

    def _placements(self, move: Move):
        if move.word not in self.words:
            raise ValueError(f"Word is not in the dictionary: {move.word!r}")
        dr, dc = DIRECTIONS[move.direction]
        end_r, end_c = move.row + dr * len(move.word), move.col + dc * len(move.word)
        if not self._inside(end_r - dr, end_c - dc):
            raise ValueError("Word extends beyond the board")
        board = self._game.board()
        for r, c in ((move.row - dr, move.col - dc), (end_r, end_c)):
            if self._inside(r, c) and board.tileAt(r, c) is not None:
                raise ValueError("Action must include the complete main word")
        rack = self._rack(self._player)
        used = set()
        placements, requests = self._List(), self._List()
        for i, letter in enumerate(move.word):
            r, c = move.row + dr * i, move.col + dc * i
            existing = board.tileAt(r, c)
            if existing is not None:
                if str(existing.letter()) != letter or i in move.blanks:
                    raise ValueError("Cannot replace an existing tile")
                continue
            required = "?" if i in move.blanks else letter
            index = next((j for j, t in enumerate(rack)
                          if j not in used and str(t.letter()) == required), None)
            if index is None:
                raise ValueError(f"Rack does not contain tile {required!r}")
            used.add(index)
            assigned = jpype.JChar(letter) if required == "?" else None
            tile = rack[index].assignBlank(assigned) if assigned is not None else rack[index]
            placements.add(self._Placement(self._Position(r, c), tile))
            requests.add(self._Request(index, r, c, assigned))
        return placements, requests

    def _preview(self, move: Move) -> int:
        placements, _ = self._placements(move)
        return int(self._game.board().preview(placements, self._validator).score())

    def legal_moves(self) -> tuple[ScoredMove, ...]:
        """Enumerate all physical placements, exchange multisets, and pass."""
        self._active()
        if self._moves is not None:
            return self._moves
        obs = self._observation()
        board = obs["board"]
        rack = dict(zip(ALPHABET, obs["rack"].tolist()))
        empty = not np.any(board)
        anchors = {
            (r, c) for r in range(SIZE) for c in range(SIZE) if not board[r, c]
            and ((empty and (r, c) == (7, 7)) or (not empty and any(
                self._inside(r + dr, c + dc) and board[r + dr, c + dc]
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))
            )))
        }
        moves: list[ScoredMove] = []
        seen = set()
        for direction, (dr, dc) in enumerate(DIRECTIONS):
            allowed = {}
            for r in range(SIZE):
                for c in range(SIZE):
                    if board[r, c]:
                        continue
                    prefix, suffix = "", ""
                    pr, pc = dc, dr
                    ar, ac = r - pr, c - pc
                    while self._inside(ar, ac) and board[ar, ac]:
                        prefix = ALPHABET[int(board[ar, ac]) - 1] + prefix
                        ar, ac = ar - pr, ac - pc
                    ar, ac = r + pr, c + pc
                    while self._inside(ar, ac) and board[ar, ac]:
                        suffix += ALPHABET[int(board[ar, ac]) - 1]
                        ar, ac = ar + pr, ac + pc
                    allowed[r, c] = (
                        {ch for ch in ALPHABET[:-1] if prefix + ch + suffix in self.words}
                        if prefix or suffix else set(ALPHABET[:-1])
                    )

            def walk(r, c, node, text, blanks, placed, touched, start_r, start_c):
                if "" in node and placed and touched and (
                    not self._inside(r, c) or not board[r, c]
                ):
                    if placed not in seen:
                        move = Move("place", start_r, start_c, direction, text, blanks)
                        moves.append(ScoredMove(move, self._preview(move)))
                        seen.add(placed)
                if not self._inside(r, c):
                    return
                if board[r, c]:
                    letter = ALPHABET[int(board[r, c]) - 1]
                    if letter in node:
                        walk(r + dr, c + dc, node[letter], text + letter, blanks,
                             placed, True, start_r, start_c)
                    return
                if len(placed) == RACK_SIZE:
                    return
                for letter, child in node.items():
                    if not letter or letter not in allowed[r, c]:
                        continue
                    for tile in (letter, "?"):
                        if not rack[tile]:
                            continue
                        rack[tile] -= 1
                        blank = tile == "?"
                        walk(r + dr, c + dc, child, text + letter,
                             blanks + (len(text),) if blank else blanks,
                             placed + ((r, c, letter, blank),),
                             touched or (r, c) in anchors, start_r, start_c)
                        rack[tile] += 1

            for r in range(SIZE):
                for c in range(SIZE):
                    if self._inside(r - dr, c - dc) and board[r - dr, c - dc]:
                        continue
                    ar, ac, empty_count = r, c, 0
                    while self._inside(ar, ac):
                        empty_count += not bool(board[ar, ac])
                        if empty_count > int(obs["rack"].sum()):
                            break
                        if (ar, ac) in anchors:
                            walk(r, c, self._trie, "", (), (), False, r, c)
                            break
                        ar, ac = ar + dr, ac + dc
        if obs["bag_size"] >= RACK_SIZE:
            tiles = [t for t in ALPHABET if rack[t]]
            for counts in product(*(range(rack[t] + 1) for t in tiles)):
                exchange = "".join(t * n for t, n in zip(tiles, counts))
                if exchange:
                    moves.append(ScoredMove(Move("exchange", exchange=exchange), 0))
        moves.append(ScoredMove(Move(), 0))
        self._moves = tuple(moves)
        return self._moves

    def step(self, action: dict | Move):
        self._active()
        move = self._decode(action)
        player = self._player
        before = self._scores()
        score = 0
        try:
            if move.kind == "place":
                _, requests = self._placements(move)
                score = int(self._game.playMove(requests).score())
            elif move.kind == "exchange":
                remaining = Counter(move.exchange)
                indexes = self._List()
                for i, tile in enumerate(self._rack(player)):
                    letter = str(tile.letter())
                    if remaining[letter]:
                        indexes.add(self._Integer(i))
                        remaining[letter] -= 1
                if any(remaining.values()):
                    raise ValueError("Exchange must select a nonempty subset of the rack")
                self._game.exchangeTiles(indexes)
            else:
                self._game.passTurn()
        except self._InvalidMove as error:
            raise ValueError(str(error.getMessage())) from error
        self._turn += 1
        terminated = bool(self._game.isGameOver())
        truncated = not terminated and self._turn >= self.max_turns
        self._done = terminated or truncated
        # Java retains the acting player at game over; the Gym API always switches.
        self._player = 1 - player
        self._moves = None
        reason = None
        if terminated:
            reason = ("six_scoreless_turns" if self._game.scorelessTurns() >= 6
                      else "rack_out")
        elif truncated:
            reason = "turn_limit"
        scores = self._scores()
        adjustments = scores - before
        adjustments[player] -= score
        info = self._info()
        info.update(acting_player=player, move=move, move_score=score,
                    score_adjustments=adjustments, end_reason=reason)
        if self._done:
            info["winner"] = None if scores[0] == scores[1] else int(np.argmax(scores))
        return (self._observation(), float(scores[player] - before[player]),
                terminated, truncated, info)

    def render(self) -> str | None:
        if self.render_mode is None:
            return None
        if self._game is None:
            raise RuntimeError("Call reset() before render()")
        obs = self._observation()
        rows = ["    " + " ".join(f"{c:2}" for c in range(SIZE))]
        for r in range(SIZE):
            cells = []
            for c in range(SIZE):
                code = int(obs["board"][r, c])
                letter = ALPHABET[code - 1] if code else "."
                cells.append(letter.lower() if obs["board_blanks"][r, c] else letter)
            rows.append(f"{r:2}  " + "  ".join(cells))
        rows.append(f"Turn {self._turn} | Player {self._player} | "
                    f"Scores {obs['scores'].tolist()} | Bag {obs['bag_size']}")
        rows.append("Rack: " + "".join(str(t.letter()) for t in self._rack(self._player)))
        return "\n".join(rows)

    def close(self) -> None:
        self._game = None
        self._moves = None
        self._done = True
        # JPype's JVM is process-wide and cannot be restarted after shutdown.


def greedy_action(env: ScrabbleEnv) -> dict:
    """Maximize immediate Java score, otherwise exchange the largest subset or pass."""
    moves = env.legal_moves()
    placements = [item for item in moves if item.move.kind == "place"]
    if placements:
        return max(placements, key=lambda item: item.score).move.to_action()
    exchanges = [item for item in moves if item.move.kind == "exchange"]
    if exchanges:
        return max(exchanges, key=lambda item: len(item.move.exchange)).move.to_action()
    return Move().to_action()
