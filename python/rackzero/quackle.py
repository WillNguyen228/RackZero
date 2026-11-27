"""Quackle's static player as a teacher and evaluation opponent.

``QuackleEngine`` keeps one native bridge process (``quackle_bridge --serve``)
alive and asks it for its best moves by static equity (score + rack-leave
value). It sees exactly what the acting player sees: the board, its own rack
and the bag size. Its moves are matched to the Java engine's legal moves, so the
Java rules engine stays the authority on legality and scoring.

With ``sim_seconds`` > 0 it plays Quackle's championship player instead (Smart
Bogowin: 2-ply simulation over sampled opponent racks and draws, chosen by win
probability; exact endgame player when the bag is empty), given the true unseen
tiles and both scores, with that many seconds per turn. The bridge seeds its own
random numbers, so simulation games are not reproducible move for move.

Requires a local Quackle build (``python -m scrabble_gym.quackle_build``).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import threading

import numpy as np

from scrabble_gym.environment import ALPHABET, DIRECTIONS, Move, ScoredMove


def _root() -> Path:
    from scrabble_gym.build import ROOT
    return ROOT


def _target() -> Path:
    return _root() / "target" / "quackle"


def quackle_available() -> bool:
    manifest = _target() / "manifest.json"
    if not manifest.is_file():
        return False
    executable = _root() / json.loads(manifest.read_text(encoding="utf-8"))["executable"]
    return executable.is_file()


def lexicon_dawg(words: frozenset[str]) -> Path:
    """Quackle DAWG for exactly ``words`` (built once, cached by content hash)."""
    from scrabble_gym.quackle_build import build_lexicon

    text = "\n".join(sorted(words)) + "\n"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    directory = _target() / "lexicons" / digest
    dawg = directory / "words.dawg"
    if not dawg.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=directory) as temporary:
            source = Path(temporary) / "words.txt"
            source.write_text(text, encoding="utf-8")
            built = Path(temporary) / "words.dawg"
            count = build_lexicon(source, built)
            if count != len(words):
                raise ValueError("Quackle lexicon dropped words outside its supported format")
            built.replace(dawg)
    return dawg


SIM_SECONDS_ENV = "RACKZERO_QUACKLE_SECONDS"


def sim_seconds_from_env(default: int = 5) -> int:
    """Seconds per turn for the ``quackle-sim`` opponent (evaluate.py --quackle-seconds)."""
    return int(os.environ.get(SIM_SECONDS_ENV, default))


def position_text(observation: dict, sim: bool = False) -> str:
    """One bridge request line from the acting player's observation (with the
    unseen tiles and both scores for the simulating player)."""
    board, blanks = observation["board"], observation["board_blanks"]
    cells = []
    for r in range(15):
        for c in range(15):
            code = int(board[r, c])
            if not code:
                cells.append(".")
            else:
                letter = ALPHABET[code - 1]
                cells.append(letter.lower() if blanks[r, c] else letter)
    rack = "".join(letter * int(n) for letter, n in zip(ALPHABET, observation["rack"]))
    text = f"{''.join(cells)} {rack} {int(observation['bag_size'])}"
    if sim:
        from rackzero.game import DISTRIBUTION, board_counts
        unseen = DISTRIBUTION - board_counts(observation) - np.asarray(observation["rack"])
        if np.any(unseen < 0):
            raise ValueError("Observation violates tile conservation")
        letters = "".join(letter * int(n) for letter, n in zip(ALPHABET, unseen))
        player = int(observation["player"])
        scores = observation["scores"]
        text += f" {letters or '-'} {int(scores[player])} {int(scores[1 - player])}"
    return text


def move_key(board: np.ndarray, move: Move) -> tuple:
    """Representation-independent identity of a move (tiles placed, or tiles exchanged)."""
    if move.kind == "pass":
        return ("pass",)
    if move.kind == "exchange":
        return ("exchange", "".join(sorted(move.exchange)))
    dr, dc = DIRECTIONS[move.direction]
    placed = []
    for i, letter in enumerate(move.word):
        r, c = move.row + dr * i, move.col + dc * i
        if not board[r, c]:
            placed.append((r, c, letter, i in move.blanks))
    return ("place", frozenset(placed))


def _parse(lines: list[str], rack: str) -> list[tuple[tuple, float, int]]:
    """Bridge output blocks -> [(move key, equity, score)] in equity order."""
    result, i = [], 0
    if lines and lines[0].startswith("MOVES "):
        i = 1
    equity = float("nan")
    while i < len(lines):
        header = lines[i].split()
        if header[0] == "EQUITY":
            equity = float(header[1])
            i += 1
        elif header[0] == "PASS":
            result.append((("pass",), equity, 0))
            i += 1
        elif header[0] == "EXCHANGE":
            letters = "".join(rack[int(j)] for j in header[2:])
            result.append((("exchange", "".join(sorted(letters))), equity, 0))
            i += 1
        elif header[0] == "PLACE":
            score, equity, count = int(header[1]), float(header[2]), int(header[3])
            placed = []
            for line in lines[i + 1:i + 1 + count]:
                r, c, _, letter, blank = line.split()
                placed.append((int(r), int(c), letter, blank == "1"))
            result.append((("place", frozenset(placed)), equity, score))
            i += 1 + count
        else:
            raise RuntimeError(f"Unexpected Quackle output line {lines[i]!r}")
    return result


class QuackleEngine:
    """A persistent Quackle static-player (or, with ``sim_seconds``, championship
    player) process for one lexicon."""

    def __init__(self, words: frozenset[str], top: int = 10, sim_seconds: int = 0):
        if not quackle_available():
            raise FileNotFoundError("Quackle is not built; run python -m scrabble_gym.quackle_build")
        manifest = json.loads((_target() / "manifest.json").read_text(encoding="utf-8"))
        root = _root()
        self.executable = (root / manifest["executable"]).resolve()
        self.dawg = lexicon_dawg(words)
        self.top = int(top)
        self.sim_seconds = int(sim_seconds)
        self.revision = manifest.get("revision")
        self._lock = threading.Lock()
        self._process = None
        self.calls = 0
        self.unmatched = 0

    def _start(self):
        if self._process is None or self._process.poll() is not None:
            root = _root()
            self._process = subprocess.Popen(
                [str(self.executable), str(self.dawg.relative_to(root)), "--serve",
                 "--top", str(self.top)]
                + (["--sim", str(self.sim_seconds)] if self.sim_seconds > 0 else []),
                cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
        return self._process

    def ranked(self, observation: dict) -> list[tuple[tuple, float, int]]:
        """Quackle's best moves for the acting player: [(move key, equity, score)]."""
        request = position_text(observation, self.sim_seconds > 0)
        rack = request.split()[1]
        with self._lock:
            process = self._start()
            process.stdin.write(request + "\n")
            process.stdin.flush()
            lines = []
            while True:
                line = process.stdout.readline()
                if not line:
                    error = process.stderr.read()
                    raise RuntimeError(f"Quackle bridge exited: {error.strip()[:500]}")
                line = line.rstrip("\r\n")
                if line == "END":
                    break
                lines.append(line)
        self.calls += 1
        if lines and lines[0].startswith("ERROR"):
            raise RuntimeError(f"Quackle: {lines[0][6:]}")
        return _parse(lines, rack)

    def rank_legal(self, observation: dict, moves: tuple[ScoredMove, ...]):
        """Quackle's ranked moves as indices into ``moves``: [(index, equity)], best first.

        Raises if Quackle's best move is not a legal Java move with the same score.
        """
        board = observation["board"]
        index = {move_key(board, m.move): i for i, m in enumerate(moves)}
        ranked = []
        for position, (key, equity, score) in enumerate(self.ranked(observation)):
            i = index.get(key)
            if i is None or (key[0] == "place" and moves[i].score != score):
                if position == 0:
                    raise RuntimeError("Quackle's choice is not a legal move with the same score")
                self.unmatched += 1
                continue
            ranked.append((i, equity))
        return ranked

    def choose(self, observation: dict, moves: tuple[ScoredMove, ...]) -> int:
        return self.rank_legal(observation, moves)[0][0]

    def close(self):
        if self._process is not None:
            try:
                self._process.stdin.close()
                self._process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self._process.kill()
                self._process.wait()
            for stream in (self._process.stdout, self._process.stderr):
                stream.close()
            self._process = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
