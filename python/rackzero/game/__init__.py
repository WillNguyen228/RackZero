"""Live game and sampled-world simulator, sharing the existing Java rules engine."""

from dataclasses import dataclass
import copy
import time

import jpype
import numpy as np

from scrabble_gym.environment import ALPHABET, DIRECTIONS, Move, ScoredMove, ScrabbleEnv


DISTRIBUTION = np.array(
    [9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2, 6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1, 2],
    dtype=np.int64,
)
POINTS = np.array([1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3, 1, 1, 3, 10,
                   1, 1, 1, 1, 4, 4, 8, 4, 10, 0], dtype=np.int64)


def counts(letters: str) -> np.ndarray:
    return np.array([letters.count(c) for c in ALPHABET], dtype=np.int64)


def letters(tiles: np.ndarray) -> str:
    if tiles.shape != (27,) or np.any(tiles < 0) or np.any(tiles != np.floor(tiles)):
        raise ValueError("Tile counts must be 27 nonnegative integers")
    return "".join(c * int(n) for c, n in zip(ALPHABET, tiles))


def board_counts(observation: dict) -> np.ndarray:
    board, blanks = observation["board"], observation["board_blanks"]
    result = np.bincount(board[(board > 0) & (blanks == 0)], minlength=27)[1:27]
    return np.append(result, int(blanks.sum())).astype(np.int64)


def used_tiles(observation: dict, move: Move) -> np.ndarray:
    result = np.zeros(27, dtype=np.int64)
    if move.kind == "exchange":
        return counts(move.exchange)
    if move.kind == "place":
        dr, dc = DIRECTIONS[move.direction]
        for offset, ch in enumerate(move.word):
            r, c = move.row + offset * dr, move.col + offset * dc
            if not observation["board"][r, c]:
                result[26 if offset in move.blanks else ALPHABET.index(ch)] += 1
    return result


@dataclass(frozen=True)
class PublicEvent:
    player: int
    kind: str
    played: tuple[int, ...]
    exchange_count: int
    score: int
    bag_before: int
    rack_before: int
    move: Move | None = None


@dataclass(frozen=True)
class InformationState:
    observation: dict
    history: tuple[PublicEvent, ...] = ()

    def __post_init__(self):
        allowed = {"board", "board_blanks", "rack", "scores", "player",
                   "opponent_rack_size", "bag_size", "scoreless_turns", "turn"}
        if set(self.observation) != allowed:
            raise ValueError("Information state contains missing or non-observable fields")
        obs = {}
        for name, value in self.observation.items():
            if isinstance(value, np.ndarray):
                value = value.copy()
                value.setflags(write=False)
            obs[name] = value
        object.__setattr__(self, "observation", obs)


def public_event(obs: dict, move: Move, score: int) -> PublicEvent:
    played = used_tiles(obs, move) if move.kind == "place" else np.zeros(27, dtype=np.int64)
    return PublicEvent(int(obs["player"]), move.kind, tuple(map(int, played)),
                       len(move.exchange), score, int(obs["bag_size"]), int(obs["rack"].sum()),
                       move if move.kind != "exchange" else None)


def premium_multipliers() -> tuple[np.ndarray, np.ndarray]:
    """Letter and word multipliers of the Java board layout (15x15 each)."""
    layout = jpype.JClass("com.masterchiefproject.scrabble.game.Board")()
    premiums = [[layout.premiumAt(r, c) for c in range(15)] for r in range(15)]
    return (np.array([[p.letterMultiplier() for p in row] for row in premiums]),
            np.array([[p.wordMultiplier() for p in row] for row in premiums]))


class Game:
    """Only this coordinator owns the actual live hidden state.

    ``movegen="fast"`` lists legal moves with the cached Python generator
    (bit-identical to the Java-preview generator; tested). Moves are still
    played, validated and scored by the Java engine.
    """

    def __init__(self, words: frozenset[str], max_turns: int = 200,
                 particles: int = 32, rng: np.random.Generator | None = None,
                 belief_likelihood=None, movegen: str = "reference"):
        if movegen not in ("fast", "reference"):
            raise ValueError("movegen must be fast or reference")
        self._env = ScrabbleEnv(words=words, max_turns=max_turns)
        self._movegen = movegen
        self._board_cache = None
        if movegen == "fast":
            from rackzero.game.movegen import BoardCache
            self._board_cache = BoardCache(self._env.words, *premium_multipliers())
        self.history: tuple[PublicEvent, ...] = ()
        self._particles = particles
        self._rng = rng if rng is not None else np.random.default_rng(0)
        self._external_rng = rng is not None
        self._likelihood = belief_likelihood
        self._beliefs = []

    def reset(self, seed: int, rng: np.random.Generator | None = None) -> InformationState:
        """Start a new game; ``rng`` (optional) replaces the belief random stream.

        Reusing one ``Game`` across games avoids rebuilding the lexicon trie.
        """
        self._env.reset(seed=seed)
        self.history = ()
        if rng is not None:
            self._rng, self._external_rng = rng, True
        elif not self._external_rng:
            self._rng = np.random.default_rng(seed)
        from rackzero.belief import RackBelief
        # Each observer gets only its own rack, even inside the coordinator.
        seeds = self._rng.integers(0, 2**63 - 1, size=2)
        self._beliefs = [
            RackBelief(self._information_for(p), self._particles,
                       np.random.default_rng(int(seeds[p])), self._likelihood)
            for p in (0, 1)
        ]
        return self.information()

    def _information_for(self, player: int) -> InformationState:
        obs = self._env._observation()
        obs["player"] = player
        obs["rack"] = counts("".join(str(t.letter()) for t in self._env._rack(player)))
        obs["opponent_rack_size"] = len(self._env._rack(1 - player))
        return InformationState(obs, self.history)

    def information(self) -> InformationState:
        return InformationState(self._env._observation(), self.history)

    @property
    def belief(self):
        return self._beliefs[self._env._player]

    def legal_moves(self) -> tuple[ScoredMove, ...]:
        if self._movegen == "reference":
            return self._env.legal_moves()
        from rackzero.game.movegen import generate
        obs = self._env._observation()
        return generate(obs["board"], obs["board_blanks"], obs["rack"], int(obs["bag_size"]),
                        self._env._trie, self._board_cache)

    @property
    def scores(self) -> np.ndarray:
        return self._env._scores()

    def step(self, move: Move) -> tuple[InformationState, bool, bool]:
        before = self._env._observation()
        _, _, terminal, truncated, info = self._env.step(move)
        event = public_event(before, move, info["move_score"])
        self.history += (event,)
        for player, belief in enumerate(self._beliefs):
            belief.update(event, self._information_for(player),
                          move if player == event.player else None)
        return self.information(), terminal, truncated

    def close(self):
        self._env.close()


def score_placement(board: np.ndarray, blanks: np.ndarray, move: Move,
                    letter_mult: np.ndarray, word_mult: np.ndarray) -> int:
    """Pure-Python mirror of Java ``Board.preview`` scoring for a generated legal move."""
    dr, dc = DIRECTIONS[move.direction]
    pr, pc = dc, dr
    main, multiplier, cross, placed = 0, 1, 0, 0
    for i, ch in enumerate(move.word):
        r, c = move.row + dr * i, move.col + dc * i
        if board[r, c]:
            main += 0 if blanks[r, c] else int(POINTS[board[r, c] - 1])
            continue
        placed += 1
        points = 0 if i in move.blanks else int(POINTS[ALPHABET.index(ch)])
        lm, wm = int(letter_mult[r, c]), int(word_mult[r, c])
        main += points * lm
        multiplier *= wm
        existing, found = 0, False
        for sign in (-1, 1):
            ar, ac = r + sign * pr, c + sign * pc
            while 0 <= ar < 15 and 0 <= ac < 15 and board[ar, ac]:
                found = True
                existing += 0 if blanks[ar, ac] else int(POINTS[board[ar, ac] - 1])
                ar, ac = ar + sign * pr, ac + sign * pc
        if found:
            cross += (existing + points * lm) * wm
    return main * multiplier + cross + (50 if placed == 7 else 0)


@dataclass
class World:
    observation: dict
    racks: tuple[np.ndarray, np.ndarray]
    bag: np.ndarray
    history: tuple[PublicEvent, ...]
    terminal: bool = False

    def information(self) -> InformationState:
        obs = dict(self.observation)
        player = int(obs["player"])
        obs["rack"] = self.racks[player]
        obs["opponent_rack_size"] = int(self.racks[1 - player].sum())
        return InformationState(obs, self.history)


class Simulator:
    """Has a lexicon and rules, but no reference to the live Game or its hidden tiles."""

    MOVEGENS = ("fast", "reference")

    def __init__(self, words: frozenset[str], max_turns: int = 200, movegen: str = "fast"):
        if movegen not in self.MOVEGENS:
            raise ValueError(f"movegen must be one of {self.MOVEGENS}")
        self._template = ScrabbleEnv(words=words, max_turns=max_turns)
        self._Tile = jpype.JClass("com.masterchiefproject.scrabble.game.Tile")
        self._letter_mult, self._word_mult = premium_multipliers()
        self.movegen = movegen
        self.stats = {"movegen_calls": 0, "movegen_seconds": 0.0,
                      "step_calls": 0, "step_seconds": 0.0}
        from rackzero.game.movegen import BoardCache
        self._board_cache = BoardCache(self._template.words, self._letter_mult, self._word_mult)

    def determinize(self, info: InformationState, opponent: np.ndarray) -> World:
        obs = info.observation
        own = np.asarray(obs["rack"], dtype=np.int64).copy()
        opponent = np.asarray(opponent)
        if (opponent.shape != (27,) or np.any(opponent < 0)
                or not np.all(np.isfinite(opponent))
                or not np.array_equal(opponent, np.floor(opponent))):
            raise ValueError("Opponent rack must contain 27 nonnegative integer counts")
        opponent = opponent.astype(np.int64)
        bag = DISTRIBUTION - board_counts(obs) - own - opponent
        if (np.any(bag < 0) or int(bag.sum()) != obs["bag_size"]
                or int(opponent.sum()) != obs["opponent_rack_size"]):
            raise ValueError("Determinization violates tile conservation")
        racks = (own, opponent) if obs["player"] == 0 else (opponent, own)
        return World(dict(obs), racks, bag, info.history)

    def _environment(self, world: World, rng: np.random.Generator) -> ScrabbleEnv:
        env = copy.copy(self._template)
        obs = world.observation
        tiles = env._List()
        for r, c in zip(*np.nonzero(obs["board"])):
            char = jpype.JChar(ALPHABET[int(obs["board"][r, c]) - 1])
            tile = (self._Tile.blankTile().assignBlank(char) if obs["board_blanks"][r, c]
                    else self._Tile.letterTile(char))
            tiles.add(env._Placement(env._Position(int(r), int(c)), tile))
        env._game = env._Game.fromSnapshot(
            env._validator, tiles, letters(world.racks[0]), letters(world.racks[1]),
            int(obs["scores"][0]), int(obs["scores"][1]), letters(world.bag),
            int(obs["player"]), int(obs["scoreless_turns"]),
            jpype.JLong(int(rng.integers(0, 2**63 - 1))),
        )
        env._player, env._turn = int(obs["player"]), int(obs["turn"])
        env._done, env._moves = False, None
        # Candidate scoring via per-move JNI previews dominates search time; transitions
        # (env.step) still go through the authoritative Java rules engine.
        board, blanks = obs["board"], obs["board_blanks"]
        env._preview = lambda move: score_placement(
            board, blanks, move, self._letter_mult, self._word_mult)
        return env

    def legal_moves(self, world: World, rng: np.random.Generator) -> tuple[ScoredMove, ...]:
        if world.terminal:
            raise ValueError("Terminal world has no moves")
        start = time.perf_counter()
        if self.movegen == "reference":
            moves = self._environment(world, rng).legal_moves()
        else:
            from rackzero.game.movegen import generate
            obs = world.observation
            moves = generate(obs["board"], obs["board_blanks"], world.racks[int(obs["player"])],
                             int(obs["bag_size"]), self._template._trie, self._board_cache)
        self.stats["movegen_calls"] += 1
        self.stats["movegen_seconds"] += time.perf_counter() - start
        return moves

    def step(self, world: World, move: Move, rng: np.random.Generator) -> World:
        if world.terminal:
            raise ValueError("Cannot step a terminal world")
        start = time.perf_counter()
        env = self._environment(world, rng)
        before = world.information().observation
        obs, _, terminal, truncated, detail = env.step(move)
        # Search horizon limits are not terminal outcomes.
        if truncated:
            env._done = False
        racks = tuple(counts("".join(str(t.letter()) for t in env._rack(p))) for p in (0, 1))
        bag = DISTRIBUTION - board_counts(obs) - racks[0] - racks[1]
        if np.any(bag < 0) or int(bag.sum()) != obs["bag_size"]:
            raise RuntimeError("Simulator lost tiles")
        self.stats["step_calls"] += 1
        self.stats["step_seconds"] += time.perf_counter() - start
        return World(obs, racks, bag,
                     world.history + (public_event(before, move, detail["move_score"]),),
                     terminal)

    def close(self):
        self._template.close()
