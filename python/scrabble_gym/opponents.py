"""Public-position adapters and learner-to-learner turns against native Quackle."""

import numpy as np

from .environment import ALPHABET, DIRECTIONS, Move, ScrabbleEnv
from .quackle_server import POINTS, QuackleBridge


def public_position(observation: dict) -> dict:
    board = []
    for r in range(15):
        row = []
        for c in range(15):
            code = int(observation["board"][r, c])
            letter = ALPHABET[code - 1] if code else None
            blank = bool(observation["board_blanks"][r, c])
            row.append(None if letter is None else {
                "letter": letter, "points": 0 if blank else POINTS[letter], "blank": blank,
            })
        board.append(row)
    rack = [{"letter": letter, "points": POINTS[letter], "blank": letter == "?"}
            for letter, count in zip(ALPHABET, observation["rack"]) for _ in range(int(count))]
    return {"board": board, "rack": rack, "bagSize": int(observation["bag_size"])}


def native_move(action: dict, position: dict) -> Move:
    """Translate physical placements into a complete Gym main word."""
    if action["kind"] == "pass":
        return Move()
    if action["kind"] == "exchange":
        return Move("exchange", exchange="".join(
            position["rack"][i]["letter"] for i in action["rackIndexes"]
        ))
    placed = {(p["row"], p["col"]): p["tile"] for p in action["placements"]}
    row, col = min(placed)
    direction = int(len({r for r, _ in placed}) > 1)
    if len(placed) == 1:
        horizontal = any(0 <= col + d < 15 and position["board"][row][col + d]
                         for d in (-1, 1))
        direction = int(not horizontal)
    dr, dc = DIRECTIONS[direction]
    while 0 <= row - dr < 15 and 0 <= col - dc < 15 and position["board"][row - dr][col - dc]:
        row, col = row - dr, col - dc
    start = row, col
    word, blanks = "", []
    while 0 <= row < 15 and 0 <= col < 15:
        tile = placed.get((row, col)) or position["board"][row][col]
        if tile is None:
            break
        if (row, col) in placed and tile["blank"]:
            blanks.append(len(word))
        word += tile["letter"]
        row, col = row + dr, col + dc
    return Move("place", *start, direction, word, tuple(blanks))


class QuackleMatch:
    """One step ends at the learner's next decision, never the opponent's."""

    def __init__(self, *, bridge: QuackleBridge | None = None, max_decisions: int = 100):
        if max_decisions < 1:
            raise ValueError("max_decisions must be positive")
        self.bridge = bridge if bridge is not None else QuackleBridge()
        self.env = ScrabbleEnv(max_turns=2 * max_decisions + 2)
        expected = frozenset(
            w for w in self.bridge_words().split()
            if w.isascii() and w.isalpha() and w.isupper() and 2 <= len(w) <= 15
        )
        if self.env.words != expected:
            self.env.close()
            raise ValueError("Java and Quackle dictionaries differ; rebuild both engines")
        self.max_decisions = max_decisions
        self.learner = 0
        self.decisions = 0
        self.done = True

    @staticmethod
    def bridge_words() -> str:
        from .quackle_server import DOCS
        return (DOCS / "words.txt").read_text(encoding="utf-8")

    def _opponent_step(self, observation):
        position = public_position(observation)
        action = self.bridge.choose(position)
        result = self.env.step(native_move(action, position))
        if action["kind"] == "place" and result[4]["move_score"] != action["score"]:
            raise RuntimeError("Quackle and Java scores disagree")
        return result

    def reset(self, *, seed: int, learner: int = 0):
        if learner not in (0, 1):
            raise ValueError("learner must be 0 or 1")
        self.learner, self.decisions, self.done = learner, 0, False
        observation, info = self.env.reset(seed=seed)
        if learner == 1:
            observation, _, terminated, truncated, info = self._opponent_step(observation)
            if terminated or truncated:
                raise RuntimeError("Quackle ended a game before the learner's first turn")
        return observation, info

    def legal_moves(self):
        if self.done:
            raise RuntimeError("Reset the match before requesting moves")
        return self.env.legal_moves()

    def step(self, move: Move):
        if self.done:
            raise RuntimeError("Reset the match before stepping")
        before = self.env._scores()
        observation, _, terminated, truncated, info = self.env.step(move)
        if not (terminated or truncated):
            observation, _, terminated, truncated, info = self._opponent_step(observation)
        if truncated:
            raise RuntimeError("Underlying turn limit reached before a complete learner transition")
        self.decisions += 1
        truncated = not terminated and self.decisions >= self.max_decisions
        self.done = terminated or truncated
        delta = np.asarray(info["scores"]) - before
        reward = float(delta[self.learner] - delta[1 - self.learner]) / 100.0
        info = {**info, "learner": self.learner, "decisions": self.decisions}
        if not terminated and observation["player"] != self.learner:
            raise RuntimeError("Nonterminal transition must return the learner's observation")
        return observation, reward, terminated, truncated, info

    def close(self):
        self.env.close()
