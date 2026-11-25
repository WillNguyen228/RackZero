"""Baselines see only the acting player's observation and legal candidates."""

import numpy as np

from rackzero.game import POINTS, used_tiles


def choose(kind, info, moves, rng: np.random.Generator, model=None):
    if not moves:
        raise ValueError("No legal candidates")
    if kind == "random":
        return moves[int(rng.integers(len(moves)))].move
    if kind == "neural":
        if model is None:
            raise ValueError("Neural agent requires a model")
        return moves[int(np.argmax(model.predict(info, moves)[0]))].move
    if kind not in ("greedy", "heuristic"):
        raise ValueError(f"Unknown agent: {kind}")
    values = []
    for candidate in moves:
        score = float(candidate.score)
        if kind == "heuristic":
            leave = info.observation["rack"] - used_tiles(info.observation, candidate.move)
            vowels = sum(leave[i] for i in (0, 4, 8, 14, 20))
            consonants = leave[:26].sum() - vowels
            score += (5 * leave[26] + 2 * leave[18] - 0.5 * (leave @ POINTS)
                      - abs(vowels - consonants) - np.maximum(leave[:26] - 1, 0).sum())
        # Prefer exchanging a full rack to passing when no placement scores.
        if candidate.move.kind == "exchange":
            score += 0.001 * len(candidate.move.exchange)
        values.append(score)
    return moves[int(np.argmax(values))].move
