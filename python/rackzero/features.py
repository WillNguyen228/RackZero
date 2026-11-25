"""Compact, observation-only position records and their batched tensor form.

A *record* is a fixed-size set of small integer/float arrays describing one
decision: the public board, the observer's own rack, public tile accounting,
an opponent-rack *belief summary*, public scalars, and up to ``K`` legal
candidates. Records are what self-play stores in the replay buffer. On the
training device they are expanded into network inputs (see ``rackzero.models``).

Information boundary: every field is derived from an ``InformationState``
(the observer's own view), the observer's belief, or a candidate move. The
builder never accepts a ``Game``, ``World`` or the opponent's actual rack.
"""

from __future__ import annotations

import numpy as np

from rackzero.game import DISTRIBUTION, InformationState, board_counts
from scrabble_gym.environment import ALPHABET, DIRECTIONS, ScoredMove


MAX_PLACED = 7
BELIEF_DIM = 54  # posterior mean (27) + P(opponent holds >= 1 of tile) (27)
# Raw public scalars: own score, opponent score, bag size, opponent rack size,
# scoreless turns, turn, and the latest opponent event's score, exchange count
# and number of tiles played.
SCALAR_DIM = 9
KINDS = ("place", "exchange", "pass")

RECORD_FIELDS = {
    # name: (per-record shape suffix, dtype). "K" means the candidate axis.
    "board": ((225,), np.int8),
    "blanks": ((225,), np.int8),
    "rack": ((27,), np.int8),
    "unseen": ((27,), np.int8),
    "belief": ((BELIEF_DIM,), np.float32),
    "scalars": ((SCALAR_DIM,), np.float32),
    "cand_cells": (("K", MAX_PLACED), np.int16),
    "cand_letters": (("K", MAX_PLACED), np.int8),
    "cand_blanks": (("K", MAX_PLACED), np.int8),
    "cand_start": (("K",), np.int16),
    "cand_dir": (("K",), np.int8),
    "cand_len": (("K",), np.int8),
    "cand_kind": (("K",), np.int8),
    "cand_score": (("K",), np.int16),
    "cand_mask": (("K",), np.bool_),
}
INDEX_FIELDS = frozenset(("board", "cand_cells", "cand_letters", "cand_start", "cand_dir",
                          "cand_len", "cand_kind"))
TARGET_FIELDS = {
    "policy": (("K",), np.float32),
    "win": ((), np.float32),
    "margin": ((), np.float32),
}


def field_shape(spec: tuple, candidates: int) -> tuple[int, ...]:
    return tuple(candidates if dim == "K" else dim for dim in spec)


def unseen_counts(info: InformationState) -> np.ndarray:
    obs = info.observation
    return DISTRIBUTION - board_counts(obs) - np.asarray(obs["rack"], dtype=np.int64)


def _p_none(tiles: np.ndarray, held: int) -> np.ndarray:
    """P(a uniform ``held``-subset of the multiset ``tiles`` has none of letter x)."""
    size = int(tiles.sum())
    result = np.ones(27, dtype=np.float64)
    for i in range(held):
        result *= np.clip(size - tiles - i, 0, None) / max(1, size - i)
    return result


def belief_summary(info: InformationState, belief=None) -> np.ndarray:
    """Posterior opponent-rack mean and P(holds >= 1) for every tile type.

    ``belief`` is the observer's ``RackBelief`` (Rao-Blackwellized groups). With
    no belief, the tile-conserving uniform prior over the observer's unseen
    tiles is used, so baselines and search nodes without a posterior still get
    well-defined features. The actual hidden rack is never an input.
    """
    if belief is None:
        unseen = unseen_counts(info)
        held = int(info.observation["opponent_rack_size"])
        total = max(1, int(unseen.sum()))
        mean = unseen * held / total
        p_any = 1.0 - _p_none(unseen, held)
    else:
        mean = belief.expected()
        p_none = np.zeros(27, dtype=np.float64)
        for weight, groups in zip(belief.weights, belief.particles):
            prob = np.ones(27, dtype=np.float64)
            for tiles, held in groups:
                if held:
                    prob *= _p_none(np.asarray(tiles), int(held))
            p_none += weight * prob
        p_any = 1.0 - p_none
    return np.concatenate((mean, p_any)).astype(np.float32)


def belief_from_mean(info: InformationState, mean: np.ndarray | None) -> np.ndarray:
    """Belief features when only an opponent-rack mean is available.

    Used for search nodes that carry a particle mean but no group structure.
    P(holds >= 1) uses the independent-slot approximation 1 - (1 - mean/n)^n.
    """
    if mean is None:
        return belief_summary(info)
    mean = np.asarray(mean, dtype=np.float64)
    if mean.shape == (BELIEF_DIM,):
        return mean.astype(np.float32)
    held = max(1, int(info.observation["opponent_rack_size"]))
    p_any = 1.0 - np.power(np.clip(1.0 - mean / held, 0.0, 1.0), held)
    return np.concatenate((mean, p_any)).astype(np.float32)


def public_scalars(info: InformationState) -> np.ndarray:
    obs = info.observation
    player = int(obs["player"])
    last = next((event for event in reversed(info.history) if event.player != player), None)
    return np.array([
        obs["scores"][player], obs["scores"][1 - player], obs["bag_size"],
        obs["opponent_rack_size"], obs["scoreless_turns"], obs["turn"],
        last.score if last else 0, last.exchange_count if last else 0,
        sum(last.played) if last else 0,
    ], dtype=np.float32)


def encode_record(info: InformationState, moves: tuple[ScoredMove, ...], belief: np.ndarray,
                  candidates: int) -> dict[str, np.ndarray]:
    """Fill one record with up to ``candidates`` legal moves (padding masked)."""
    if not moves:
        raise ValueError("A nonempty legal candidate set is required")
    if len(moves) > candidates:
        raise ValueError(f"{len(moves)} candidates exceed the record capacity {candidates}")
    belief = np.asarray(belief, dtype=np.float32)
    if belief.shape != (BELIEF_DIM,):
        raise ValueError(f"belief features must have shape ({BELIEF_DIM},)")
    obs = info.observation
    board = np.asarray(obs["board"])
    record = {name: np.zeros(field_shape(shape, candidates), dtype=dtype)
              for name, (shape, dtype) in RECORD_FIELDS.items()}
    record["board"][:] = board.reshape(-1)
    record["blanks"][:] = np.asarray(obs["board_blanks"]).reshape(-1)
    record["rack"][:] = obs["rack"]
    record["unseen"][:] = unseen_counts(info)
    record["belief"][:] = belief
    record["scalars"][:] = public_scalars(info)
    record["cand_cells"][:] = -1
    for k, candidate in enumerate(moves):
        move = candidate.move
        record["cand_kind"][k] = KINDS.index(move.kind)
        record["cand_score"][k] = candidate.score
        record["cand_mask"][k] = True
        if move.kind == "place":
            dr, dc = DIRECTIONS[move.direction]
            record["cand_start"][k] = move.row * 15 + move.col
            record["cand_dir"][k] = move.direction
            record["cand_len"][k] = len(move.word)
            slot = 0
            for i, letter in enumerate(move.word):
                r, c = move.row + dr * i, move.col + dc * i
                if board[r, c]:
                    continue
                record["cand_cells"][k, slot] = r * 15 + c
                record["cand_letters"][k, slot] = ALPHABET.index(letter) + 1
                record["cand_blanks"][k, slot] = i in move.blanks
                slot += 1
        elif move.kind == "exchange":
            for slot, letter in enumerate(move.exchange):
                record["cand_letters"][k, slot] = ALPHABET.index(letter) + 1
    return record


def pad_record(record: dict[str, np.ndarray], candidates: int) -> dict[str, np.ndarray]:
    """Widen a record's candidate axis to ``candidates`` (new slots are masked padding)."""
    current = len(record["cand_mask"])
    if current == candidates:
        return record
    if current > candidates:
        raise ValueError("Cannot shrink a record's candidate axis")
    result = dict(record)
    for name, array in record.items():
        spec = RECORD_FIELDS.get(name, TARGET_FIELDS.get(name, ((),)))[0]
        if spec and spec[0] == "K":
            width = [(0, candidates - current)] + [(0, 0)] * (array.ndim - 1)
            result[name] = np.pad(array, width, constant_values=-1 if name == "cand_cells" else 0)
    return result


def stack_records(records: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    width = max(len(r["cand_mask"]) for r in records)
    records = [pad_record(r, width) for r in records]
    return {name: np.stack([r[name] for r in records]) for name in records[0]}


def to_tensors(batch: dict[str, np.ndarray], device, non_blocking: bool = False) -> dict:
    """NumPy batch -> torch tensors on ``device`` (integer fields widened)."""
    import torch

    result = {}
    for name, array in batch.items():
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if tensor.dtype in (torch.int8, torch.int16):
            tensor = tensor.to(torch.int64) if name in INDEX_FIELDS else tensor.float()
        if device is not None and str(device) != "cpu":
            tensor = tensor.to(device, non_blocking=non_blocking)
        result[name] = tensor
    return result
