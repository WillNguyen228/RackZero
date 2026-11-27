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
                          "cand_len", "cand_kind", "action"))
TARGET_FIELDS = {
    "policy": (("K",), np.float32),
    "win": ((), np.float32),
    "margin": ((), np.float32),
    # Root value backed up by search (player-to-move view); mixed into the value
    # target with training.value_target_q_weight.
    "search_value": ((), np.float32),
    # 1 if ``policy`` is a training target (full search / teacher), 0 otherwise
    # (playout cap randomization's fast searches).
    "policy_weight": ((), np.float32),
    # TD(lambda) move-value target for the candidate actually played: spread-to-go
    # / 100 (q_weight 1), or no Q target (q_weight 0, e.g. teacher/older records).
    "action": ((), np.int16),
    "q_target": ((), np.float32),
    "q_weight": ((), np.float32),
}


def fill_target_defaults(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Add targets missing from older records so they train exactly as before:
    search_value = win (q-mixing has no effect) and policy_weight = 1."""
    batch = dict(batch)
    if "search_value" not in batch and "win" in batch:
        batch["search_value"] = np.asarray(batch["win"], dtype=np.float32).copy()
    if "policy_weight" not in batch and "policy" in batch:
        batch["policy_weight"] = np.ones(len(batch["policy"]), dtype=np.float32)
    if "policy" in batch:
        n = len(batch["policy"])
        for name, dtype in (("action", np.int16), ("q_target", np.float32),
                            ("q_weight", np.float32)):
            if name not in batch:
                batch[name] = np.zeros(n, dtype=dtype)
    return batch


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


def record_moves(record: dict[str, np.ndarray]) -> tuple[ScoredMove, ...]:
    """The candidate moves stored in a record, in slot order (inverse of encode_record)."""
    from scrabble_gym.environment import Move

    board = np.asarray(record["board"]).reshape(15, 15)
    moves = []
    for k in np.flatnonzero(record["cand_mask"]):
        kind = KINDS[int(record["cand_kind"][k])]
        score = int(record["cand_score"][k])
        if kind == "pass":
            moves.append(ScoredMove(Move(), score))
        elif kind == "exchange":
            letters = "".join(ALPHABET[int(x) - 1] for x in record["cand_letters"][k] if x > 0)
            moves.append(ScoredMove(Move("exchange", exchange=letters), score))
        else:
            start, direction = int(record["cand_start"][k]), int(record["cand_dir"][k])
            row, col = divmod(start, 15)
            dr, dc = DIRECTIONS[direction]
            word, blanks, slot = "", [], 0
            for i in range(int(record["cand_len"][k])):
                r, c = row + dr * i, col + dc * i
                if board[r, c]:
                    word += ALPHABET[int(board[r, c]) - 1]
                    continue
                word += ALPHABET[int(record["cand_letters"][k, slot]) - 1]
                if record["cand_blanks"][k, slot]:
                    blanks.append(i)
                slot += 1
            moves.append(ScoredMove(Move("place", row, col, direction, word, tuple(blanks)),
                                    score))
    return tuple(moves)


def record_information(record: dict[str, np.ndarray]):
    """(InformationState, candidate moves) reconstructed from a stored record.

    The observer is placed in seat 0 (scores are stored as own/opponent). The
    opponent's latest public event is restored from the record's scalars so the
    network sees the same inputs; earlier history is not stored.
    """
    from rackzero.game import PublicEvent

    s = np.asarray(record["scalars"], dtype=np.float64)
    observation = {
        "board": np.asarray(record["board"], dtype=np.int8).reshape(15, 15),
        "board_blanks": np.asarray(record["blanks"], dtype=np.int8).reshape(15, 15),
        "rack": np.asarray(record["rack"], dtype=np.int64),
        "scores": np.array([int(s[0]), int(s[1])], dtype=np.int64),
        "player": 0, "opponent_rack_size": int(s[3]), "bag_size": int(s[2]),
        "scoreless_turns": int(s[4]), "turn": int(s[5]),
    }
    history = ()
    if s[6] or s[7] or s[8]:
        played = [0] * 27
        played[26] = int(s[8])   # only the count of tiles played enters the features
        kind = "place" if s[8] else ("exchange" if s[7] else "pass")
        history = (PublicEvent(1, kind, tuple(played), int(s[7]), int(s[6]), 0, 0),)
    return InformationState(observation, history), record_moves(record)


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
