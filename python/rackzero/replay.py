"""Fixed-capacity replay ring of compact records (see ``rackzero.features``).

Storage is a set of preallocated NumPy arrays, so sampling a minibatch is a
vectorized gather and the whole buffer checkpoints as a few arrays (no per-sample
pickling). Sampling is uniform with replacement, as in the original trainer.
"""

from __future__ import annotations

import numpy as np

from rackzero.features import RECORD_FIELDS, TARGET_FIELDS, field_shape, fill_target_defaults


FIELDS = {**RECORD_FIELDS, **TARGET_FIELDS}


class ReplayBuffer:
    def __init__(self, capacity: int, candidates: int):
        if capacity < 1 or candidates < 1:
            raise ValueError("Replay capacity and candidate count must be positive")
        self.capacity = int(capacity)
        self.candidates = int(candidates)
        self.arrays = {name: np.zeros((self.capacity,) + field_shape(shape, self.candidates), dtype)
                       for name, (shape, dtype) in FIELDS.items()}
        self.size = 0
        self.head = 0          # next write slot
        self.total_added = 0

    def __len__(self) -> int:
        return self.size

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.arrays.values())

    def add(self, batch: dict[str, np.ndarray]) -> None:
        """Append a stacked batch of records (oldest entries are overwritten)."""
        count = len(batch["policy"])
        if not count:
            return
        batch = fill_target_defaults(batch)
        if set(batch) != set(FIELDS):
            raise ValueError(f"Replay batch fields differ: {sorted(set(FIELDS) ^ set(batch))}")
        if count > self.capacity:
            batch = {k: v[-self.capacity:] for k, v in batch.items()}
            count = self.capacity
        slots = (self.head + np.arange(count)) % self.capacity
        for name, array in self.arrays.items():
            value = np.asarray(batch[name])
            if value.shape[1:] != array.shape[1:]:
                raise ValueError(f"Replay field {name} has shape {value.shape[1:]}, "
                                 f"expected {array.shape[1:]}")
            array[slots] = value
        self.head = int((self.head + count) % self.capacity)
        self.size = min(self.capacity, self.size + count)
        self.total_added += count

    def sample(self, rng: np.random.Generator, batch_size: int) -> dict[str, np.ndarray]:
        if not self.size:
            raise RuntimeError("No replay positions available")
        index = rng.integers(0, self.size, size=batch_size)
        return {name: array[index] for name, array in self.arrays.items()}

    def positions(self, slots: np.ndarray) -> dict[str, np.ndarray]:
        """Copies of the records stored in ``slots`` (indices into the ring)."""
        return {name: array[slots].copy() for name, array in self.arrays.items()}

    def update(self, slots: np.ndarray, values: dict[str, np.ndarray]) -> None:
        """Overwrite target fields of stored records (Reanalyse)."""
        for name, value in values.items():
            if name not in TARGET_FIELDS:
                raise ValueError(f"Only target fields can be updated, not {name!r}")
            self.arrays[name][slots] = value

    def ordered(self) -> dict[str, np.ndarray]:
        """Stored records, oldest first."""
        if self.size < self.capacity:
            return {k: v[:self.size].copy() for k, v in self.arrays.items()}
        order = (self.head + np.arange(self.capacity)) % self.capacity
        return {k: v[order] for k, v in self.arrays.items()}

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "candidates": self.candidates,
                "total_added": self.total_added, "records": self.ordered()}

    def load_state_dict(self, state: dict) -> None:
        if state["candidates"] != self.candidates:
            raise ValueError(f"Replay candidate width {state['candidates']} differs from "
                             f"the configured search.candidates {self.candidates}")
        records = state["records"]
        self.size = self.head = 0
        self.add(records)
        self.total_added = int(state["total_added"])
