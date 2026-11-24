"""Legal-candidate Double DQN; no fixed or state-dependent action-index outputs."""

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from .environment import ALPHABET, DIRECTIONS, ScoredMove


SCHEMA = 3
CELL_SIZE = 35
STATE_SIZE = 15 * 15 * CELL_SIZE + 27 + 5
ACTION_SIZE = 15 * 15 * CELL_SIZE + 27 * 2 + 3 + 1
HIDDEN = (256, 128, 64)


def _square_features():
    rows, cols = np.indices((15, 15))
    layout = np.stack((rows / 14, cols / 14, np.zeros_like(rows),
                       np.zeros_like(rows)), axis=-1).astype(np.float32)
    premiums = {
        (3, 0): [(0, 0), (0, 7), (0, 14), (7, 0), (7, 14), (14, 0), (14, 7), (14, 14)],
        (2, 0): [(i, j) for n in range(1, 5)
                 for i, j in ((n, n), (n, 14-n), (14-n, n), (14-n, 14-n))] + [(7, 7)],
        (3, 1): [(1, 5), (1, 9), (5, 1), (5, 5), (5, 9), (5, 13),
                 (9, 1), (9, 5), (9, 9), (9, 13), (13, 5), (13, 9)],
        (2, 1): [(0, 3), (0, 11), (2, 6), (2, 8), (3, 0), (3, 7), (3, 14),
                 (6, 2), (6, 6), (6, 8), (6, 12), (7, 3), (7, 11), (8, 2),
                 (8, 6), (8, 8), (8, 12), (11, 0), (11, 7), (11, 14),
                 (12, 6), (12, 8), (14, 3), (14, 11)],
    }
    for (multiplier, letter), squares in premiums.items():
        for row, col in squares:
            layout[row, col, 3 if letter else 2] = (multiplier - 1) / 2
    return layout


SQUARE_FEATURES = _square_features()


def board_features(cells: np.ndarray) -> np.ndarray:
    """One-hot letters, blanks, coordinates, N/S/W/E occupancy and premiums."""
    grid = cells.reshape(*cells.shape[:-1], 15, 15)
    letters = np.where(grid > 26, grid - 26, grid)
    occupied = grid != 0
    neighbors = np.zeros((*grid.shape, 4), dtype=np.float32)
    neighbors[..., 1:, :, 0] = occupied[..., :-1, :]
    neighbors[..., :-1, :, 1] = occupied[..., 1:, :]
    neighbors[..., :, 1:, 2] = occupied[..., :, :-1]
    neighbors[..., :, :-1, 3] = occupied[..., :, 1:]
    squares = SQUARE_FEATURES * occupied[..., None]
    features = np.concatenate((
        (letters[..., None] == np.arange(1, 27)).astype(np.float32),
        (grid > 26)[..., None], squares[..., :2], neighbors, squares[..., 2:],
    ), axis=-1)
    return features.reshape(*cells.shape[:-1], -1)


def encode_state(observation: dict) -> np.ndarray:
    cells = (observation["board"] + 26 * observation["board_blanks"]).reshape(225)
    player = observation["player"]
    scores = observation["scores"]
    return np.concatenate((
        board_features(cells), np.asarray(observation["rack"], dtype=np.float32) / 7,
        [scores[player] / 1000, scores[1 - player] / 1000,
         observation["bag_size"] / 100, observation["opponent_rack_size"] / 7,
         observation["scoreless_turns"] / 6],
    )).astype(np.float32)


@dataclass
class Candidates:
    cells: np.ndarray
    leave: np.ndarray
    exchange: np.ndarray
    kinds: np.ndarray
    scores: np.ndarray

    @classmethod
    def from_moves(cls, observation: dict, moves: tuple[ScoredMove, ...]):
        count = len(moves)
        cells = np.zeros((count, 225), dtype=np.uint8)
        leave = np.tile(observation["rack"], (count, 1)).astype(np.uint8)
        exchange = np.zeros((count, 27), dtype=np.uint8)
        kinds = np.empty(count, dtype=np.uint8)
        scores = np.empty(count, dtype=np.float32)
        for i, candidate in enumerate(moves):
            move = candidate.move
            kinds[i] = ("place", "exchange", "pass").index(move.kind)
            scores[i] = candidate.score
            if move.kind == "exchange":
                for letter in move.exchange:
                    exchange[i, ALPHABET.index(letter)] += 1
                leave[i] -= exchange[i]
            elif move.kind == "place":
                dr, dc = DIRECTIONS[move.direction]
                for offset, letter in enumerate(move.word):
                    r, c = move.row + dr * offset, move.col + dc * offset
                    if observation["board"][r, c]:
                        continue
                    blank = offset in move.blanks
                    cells[i, r * 15 + c] = ALPHABET.index(letter) + 1 + 26 * blank
                    leave[i, 26 if blank else ALPHABET.index(letter)] -= 1
        return cls(cells, leave, exchange, kinds, scores)

    def __len__(self):
        return len(self.kinds)

    def features(self, indexes) -> np.ndarray:
        return np.concatenate((
            board_features(self.cells[indexes]), self.leave[indexes].astype(np.float32) / 7,
            self.exchange[indexes].astype(np.float32) / 7,
            np.eye(3, dtype=np.float32)[self.kinds[indexes]],
            self.scores[indexes, None] / 100,
        ), axis=-1)

    def take(self, index: int):
        return self.subset([index])

    def subset(self, indexes):
        return Candidates(*(getattr(self, name)[indexes].copy()
                            for name in ("cells", "leave", "exchange", "kinds", "scores")))

    def top(self, count: int | None):
        """Highest-scoring candidates plus pass: the DQN's acting and Bellman max set."""
        keep = self.top_indexes(count)
        return self if keep is None else self.subset(keep)

    def top_indexes(self, count: int | None):
        if count is None or len(self) <= count:
            return None
        order = np.lexsort((self.kinds, -self.scores))[:count]
        passes = np.flatnonzero(self.kinds == 2)
        if len(passes) and passes[0] not in order:
            order[-1] = passes[0]
        return np.sort(order)

    def tensors(self):
        return {name: torch.from_numpy(getattr(self, name))
                for name in ("cells", "leave", "exchange", "kinds", "scores")}

    @classmethod
    def from_tensors(cls, data):
        return cls(**{name: value.numpy() for name, value in data.items()})

    @property
    def nbytes(self):
        return sum(array.nbytes for array in (
            self.cells, self.leave, self.exchange, self.kinds, self.scores
        ))


class QNetwork(nn.Module):
    """Q = immediate move score / 100 (the action's last feature) + learned residual.

    The residual starts at zero, so an untrained network ranks moves by score;
    Bellman updates learn leave, defensive and future value on top of it.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(STATE_SIZE + ACTION_SIZE, HIDDEN[0]), nn.ReLU(),
            nn.Linear(HIDDEN[0], HIDDEN[1]), nn.ReLU(),
            nn.Linear(HIDDEN[1], HIDDEN[2]), nn.ReLU(), nn.Linear(HIDDEN[2], 1),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, states, actions):
        return self.layers(torch.cat((states, actions), dim=-1)).squeeze(-1) + actions[..., -1]


@dataclass
class Transition:
    state: np.ndarray
    action: Candidates
    reward: float
    next_state: np.ndarray
    next_actions: Candidates | None
    terminated: bool

    @property
    def nbytes(self):
        return (self.state.nbytes + self.next_state.nbytes + self.action.nbytes
                + (0 if self.next_actions is None else self.next_actions.nbytes))


class Replay:
    def __init__(self, capacity=5000, max_bytes=128 * 1024 * 1024):
        if capacity < 1 or max_bytes < 1:
            raise ValueError("Replay limits must be positive")
        self.capacity, self.max_bytes = capacity, max_bytes
        self.items: deque[Transition] = deque()
        self.bytes = 0

    def add(self, transition: Transition):
        if transition.nbytes > self.max_bytes:
            raise ValueError("A transition exceeds the replay memory budget; increase --replay-mb")
        while self.items and (len(self.items) >= self.capacity
                              or self.bytes + transition.nbytes > self.max_bytes):
            self.bytes -= self.items.popleft().nbytes
        self.items.append(transition)
        self.bytes += transition.nbytes

    def sample(self, size, rng):
        return rng.sample(list(self.items), size)

    def __len__(self):
        return len(self.items)


class DoubleDQN:
    def __init__(self, *, seed=42, learning_rate=0.001, gamma=0.99, target_interval=100,
                 device="cpu", replay_capacity=5000, replay_mb=128, top_candidates=None):
        if (not 0 <= gamma <= 1 or target_interval < 1
                or not math.isfinite(learning_rate) or learning_rate <= 0):
            raise ValueError("Invalid DQN hyperparameters")
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.online = QNetwork().to(self.device)
        self.target = QNetwork().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.gamma, self.target_interval = gamma, target_interval
        self.replay = Replay(replay_capacity, replay_mb * 1024 * 1024)
        self.rng = random.Random(seed)
        self.updates = 0
        self.decisions = 0
        self.episodes = 0
        self.seed = seed
        self.learning_rate = learning_rate
        self.top_candidates = top_candidates
        self.opponent_revision = ""

    @staticmethod
    def _split(network):
        first = network.layers[0]
        return (first.weight[:, :STATE_SIZE].contiguous(),
                first.weight[:, STATE_SIZE:].t().contiguous(), first.bias, network.layers[1:])

    @torch.no_grad()
    def values(self, state: np.ndarray, actions: Candidates, *, network=None, split=None) -> np.ndarray:
        state_weight, action_weight, bias, rest = split or self._split(
            self.online if network is None else network)
        values = []
        state_tensor = torch.from_numpy(state).to(self.device)
        shared = nn.functional.linear(state_tensor, state_weight, bias)
        for start in range(0, len(actions), 256):
            features = torch.from_numpy(actions.features(slice(start, start + 256))).to(self.device)
            # Candidate features are mostly zero; do not repeat the state projection.
            hidden = torch.sparse.mm(features.to_sparse(), action_weight) + shared
            values.append((rest(hidden).squeeze(-1) + features[:, -1]).cpu().numpy())
        if not values:
            raise ValueError("Legal candidates must include at least pass")
        result = np.concatenate(values)
        if not np.all(np.isfinite(result)):
            raise RuntimeError("Nonfinite Q values")
        return result

    def choose(self, state, actions, epsilon=0.0, *, focused=0.0):
        """Epsilon-greedy; a `focused` share of exploration samples the top-10 scoring moves."""
        if not 0 <= epsilon <= 1 or not 0 <= focused <= 1:
            raise ValueError("epsilon and focused must be from 0 to 1")
        if epsilon and self.rng.random() < epsilon:
            if focused and self.rng.random() < focused:
                best = np.lexsort((actions.kinds, -actions.scores))[:10]
                return int(best[self.rng.randrange(len(best))])
            return self.rng.randrange(len(actions))
        return int(self.values(state, actions).argmax())

    @torch.no_grad()
    def _batch_values(self, split, states: np.ndarray, groups: list[Candidates]) -> torch.Tensor:
        """Q for every candidate of every state in one sparse first-layer product."""
        state_weight, action_weight, bias, rest = split
        shared = nn.functional.linear(torch.from_numpy(states).to(self.device), state_weight, bias)
        owners = torch.repeat_interleave(torch.arange(len(groups), device=self.device),
                                         torch.tensor([len(g) for g in groups], device=self.device))
        parts = [torch.from_numpy(g.features(slice(None))).to_sparse() for g in groups]
        hidden = torch.sparse.mm(torch.cat(parts).to(self.device), action_weight) + shared[owners]
        prior = torch.from_numpy(np.concatenate([g.scores for g in groups]) / 100).to(self.device)
        return rest(hidden).squeeze(-1) + prior

    @torch.no_grad()
    def bellman_targets(self, batch: list[Transition]) -> torch.Tensor:
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        live = [i for i, t in enumerate(batch) if not t.terminated]
        if any(batch[i].next_actions is None for i in live):
            raise ValueError("Nonterminal replay needs next legal actions")
        if not live:
            return rewards
        states = np.stack([batch[i].next_state for i in live])
        groups = [batch[i].next_actions for i in live]
        online = self._batch_values(self._split(self.online), states, groups)
        if not torch.all(torch.isfinite(online)):
            raise RuntimeError("Nonfinite Q values")
        chosen, start = [], 0
        for group in groups:
            chosen.append(group.take(int(online[start:start + len(group)].argmax())))
            start += len(group)
        future = self._batch_values(self._split(self.target), states, chosen)
        rewards[live] += self.gamma * future
        return rewards

    def learn(self, batch_size=32):
        if len(self.replay) < batch_size:
            return None
        batch = self.replay.sample(batch_size, self.rng)
        targets = self.bellman_targets(batch)
        states = torch.from_numpy(np.stack([t.state for t in batch])).to(self.device)
        actions = torch.from_numpy(np.concatenate([t.action.features(slice(None)) for t in batch])).to(self.device)
        predictions = self.online(states, actions)
        loss = nn.functional.smooth_l1_loss(predictions, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite DQN loss")
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10, error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.target_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach())

    def save(self, path: Path, dictionary_sha256: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema": SCHEMA, "dictionary_sha256": dictionary_sha256,
            "online": self.online.state_dict(), "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(), "gamma": self.gamma,
            "target_interval": self.target_interval, "learning_rate": self.learning_rate,
            "top_candidates": self.top_candidates,
            "updates": self.updates, "decisions": self.decisions, "episodes": self.episodes,
            "seed": self.seed, "rng": self.rng.getstate(), "torch_rng": torch.get_rng_state(),
            "opponent_revision": self.opponent_revision,
            "replay": [{
                "state": torch.from_numpy(t.state), "action": t.action.tensors(),
                "reward": t.reward, "next_state": torch.from_numpy(t.next_state),
                "next_actions": None if t.next_actions is None else t.next_actions.tensors(),
                "terminated": t.terminated,
            } for t in self.replay.items],
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(data, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path, dictionary_sha256: str, *, device="cpu", **kwargs):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data["schema"] != SCHEMA or data["dictionary_sha256"] != dictionary_sha256:
            raise ValueError("Incompatible checkpoint schema or dictionary; do not reuse this model")
        agent = cls(seed=data["seed"], learning_rate=data["learning_rate"], gamma=data["gamma"],
                    target_interval=data["target_interval"], device=device,
                    top_candidates=data.get("top_candidates"), **kwargs)
        agent.online.load_state_dict(data["online"])
        agent.target.load_state_dict(data["target"])
        agent.optimizer.load_state_dict(data["optimizer"])
        for name in ("updates", "decisions", "episodes"):
            setattr(agent, name, data[name])
        agent.rng.setstate(data["rng"])
        torch.set_rng_state(data["torch_rng"])
        agent.opponent_revision = data["opponent_revision"]
        for item in data["replay"]:
            agent.replay.add(Transition(
                item["state"].numpy(), Candidates.from_tensors(item["action"]), item["reward"],
                item["next_state"].numpy(),
                None if item["next_actions"] is None else Candidates.from_tensors(item["next_actions"]),
                item["terminated"],
            ))
        return agent

    def export(self, path: Path, dictionary_sha256: str):
        layers = [layer for layer in self.online.layers if isinstance(layer, nn.Linear)]
        data = {
            "schema": SCHEMA, "dictionary_sha256": dictionary_sha256,
            "algorithm": "Double DQN", "opponent": "Quackle static score-plus-leave",
            "opponent_revision": self.opponent_revision,
            "episodes": self.episodes, "updates": self.updates,
            "top_candidates": self.top_candidates,
            "state_size": STATE_SIZE, "action_size": ACTION_SIZE,
            "layers": [{"weight": layer.weight.detach().cpu().tolist(),
                        "bias": layer.bias.detach().cpu().tolist()} for layer in layers],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False), encoding="utf-8")
        temporary.replace(path)


def dictionary_hash(word_list: Path | None = None) -> str:
    from .quackle_server import DOCS
    return hashlib.sha256((word_list or DOCS / "words.txt").read_bytes()).hexdigest()
