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


SCHEMA = 1
STATE_SIZE = 15 * 15 * 6 + 27 + 5
ACTION_SIZE = 15 * 15 * 6 + 27 * 2 + 3 + 1
HIDDEN = (64, 32)


def board_features(cells: np.ndarray) -> np.ndarray:
    """Five letter-identity bits plus a separate assigned-blank bit per square."""
    letters = np.where(cells > 26, cells - 26, cells)
    bits = ((letters[..., None] >> np.arange(5)) & 1).astype(np.float32)
    return np.concatenate((bits, (cells > 26)[..., None]), axis=-1).reshape(*cells.shape[:-1], -1)


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
        return Candidates(*(getattr(self, name)[index:index + 1].copy()
                            for name in ("cells", "leave", "exchange", "kinds", "scores")))

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
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(STATE_SIZE + ACTION_SIZE, HIDDEN[0]), nn.ReLU(),
            nn.Linear(HIDDEN[0], HIDDEN[1]), nn.ReLU(), nn.Linear(HIDDEN[1], 1),
        )

    def forward(self, states, actions):
        return self.layers(torch.cat((states, actions), dim=-1)).squeeze(-1)


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
                 device="cpu", replay_capacity=5000, replay_mb=128):
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
        self.opponent_revision = ""

    @torch.no_grad()
    def values(self, state: np.ndarray, actions: Candidates, *, network=None) -> np.ndarray:
        network = self.online if network is None else network
        values = []
        for start in range(0, len(actions), 256):
            features = torch.from_numpy(actions.features(slice(start, start + 256))).to(self.device)
            states = torch.from_numpy(state).to(self.device).expand(len(features), -1)
            values.append(network(states, features).cpu().numpy())
        if not values:
            raise ValueError("Legal candidates must include at least pass")
        result = np.concatenate(values)
        if not np.all(np.isfinite(result)):
            raise RuntimeError("Nonfinite Q values")
        return result

    def choose(self, state, actions, epsilon=0.0):
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be from 0 to 1")
        if epsilon and self.rng.random() < epsilon:
            return self.rng.randrange(len(actions))
        return int(self.values(state, actions).argmax())

    @torch.no_grad()
    def bellman_targets(self, batch: list[Transition]) -> torch.Tensor:
        values = []
        for transition in batch:
            future = 0.0
            if not transition.terminated:
                if transition.next_actions is None:
                    raise ValueError("Nonterminal replay needs next legal actions")
                best = self.choose(transition.next_state, transition.next_actions)
                future = float(self.values(transition.next_state,
                               transition.next_actions.take(best), network=self.target)[0])
            values.append(transition.reward + self.gamma * future)
        return torch.tensor(values, dtype=torch.float32, device=self.device)

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
                    target_interval=data["target_interval"], device=device, **kwargs)
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
            "state_size": STATE_SIZE, "action_size": ACTION_SIZE,
            "layers": [{"weight": layer.weight.detach().cpu().tolist(),
                        "bias": layer.bias.detach().cpu().tolist()} for layer in layers],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False), encoding="utf-8")
        temporary.replace(path)


def dictionary_hash() -> str:
    from .quackle_server import DOCS
    return hashlib.sha256((DOCS / "words.txt").read_bytes()).hexdigest()
