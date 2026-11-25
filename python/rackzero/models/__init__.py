"""Variable-candidate PyTorch policy and two information-state value heads."""

import numpy as np
import torch
from torch import nn

from rackzero.belief import unseen_tiles
from rackzero.game import InformationState, used_tiles
from scrabble_gym.environment import ALPHABET, DIRECTIONS, ScoredMove


STATE_DIM = 225 * 28 + 27 * 3 + 6
ACTION_DIM = 225 * 2 + 27 * 3 + 3 + 5


def encode_state(info: InformationState, belief_mean: np.ndarray | None = None) -> np.ndarray:
    obs = info.observation
    board = obs["board"]
    board_features = np.concatenate((
        (board[..., None] == np.arange(27)).astype(np.float32),
        obs["board_blanks"][..., None],
    ), axis=-1).reshape(-1)
    unseen = unseen_tiles(info)
    if belief_mean is None:
        belief_mean = (unseen * obs["opponent_rack_size"] / max(1, int(unseen.sum())))
    player = obs["player"]
    return np.concatenate((
        board_features, obs["rack"] / 7, unseen / 12, belief_mean / 7,
        [obs["scores"][player] / 100, obs["scores"][1 - player] / 100,
         obs["bag_size"] / 100, obs["opponent_rack_size"] / 7,
         obs["scoreless_turns"] / 6, obs["turn"] / 200],
    )).astype(np.float32)


def encode(info: InformationState, moves: tuple[ScoredMove, ...],
           belief_mean: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    obs = info.observation
    board = obs["board"]
    state = encode_state(info, belief_mean)
    actions = []
    for candidate in moves:
        move = candidate.move
        placed = np.zeros((15, 15), dtype=np.float32)
        blanks = np.zeros((15, 15), dtype=np.float32)
        if move.kind == "place":
            dr, dc = DIRECTIONS[move.direction]
            for i, letter in enumerate(move.word):
                r, c = move.row + dr * i, move.col + dc * i
                if not board[r, c]:
                    placed[r, c] = (ALPHABET.index(letter) + 1) / 26
                    blanks[r, c] = i in move.blanks
        spent = used_tiles(obs, move)
        leave = obs["rack"] - spent
        kind = np.eye(3)[("place", "exchange", "pass").index(move.kind)]
        actions.append(np.concatenate((
            placed.reshape(-1), blanks.reshape(-1), spent / 7, leave / 7,
            np.array([move.exchange.count(ch) for ch in ALPHABET]) / 7, kind,
            [candidate.score / 100, move.row / 14, move.col / 14,
             move.direction, len(move.word) / 15],
        )))
    if not actions:
        raise ValueError("A nonempty legal candidate set is required")
    return state, np.asarray(actions, dtype=np.float32)


class PolicyValueNet(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.hidden = hidden
        self.state_encoder = nn.Sequential(nn.Linear(STATE_DIM, hidden), nn.ReLU(),
                                           nn.Linear(hidden, hidden), nn.ReLU())
        self.action_encoder = nn.Sequential(nn.Linear(ACTION_DIM, hidden), nn.ReLU())
        self.policy = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.win = nn.Linear(hidden, 1)
        self.margin = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor, actions: torch.Tensor):
        embedding = self.state_encoder(state)
        candidates = self.action_encoder(actions)
        if state.ndim == 1:
            context = embedding.expand(actions.shape[0], -1)
        else:
            context = embedding.unsqueeze(-2).expand(*candidates.shape[:-1], embedding.shape[-1])
        logits = self.policy(torch.cat((context, candidates), dim=-1)).squeeze(-1)
        return logits, self.win(embedding).squeeze(-1).tanh(), self.margin(embedding).squeeze(-1)

    @torch.no_grad()
    def predict(self, info, moves, belief_mean=None):
        state, actions = encode(info, moves, belief_mean)
        device = next(self.parameters()).device
        logits, value, margin = self(torch.from_numpy(state).to(device),
                                     torch.from_numpy(actions).to(device))
        return logits.softmax(-1).cpu().numpy(), float(value.item()), float(margin.item()) * 100

    @torch.no_grad()
    def predict_value(self, info, belief_mean=None):
        """Value heads only, so leaf evaluation needs no legal-move generation."""
        state = torch.from_numpy(encode_state(info, belief_mean)).to(next(self.parameters()).device)
        embedding = self.state_encoder(state)
        return float(self.win(embedding).tanh().item()), float(self.margin(embedding).item()) * 100
