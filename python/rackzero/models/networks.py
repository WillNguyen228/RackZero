"""Configurable RackZero networks over compact observation records.

Two architectures consume the same batched record tensors:

* ``legacy_mlp``: the original ``PolicyValueNet``, unchanged and fed with
  features identical to ``rackzero.models.encode``. It is kept as a baseline.
* ``resnet`` / ``cnn``: a modular Scrabble network:

  - board encoder: 15x15 planes (26 letters, blank, occupied, 4 premium types),
    stem convolution conditioned on the context vector, then residual blocks
    (``resnet``) or plain conv layers (``cnn``);
  - rack encoder: own tile counts (blank separate) -> MLP;
  - belief encoder: unseen counts, posterior opponent-rack mean and
    P(opponent holds >= 1 of each tile) -> MLP;
  - global encoder: public scores, bag/rack sizes, counters, last opponent event;
  - policy head: scores each generator-supplied legal move. Board features are
    pooled over the move's newly placed squares and over its full main word.
    These are combined with move features (kind, score, tiles, leave, blanks,
    geometry) and the shared latent. Padding candidates are masked;
  - value heads: expected outcome (``scalar`` tanh, or ``wdl`` logits) and
    normalized final score margin.

The legal-move generator remains the only source of candidates: the network
scores the candidates it is given and cannot invent or alter a move.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from functools import lru_cache

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from rackzero.features import BELIEF_DIM, MAX_PLACED, encode_record, belief_from_mean, \
    belief_summary, stack_records, to_tensors
from rackzero.models import PolicyValueNet


ARCHITECTURES = ("resnet", "cnn", "legacy_mlp")
VOWELS = (0, 4, 8, 14, 20)


@dataclass
class ModelConfig:
    arch: str = "resnet"
    channels: int = 64
    blocks: int = 6
    latent: int = 256
    rack_dim: int = 64
    belief_dim: int = 64
    global_dim: int = 32
    move_dim: int = 128
    policy_hidden: int = 256
    value_hidden: int = 256
    policy_attention_layers: int = 0
    attention_heads: int = 4
    dropout: float = 0.0
    norm: str = "batch"
    value_head: str = "scalar"
    margin_head: bool = True
    # Per-candidate move value Q(s, a): predicted spread-to-go (final margin minus
    # the current margin, / 100) after playing the candidate; trained by TD(lambda).
    q_head: bool = False
    # Afterstate head (TD-Gammon style): value of the position *after* a candidate is
    # played (mover's view, spread-to-go / 100), added to Q for the ``after_top``
    # candidates ranked highest by log prior + Q. Needs ``q_head``.
    after_head: bool = False
    after_top: int = 12
    legacy_hidden: int = 128

    def __post_init__(self):
        if self.arch not in ARCHITECTURES:
            raise ValueError(f"model.arch must be one of {ARCHITECTURES}")
        for name in ("channels", "blocks", "latent", "rack_dim", "belief_dim", "global_dim",
                     "move_dim", "policy_hidden", "value_hidden", "attention_heads",
                     "legacy_hidden", "after_top"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"model.{name} must be a positive integer")
        if (isinstance(self.policy_attention_layers, bool)
                or not isinstance(self.policy_attention_layers, int)
                or self.policy_attention_layers < 0):
            raise ValueError("model.policy_attention_layers must be a nonnegative integer")
        if self.policy_attention_layers and self.move_dim % self.attention_heads:
            raise ValueError("model.move_dim must be divisible by model.attention_heads")
        if not 0 <= float(self.dropout) < 1:
            raise ValueError("model.dropout must be in [0, 1)")
        if self.norm not in ("batch", "group", "none"):
            raise ValueError("model.norm must be batch, group or none")
        if self.value_head not in ("scalar", "wdl"):
            raise ValueError("model.value_head must be scalar or wdl")
        if not isinstance(self.margin_head, bool):
            raise ValueError("model.margin_head must be a boolean")
        if self.after_head and not self.q_head:
            raise ValueError("model.after_head requires model.q_head")

    @classmethod
    def from_dict(cls, values: dict) -> "ModelConfig":
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown model keys: {sorted(unknown)}")
        return cls(**values)


@lru_cache(maxsize=1)
def premium_planes() -> np.ndarray:
    """[4, 15, 15] float planes: DL, TL, DW, TW, read from the Java ``Board``."""
    import jpype
    from scrabble_gym.environment import JAVA_PACKAGE, _jar_path

    jpype.addClassPath(str(_jar_path(None)))
    if not jpype.isJVMStarted():
        jpype.startJVM()
    layout = jpype.JClass(JAVA_PACKAGE + "game.Board")()
    lm = np.array([[layout.premiumAt(r, c).letterMultiplier() for c in range(15)]
                   for r in range(15)])
    wm = np.array([[layout.premiumAt(r, c).wordMultiplier() for c in range(15)]
                   for r in range(15)])
    return np.stack((lm == 2, lm == 3, wm == 2, wm == 3)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Shared tensor featurization (device side)
# --------------------------------------------------------------------------- #

def _scatter_cells(index: torch.Tensor, valid: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """[B, K, n] cell indices -> [B, K, 225] planes (invalid entries dropped)."""
    shape = index.shape[:-1] + (226,)
    out = torch.zeros(shape, dtype=values.dtype, device=index.device)
    out.scatter_add_(-1, torch.where(valid, index, torch.full_like(index, 225)),
                     torch.where(valid, values, torch.zeros_like(values)))
    return out[..., :225]


def candidate_tensors(batch: dict) -> dict:
    """Derived per-candidate tensors shared by both architectures."""
    cells, letters = batch["cand_cells"], batch["cand_letters"]
    blanks = batch["cand_blanks"]
    valid_cells = cells >= 0
    placed = _scatter_cells(cells, valid_cells, torch.ones_like(blanks))
    steps = torch.arange(15, device=cells.device)
    stride = torch.where(batch["cand_dir"] == 1, 15, 1)
    word_index = batch["cand_start"].unsqueeze(-1) + steps * stride.unsqueeze(-1)
    word_valid = steps < batch["cand_len"].unsqueeze(-1)
    word = _scatter_cells(word_index, word_valid, torch.ones(word_index.shape, device=cells.device))
    tile = torch.where(blanks > 0, torch.full_like(letters, 26), letters - 1)
    spent = torch.zeros(letters.shape[:-1] + (28,), device=cells.device)
    spent.scatter_add_(-1, torch.where(letters > 0, tile, torch.full_like(tile, 27)),
                       (letters > 0).float())
    return {"placed": placed, "word": word, "spent": spent[..., :27],
            "valid_cells": valid_cells, "tile": tile}


def afterstate_batch(batch: dict, index: torch.Tensor) -> dict:
    """State inputs of the positions after playing candidates ``index`` [B, M].

    The mover keeps the perspective: placed tiles go on the board, the rack becomes
    the leave (draws are unknown), exchanged tiles return to the unseen pool and the
    move score is added to the mover's score. Returns B * M flattened records.
    """
    B, M = index.shape

    def take(name):
        x = batch[name]
        idx = index.view(B, M, *([1] * (x.dim() - 2))).expand(B, M, *x.shape[2:])
        return x.gather(1, idx).reshape(B * M, *x.shape[2:])

    def repeat(name):
        return batch[name].repeat_interleave(M, 0)

    cells, letters, cand_blanks = take("cand_cells"), take("cand_letters"), take("cand_blanks")
    kind, score = take("cand_kind"), take("cand_score")
    valid = cells >= 0
    pos = torch.where(valid, cells, torch.full_like(cells, 225)).long()

    def place(plane, values):
        padded = torch.cat((plane, plane.new_zeros(plane.shape[0], 1)), dim=1)
        values = torch.where(valid, values.to(plane.dtype), torch.zeros_like(values, dtype=plane.dtype))
        return padded.scatter(1, pos, values)[:, :225]

    board = place(repeat("board"), letters)
    blanks = place(repeat("blanks"), (cand_blanks > 0))
    tile = torch.where(cand_blanks > 0, torch.full_like(letters, 26), letters - 1).long()
    spent = torch.zeros(B * M, 28, device=letters.device)
    spent.scatter_add_(1, torch.where(letters > 0, tile, torch.full_like(tile, 27)),
                       (letters > 0).float())
    spent = spent[:, :27]
    rack = repeat("rack")
    unseen = repeat("unseen")
    scalars = repeat("scalars").clone()
    scalars[:, 0] = scalars[:, 0] + score.to(scalars.dtype)
    return {"board": board, "blanks": blanks,
            "rack": (rack - spent.to(rack.dtype)).clamp(min=0),
            "unseen": unseen + (spent * (kind == 1).unsqueeze(-1)).to(unseen.dtype),
            "belief": repeat("belief"), "scalars": scalars}


def normalized_scalars(scalars: torch.Tensor) -> torch.Tensor:
    own, opp, bag, opp_rack, scoreless, turn, last_score, last_exchange, last_played = \
        scalars.unbind(-1)
    return torch.stack((
        own / 100, opp / 100, (own - opp) / 100, bag / 100, opp_rack / 7, scoreless / 6,
        turn / 100, (bag == 0).float(), (bag < 7).float(), last_score / 50,
        last_exchange / 7, last_played / 7,
    ), dim=-1)


def _norm(kind: str, channels: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        return nn.GroupNorm(min(8, channels), channels)
    return nn.Identity()


def _mlp(sizes: list[int], dropout: float = 0.0, final_activation: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2 or final_activation:
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, norm: str, residual: bool = True):
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=norm == "none")
        self.norm1 = _norm(norm, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=norm == "none")
        self.norm2 = _norm(norm, channels)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x + y if self.residual else y)


class RackZeroNet(nn.Module):
    BOARD_PLANES = 26 + 1 + 1 + 4
    RACK_IN = 28
    BELIEF_IN = 27 + BELIEF_DIM
    GLOBAL_IN = 12
    MOVE_IN = 69

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        c = config
        self.register_buffer("premiums", torch.from_numpy(premium_planes()).unsqueeze(0))
        self.rack_encoder = _mlp([self.RACK_IN, c.rack_dim, c.rack_dim], c.dropout)
        self.belief_encoder = _mlp([self.BELIEF_IN, c.belief_dim, c.belief_dim], c.dropout)
        self.global_encoder = _mlp([self.GLOBAL_IN, c.global_dim, c.global_dim], c.dropout)
        context = c.rack_dim + c.belief_dim + c.global_dim
        self.stem = nn.Conv2d(self.BOARD_PLANES, c.channels, 3, padding=1, bias=False)
        self.stem_context = nn.Linear(context, c.channels)
        self.stem_norm = _norm(c.norm, c.channels)
        self.blocks = nn.Sequential(*(ResidualBlock(c.channels, c.norm, c.arch == "resnet")
                                      for _ in range(c.blocks)))
        self.latent = _mlp([2 * c.channels + context, c.latent], c.dropout)
        self.spatial = nn.Linear(2 * c.channels, c.move_dim)
        self.move_encoder = _mlp([self.MOVE_IN, c.move_dim, c.move_dim], c.dropout)
        self.latent_to_move = nn.Linear(c.latent, c.move_dim)
        self.attention = (nn.TransformerEncoder(
            nn.TransformerEncoderLayer(c.move_dim, c.attention_heads, 2 * c.move_dim,
                                       dropout=c.dropout, batch_first=True),
            c.policy_attention_layers, enable_nested_tensor=False,
        ) if c.policy_attention_layers else None)
        self.policy = _mlp([3 * c.move_dim, c.policy_hidden, 1], c.dropout, final_activation=False)
        self.value_trunk = _mlp([c.latent, c.value_hidden], c.dropout)
        self.value = nn.Linear(c.value_hidden, 3 if c.value_head == "wdl" else 1)
        self.margin = nn.Linear(c.value_hidden, 1) if c.margin_head else None
        self.q = None
        if c.q_head:
            self.q = _mlp([3 * c.move_dim, c.policy_hidden, 1], c.dropout, final_activation=False)
            # Zero output at initialization: a warm-started network first plays its policy.
            nn.init.zeros_(self.q[-1].weight)
            nn.init.zeros_(self.q[-1].bias)
        self.after = None
        if c.after_head:
            self.after = nn.Linear(c.value_hidden, 1)
            # Zero at initialization: the network first plays exactly as its Q head.
            nn.init.zeros_(self.after.weight)
            nn.init.zeros_(self.after.bias)

    def encode(self, batch: dict):
        board = batch["board"]
        batch_size = board.shape[0]
        letters = F.one_hot(board.clamp(0, 26), 27)[..., 1:].float()
        planes = torch.cat((
            letters, batch["blanks"].unsqueeze(-1), (board > 0).float().unsqueeze(-1),
        ), dim=-1).reshape(batch_size, 15, 15, 28).permute(0, 3, 1, 2)
        planes = torch.cat((planes, self.premiums.expand(batch_size, -1, -1, -1)), dim=1)
        rack = batch["rack"]
        rack_in = torch.cat((rack / 7, rack.sum(-1, keepdim=True) / 7), dim=-1)
        belief_in = torch.cat((batch["unseen"] / 12, batch["belief"][:, :27] / 7,
                               batch["belief"][:, 27:]), dim=-1)
        context = torch.cat((self.rack_encoder(rack_in), self.belief_encoder(belief_in),
                             self.global_encoder(normalized_scalars(batch["scalars"]))), dim=-1)
        x = self.stem(planes) + self.stem_context(context)[:, :, None, None]
        x = self.blocks(F.relu(self.stem_norm(x)))
        pooled = torch.cat((x.mean((2, 3)), x.amax((2, 3)), context), dim=-1)
        return x, self.latent(pooled)

    def move_features(self, batch: dict, derived: dict) -> torch.Tensor:
        kind = batch["cand_kind"]
        spent = derived["spent"]
        placed_count = derived["valid_cells"].sum(-1).float()
        leave = (batch["rack"].unsqueeze(1) - spent).clamp(min=0)
        vowels = leave[..., list(VOWELS)].sum(-1)
        consonants = leave[..., :26].sum(-1) - vowels
        start = batch["cand_start"]
        length = batch["cand_len"].float()
        return torch.cat((
            F.one_hot(kind, 3).float(), (batch["cand_score"] / 50).unsqueeze(-1),
            (placed_count / 7).unsqueeze(-1), (placed_count == 7).float().unsqueeze(-1),
            (batch["cand_blanks"].sum(-1) / 2).unsqueeze(-1), spent / 7, leave / 7,
            (vowels / 7).unsqueeze(-1), (consonants / 7).unsqueeze(-1),
            leave[..., 26:27], batch["cand_dir"].float().unsqueeze(-1),
            (start // 15).float().unsqueeze(-1) / 14, (start % 15).float().unsqueeze(-1) / 14,
            (length / 15).unsqueeze(-1), ((length - placed_count).clamp(min=0) / 15).unsqueeze(-1),
        ), dim=-1)

    def forward(self, batch: dict, policy: bool = True) -> dict:
        features, latent = self.encode(batch)
        hidden = self.value_trunk(latent)
        value = self.value(hidden)
        out = {}
        if self.config.value_head == "wdl":
            out["wdl_logits"] = value
            probs = value.float().softmax(-1)
            out["value"] = probs[:, 0] - probs[:, 2]
        else:
            out["value"] = value.squeeze(-1).tanh()
        out["margin"] = (self.margin(hidden).squeeze(-1) if self.margin is not None
                         else torch.zeros_like(out["value"]))
        if self.after is not None:
            out["after"] = self.after(hidden).squeeze(-1).float()
        if not policy:
            return out
        derived = candidate_tensors(batch)
        flat = features.flatten(2).transpose(1, 2)  # [B, 225, C]
        placed, word = derived["placed"], derived["word"]
        pooled = torch.cat((
            torch.bmm(placed.to(flat.dtype), flat) / placed.sum(-1, keepdim=True).clamp(min=1),
            torch.bmm(word.to(flat.dtype), flat) / word.sum(-1, keepdim=True).clamp(min=1),
        ), dim=-1)
        moves = self.move_encoder(self.move_features(batch, derived).to(pooled.dtype))
        mask = batch["cand_mask"]
        if self.attention is not None:
            moves = self.attention(moves, src_key_padding_mask=~mask)
        context = self.latent_to_move(latent).unsqueeze(1).expand_as(moves)
        joint = torch.cat((self.spatial(pooled), moves, context), dim=-1)
        logits = self.policy(joint)
        out["logits"] = logits.squeeze(-1).float().masked_fill(~mask, float("-inf"))
        if self.q is not None:
            out["q"] = self.q(joint).squeeze(-1).float().masked_fill(~mask, 0.0)
        if self.after is not None:
            if self.training:
                # Training: afterstate value of the move actually played.
                if "action" in batch:
                    action = batch["action"].long().unsqueeze(1)
                    out["after_taken"] = self.afterstate_values(batch, action)[:, 0]
            else:
                q = out["q"]
                k = min(self.config.after_top, q.shape[1])
                rank = (torch.log_softmax(out["logits"], -1) + 50 * q).masked_fill(
                    ~mask, float("-inf"))
                top = rank.topk(k, dim=-1).indices
                total = q.gather(1, top) + self.afterstate_values(batch, top)
                total = total.masked_fill(~mask.gather(1, top), float("-inf"))
                out["q_after"] = torch.full_like(q, float("-inf")).scatter(1, top, total)
        return out

    def afterstate_values(self, batch: dict, index: torch.Tensor) -> torch.Tensor:
        """Afterstate head (spread-to-go / 100, mover's view) for candidates [B, M]."""
        after = afterstate_batch(batch, index)
        return self.forward(after, policy=False)["after"].view(index.shape)


class LegacyNet(nn.Module):
    """Original ``PolicyValueNet`` behind the record interface (baseline)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.net = PolicyValueNet(hidden=config.legacy_hidden)

    @staticmethod
    def state_features(batch: dict) -> torch.Tensor:
        board = batch["board"]
        planes = torch.cat((F.one_hot(board.clamp(0, 26), 27).float(),
                            batch["blanks"].unsqueeze(-1)), dim=-1).flatten(1)
        s = batch["scalars"]
        return torch.cat((planes, batch["rack"] / 7, batch["unseen"] / 12,
                          batch["belief"][:, :27] / 7,
                          torch.stack((s[:, 0] / 100, s[:, 1] / 100, s[:, 2] / 100,
                                       s[:, 3] / 7, s[:, 4] / 6, s[:, 5] / 200), dim=-1)),
                         dim=-1)

    @staticmethod
    def action_features(batch: dict) -> torch.Tensor:
        derived = candidate_tensors(batch)
        cells, valid = batch["cand_cells"], derived["valid_cells"]
        letters = batch["cand_letters"].float()
        placed = _scatter_cells(cells, valid, letters / 26)
        blanks = _scatter_cells(cells, valid, batch["cand_blanks"])
        spent = derived["spent"]
        leave = batch["rack"].unsqueeze(1) - spent
        kind = batch["cand_kind"]
        exchange = spent * (kind == 1).float().unsqueeze(-1)
        start = batch["cand_start"]
        return torch.cat((
            placed, blanks, spent / 7, leave / 7, exchange / 7, F.one_hot(kind, 3).float(),
            torch.stack(((batch["cand_score"] / 100), (start // 15).float() / 14,
                         (start % 15).float() / 14, batch["cand_dir"].float(),
                         batch["cand_len"].float() / 15), dim=-1),
        ), dim=-1)

    def forward(self, batch: dict, policy: bool = True) -> dict:
        state = self.state_features(batch)
        if not policy:
            embedding = self.net.state_encoder(state)
            return {"value": self.net.win(embedding).squeeze(-1).tanh(),
                    "margin": self.net.margin(embedding).squeeze(-1)}
        logits, value, margin = self.net(state, self.action_features(batch))
        logits = logits.float().masked_fill(~batch["cand_mask"], float("-inf"))
        return {"logits": logits, "value": value, "margin": margin}


def build_model(config: ModelConfig | dict) -> nn.Module:
    if isinstance(config, dict):
        config = ModelConfig.from_dict(config)
    return LegacyNet(config) if config.arch == "legacy_mlp" else RackZeroNet(config)


def model_config_dict(model: nn.Module) -> dict:
    return asdict(model.config)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def q_outputs(out: dict, n: int):
    """(policy probabilities, move values / 100) from one record's network outputs.

    With an afterstate head the values are Q + afterstate value for the top
    candidates and -inf for the rest (never chosen)."""
    logits = out["logits"][:n]
    probs = np.exp(logits - logits.max())
    return probs / probs.sum(), np.asarray(out.get("q_after", out["q"]))[:n]


class NetworkEvaluator:
    """Search-facing adapter: ``predict``/``predict_value`` on one information state.

    It builds the same records used for training from observation-only inputs and
    keeps timing counters for throughput reporting. ``model`` may live on any device.
    """

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu",
                 precision: str = "fp32"):
        self.model = model
        self.device = torch.device(device)
        self.dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
        self.calls = 0
        self.positions = 0
        self.seconds = 0.0

    def eval(self):
        self.model.eval()
        return self

    def belief_features(self, info, belief=None) -> np.ndarray:
        return belief_summary(info, belief)

    def _run(self, records: list[dict], policy: bool) -> dict:
        import time

        start = time.perf_counter()
        batch = to_tensors(stack_records(records), self.device)
        with torch.inference_mode(), torch.autocast(
                self.device.type, dtype=self.dtype or torch.float32,
                enabled=self.dtype is not None):
            out = self.model(batch, policy=policy)
        out = {k: v.float().cpu().numpy() for k, v in out.items()}
        self.calls += 1
        self.positions += len(records)
        self.seconds += time.perf_counter() - start
        return out

    def predict(self, info, moves, belief_mean=None):
        record = encode_record(info, moves, belief_from_mean(info, belief_mean), len(moves))
        out = self._run([record], policy=True)
        logits = out["logits"][0]
        probs = np.exp(logits - logits.max())
        return probs / probs.sum(), float(out["value"][0]), float(out["margin"][0]) * 100

    def predict_q(self, info, moves, belief_mean=None):
        """(policy probabilities, Q values in points / 100) for every candidate."""
        record = encode_record(info, moves, belief_from_mean(info, belief_mean), len(moves))
        out = self._run([record], policy=True)
        if "q" not in out:
            raise ValueError("This network has no Q head (model.q_head: false)")
        return q_outputs({k: v[0] for k, v in out.items()}, len(moves))

    def predict_value(self, info, belief_mean=None):
        from scrabble_gym.environment import Move, ScoredMove

        record = encode_record(info, (ScoredMove(Move(), 0),),
                               belief_from_mean(info, belief_mean), 1)
        out = self._run([record], policy=False)
        return float(out["value"][0]), float(out["margin"][0]) * 100

    def stats(self) -> dict:
        return {"inference_calls": self.calls, "inference_positions": self.positions,
                "inference_seconds": self.seconds}
