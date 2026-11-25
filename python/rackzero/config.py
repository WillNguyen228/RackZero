"""Validated, nested experiment configuration.

YAML files may ``extends:`` another file (path relative to the extending
file); mappings are merged recursively. Command-line ``--set a.b=value``
overrides are applied last and parsed as YAML scalars. Unknown keys are errors,
never silently ignored. The original flat RackZero format (schema 1) is still
accepted and migrated (see ``migrate_flat``).
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
import math
from pathlib import Path
import re
from typing import Any

import yaml


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads scientific notation without a dot (e.g. 3e-4)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*)(?:\.[0-9_]*)?(?:[eE][-+]?[0-9]+)$"),
    list("-+0123456789"),
)


def yaml_load(text):
    return yaml.load(text, Loader=_Loader)

from rackzero.models.networks import ModelConfig
from rackzero.search import SearchConfig


def _positive_int(owner: str, name: str, value, minimum: int = 1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{owner}.{name} must be an integer >= {minimum}")


def _finite(owner: str, name: str, value, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{owner}.{name} must be a finite number")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{owner}.{name} must lie in [{low}, {high}]")


@dataclass
class RuntimeConfig:
    device: str = "auto"          # auto | cpu | cuda (auto = cuda when available)
    threads: int = 1              # torch intra-op threads in the trainer process
    precision: str = "fp32"       # fp32 | bf16 | fp16 autocast for training (CUDA)
    tf32: bool = True             # allow TF32 matmuls/convs on Ampere+ GPUs
    deterministic: bool = False   # cuDNN deterministic algorithms (slower)

    def __post_init__(self):
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("runtime.device must be auto, cpu or cuda")
        _positive_int("runtime", "threads", self.threads)
        if self.precision not in ("fp32", "bf16", "fp16"):
            raise ValueError("runtime.precision must be fp32, bf16 or fp16")


@dataclass
class SelfPlayConfig:
    games_per_iteration: int = 8
    workers: int = 0                 # 0 = run games in the trainer process
    worker_threads: int = 1          # torch threads per self-play worker
    inference: str = "local"         # local (CPU per worker) | server (batched, trainer device)
    inference_batch_size: int = 32   # server: max positions per forward pass
    inference_max_wait_ms: float = 2.0
    inference_precision: str = "fp32"
    temperature: float = 1.0
    temperature_moves: int | None = None   # plies using `temperature`; None = whole game
    final_temperature: float = 0.0
    # Playout cap randomization (KataGo, Wu 2019): each move is a full search
    # (full_simulations, root noise, recorded as a policy target) with this
    # probability, otherwise a fast search (fast_simulations, no noise, used only
    # for value targets). 1.0 = every move is a full search (original).
    full_search_probability: float = 1.0
    fast_simulations: int | None = None
    # Self-play full-search budget; None = search.simulations. Evaluation and the
    # arena always use search.simulations, so results stay comparable.
    full_simulations: int | None = None

    def __post_init__(self):
        _positive_int("selfplay", "games_per_iteration", self.games_per_iteration)
        _positive_int("selfplay", "workers", self.workers, 0)
        _positive_int("selfplay", "worker_threads", self.worker_threads)
        _positive_int("selfplay", "inference_batch_size", self.inference_batch_size)
        if self.inference not in ("local", "server"):
            raise ValueError("selfplay.inference must be local or server")
        if self.inference_precision not in ("fp32", "bf16", "fp16"):
            raise ValueError("selfplay.inference_precision must be fp32, bf16 or fp16")
        _finite("selfplay", "inference_max_wait_ms", self.inference_max_wait_ms, 0)
        _finite("selfplay", "temperature", self.temperature, 0)
        _finite("selfplay", "final_temperature", self.final_temperature, 0)
        if self.temperature_moves is not None:
            _positive_int("selfplay", "temperature_moves", self.temperature_moves, 0)
        _finite("selfplay", "full_search_probability", self.full_search_probability, 0, 1)
        if self.full_search_probability <= 0:
            raise ValueError("selfplay.full_search_probability must be positive")
        if self.full_search_probability < 1 and self.fast_simulations is None:
            raise ValueError("selfplay.full_search_probability < 1 requires selfplay.fast_simulations")
        if self.fast_simulations is not None:
            _positive_int("selfplay", "fast_simulations", self.fast_simulations)
        if self.full_simulations is not None:
            _positive_int("selfplay", "full_simulations", self.full_simulations)


@dataclass
class ReplayConfig:
    capacity: int = 20000

    def __post_init__(self):
        _positive_int("replay", "capacity", self.capacity)


@dataclass
class TrainingConfig:
    steps_per_iteration: int = 64
    # If set, each iteration trains ceil(samples_per_position * new_positions /
    # batch_size) steps instead of steps_per_iteration, holding the replay ratio
    # (gradient samples per generated position) fixed. steps_per_iteration then
    # only sets the LR-schedule horizon estimate.
    samples_per_position: float | None = None
    batch_size: int = 32
    min_replay: int = 1               # skip optimization until replay has this many positions
    optimizer: str = "adam"           # adam | adamw | sgd
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    momentum: float = 0.9             # sgd only
    lr_schedule: str = "constant"     # constant | cosine | step
    warmup_steps: int = 0
    lr_min_factor: float = 0.1        # cosine floor as a fraction of learning_rate
    lr_step_size: int = 1000          # step schedule: optimizer steps per decay
    lr_gamma: float = 0.5             # step schedule decay factor
    grad_clip: float = 5.0            # 0 disables clipping
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    margin_loss_weight: float = 1.0
    margin_scale: float = 100.0       # margin target = final score difference / margin_scale
    # Value target = (1 - w) * game outcome z + w * search root value q (as in
    # Leela Chess Zero / KataGo). 0 = outcome only (original).
    value_target_q_weight: float = 0.0

    def __post_init__(self):
        for name in ("steps_per_iteration", "batch_size", "min_replay", "lr_step_size"):
            _positive_int("training", name, getattr(self, name))
        _positive_int("training", "warmup_steps", self.warmup_steps, 0)
        if self.samples_per_position is not None:
            _finite("training", "samples_per_position", self.samples_per_position)
            if self.samples_per_position <= 0:
                raise ValueError("training.samples_per_position must be positive")
        if self.optimizer not in ("adam", "adamw", "sgd"):
            raise ValueError("training.optimizer must be adam, adamw or sgd")
        if self.lr_schedule not in ("constant", "cosine", "step"):
            raise ValueError("training.lr_schedule must be constant, cosine or step")
        _finite("training", "learning_rate", self.learning_rate)
        if self.learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive")
        for name in ("weight_decay", "momentum", "grad_clip", "policy_loss_weight",
                     "value_loss_weight", "margin_loss_weight"):
            _finite("training", name, getattr(self, name), 0)
        _finite("training", "lr_min_factor", self.lr_min_factor, 0, 1)
        _finite("training", "lr_gamma", self.lr_gamma, 0, 1)
        _finite("training", "margin_scale", self.margin_scale)
        _finite("training", "value_target_q_weight", self.value_target_q_weight, 0, 1)
        if self.margin_scale <= 0:
            raise ValueError("training.margin_scale must be positive")


@dataclass
class ReanalyseConfig:
    # Reanalyse (MuZero Reanalyse, Schrittwieser et al. 2021): each iteration,
    # re-search fraction x new_positions stored positions with the current
    # network and replace their policy (and search-value) targets. 0 = off.
    fraction: float = 0.0
    positions_per_task: int = 8

    def __post_init__(self):
        _finite("reanalyse", "fraction", self.fraction, 0)
        _positive_int("reanalyse", "positions_per_task", self.positions_per_task)


@dataclass
class ArenaConfig:
    mode: str = "gate"          # gate (promote/roll back) | monitor (always accept) | off
    games: int = 8              # even; each seed is played in both seats
    threshold: float = 0.55     # gate: minimum half-draw win rate to promote
    criterion: str = "threshold"  # threshold | wilson (lower 95% bound must exceed threshold)
    every: int = 1              # monitor mode: run the arena every N iterations

    def __post_init__(self):
        if self.mode is False:  # YAML 1.1 reads a bare `off` as boolean false
            self.mode = "off"
        if self.mode not in ("gate", "monitor", "off"):
            raise ValueError("arena.mode must be gate, monitor or off")
        if self.criterion not in ("threshold", "wilson"):
            raise ValueError("arena.criterion must be threshold or wilson")
        if self.mode != "off":
            _positive_int("arena", "games", self.games, 2)
            if self.games % 2:
                raise ValueError("arena.games must be even for paired seats")
        _finite("arena", "threshold", self.threshold, 0, 1)
        _positive_int("arena", "every", self.every)


@dataclass
class EvaluationConfig:
    every: int = 1                     # iterations between baseline evaluations; 0 = never
    games: int = 8                     # even, per opponent
    opponents: list[str] = field(default_factory=lambda: ["greedy"])
    seed: int = 1000                   # fixed across iterations and experiments

    def __post_init__(self):
        _positive_int("evaluation", "every", self.every, 0)
        if self.every:
            _positive_int("evaluation", "games", self.games, 2)
            if self.games % 2:
                raise ValueError("evaluation.games must be even for paired seats")
        allowed = {"random", "greedy", "heuristic", "quackle"}
        if not isinstance(self.opponents, list) or not set(self.opponents) <= allowed:
            raise ValueError(f"evaluation.opponents must be a list drawn from {sorted(allowed)}")
        _positive_int("evaluation", "seed", self.seed, 0)


@dataclass
class CheckpointConfig:
    every: int = 1                 # iterations between iteration_NNNN.pt archives; 0 = none
    archive_replay: bool = False   # include the replay buffer in archives (latest.pt always does)

    def __post_init__(self):
        _positive_int("checkpoint", "every", self.every, 0)


@dataclass
class LoggingConfig:
    tensorboard: bool = False
    hardware_interval: float = 2.0   # seconds between GPU/CPU samples; 0 disables
    progress: bool = True

    def __post_init__(self):
        _finite("logging", "hardware_interval", self.hardware_interval, 0)


SECTIONS = {
    "runtime": RuntimeConfig, "model": ModelConfig, "search": SearchConfig,
    "selfplay": SelfPlayConfig, "replay": ReplayConfig, "training": TrainingConfig,
    "reanalyse": ReanalyseConfig, "arena": ArenaConfig, "evaluation": EvaluationConfig, "checkpoint": CheckpointConfig,
    "logging": LoggingConfig,
}


@dataclass
class Config:
    name: str = "rackzero"
    seed: int = 42
    iterations: int = 10
    output_root: str = "runs"
    dictionary: str | None = "WOW24v3.zip"
    max_turns: int = 200
    # Initial network weights for a new run (e.g. a Quackle warm start from
    # ``python -m rackzero.warmstart pretrain``). Ignored when resuming.
    init_from: str | None = None
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    reanalyse: ReanalyseConfig = field(default_factory=ReanalyseConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a nonempty string")
        _positive_int("config", "seed", self.seed, 0)
        _positive_int("config", "iterations", self.iterations)
        _positive_int("config", "max_turns", self.max_turns)
        if not isinstance(self.output_root, str) or not self.output_root:
            raise ValueError("output_root must be a nonempty path")
        if self.init_from is not None and not isinstance(self.init_from, str):
            raise ValueError("init_from must be a checkpoint path or null")
        if self.dictionary is not None and not isinstance(self.dictionary, str):
            raise ValueError("dictionary must be a path or null (built-in tiny lexicon)")
        if self.selfplay.inference == "server" and self.selfplay.workers < 1:
            raise ValueError("selfplay.inference: server requires selfplay.workers >= 1")

    def to_dict(self) -> dict:
        return to_plain(self)

    @classmethod
    def from_dict(cls, values: dict) -> "Config":
        if not isinstance(values, dict):
            raise ValueError("configuration must be a mapping")
        if is_flat(values):
            values = migrate_flat(values)
        values = dict(values)
        values.pop("extends", None)
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        kwargs = {}
        for key, value in values.items():
            if key in SECTIONS:
                if value is None:
                    value = {}
                if not isinstance(value, dict):
                    raise ValueError(f"{key} must be a mapping")
                section = SECTIONS[key]
                bad = set(value) - {f.name for f in fields(section)}
                if bad:
                    raise ValueError(f"Unknown {key} keys: {sorted(bad)}")
                kwargs[key] = section(**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)


def to_plain(value):
    if is_dataclass(value):
        return {k: to_plain(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    return value


FLAT_KEYS = {"games_per_iteration", "training_steps", "hidden", "arena_games",
             "promotion_threshold", "replay_capacity", "output_dir", "threads", "device",
             "batch_size", "learning_rate", "weight_decay", "temperature"}


def is_flat(values: dict) -> bool:
    return bool(FLAT_KEYS & set(values))


def migrate_flat(old: dict) -> dict:
    """Schema-1 flat configuration -> equivalent nested configuration.

    The original network is selected (``model.arch: legacy_mlp``), and gating,
    evaluation and replay settings keep their original meaning: every iteration
    a paired arena of ``arena_games`` gates promotion and a greedy baseline of
    the same size is measured.
    """
    known = FLAT_KEYS | {"dictionary", "seed", "iterations", "max_turns", "search"}
    unknown = set(old) - known
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    arena_games = old.get("arena_games", 8)
    old_name = Path(str(old.get("output_dir", ""))).name
    nested = {
        "name": f"legacy-{old_name}" if old_name not in ("", "checkpoints") else "legacy",
        "seed": old.get("seed", 42),
        "iterations": old.get("iterations", 10),
        "dictionary": old.get("dictionary", "WOW24v3.zip"),
        "max_turns": old.get("max_turns", 200),
        "runtime": {"device": old.get("device", "cpu"), "threads": old.get("threads", 1),
                    "deterministic": True},
        "model": {"arch": "legacy_mlp", "legacy_hidden": old.get("hidden", 128)},
        "search": dict(old.get("search") or {}),
        "selfplay": {"games_per_iteration": old.get("games_per_iteration", 8),
                     "temperature": old.get("temperature", 1.0)},
        "replay": {"capacity": old.get("replay_capacity", 20000)},
        "training": {"steps_per_iteration": old.get("training_steps", 64),
                     "batch_size": old.get("batch_size", 32),
                     "learning_rate": old.get("learning_rate", 0.001),
                     "weight_decay": old.get("weight_decay", 0.0001)},
        "arena": {"mode": "gate", "games": arena_games,
                  "threshold": old.get("promotion_threshold", 0.55)},
        "evaluation": {"every": 1, "games": arena_games, "opponents": ["greedy"],
                       "seed": old.get("seed", 42)},
    }
    if "output_dir" in old:
        parent = Path(str(old["output_dir"])).parent
        nested["output_root"] = str(parent) if str(parent) not in ("", ".") else "runs"
    return nested


def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: str | Path, _seen: tuple = ()) -> dict:
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"Circular config extends: {path}")
    with path.open(encoding="utf-8") as handle:
        values = yaml_load(handle) or {}
    if not isinstance(values, dict):
        raise ValueError(f"{path}: configuration must be a mapping")
    parent = values.pop("extends", None)
    if parent is None:
        return values
    base = load_yaml(path.parent / parent, _seen + (path,))
    return deep_merge(base, values)


def parse_override(text: str) -> tuple[list[str], Any]:
    if "=" not in text:
        raise ValueError(f"Override {text!r} must look like key.subkey=value")
    key, raw = text.split("=", 1)
    keys = key.strip().split(".")
    if not all(keys):
        raise ValueError(f"Invalid override key {key!r}")
    return keys, yaml_load(raw)


def apply_overrides(values: dict, overrides: list[str]) -> dict:
    values = copy.deepcopy(values)
    for text in overrides or ():
        keys, value = parse_override(text)
        target = values
        for key in keys[:-1]:
            target = target.setdefault(key, {})
            if not isinstance(target, dict):
                raise ValueError(f"Override {text!r} descends into a non-mapping")
        target[keys[-1]] = value
    return values


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    values = load_yaml(path)
    if is_flat(values):
        values = migrate_flat(values)
    return Config.from_dict(apply_overrides(values, overrides or []))


# Settings that change throughput or reporting but not the training trajectory
# (self-play games are seeded per game, so worker count cannot change results).
RESUME_NEUTRAL = {
    "iterations", "name", "output_root", "init_from",
    "runtime.threads",
    "selfplay.workers", "selfplay.worker_threads",
    "logging", "checkpoint", "evaluation",
}
# Settings that change numerics on resume (allowed with a recorded warning).
RESUME_NUMERIC = {
    "runtime.device", "runtime.precision", "runtime.tf32", "runtime.deterministic",
    "selfplay.inference", "selfplay.inference_batch_size", "selfplay.inference_max_wait_ms",
    "selfplay.inference_precision",
}


def flatten(values: dict, prefix: str = "") -> dict:
    result = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            result.update(flatten(value, name + "."))
        else:
            result[name] = value
    return result


def config_differences(old: dict, new: dict) -> dict[str, tuple]:
    """Changed settings. A setting absent from ``old`` (saved before it existed)
    is unchanged if ``new`` holds its default, which reproduces the old behaviour."""
    a, b = flatten(old), flatten(new)
    defaults = flatten(Config().to_dict())
    return {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b))
            if a.get(k) != b.get(k) and not (k not in a and b.get(k) == defaults.get(k))}


def classify_resume_change(key: str) -> str:
    def matches(group):
        return any(key == item or key.startswith(item + ".") for item in group)
    if matches(RESUME_NEUTRAL):
        return "neutral"
    if matches(RESUME_NUMERIC):
        return "numeric"
    return "trajectory"
