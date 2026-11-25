"""RackZero AlphaZero-style training loop.

Each iteration:

1. Self-play: games played by neural-guided search with the current network
   (optionally in parallel workers). Every game has its own seed, derived from
   (experiment seed, iteration, game index).
2. Replay: observation-only records, MCTS visit-count policy targets and
   natural terminal outcomes (win sign, score margin). Truncated games produce
   no targets.
3. Optimization: batched policy cross-entropy + value + margin losses.
4. Arena: candidate vs incumbent in paired seats. In ``gate`` mode the candidate
   is promoted only if it passes the configured criterion; otherwise its weights
   and optimizer state are rolled back together. ``monitor`` mode always accepts
   and only reports.
5. Baseline evaluation on a fixed seed schedule (comparable across iterations
   and experiments); it never decides promotion.
6. Atomic checkpoints with model, optimizer, scheduler, scaler, replay,
   counters, history, configuration and all RNG states.

Checkpoints use Python pickle via ``torch.load(weights_only=False)``: load only
files you trust. Recovery resumes from the last complete iteration.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from rackzero.checkpoints import (
    SCHEMA_VERSION, atomic_checkpoint, capture_rng, load_checkpoint, resolve_checkpoint,
    restore_rng,
)
from rackzero.config import (
    Config, apply_overrides, classify_resume_change, config_differences, load_config, to_plain,
)
from rackzero.dictionary import FileLexicon, TinyLexicon
from rackzero.features import to_tensors
from rackzero.models.networks import ModelConfig, build_model, model_config_dict, parameter_count
from rackzero.replay import ReplayBuffer
from rackzero.stats import wilson_interval
from rackzero.telemetry import HardwareMonitor, MetricsWriter, environment_report
from rackzero.workers import (
    Executor, InferenceServer, ReanalyseTask, SelfPlayTask, WorkerSpec, evaluate_parallel,
    selfplay_seed, write_model,
)


__all__ = ["Config", "Trainer", "atomic_checkpoint", "load_checkpoint", "load_config", "train"]


def lexicon_for_path(path):
    return TinyLexicon() if path is None else FileLexicon(path)


def lexicon_for(config: Config):
    return lexicon_for_path(config.dictionary)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use runtime.device: cpu "
                         "(configs/cpu.yaml) or install a CUDA build of PyTorch")
    return device


def configure_runtime(config: Config) -> str:
    torch.set_num_threads(config.runtime.threads)
    device = resolve_device(config.runtime.device)
    if device == "cpu" and config.runtime.precision != "fp32":
        raise ValueError("runtime.precision bf16/fp16 requires CUDA; use fp32 on CPU")
    if device == "cuda" and config.runtime.precision == "bf16" \
            and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16; use fp16 or fp32")
    torch.backends.cuda.matmul.allow_tf32 = config.runtime.tf32
    torch.backends.cudnn.allow_tf32 = config.runtime.tf32
    torch.backends.cudnn.benchmark = not config.runtime.deterministic
    torch.backends.cudnn.deterministic = config.runtime.deterministic
    return device


def build_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                                weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                 weight_decay=cfg.weight_decay)
    return torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum,
                           nesterov=cfg.momentum > 0, weight_decay=cfg.weight_decay)


def lr_factor(cfg, total_steps: int):
    def factor(step: int) -> float:
        if cfg.warmup_steps and step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        if cfg.lr_schedule == "cosine":
            span = max(1, total_steps - cfg.warmup_steps)
            progress = min(1.0, (step - cfg.warmup_steps) / span)
            return cfg.lr_min_factor + (1 - cfg.lr_min_factor) * 0.5 * (1 + math.cos(math.pi * progress))
        if cfg.lr_schedule == "step":
            return cfg.lr_gamma ** ((step - cfg.warmup_steps) // cfg.lr_step_size)
        return 1.0
    return factor


def compute_losses(out: dict, batch: dict, training, value_head: str) -> dict:
    mask = batch["cand_mask"]
    logits = out["logits"].float()
    log_probs = torch.log_softmax(logits, dim=-1).masked_fill(~mask, 0.0)
    target = batch["policy"]
    per_position = -(target * log_probs).sum(-1)
    # Positions whose policy is not a training target (playout cap randomization's
    # fast searches) have weight 0; records without the field weigh 1.
    weight = batch.get("policy_weight")
    if weight is None:
        weight = torch.ones_like(per_position)
    weight_sum = weight.sum().clamp_min(1.0)
    policy_loss = (weight * per_position).sum() / weight_sum
    win = batch["win"]
    q_weight = getattr(training, "value_target_q_weight", 0.0)
    search_value = batch.get("search_value", win)
    if value_head == "wdl":
        classes = (1 - win).round().long().clamp(0, 2)  # win -> 0, draw -> 1, loss -> 2
        wdl_logits = out["wdl_logits"].float()
        if q_weight:
            soft = F.one_hot(classes, 3).float() * (1 - q_weight)
            q = search_value.float().clamp(-1, 1)
            soft[:, 0] += q_weight * (1 + q) / 2
            soft[:, 2] += q_weight * (1 - q) / 2
            value_loss = -(soft * torch.log_softmax(wdl_logits, -1)).sum(-1).mean()
        else:
            value_loss = F.cross_entropy(wdl_logits, classes)
    else:
        value_target = win if not q_weight else (1 - q_weight) * win + q_weight * search_value
        value_loss = (out["value"].float() - value_target).square().mean()
    margin_loss = (out["margin"].float() - batch["margin"] / training.margin_scale).square().mean()
    total = (training.policy_loss_weight * policy_loss + training.value_loss_weight * value_loss
             + training.margin_loss_weight * margin_loss)
    with torch.no_grad():
        probs = log_probs.exp().masked_fill(~mask, 0.0)
        entropy = ((weight * -(probs * log_probs).sum(-1)).sum() / weight_sum)
        top1 = (weight * (logits.argmax(-1) == target.argmax(-1)).float()).sum() / weight_sum
        value_mse = (out["value"].float() - win).square().mean()
    return {"total": total, "policy": policy_loss, "value": value_loss, "margin": margin_loss,
            "policy_entropy": entropy, "policy_top1": top1, "value_mse": value_mse}


def new_run_dir(config: Config) -> Path:
    base = Path(config.output_root) / f"{config.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    path, suffix = base, 1
    while path.exists():
        suffix += 1
        path = base.with_name(f"{base.name}-{suffix}")
    return path


def run_dir_of(checkpoint_path: Path) -> Path:
    checkpoint_path = resolve_checkpoint(checkpoint_path)
    parent = checkpoint_path.parent
    return parent.parent if parent.name == "checkpoints" else parent


def promote(arena: dict, cfg) -> bool:
    if cfg.criterion == "wilson":
        low, _ = wilson_interval(arena["wins"] + 0.5 * arena["draws"], arena["games"])
        return low > cfg.threshold
    return arena["win_rate"] >= cfg.threshold


class Trainer:
    def __init__(self, config: Config, run_dir=None, resume=None,
                 allow_config_change: bool = False):
        self.config = config
        self.device = configure_runtime(config)
        self.lexicon = lexicon_for(config)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.model = build_model(config.model).to(self.device)
        self.optimizer = build_optimizer(self.model, config.training)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_factor(config.training, self.total_steps))
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
            config.runtime.precision)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.device == "cuda" and config.runtime.precision == "fp16")
        self.replay = ReplayBuffer(config.replay.capacity, config.search.candidates)
        self.best_model = self._cpu_state()
        self.iteration = 0
        self.counters = {"optimizer_steps": 0, "training_samples": 0, "selfplay_games": 0,
                         "selfplay_positions": 0, "promotions": 0, "wall_seconds": 0.0}
        self.history: list[dict] = []
        self.config_changes: dict = {}
        self._published = 0
        checkpoint = None
        if resume is not None:
            checkpoint = load_checkpoint(resume)
            run_dir = run_dir or run_dir_of(Path(resume))
        self.run_dir = Path(run_dir) if run_dir is not None else new_run_dir(config)
        if checkpoint is None and (self.run_dir / "checkpoints" / "latest.pt").exists():
            raise ValueError(f"{self.run_dir} already holds a run; resume it with --resume "
                             "or choose a new --run-dir")
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "models").mkdir(exist_ok=True)
        if checkpoint is not None:
            self.restore(checkpoint, allow_config_change)
        elif config.init_from:
            self.initialize_from(config.init_from)
        with (self.run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
        report = environment_report(self.device)
        report["parameters"] = parameter_count(self.model)
        report["replay_bytes"] = self.replay.nbytes
        with (self.run_dir / "environment.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        self.metrics = MetricsWriter(self.run_dir, config.logging.tensorboard)
        self.executor: Executor | None = None
        self.server: InferenceServer | None = None

    # -- state ------------------------------------------------------------- #

    @property
    def total_steps(self) -> int:
        return self.config.iterations * self.config.training.steps_per_iteration

    def _cpu_state(self, model=None) -> dict:
        model = model or self.model
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def checkpoint(self, include_replay: bool = True) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "config": self.config.to_dict(),
            "model_config": model_config_dict(self.model),
            "dictionary_fingerprint": self.lexicon.fingerprint,
            "iteration": self.iteration,
            "counters": dict(self.counters),
            "model": self._cpu_state(),
            "best_model": copy.deepcopy(self.best_model),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": capture_rng(self.rng),
            "replay": self.replay.state_dict() if include_replay else None,
            "history": copy.deepcopy(self.history),
        }

    def restore(self, checkpoint: dict, allow_config_change: bool = False) -> None:
        if checkpoint["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Schema-1 checkpoints can be evaluated (evaluate.py) but not "
                             "resumed: their replay format predates compact records")
        if checkpoint["dictionary_fingerprint"] != self.lexicon.fingerprint:
            raise ValueError("Checkpoint dictionary fingerprint does not match")
        if checkpoint.get("replay") is None:
            raise ValueError("This checkpoint has no replay buffer; resume from "
                             "checkpoints/latest.pt (or an archive saved with archive_replay)")
        differences = config_differences(checkpoint["config"], self.config.to_dict())
        blocked = {k: v for k, v in differences.items()
                   if classify_resume_change(k) == "trajectory"}
        numeric = {k: v for k, v in differences.items()
                   if classify_resume_change(k) == "numeric"}
        if blocked and not allow_config_change:
            raise ValueError("Resume config differs in trajectory-changing settings "
                             f"{sorted(blocked)}; pass --allow-config-change to accept")
        if numeric:
            warnings.warn(f"Resume changes numerics-affecting settings {sorted(numeric)}",
                          RuntimeWarning, stacklevel=2)
        self.config_changes = {k: list(v) for k, v in differences.items()}
        if asdict(self.config.model) != asdict(ModelConfig.from_dict(checkpoint["model_config"])):
            raise ValueError("Model architecture differs from the checkpoint")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if not blocked:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            # Apply the (possibly changed) schedule and learning rate, continuing
            # from the saved optimizer step count.
            step = int(checkpoint["scheduler"]["last_epoch"])
            factor = lr_factor(self.config.training, self.total_steps)
            self.scheduler.base_lrs = [self.config.training.learning_rate] * len(
                self.optimizer.param_groups)
            self.scheduler.last_epoch = step
            for group in self.optimizer.param_groups:
                group["initial_lr"] = self.config.training.learning_rate
                group["lr"] = self.config.training.learning_rate * factor(step)
                group["weight_decay"] = self.config.training.weight_decay
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.best_model = copy.deepcopy(checkpoint["best_model"])
        self.replay.load_state_dict(checkpoint["replay"])
        self.history = copy.deepcopy(checkpoint["history"])
        self.iteration = int(checkpoint["iteration"])
        self.counters.update(checkpoint["counters"])
        restore_rng(checkpoint["rng"], self.rng, allow_missing_cuda=allow_config_change)

    def initialize_from(self, path) -> None:
        """Start a new run from another checkpoint's network (e.g. a warm start).

        Only the weights are copied: optimizer, replay, counters and RNG start fresh.
        """
        source = load_checkpoint(path)
        if source["dictionary_fingerprint"] != self.lexicon.fingerprint:
            raise ValueError("init_from checkpoint dictionary fingerprint does not match")
        if asdict(self.config.model) != asdict(ModelConfig.from_dict(source["model_config"])):
            raise ValueError("init_from checkpoint has a different model architecture; "
                             "use the same model settings as the warm start")
        self.model.load_state_dict(source["best_model"])
        self.best_model = self._cpu_state()

    # -- workers ----------------------------------------------------------- #

    def _ensure_executor(self) -> Executor:
        if self.executor is None:
            cfg = self.config.selfplay
            if cfg.inference == "server":
                self.server = InferenceServer(cfg.workers, self.device, cfg.inference_precision,
                                              cfg.inference_batch_size, cfg.inference_max_wait_ms)
            self.executor = Executor(WorkerSpec.from_config(self.config, self.lexicon.fingerprint),
                                     cfg.workers, self.server)
        return self.executor

    def _publish(self, role: str, state: dict):
        self._published += 1
        tag = f"{role}-{self.iteration + 1}-{self._published}"
        ref = write_model(self.run_dir / "models", role, model_config_dict(self.model), state)
        ref = type(ref)(ref.path, tag)
        if self.server is not None:
            self.server.load(ref)
        return ref

    def close(self) -> None:
        if self.executor is not None:
            self.executor.close()
            self.executor = None
        if self.server is not None:
            self.server.stop()
            self.server = None
        self.metrics.close()

    def _progress(self, label: str):
        if not self.config.logging.progress:
            return None
        iteration = self.iteration + 1

        def report(done, total):
            print(f"iteration {iteration}: {label} {done}/{total}", flush=True)
        return report

    # -- phases ------------------------------------------------------------ #

    def self_play(self) -> dict:
        cfg = self.config
        executor = self._ensure_executor()
        if self.server is not None:
            self.server.reset_stats()
        ref = self._publish("selfplay", self._cpu_state())
        tasks = [SelfPlayTask(
            selfplay_seed(cfg.seed, self.iteration, i), ref, asdict(cfg.search),
            cfg.search.candidates, cfg.max_turns, cfg.selfplay.temperature,
            cfg.selfplay.temperature_moves, cfg.selfplay.final_temperature,
            cfg.selfplay.full_search_probability, cfg.selfplay.fast_simulations,
            cfg.selfplay.full_simulations,
        ) for i in range(cfg.selfplay.games_per_iteration)]
        started = time.perf_counter()
        results = executor.map("selfplay", tasks, self._progress("self-play"))
        wall = time.perf_counter() - started
        totals: dict = {}
        for result in results:
            for key, value in result["stats"].items():
                totals[key] = totals.get(key, 0) + value
            if result["records"] is not None:
                self.replay.add(result["records"])
        if not totals.get("natural_games"):
            raise RuntimeError("No naturally completed self-play games; increase max_turns")
        plies = max(1, totals["plies"])
        stats = {
            "games": totals["games"], "natural_games": totals["natural_games"],
            "truncated_games": totals["truncated_games"], "new_positions": totals["positions"],
            "plies": totals["plies"], "mean_game_length": totals["plies"] / totals["games"],
            "wall_seconds": wall, "games_per_hour": 3600 * totals["games"] / wall,
            "positions_per_second": totals["positions"] / wall,
            "decision_seconds_mean": totals["search_seconds"] / plies,
            "search_seconds": totals["search_seconds"],
            "simulator_movegen_seconds": totals["movegen_seconds"],
            "simulator_movegen_calls": totals["movegen_calls"],
            "simulator_step_seconds": totals["step_seconds"],
            "live_movegen_seconds": totals["live_movegen_seconds"],
            "game_step_seconds": totals["game_step_seconds"],
            "inference_seconds": totals.get("inference_seconds", 0.0),
            "inference_calls": totals.get("inference_calls", 0),
            "inference_positions_per_second": (
                totals.get("inference_positions", 0) / totals["inference_seconds"]
                if totals.get("inference_seconds") else None),
            "simulations_per_move": cfg.selfplay.full_simulations or cfg.search.simulations,
            "belief_resets": totals["belief_resets"],
            "full_search_fraction": totals.get("full_searches", totals["plies"]) / plies,
            "workers": cfg.selfplay.workers,
        }
        if self.server is not None:
            stats.update(self.server.stats())
        self.counters["selfplay_games"] += totals["games"]
        self.counters["selfplay_positions"] += totals["positions"]
        return stats

    def reanalyse(self, new_positions: int) -> dict | None:
        """Refresh policy/search-value targets of stored positions with the current network."""
        cfg = self.config
        count = min(len(self.replay), int(round(cfg.reanalyse.fraction * new_positions)))
        if count < 1:
            return None
        from rackzero.evaluation import evaluation_search_config
        slots = self.rng.choice(len(self.replay), size=count, replace=False)
        ref = self._publish("reanalyse", self._cpu_state())
        search = asdict(evaluation_search_config(cfg.search))
        per = cfg.reanalyse.positions_per_task
        chunks = [slots[i:i + per] for i in range(0, count, per)]
        tasks = [ReanalyseTask(int(np.random.SeedSequence([cfg.seed, self.iteration, 11, i])
                                   .generate_state(1)[0] % 2**31),
                               ref, search, self.replay.positions(chunk))
                 for i, chunk in enumerate(chunks)]
        started = time.perf_counter()
        results = self._ensure_executor().map("reanalyse", tasks, self._progress("reanalyse"))
        changed = 0.0
        for chunk, result in zip(chunks, results):
            old = self.replay.arrays["policy"][chunk]
            changed += float(0.5 * np.abs(old - result["policy"]).sum())
            self.replay.update(chunk, {"policy": result["policy"],
                                       "search_value": result["search_value"],
                                       "policy_weight": np.ones(len(chunk), dtype=np.float32)})
        wall = time.perf_counter() - started
        return {"positions": count, "wall_seconds": wall,
                "seconds_per_position": sum(r["stats"]["seconds"] for r in results) / count,
                "mean_policy_change": changed / count}

    def training_steps(self, new_positions: int) -> int:
        cfg = self.config.training
        if cfg.samples_per_position is None:
            return cfg.steps_per_iteration
        return max(1, math.ceil(cfg.samples_per_position * new_positions / cfg.batch_size))

    def optimize(self, new_positions: int | None = None) -> dict:
        cfg = self.config.training
        if len(self.replay) < cfg.min_replay:
            return {"skipped": True, "replay_size": len(self.replay)}
        steps = (cfg.steps_per_iteration if new_positions is None
                 else self.training_steps(new_positions))
        cuda = self.device == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        self.model.train()
        names = ("total", "policy", "value", "margin", "policy_entropy", "policy_top1",
                 "value_mse")
        # Accumulate on the device; one host synchronization per iteration.
        sums = {name: torch.zeros((), device=self.device) for name in names}
        grad_norm_sum = torch.zeros((), device=self.device)
        finite = torch.ones((), dtype=torch.bool, device=self.device)
        data_seconds = 0.0
        started = time.perf_counter()
        for _ in range(steps):
            t0 = time.perf_counter()
            batch = to_tensors(self.replay.sample(self.rng, cfg.batch_size), self.device)
            data_seconds += time.perf_counter() - t0
            with torch.autocast(self.device, dtype=self.amp_dtype or torch.float32,
                                enabled=self.amp_dtype is not None):
                out = self.model(batch)
            losses = compute_losses(out, batch, cfg, self.config.model.value_head)
            finite &= torch.isfinite(losses["total"])
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(losses["total"]).backward()
            if cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm_sum += torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.grad_clip).detach().float()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            for name in names:
                sums[name] += losses[name].detach().float()
        if not bool(finite):
            # The run stops here; checkpoints/latest.pt still holds the last good iteration.
            raise RuntimeError("Nonfinite training loss")
        sums = {name: float(value) for name, value in sums.items()}
        grad_norm_sum = float(grad_norm_sum)
        seconds = time.perf_counter() - started
        self.counters["optimizer_steps"] += steps
        self.counters["training_samples"] += steps * cfg.batch_size
        self.model.eval()
        stats = {f"{name}_loss" if name in ("total", "policy", "value", "margin") else name:
                 value / steps for name, value in sums.items()}
        stats.update(
            loss=sums["total"] / steps, steps=steps, batch_size=cfg.batch_size,
            learning_rate=self.optimizer.param_groups[0]["lr"],
            grad_norm=grad_norm_sum / steps if cfg.grad_clip > 0 else None,
            seconds=seconds, data_seconds=data_seconds,
            samples_per_second=steps * cfg.batch_size / seconds,
            replay_size=len(self.replay),
        )
        if cuda:
            stats["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 2**20
        return stats

    def _evaluate(self, ref, opponent, games, seed, opponent_ref=None) -> dict:
        return evaluate_parallel(self._ensure_executor(), self.config, ref, opponent=opponent,
                                 games=games, seed=seed, opponent_model=opponent_ref,
                                 progress=self._progress(f"eval vs {opponent}"))

    def run(self) -> list[dict]:
        cfg = self.config
        latest = self.run_dir / "checkpoints" / "latest.pt"
        # Also gives an interrupted first iteration a valid recovery boundary.
        atomic_checkpoint(latest, self.checkpoint())
        try:
            while self.iteration < cfg.iterations:
                self._iteration(latest)
        finally:
            self.close()
        return self.history

    def _iteration(self, latest: Path) -> None:
        cfg = self.config
        start = time.perf_counter()
        number = self.iteration + 1
        incumbent_state = self._cpu_state()
        incumbent_optimizer = copy.deepcopy(self.optimizer.state_dict())
        self._ensure_executor()
        monitor = HardwareMonitor(cfg.logging.hardware_interval, self.device)
        with monitor:
            selfplay = self.self_play()
        hardware = {"selfplay": monitor.summary()}
        reanalysed = None
        if cfg.reanalyse.fraction > 0:
            with monitor:
                reanalysed = self.reanalyse(selfplay.get("new_positions", 0))
            hardware["reanalyse"] = monitor.summary()
        with monitor:
            training = self.optimize(selfplay.get("new_positions"))
        hardware["training"] = monitor.summary()
        if not training.get("skipped") and selfplay.get("new_positions"):
            # Gradient samples drawn per newly generated position (overfitting indicator).
            training["samples_per_new_position"] = (
                training["steps"] * training["batch_size"] / selfplay["new_positions"])
        record = {"iteration": number, "selfplay": selfplay, "training": training,
                  "loss": training.get("loss"), "replay_size": len(self.replay)}
        if reanalysed is not None:
            record["reanalyse"] = reanalysed
        candidate_ref = None
        arena = None
        run_arena = cfg.arena.mode == "gate" or (
            cfg.arena.mode == "monitor" and number % cfg.arena.every == 0)
        with monitor:
            if run_arena:
                candidate_ref = self._publish("candidate", self._cpu_state())
                incumbent_ref = self._publish("incumbent", incumbent_state)
                arena_seed = int(np.random.SeedSequence([cfg.seed, number, 7])
                                 .generate_state(1)[0] % 2**31)
                arena = self._evaluate(candidate_ref, "search", cfg.arena.games, arena_seed,
                                       incumbent_ref)
            if cfg.arena.mode == "gate":
                promoted = promote(arena, cfg.arena)
            else:
                promoted = True
            baselines = {}
            if cfg.evaluation.every and number % cfg.evaluation.every == 0:
                candidate_ref = candidate_ref or self._publish("candidate", self._cpu_state())
                for opponent in cfg.evaluation.opponents:
                    baselines[opponent] = self._evaluate(
                        candidate_ref, opponent, cfg.evaluation.games, cfg.evaluation.seed)
        hardware["evaluation"] = monitor.summary()
        if promoted:
            self.best_model = self._cpu_state()
            self.counters["promotions"] += 1
        else:
            self.model.load_state_dict(incumbent_state)
            self.optimizer.load_state_dict(incumbent_optimizer)
        self.iteration = number
        seconds = time.perf_counter() - start
        self.counters["wall_seconds"] += seconds
        record.update(promoted=promoted, arena=arena, baselines=baselines,
                      hardware=hardware, counters=dict(self.counters), seconds=seconds)
        if self.config_changes:
            record["config_changes"] = self.config_changes
            self.config_changes = {}
        self.history.append(record)
        payload = self.checkpoint()
        atomic_checkpoint(latest, payload)
        light = dict(payload, replay=None)
        if cfg.checkpoint.every and number % cfg.checkpoint.every == 0:
            archive = payload if cfg.checkpoint.archive_replay else light
            atomic_checkpoint(self.run_dir / "checkpoints" / f"iteration_{number:04d}.pt", archive)
        atomic_checkpoint(self.run_dir / "checkpoints" / "best.pt", light)
        self.metrics.write(record, self.history)
        print(summary_line(record), flush=True)


def summary_line(record: dict) -> str:
    sp, tr = record["selfplay"], record["training"]
    parts = [f"[iter {record['iteration']}]",
             f"selfplay {sp['games']}g/{sp['new_positions']}pos "
             f"{sp['games_per_hour']:.0f}g/h {sp['decision_seconds_mean']:.2f}s/move"]
    if record.get("reanalyse"):
        ra = record["reanalyse"]
        parts.append(f"reanalyse {ra['positions']}pos {ra['wall_seconds']:.0f}s")
    if not tr.get("skipped"):
        parts.append(f"loss {tr['loss']:.3f} (p {tr['policy_loss']:.3f} v {tr['value_loss']:.3f} "
                     f"m {tr['margin_loss']:.3f}) {tr['samples_per_second']:.0f} samples/s")
    if record.get("arena"):
        parts.append(f"arena {record['arena']['win_rate']:.2f}"
                     f"{' promoted' if record['promoted'] else ' rejected'}")
    for name, result in (record.get("baselines") or {}).items():
        low, high = result["win_rate_ci95"]
        parts.append(f"{name} {result['win_rate']:.2f} [{low:.2f},{high:.2f}] "
                     f"{result['mean_score_difference']:+.0f}")
    return " | ".join(parts)


def train(config: Config, resume=None, run_dir=None, allow_config_change=False):
    trainer = Trainer(config, run_dir=run_dir, resume=resume,
                      allow_config_change=allow_config_change)
    return trainer.run()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML configuration (required for a new run)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override any setting, e.g. --set training.batch_size=256")
    parser.add_argument("--resume", help="TRUSTED run directory or checkpoint to continue")
    parser.add_argument("--run-dir", help="explicit run directory (default: "
                        "<output_root>/<name>-<timestamp>)")
    parser.add_argument("--allow-config-change", action="store_true",
                        help="accept trajectory-changing config differences on resume")
    parser.add_argument("--print-config", action="store_true",
                        help="print the resolved configuration and exit")
    args = parser.parse_args(argv)
    if args.resume:
        saved = load_checkpoint(args.resume)
        values = (load_config(args.config).to_dict() if args.config
                  else Config.from_dict(saved["config"]).to_dict())
        config = Config.from_dict(apply_overrides(values, args.set))
        del saved
    elif args.config:
        config = load_config(args.config, args.set)
    else:
        parser.error("--config is required for a new run")
    if args.print_config:
        yaml.safe_dump(to_plain(config), sys.stdout, sort_keys=False)
        return 0
    trainer = Trainer(config, run_dir=args.run_dir, resume=args.resume,
                      allow_config_change=args.allow_config_change)
    print(f"run directory: {trainer.run_dir}", flush=True)
    try:
        trainer.run()
    except KeyboardInterrupt:
        latest = trainer.run_dir / "checkpoints" / "latest.pt"
        print(f"Interrupted; resume the last complete iteration with --resume {trainer.run_dir}. "
              "No partial iteration was saved." if latest.is_file() else
              "Interrupted before the first checkpoint; nothing to resume.", flush=True)
        return 130
    return 0
