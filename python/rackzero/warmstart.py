"""Warm start: supervised pre-training on Quackle's static player.

Pure self-play from random weights spends its first iterations rediscovering
what a static evaluator already knows (score now, keep a good rack leave).
AlphaGo started from supervised learning on expert moves; here the expert is
Quackle's static player (score + rack-leave equity), already in this repo.

1. ``generate`` plays Quackle-vs-Quackle games with the Java engine as the rules
   authority. Each decision becomes a normal replay record: the acting
   player's observation, belief features, a candidate set (Quackle's ranked
   moves plus the search shortlist), a policy target softmax(equity / T) over
   Quackle's ranked moves, and the game's outcome as value/margin targets.
2. ``pretrain`` fits a RackZero network to those records and writes a standard
   checkpoint, usable by ``evaluate.py`` and as ``init_from`` for ``train.py``.

    python -m rackzero.warmstart generate --config configs/warm.yaml \\
        --games 2000 --workers 8 --output data/quackle-2000.npz
    python -m rackzero.warmstart pretrain --config configs/warm.yaml \\
        --data data/quackle-2000.npz --epochs 10
    python train.py --config configs/warm.yaml \\
        --set init_from=runs/<pretrain-run>/checkpoints/pretrained.pt

Quackle is needed only for ``generate`` (and for the ``quackle`` evaluation
opponent); the dataset and checkpoint are plain files that can be copied to a
cluster without Quackle.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from rackzero.features import fill_target_defaults


def _load_config(args):
    from rackzero.config import load_config
    return load_config(args.config, args.set)


# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #

def generate(config, games: int, workers: int, output: Path, seed: int = 0, top: int = 10,
             temperature: float = 2.0, progress: bool = True) -> dict:
    from rackzero.training import lexicon_for
    from rackzero.workers import Executor, TeacherGameTask, WorkerSpec

    from rackzero.quackle import lexicon_dawg

    lexicon = lexicon_for(config)
    lexicon_dawg(lexicon.words)   # build once here, not concurrently in every worker
    tasks = [TeacherGameTask(
        int(np.random.SeedSequence([seed, 77, i]).generate_state(1)[0] % 2**31),
        config.search.candidates, config.max_turns, top, temperature, config.search.shortlist)
        for i in range(games)]
    executor = Executor(WorkerSpec.from_config(config, lexicon.fingerprint), workers)
    started = time.perf_counter()

    def report(done, total):
        if progress and (done == total or done % max(1, total // 20) == 0):
            elapsed = time.perf_counter() - started
            print(f"teacher games {done}/{total}  {3600 * done / elapsed:.0f} games/h", flush=True)

    try:
        results = executor.map("teacher_game", tasks, report)
    finally:
        executor.close()
    wall = time.perf_counter() - started
    batches, game_ids = [], []
    totals = {"decisions": 0, "quackle_in_shortlist": 0, "quackle_in_score_shortlist": 0,
              "natural_games": 0}
    for index, result in enumerate(results):
        for key in totals:
            totals[key] += result["stats"][key]
        if result["records"] is not None:
            batches.append(result["records"])
            game_ids.append(np.full(len(result["records"]["policy"]), index, dtype=np.int32))
    if not batches:
        raise RuntimeError("No naturally completed teacher games")
    data = {name: np.concatenate([b[name] for b in batches]) for name in batches[0]}
    data["game"] = np.concatenate(game_ids)
    meta = {
        "kind": "quackle-teacher", "games": games, "natural_games": totals["natural_games"],
        "positions": int(len(data["game"])), "candidates": config.search.candidates,
        "shortlist": config.search.shortlist, "top": top, "temperature": temperature,
        "seed": seed, "dictionary": config.dictionary,
        "dictionary_fingerprint": lexicon.fingerprint, "wall_seconds": wall,
        "quackle_in_shortlist": totals["quackle_in_shortlist"] / max(1, totals["decisions"]),
        "quackle_in_score_shortlist": (totals["quackle_in_score_shortlist"]
                                       / max(1, totals["decisions"])),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, meta=np.array(json.dumps(meta)), **data)
    return meta


def load_dataset(paths: list[str], fingerprint: str, candidates: int) -> tuple[dict, list]:
    parts, metas, offset = [], [], 0
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"]))
            if meta["dictionary_fingerprint"] != fingerprint:
                raise ValueError(f"{path}: dataset dictionary does not match the config's")
            if meta["candidates"] != candidates:
                raise ValueError(f"{path}: {meta['candidates']} candidates per record, "
                                 f"but search.candidates is {candidates}")
            data = {name: archive[name] for name in archive.files if name != "meta"}
        data["game"] = data["game"].astype(np.int64) + offset
        offset = int(data["game"].max()) + 1
        parts.append(fill_target_defaults(data))
        metas.append(meta)
    data = {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}
    return data, metas


# --------------------------------------------------------------------------- #
# Supervised pre-training
# --------------------------------------------------------------------------- #

def pretrain(config, data_paths: list[str], epochs: int, batch_size: int, learning_rate: float,
             validation: float = 0.05, run_dir=None, value_weight: float = 0.0,
             margin_weight: float = 0.0, value_epochs: int = 2) -> Path:
    """Fit the network to teacher data in two phases.

    1. Policy: the whole network is trained on the policy loss (plus
       ``value_weight``/``margin_weight``, default 0) for ``epochs``; the epoch
       with the lowest validation policy loss is kept.
    2. Value: every position of a game shares one outcome, so value heads
       trained for many epochs memorize the games (measured on 4,000 games:
       validation value loss 0.69 -> 0.91 over 10 joint epochs). The value and
       margin heads are re-initialized, then the whole network is trained on
       policy + value + margin for up to ``value_epochs`` (default 2), keeping
       the epoch with the lowest validation policy + value + margin loss.
       (Fitting only the heads on the frozen policy trunk was tried and gave a
       much worse value: validation 0.87 vs 0.68.)
    """
    from dataclasses import replace

    import torch

    from rackzero.checkpoints import atomic_checkpoint
    from rackzero.features import to_tensors
    from rackzero.training import Trainer, compute_losses

    trainer = Trainer(config, run_dir=run_dir)
    try:
        data, metas = load_dataset(data_paths, trainer.lexicon.fingerprint,
                                   config.search.candidates)
        games = np.unique(data["game"])
        rng = np.random.default_rng(config.seed)
        held_out = set(rng.choice(games, size=max(1, int(round(validation * len(games)))),
                                  replace=False).tolist())
        is_val = np.isin(data["game"], list(held_out))
        fields = [name for name in data if name != "game"]
        train_index, val_index = np.flatnonzero(~is_val), np.flatnonzero(is_val)
        print(f"pretrain: {len(train_index)} training / {len(val_index)} validation positions "
              f"from {len(games)} games; run directory {trainer.run_dir}", flush=True)
        model, device = trainer.model, trainer.device
        amp = trainer.amp_dtype
        log_path = trainer.run_dir / "pretrain.jsonl"
        head_modules = [m for name, m in model.named_children()
                        if name in ("value_trunk", "value", "margin") and m is not None]
        heads = [p for m in head_modules for p in m.parameters()]

        def batches(index, shuffle):
            order = rng.permutation(index) if shuffle else index
            for start in range(0, len(order) - (batch_size - 1 if shuffle else 0), batch_size):
                chunk = order[start:start + batch_size]
                if len(chunk):
                    yield to_tensors({name: data[name][chunk] for name in fields}, device)

        def evaluate(losses_config):
            model.eval()
            sums, count = {}, 0
            with torch.no_grad():
                for batch in batches(val_index, False):
                    with torch.autocast(device, dtype=amp or torch.float32,
                                        enabled=amp is not None):
                        out = model(batch)
                    losses = compute_losses(out, batch, losses_config, config.model.value_head)
                    n = len(batch["policy"])
                    for name in ("total", "policy", "value", "margin", "policy_top1"):
                        sums[name] = sums.get(name, 0.0) + float(losses[name]) * n
                    count += n
            return {name: value / max(1, count) for name, value in sums.items()}

        def run_phase(phase, epochs, params, losses_config, train_mode, select):
            optimizer = torch.optim.AdamW(params, lr=learning_rate,
                                          weight_decay=config.training.weight_decay)
            total = max(1, epochs * max(1, len(train_index) // batch_size))
            warmup = min(500, total // 10)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (
                (step + 1) / warmup if step < warmup else
                0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) /
                                                                max(1, total - warmup))))))
            best = (math.inf, trainer._cpu_state(), 0, evaluate(losses_config))
            for epoch in range(1, epochs + 1):
                model.train(train_mode)
                started = time.perf_counter()
                sums, steps = {}, 0
                for batch in batches(train_index, True):
                    with torch.autocast(device, dtype=amp or torch.float32,
                                        enabled=amp is not None):
                        out = model(batch)
                    losses = compute_losses(out, batch, losses_config, config.model.value_head)
                    optimizer.zero_grad(set_to_none=True)
                    losses["total"].backward()
                    if config.training.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(params, config.training.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    steps += 1
                    for name in ("total", "policy", "value", "margin", "policy_top1"):
                        sums[name] = sums.get(name, 0.0) + float(losses[name].detach())
                train = {name: value / max(1, steps) for name, value in sums.items()}
                val = evaluate(losses_config)
                record = {"phase": phase, "epoch": epoch, "steps": steps,
                          "seconds": time.perf_counter() - started,
                          "learning_rate": optimizer.param_groups[0]["lr"],
                          "train": train, "validation": val}
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(f"[{phase} {epoch}] train p {train['policy']:.3f} v {train['value']:.3f} "
                      f"m {train['margin']:.3f} | validation p {val['policy']:.3f} "
                      f"v {val['value']:.3f} m {val['margin']:.3f} "
                      f"top1 (agrees with Quackle) {val['policy_top1']:.3f} "
                      f"| {record['seconds']:.0f}s", flush=True)
                if select(val) < best[0]:
                    best = (select(val), trainer._cpu_state(), epoch, val)
            model.load_state_dict(best[1])
            print(f"{phase}: keeping epoch {best[2]}", flush=True)
            return best[2], best[3]

        policy_config = replace(config.training, value_loss_weight=value_weight,
                                margin_loss_weight=margin_weight)
        policy_epoch, policy_val = run_phase(
            "policy", epochs, list(model.parameters()), policy_config, True,
            lambda val: val["policy"])
        value_epoch, value_val = 0, policy_val
        if value_epochs > 0 and heads:
            for module in head_modules:
                for layer in module.modules():
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()
            value_config = replace(config.training, policy_loss_weight=1.0,
                                   value_loss_weight=1.0, margin_loss_weight=1.0)
            value_epoch, value_val = run_phase(
                "value", value_epochs, list(model.parameters()), value_config, True,
                lambda val: val["policy"] + val["value"] + val["margin"])
        model.eval()
        trainer.best_model = trainer._cpu_state()
        trainer.history = [{"pretrain": {
            "datasets": metas, "epochs": epochs, "policy_epoch": policy_epoch,
            "value_epochs": value_epochs, "value_epoch": value_epoch,
            "validation": {"policy": policy_val["policy"],
                           "policy_top1": policy_val["policy_top1"],
                           "value": value_val["value"], "margin": value_val["margin"]},
            "value_weight": value_weight, "margin_weight": margin_weight}}]
        payload = trainer.checkpoint(include_replay=False)
        path = trainer.run_dir / "checkpoints" / "pretrained.pt"
        atomic_checkpoint(path, payload)
        atomic_checkpoint(trainer.run_dir / "checkpoints" / "best.pt", payload)
        return path
    finally:
        trainer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="Quackle-vs-Quackle games -> teacher dataset (.npz)")
    gen.add_argument("--config", required=True, help="training config (dictionary, candidates, "
                     "shortlist and particles must match the run you will warm-start)")
    gen.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    gen.add_argument("--games", type=int, required=True)
    gen.add_argument("--workers", type=int, default=0, help="parallel processes (0 = inline)")
    gen.add_argument("--output", required=True)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--top", type=int, default=10, help="Quackle moves ranked per decision")
    gen.add_argument("--temperature", type=float, default=2.0,
                     help="policy target softmax(equity / T) in points; 0 = Quackle's move only")
    pre = sub.add_parser("pretrain", help="fit a network to teacher datasets")
    pre.add_argument("--config", required=True)
    pre.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    pre.add_argument("--data", nargs="+", required=True)
    pre.add_argument("--epochs", type=int, default=10)
    pre.add_argument("--batch-size", type=int, default=256)
    pre.add_argument("--lr", type=float, default=1e-3)
    pre.add_argument("--validation", type=float, default=0.05,
                     help="fraction of games held out for validation")
    pre.add_argument("--value-weight", type=float, default=0.0,
                     help="value loss weight in the policy phase (outcomes are shared by a "
                          "whole game and overfit; the value phase fits the value heads)")
    pre.add_argument("--margin-weight", type=float, default=0.0)
    pre.add_argument("--value-epochs", type=int, default=2,
                     help="joint policy+value epochs after re-initializing the value heads")
    pre.add_argument("--run-dir")
    args = parser.parse_args(argv)
    config = _load_config(args)
    if args.command == "generate":
        if args.games < 1 or args.top < 1 or args.temperature < 0:
            parser.error("--games and --top must be positive and --temperature nonnegative")
        meta = generate(config, args.games, args.workers, Path(args.output), args.seed,
                        args.top, args.temperature)
        print(json.dumps(meta, indent=2), flush=True)
    else:
        config.name = f"{config.name}-pretrain"
        config.init_from = None
        path = pretrain(config, args.data, args.epochs, args.batch_size, args.lr,
                        args.validation, args.run_dir, args.value_weight, args.margin_weight,
                        args.value_epochs)
        print(f"pretrained checkpoint: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
