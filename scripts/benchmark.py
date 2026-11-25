"""Measure where RackZero spends time, on the current machine.

Components (run any subset; ``all`` runs every one):

  movegen    fast vs reference simulator move generation on real game positions
  inference  network forward latency/throughput by model size, batch and device
  training   optimizer samples/s and peak GPU memory by model size/batch/precision
  search     one search decision, split into move generation / inference / other

Examples (from the repository root):

  python scripts/benchmark.py movegen --dictionary WOW24v3.zip
  python scripts/benchmark.py inference training --device cuda --precision bf16
  python scripts/benchmark.py search --config configs/medium.yaml
  python scripts/benchmark.py all --device cuda --output runs/benchmark.jsonl

Results are printed and, with --output, appended as JSON lines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from rackzero.agents import choose  # noqa: E402
from rackzero.config import load_config  # noqa: E402
from rackzero.features import belief_summary, encode_record, stack_records, to_tensors  # noqa: E402
from rackzero.game import Game, Simulator  # noqa: E402
from rackzero.belief import RackBelief  # noqa: E402
from rackzero.models.networks import NetworkEvaluator, build_model, parameter_count  # noqa: E402
from rackzero.replay import ReplayBuffer  # noqa: E402
from rackzero.search import Search, SearchConfig, shortlist  # noqa: E402
from rackzero.training import compute_losses, lexicon_for_path  # noqa: E402


SIZES = ("small", "medium", "large")


def emit(results: list, output: str | None, record: dict) -> None:
    results.append(record)
    print(json.dumps(record), flush=True)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def sample_positions(words, count: int, seed: int = 0, particles: int = 32):
    """(info, belief, live legal moves) from greedy/random self-play games."""
    rng = np.random.default_rng(seed)
    positions = []
    game = Game(words, particles=particles)
    while len(positions) < count:
        info = game.reset(int(rng.integers(2**31)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for turn in range(200):
                moves = game.legal_moves()
                positions.append((info, game.belief, moves))
                kind = "greedy" if rng.random() < 0.8 else "random"
                info, terminal, truncated = game.step(choose(kind, info, moves, rng))
                if terminal or truncated or len(positions) >= count:
                    break
    game.close()
    return positions


def bench_movegen(args, words, results):
    positions = sample_positions(words, args.positions)
    rng = np.random.default_rng(1)
    worlds = []
    simulators = {kind: Simulator(words, movegen=kind) for kind in ("fast", "reference")}
    for info, _, _ in positions:
        worlds.append(simulators["fast"].determinize(info, RackBelief(info, 1, rng).sample()))
    for kind, simulator in simulators.items():
        start = time.perf_counter()
        total = 0
        for world in worlds:
            total += len(simulator.legal_moves(world, rng))
        seconds = time.perf_counter() - start
        emit(results, args.output, {
            "benchmark": "movegen", "movegen": kind, "positions": len(worlds),
            "moves": total, "ms_per_position": 1000 * seconds / len(worlds),
            "moves_per_second": total / seconds})
        simulator.close()


def model_section(size: str):
    return load_config(ROOT / "configs" / f"{size}.yaml").model


def records_for(positions, candidates: int):
    records = []
    for info, belief, moves in positions:
        moves = shortlist(moves, candidates)
        record = encode_record(info, moves, belief_summary(info, belief), candidates)
        policy = np.random.default_rng(len(records)).dirichlet(np.ones(len(moves)))
        record["policy"] = np.zeros(candidates, dtype=np.float32)
        record["policy"][:len(moves)] = policy
        record["win"] = np.float32(1.0)
        record["margin"] = np.float32(20.0)
        records.append(record)
    return records


def bench_inference(args, words, results):
    positions = sample_positions(words, 64)
    records = records_for(positions, 16)
    device = args.device
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.precision)
    for size in args.sizes:
        model = build_model(model_section(size)).to(device).eval()
        for batch_size in args.batch_sizes:
            batch = to_tensors(stack_records((records * (batch_size // len(records) + 1))
                                             [:batch_size]), device)
            with torch.inference_mode(), torch.autocast(
                    torch.device(device).type, dtype=dtype or torch.float32,
                    enabled=dtype is not None):
                for _ in range(3):
                    model(batch)
                if device == "cuda":
                    torch.cuda.synchronize()
                repeats = max(5, min(200, 2000 // batch_size))
                start = time.perf_counter()
                for _ in range(repeats):
                    model(batch)
                if device == "cuda":
                    torch.cuda.synchronize()
                seconds = (time.perf_counter() - start) / repeats
            emit(results, args.output, {
                "benchmark": "inference", "size": size, "params": parameter_count(model),
                "device": device, "precision": args.precision if dtype else "fp32",
                "threads": torch.get_num_threads(), "batch_size": batch_size,
                "ms_per_batch": 1000 * seconds, "positions_per_second": batch_size / seconds})


def bench_training(args, words, results):
    from rackzero.config import TrainingConfig
    from rackzero.training import build_optimizer

    positions = sample_positions(words, 256)
    candidates = 16
    replay = ReplayBuffer(4096, candidates)
    replay.add(stack_records(records_for(positions, candidates)))
    device = args.device
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.precision)
    rng = np.random.default_rng(0)
    for size in args.sizes:
        for batch_size in args.train_batch_sizes:
            model = build_model(model_section(size)).to(device)
            cfg = TrainingConfig(batch_size=batch_size)
            optimizer = build_optimizer(model, cfg)
            scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and args.precision == "fp16")
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()

            def step():
                batch = to_tensors(replay.sample(rng, batch_size), device)
                with torch.autocast(torch.device(device).type, dtype=dtype or torch.float32,
                                    enabled=dtype is not None):
                    out = model(batch)
                loss = compute_losses(out, batch, cfg, model.config.value_head)["total"]
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()

            model.train()
            for _ in range(5):
                step()
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            data = 0.0
            for _ in range(args.steps):
                t0 = time.perf_counter()
                replay.sample(rng, batch_size)
                data += time.perf_counter() - t0
                step()
            if device == "cuda":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            record = {"benchmark": "training", "size": size, "params": parameter_count(model),
                      "device": device, "precision": args.precision if dtype else "fp32",
                      "batch_size": batch_size, "steps": args.steps,
                      "ms_per_step": 1000 * seconds / args.steps,
                      "samples_per_second": args.steps * batch_size / seconds,
                      "replay_sample_ms": 1000 * data / args.steps}
            if device == "cuda":
                record["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 2**20
            emit(results, args.output, record)


def bench_search(args, words, results):
    config = load_config(args.config, args.set)
    search_config = SearchConfig(**{**config.search.__dict__, "root_noise_fraction": 0.0})
    positions = sample_positions(words, args.decisions, seed=3,
                                 particles=search_config.particles)[::1]
    simulator = Simulator(words, movegen=search_config.movegen)
    model = build_model(config.model).eval()
    evaluator = NetworkEvaluator(model, "cpu")
    search = Search(simulator, evaluator, search_config, np.random.default_rng(0))
    start = time.perf_counter()
    for info, belief, moves in positions:
        search.run(info, moves=moves, belief=belief)
    seconds = time.perf_counter() - start
    movegen = simulator.stats["movegen_seconds"]
    inference = evaluator.seconds
    emit(results, args.output, {
        "benchmark": "search", "config": str(args.config), "method": search_config.method,
        "simulations": search_config.simulations, "depth": search_config.depth,
        "candidates": search_config.candidates, "model": config.model.arch,
        "params": parameter_count(model), "decisions": len(positions),
        "seconds_per_decision": seconds / len(positions),
        "movegen_fraction": movegen / seconds, "inference_fraction": inference / seconds,
        "simulator_step_fraction": simulator.stats["step_seconds"] / seconds,
        "movegen_calls_per_decision": simulator.stats["movegen_calls"] / len(positions),
        "inference_calls_per_decision": evaluator.calls / len(positions),
        "inference_ms_per_call": 1000 * inference / max(1, evaluator.calls)})
    simulator.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("components", nargs="+",
                        choices=("movegen", "inference", "training", "search", "all"))
    parser.add_argument("--dictionary", default="WOW24v3.zip",
                        help="lexicon path, or 'tiny' for the built-in test lexicon")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--threads", type=int, default=1, help="torch CPU threads")
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=list(SIZES))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 16, 64, 256])
    parser.add_argument("--train-batch-sizes", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--positions", type=int, default=40)
    parser.add_argument("--decisions", type=int, default=6)
    parser.add_argument("--config", default=str(ROOT / "configs" / "medium.yaml"),
                        help="config whose search/model settings the search benchmark uses")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output", help="append JSON lines here")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is not available in this environment")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    words = lexicon_for_path(None if args.dictionary == "tiny" else args.dictionary).words
    components = (("movegen", "inference", "training", "search")
                  if "all" in args.components else args.components)
    results: list = []
    print(json.dumps({"benchmark": "environment", "device": args.device,
                      "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                      "torch": torch.__version__, "threads": args.threads}), flush=True)
    for name in components:
        {"movegen": bench_movegen, "inference": bench_inference,
         "training": bench_training, "search": bench_search}[name](args, words, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
