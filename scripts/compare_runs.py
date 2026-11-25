"""Compare RackZero experiment runs side by side.

Reads each run directory's config.yaml and metrics.jsonl. For every run it
prints the settings that differ between runs, plus throughput, the latest
losses, and baseline results with 95% intervals. Evaluation seeds are fixed by
``evaluation.seed``, so runs that share it played the same evaluation deals.

  python scripts/compare_runs.py runs/medium-* runs/large-*
  python scripts/compare_runs.py runs/* --csv runs/comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from rackzero.config import flatten  # noqa: E402


def load_run(path: Path):
    config = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
    metrics = [json.loads(line) for line in (path / "metrics.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    return config, metrics


def summarize(path: Path, config: dict, metrics: list[dict]) -> dict:
    last = metrics[-1]
    sp, tr = last["selfplay"], last["training"]
    row = {
        "run": path.name, "iterations": last["iteration"],
        "hours": sum(m["seconds"] for m in metrics) / 3600,
        "games_per_hour": sum(m["selfplay"]["games"] for m in metrics)
        / max(1e-9, sum(m["selfplay"]["wall_seconds"] for m in metrics)) * 3600,
        "decision_s": sp["decision_seconds_mean"],
        "train_samples_per_s": tr.get("samples_per_second"),
        "peak_gpu_mb": tr.get("peak_gpu_memory_mb"),
        "gpu_util_train": last.get("hardware", {}).get("training", {}).get("gpu_util_mean"),
        "loss": tr.get("loss"), "policy_loss": tr.get("policy_loss"),
        "value_loss": tr.get("value_loss"),
    }
    latest = {}
    for record in metrics:
        for name, result in (record.get("baselines") or {}).items():
            latest[name] = (record["iteration"], result)
    for name, (iteration, result) in sorted(latest.items()):
        low, high = result["win_rate_ci95"]
        row[f"{name}_win"] = result["win_rate"]
        row[f"{name}_ci95"] = f"[{low:.2f},{high:.2f}]"
        row[f"{name}_margin"] = result["mean_score_difference"]
        row[f"{name}_at_iter"] = iteration
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="run directories (globs allowed)")
    parser.add_argument("--csv", help="also write the table to this CSV file")
    args = parser.parse_args(argv)
    paths = []
    for pattern in args.runs:
        paths.extend(Path(p) for p in sorted(glob.glob(pattern)) or [pattern])
    runs = []
    for path in paths:
        if (path / "metrics.jsonl").is_file() and (path / "config.yaml").is_file():
            config, metrics = load_run(path)
            if metrics:
                runs.append((path, config, metrics))
        else:
            print(f"skipping {path}: no config.yaml/metrics.jsonl", file=sys.stderr)
    if not runs:
        print("No completed runs found", file=sys.stderr)
        return 1
    flat = [flatten(config) for _, config, _ in runs]
    keys = sorted(set().union(*flat))
    varying = [k for k in keys if len({json.dumps(f.get(k)) for f in flat}) > 1
               and k not in ("name", "output_root")]
    print("Settings that differ:")
    for key in varying:
        print(f"  {key}: " + "  ".join(f"{p.name}={f.get(key)}" for (p, _, _), f in zip(runs, flat)))
    rows = [summarize(path, config, metrics) for path, config, metrics in runs]
    columns = []
    for row in rows:
        columns.extend(k for k in row if k not in columns)
    print()
    widths = {c: max(len(c), *(len(fmt(r.get(c))) for r in rows)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    for row in rows:
        print("  ".join(fmt(row.get(c)).ljust(widths[c]) for c in columns))
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    return 0


def fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 100 else f"{value:.0f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
