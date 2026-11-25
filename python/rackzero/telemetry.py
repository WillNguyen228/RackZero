"""Experiment logging: JSONL/CSV metrics, optional TensorBoard, hardware sampling.

GPU utilization and memory come from ``nvidia-smi`` (sampled in a background
thread). They are *measured*, not inferred from the configured device. When
``nvidia-smi`` is unavailable those fields are ``null``.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time


def flatten_metrics(record: dict, prefix: str = "") -> dict:
    result = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            result.update(flatten_metrics(value, name + "/"))
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    result[f"{name}/{i}"] = item
        elif isinstance(value, bool):
            result[name] = int(value)
        elif isinstance(value, (int, float)) or value is None or isinstance(value, str):
            result[name] = value
    return result


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class MetricsWriter:
    """Appends JSONL, rewrites a flattened CSV and optionally logs to TensorBoard."""

    def __init__(self, run_dir: Path, tensorboard: bool = False):
        self.run_dir = Path(run_dir)
        self.jsonl = self.run_dir / "metrics.jsonl"
        self.csv = self.run_dir / "metrics.csv"
        self.tensorboard = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tensorboard = SummaryWriter(str(self.run_dir / "tensorboard"))
            except Exception as error:  # tensorboard is optional
                print(f"TensorBoard disabled: {error}", flush=True)

    def write(self, record: dict, history: list[dict]) -> None:
        record = _json_safe(record)
        with self.jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        rows = [flatten_metrics(_json_safe(item)) for item in history]
        columns = []
        for row in rows:
            columns.extend(k for k in row if k not in columns)
        staging = self.csv.with_suffix(".csv.partial")
        with staging.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        from rackzero.checkpoints import _replace_with_retry
        _replace_with_retry(staging, self.csv)
        if self.tensorboard is not None:
            step = int(record.get("iteration", 0))
            for key, value in flatten_metrics(record).items():
                if isinstance(value, (int, float)) and value is not None:
                    self.tensorboard.add_scalar(key, value, step)
            self.tensorboard.flush()

    def close(self) -> None:
        if self.tensorboard is not None:
            self.tensorboard.close()


def _gpu_query(uuid: str | None):
    if shutil.which("nvidia-smi") is None:
        return None
    command = ["nvidia-smi", "--query-gpu=uuid,utilization.gpu,memory.used,memory.total",
               "--format=csv,noheader,nounits"]
    try:
        output = subprocess.run(command, capture_output=True, text=True, timeout=10,
                                check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    rows = [line.split(", ") for line in output.strip().splitlines() if line.strip()]
    if not rows:
        return None
    row = next((r for r in rows if uuid and r[0].strip() == uuid), rows[0])
    try:
        return float(row[1]), float(row[2]), float(row[3])
    except (IndexError, ValueError):
        return None


def _cpu_times():
    try:
        import psutil
        return ("psutil", psutil.cpu_percent(interval=None))
    except ImportError:
        pass
    if hasattr(os, "getloadavg"):
        return ("loadavg", os.getloadavg()[0])
    return (None, None)


class HardwareMonitor:
    """Samples GPU utilization/memory and CPU load while a phase runs."""

    def __init__(self, interval: float, device: str):
        self.interval = interval
        self.device = device
        self.uuid = None
        if device.startswith("cuda"):
            try:
                import torch
                self.uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
            except Exception:
                self.uuid = None
        self.samples: list[tuple] = []
        self._thread = None
        self._stop = threading.Event()

    def __enter__(self):
        self.samples = []
        if self.interval > 0:
            _cpu_times()  # psutil's first cpu_percent() call only primes its counter
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None
            self._sample()  # covers the tail of the phase (and very short phases)

    def _sample(self):
        gpu = _gpu_query(self.uuid) if self.device.startswith("cuda") else None
        cpu_kind, cpu = _cpu_times()
        self.samples.append((gpu, cpu_kind, cpu))

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def summary(self) -> dict:
        gpu = [s[0] for s in self.samples if s[0] is not None]
        cpu = [s[2] for s in self.samples if s[2] is not None]
        kind = next((s[1] for s in self.samples if s[1]), None)
        result = {"samples": len(self.samples)}
        if gpu:
            result.update(
                gpu_util_mean=sum(g[0] for g in gpu) / len(gpu),
                gpu_util_max=max(g[0] for g in gpu),
                gpu_mem_used_mb_max=max(g[1] for g in gpu),
                gpu_mem_total_mb=gpu[0][2],
            )
        if cpu:
            key = "cpu_percent" if kind == "psutil" else "cpu_loadavg"
            result[f"{key}_mean"] = sum(cpu) / len(cpu)
            result[f"{key}_max"] = max(cpu)
        return result


def environment_report(device: str) -> dict:
    import numpy
    import torch

    report = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "numpy": numpy.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "hostname": platform.node(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        report.update(cuda_version=torch.version.cuda, gpu_name=props.name,
                      gpu_memory_gb=round(props.total_memory / 2**30, 2),
                      cudnn=torch.backends.cudnn.version(),
                      bf16_supported=torch.cuda.is_bf16_supported())
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=5, cwd=Path(__file__).resolve().parent).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               timeout=10, cwd=Path(__file__).resolve().parent).stdout.strip()
        report["git_commit"] = commit or None
        report["git_dirty"] = bool(dirty)
    except (OSError, subprocess.SubprocessError):
        report["git_commit"] = None
    report["device"] = device
    report["time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return report
