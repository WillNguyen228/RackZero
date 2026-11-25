"""Checkpoint files, RNG capture and model (de)serialization.

Full training checkpoints use Python pickle via ``torch.load(weights_only=False)``:
**load only files you trust**. Worker model files exchanged inside one run use
``weights_only=True``.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from rackzero.config import Config
from rackzero.models.networks import ModelConfig, build_model


SCHEMA_VERSION = 2
V1_REQUIRED = {"config", "iteration", "model", "best_model", "optimizer", "rng",
               "replay", "history", "dictionary_fingerprint"}
V2_REQUIRED = {"config", "iteration", "model", "best_model", "optimizer", "scheduler", "rng",
               "history", "counters", "dictionary_fingerprint", "model_config"}


def capture_rng(rng: np.random.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "generator": copy.deepcopy(rng.bit_generator.state),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict, rng: np.random.Generator, allow_missing_cuda: bool = False) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    rng.bit_generator.state = state["generator"]
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            if allow_missing_cuda:
                return
            raise ValueError("CUDA RNG checkpoint requires CUDA to resume exactly "
                             "(pass --allow-config-change to resume on CPU anyway)")
        states = [item.cpu() for item in state["cuda"]]
        torch.cuda.set_rng_state_all(states[:torch.cuda.device_count()])


def atomic_checkpoint(path, payload) -> None:
    """Atomically replace a checkpoint, leaving the previous file intact on error."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".partial")
    try:
        with staging.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def _replace_with_retry(source: Path, target: Path, attempts: int = 20) -> None:
    """``os.replace``, retrying transient Windows sharing violations.

    Sync clients (e.g. OneDrive) and virus scanners briefly lock files that were
    just written; on Windows that makes the rename fail with PermissionError.
    """
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if os.name != "nt" or attempt == attempts - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def resolve_checkpoint(path) -> Path:
    """A checkpoint file, or a run directory (-> its checkpoints/latest.pt)."""
    path = Path(path)
    if path.is_dir():
        for candidate in (path / "checkpoints" / "latest.pt", path / "latest.pt"):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"No checkpoints/latest.pt in run directory {path}")
    return path


def load_checkpoint(path) -> dict:
    """Load a TRUSTED checkpoint only: pickle can execute arbitrary code."""
    checkpoint = torch.load(resolve_checkpoint(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported RackZero checkpoint schema")
    version = checkpoint.get("schema_version")
    required = {1: V1_REQUIRED, 2: V2_REQUIRED}.get(version)
    if required is None:
        raise ValueError("Unsupported RackZero checkpoint schema")
    if required - checkpoint.keys():
        raise ValueError(f"Incomplete checkpoint: {sorted(required - checkpoint.keys())}")
    return checkpoint


def checkpoint_config(checkpoint: dict) -> Config:
    return Config.from_dict(checkpoint["config"])


def model_from_checkpoint(checkpoint: dict, which: str = "best_model") -> torch.nn.Module:
    """Rebuild a network from a v1 or v2 checkpoint (CPU, eval mode)."""
    if checkpoint["schema_version"] == 1:
        model = build_model({"arch": "legacy_mlp",
                             "legacy_hidden": checkpoint["config"].get("hidden", 128)})
        model.net.load_state_dict(checkpoint[which])
    else:
        model = build_model(checkpoint["model_config"])
        model.load_state_dict(checkpoint[which])
    return model.eval()


def save_model_file(path, model_config: dict, state_dict: dict) -> None:
    """Weights-only model file for self-play/evaluation workers."""
    atomic_checkpoint(path, {"model_config": model_config,
                             "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()}})


def load_model_file(path, device="cpu") -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ModelConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()
