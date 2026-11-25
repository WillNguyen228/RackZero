"""Small, dependency-free statistics for match results."""

from __future__ import annotations

import math

import numpy as np


Z95 = 1.959963984540054


def wilson_interval(successes: float, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a proportion (draws may count as 0.5 successes)."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_interval(values, z: float = Z95) -> tuple[float, float, float]:
    """Mean and normal-approximation 95% interval (paired-seed games are summarized as-is)."""
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return (0.0, 0.0, 0.0)
    mean = float(values.mean())
    if len(values) < 2:
        return (mean, mean, mean)
    half = z * float(values.std(ddof=1)) / math.sqrt(len(values))
    return (mean, mean - half, mean + half)


def games_for_margin(half_width: float, p: float = 0.5, z: float = Z95) -> int:
    """Games needed for a win-rate interval of +/- half_width (normal approximation)."""
    return math.ceil((z / half_width) ** 2 * p * (1 - p))
