"""Paired bootstrap confidence intervals (G4 size steps, the BM25 trigger)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    """The mean of `a - b` and its two-sided confidence interval, in the input's units."""

    mean: float
    low: float
    high: float
    samples: int
    n: int

    def scaled(self, factor: float) -> BootstrapCI:
        return BootstrapCI(
            self.mean * factor, self.low * factor, self.high * factor, self.samples, self.n
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean,
            "low": self.low,
            "high": self.high,
            "samples": self.samples,
            "n": self.n,
        }


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> BootstrapCI:
    """Percentile bootstrap of the mean paired difference `a - b`.

    Questions are resampled with replacement, keeping each question's pair
    together, `samples` times with a fixed `seed`.
    """
    if len(a) != len(b):
        raise ValueError(f"paired samples differ in length: {len(a)} and {len(b)}")
    x = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(x)
    if n == 0:
        raise ValueError("no paired samples")
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    step = max(1, 2_000_000 // n)  # bound memory: at most ~2M indices at once
    for start in range(0, samples, step):
        stop = min(samples, start + step)
        idx = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = x[idx].mean(axis=1)
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(means, [tail, 100.0 - tail])
    return BootstrapCI(float(x.mean()), float(low), float(high), samples, n)
