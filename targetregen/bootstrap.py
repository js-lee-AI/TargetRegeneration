"""The paper's prompt bootstrap.

Each of the 10,000 resamples draws 99 prompt indices with replacement inside each of the three
domains, from one NumPy PCG64 stream with seed 20260928, and applies the same indices to every run
being compared. Each run's MAT is recomputed from the drawn prompts, the runs of a cell are averaged,
and increments and ratios of ratios are formed inside each resample. Intervals are the 2.5 and 97.5
percentiles with linear interpolation. The intervals describe prompt uncertainty for these trained
checkpoints; they do not resample training runs.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

SEED = 20260928
RESAMPLES = 10_000


class PromptBootstrap:
    """Domain-stratified prompt resamples, shared by every run they are applied to.

    Indices are drawn one resample at a time, as rng.integers(0, P, size=(D, P)), so the draws are
    identical to those behind the paper's intervals.
    """

    def __init__(self, n_domains: int = 3, n_prompts: int = 99, resamples: int = RESAMPLES, seed: int = SEED):
        rng = np.random.Generator(np.random.PCG64(seed))
        self.shape = (n_domains, n_prompts)
        self.index = np.stack([rng.integers(0, n_prompts, size=self.shape) for _ in range(resamples)])
        self._rows = np.arange(n_domains)[None, :, None]

    @property
    def resamples(self) -> int:
        return self.index.shape[0]

    def run(self, committed: np.ndarray, rounds: np.ndarray) -> np.ndarray:
        """Geometric-mean MAT of one run in every resample. Inputs have shape (domains, prompts)."""
        committed = np.asarray(committed, dtype=np.float64)
        rounds = np.asarray(rounds, dtype=np.float64)
        if committed.shape != self.shape or rounds.shape != self.shape:
            raise ValueError(f"expected arrays of shape {self.shape}")
        c = committed[self._rows, self.index].sum(-1)
        r = rounds[self._rows, self.index].sum(-1)
        return np.exp(np.log(c / r).mean(-1))

    def cell(self, runs: Sequence[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """Mean over the runs of a cell in every resample."""
        return np.mean([self.run(c, r) for c, r in runs], axis=0)


def ci95(ratio_draws: np.ndarray, log_scale: bool = False) -> tuple[float, float]:
    """95% percentile interval of a ratio, in percent change.

    log_scale takes the percentiles of the log ratio, the convention of the frozen factorial analysis;
    both scales agree to the printed precision.
    """
    x = np.asarray(ratio_draws, dtype=np.float64)
    if log_scale:
        lo, hi = np.percentile(np.log(x), [2.5, 97.5], method="linear")
        return 100.0 * (np.exp(lo) - 1.0), 100.0 * (np.exp(hi) - 1.0)
    lo, hi = np.percentile(x, [2.5, 97.5], method="linear")
    return 100.0 * (lo - 1.0), 100.0 * (hi - 1.0)
