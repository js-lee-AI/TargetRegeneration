"""targetregen: draft-training objectives measured against a target-regenerated baseline.

    from targetregen import mat, geomean, increment, ratio_of_ratios

    mat([30, 12], [10, 6])               # pooled MAT of two prompts: 42 tokens in 16 rounds, 2.625
    geomean([2.0, 3.0, 4.5])             # a run's MAT, the geometric mean of its domain MATs, 3.0
    ratio_of_ratios(3.0, 15.0)           # a 15% increment that falls to 3% shrinks by 10.4%

The metrics need only the standard library. targetregen.bootstrap (the paper's prompt bootstrap)
and targetregen.objectives (numpy forms of the training objectives) need numpy. The GPU pipeline
that regenerates the corpus, trains the drafters and evaluates them lives in pipeline/.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .metrics import cell, geomean, increment, mat, ratio_of_ratios, run_mat

__all__ = ["__version__", "mat", "geomean", "run_mat", "cell", "increment", "ratio_of_ratios"]
