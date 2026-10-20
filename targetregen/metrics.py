"""Mean acceptance length (MAT) and the comparisons the paper reports.

A verification round commits the draft tokens the target accepts plus the token the target adds
itself. The MAT of a domain pools its prompts: committed tokens over rounds. A run's MAT is the
geometric mean of its domain MATs, a cell's MAT the arithmetic mean of its runs.
"""
from __future__ import annotations

import math
import statistics
from typing import Iterable, Mapping, Sequence


def mat(committed: Sequence[float], rounds: Sequence[float]) -> float:
    """Pooled mean acceptance length: total committed tokens over total verification rounds."""
    total = sum(rounds)
    if total <= 0:
        raise ValueError("MAT needs at least one verification round")
    return sum(committed) / total


def geomean(values: Iterable[float]) -> float:
    values = list(values)
    if not values or any(v <= 0 for v in values):
        raise ValueError("a geometric mean needs positive, nonempty values")
    return math.exp(statistics.fmean(math.log(v) for v in values))


def run_mat(domains: Mapping[str, tuple[Sequence[float], Sequence[float]]]) -> tuple[dict[str, float], float]:
    """Domain MATs and their geometric mean for one run, from {domain: (committed, rounds)}."""
    by_domain = {d: mat(c, r) for d, (c, r) in domains.items()}
    return by_domain, geomean(by_domain.values())


def cell(run_values: Sequence[float]) -> tuple[float, float | None]:
    """Mean over the runs of a cell and their sample standard deviation (None for a single run)."""
    if not run_values:
        raise ValueError("a cell needs at least one run")
    sd = statistics.stdev(run_values) if len(run_values) > 1 else None
    return statistics.fmean(run_values), sd


def increment(value: float, reference: float) -> float:
    """Relative increment in percent, for example KD over CE."""
    return 100.0 * (value / reference - 1.0)


def ratio_of_ratios(increment_given: float, increment_reference: float) -> float:
    """Percent change of an increment's ratio on given responses against the same ratio on external ones.

    (1 + G_given) / (1 + G_external) - 1, with both increments in percent. Negative values mean the
    increment shrinks.
    """
    return 100.0 * ((1 + increment_given / 100.0) / (1 + increment_reference / 100.0) - 1.0)
