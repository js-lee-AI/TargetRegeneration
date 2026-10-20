"""MAT of evaluation files written by pipeline/evaluate_vision.py.

An evaluation file holds one cell for each domain and temperature ("caption/T0"), and each cell holds
one row for each scored prompt with its mean accepted length and its number of verification rounds.
A file is one training run: its MAT at a temperature is the geometric mean over domains of the
round-pooled domain MAT, and a cell of several runs reports their mean and standard deviation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .metrics import cell, mat, run_mat


def read_run(path) -> dict[str, dict[str, tuple[list[float], list[float]]]]:
    """{temperature: {domain: (committed, rounds)}} for one evaluation file."""
    doc = json.loads(Path(path).read_text())
    out: dict[str, dict[str, tuple[list[float], list[float]]]] = {}
    for key, value in doc["cells"].items():
        domain, temperature = key.rsplit("/", 1)
        rows = value["rows"]
        committed = [r["draft"]["tau"] * r["draft"]["rounds"] for r in rows]
        rounds = [r["draft"]["rounds"] for r in rows]
        out.setdefault(temperature, {})[domain] = (committed, rounds)
    return out


def summarize(paths: Iterable) -> dict[str, dict]:
    """Domain MATs and the cell MAT of the given runs at every temperature they share."""
    runs = [read_run(p) for p in paths]
    if not runs:
        raise ValueError("no evaluation files")
    out = {}
    for temperature in sorted(set.intersection(*(set(r) for r in runs))):
        domains = list(runs[0][temperature])  # in the order of the first file
        if any(set(r[temperature]) != set(domains) for r in runs):
            raise ValueError(f"the runs cover different domains at {temperature}")
        values = [run_mat({d: r[temperature][d] for d in domains})[1] for r in runs]
        mean, sd = cell(values)
        out[temperature] = {
            "domains": {d: sum(mat(*r[temperature][d]) for r in runs) / len(runs) for d in domains},
            "runs": values,
            "mat": mean,
            "sd": sd,
        }
    return out
