"""Loaders for data/ and the check that compares each rebuilt value with the printed one."""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))

from targetregen.bootstrap import PromptBootstrap, ci95  # noqa: E402
from targetregen.metrics import cell, geomean, increment, mat  # noqa: E402

DOMAINS = ("caption", "textvqa", "docvqa")
OBJECTIVES = ("ce", "kd", "gate")
NAMES = {"external": "External", "full": "Full", "truncated": "Truncated", "window": "Loss window",
         "sampled_t1": "Sampled, T=1", "sampled_t07": "Sampled, T=0.7"}


@lru_cache(maxsize=None)
def runs():
    out = {}
    with open(DATA / "vision" / "runs.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            for k in ("gamma", "seed", "order", "temperature"):
                r[k] = int(r[k])
            r["epochs"] = float(r["epochs"])
            out[r["run"]] = r
    return out


@lru_cache(maxsize=None)
def prompts():
    """run -> (committed, rounds), each an array of shape (3 domains, 99 prompts)."""
    rows = defaultdict(lambda: {d: [] for d in DOMAINS})
    with open(DATA / "vision" / "acceptance.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            rows[r["run"]][r["domain"]].append((int(r["prompt"]), int(r["committed"]), int(r["rounds"])))
    out = {}
    for run, by_domain in rows.items():
        c, n = [], []
        for d in DOMAINS:
            ordered = sorted(by_domain[d])
            assert [p for p, _, _ in ordered] == list(range(1, 100)), (run, d)
            c.append([x for _, x, _ in ordered])
            n.append([x for _, _, x in ordered])
        out[run] = (np.asarray(c, dtype=np.int64), np.asarray(n, dtype=np.int64))
    return out


def find(corpus="allava", responses="external", objective="ce", setting="main", gamma=7, epochs=3, order=0,
         temperature=0, decoder="tree", seeds=None):
    """Run ids of one cell, ordered by anchor seed."""
    key = (corpus, responses, objective, setting, gamma, float(epochs), order, temperature, decoder)
    hits = [r for r in runs().values()
            if (r["corpus"], r["responses"], r["objective"], r["setting"], r["gamma"], r["epochs"], r["order"],
                r["temperature"], r["decoder"]) == key and (seeds is None or r["seed"] in seeds)]
    return [r["run"] for r in sorted(hits, key=lambda r: r["seed"])]


def run_value(run):
    """Geometric mean over domains of the round-pooled domain MAT."""
    c, n = prompts()[run]
    return geomean(mat(c[d], n[d]) for d in range(len(DOMAINS)))


def domain_values(run):
    c, n = prompts()[run]
    return [mat(c[d], n[d]) for d in range(len(DOMAINS))]


@lru_cache(maxsize=None)
def boot():
    return PromptBootstrap()


def cell_draws(run_ids):
    return boot().cell([prompts()[r] for r in run_ids])


def cell_value(run_ids):
    return cell([run_value(r) for r in run_ids])


def gain(num_runs, den_runs, log_scale=False):
    """Increment of one cell over another in percent, with its 95% prompt-bootstrap interval."""
    estimate = increment(cell_value(num_runs)[0], cell_value(den_runs)[0])
    lo, hi = ci95(cell_draws(num_runs) / cell_draws(den_runs), log_scale=log_scale)
    return estimate, lo, hi


def ratio_of_ratios(given, reference, log_scale=False):
    """(num/den on given responses) over (num/den on reference responses), in percent, with its interval.

    given and reference are (numerator runs, denominator runs) pairs."""
    point = (cell_value(given[0])[0] / cell_value(given[1])[0]) / (cell_value(reference[0])[0] / cell_value(reference[1])[0])
    draws = (cell_draws(given[0]) / cell_draws(given[1])) / (cell_draws(reference[0]) / cell_draws(reference[1]))
    lo, hi = ci95(draws, log_scale=log_scale)
    return 100.0 * (point - 1.0), lo, hi


# printing, in the paper's format

def fmt_mat(x):
    return f"{x:.3f}"


def fmt_sd(x):
    s = f"{x:.3f}".lstrip("0")
    return "<.001" if s == ".000" else s


def fmt_pct(x, digits=1):
    s = f"{x:+.{digits}f}"
    return f"{0:.{digits}f}" if float(s) == 0 else s


def fmt_bound(x, digits=1):
    s = f"{x:.{digits}f}"
    return f"{0:.{digits}f}" if float(s) == 0 else s


def fmt_cell(run_ids):
    mean, sd = cell_value(run_ids)
    return fmt_mat(mean) + ("" if sd is None else " ±" + fmt_sd(sd))


def fmt_gain(estimate, lo, hi):
    return f"{fmt_pct(estimate)} [{fmt_bound(lo)}, {fmt_bound(hi)}]"


class Check:
    """Counts rebuilt values that equal the printed strings."""

    def __init__(self):
        self.n = 0
        self.failed = []

    def __call__(self, label, rebuilt, printed):
        self.n += 1
        if rebuilt != printed:
            self.failed.append((label, rebuilt, printed))
        return rebuilt

    def done(self):
        for label, rebuilt, printed in self.failed:
            print(f"MISMATCH {label}: rebuilt {rebuilt!r}, printed {printed!r}")
        print(f"\n{self.n - len(self.failed)}/{self.n} values match the paper")
        return 1 if self.failed else 0
