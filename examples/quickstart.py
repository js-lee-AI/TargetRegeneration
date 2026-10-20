"""The CPU quickstart: KD over CE on external ALLaVA responses, from the released prompt-level counts."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from targetregen import cell, increment, run_mat
from targetregen.bootstrap import PromptBootstrap, ci95

DATA = Path(__file__).resolve().parents[1] / "data" / "vision" / "acceptance.csv"
DOMAINS = ("caption", "textvqa", "docvqa")

counts = defaultdict(lambda: {d: ([], []) for d in DOMAINS})  # run -> domain -> (committed, rounds)
with open(DATA, newline="") as fh:
    for row in csv.DictReader(fh):  # sorted by run, domain and prompt
        committed, rounds = counts[row["run"]][row["domain"]]
        committed.append(int(row["committed"]))
        rounds.append(int(row["rounds"]))


def arrays(run):
    """(committed, rounds) of one run as arrays of shape (domains, prompts), for the bootstrap."""
    return tuple(np.array([counts[run][d][i] for d in DOMAINS]) for i in (0, 1))


ce = [f"allava-external-ce-g7-e3-s{seed}-t0" for seed in (0, 1)]  # the two training runs of each cell
kd = [f"allava-external-kd-g7-e3-s{seed}-t0" for seed in (0, 1)]
mat_ce, sd_ce = cell([run_mat(counts[r])[1] for r in ce])
mat_kd, sd_kd = cell([run_mat(counts[r])[1] for r in kd])
print(f"CE {mat_ce:.3f} ±{sd_ce:.3f}, KD {mat_kd:.3f} ±{sd_kd:.3f}")

boot = PromptBootstrap()  # 10,000 prompt resamples within each domain, shared by every run
lo, hi = ci95(boot.cell([arrays(r) for r in kd]) / boot.cell([arrays(r) for r in ce]))
print(f"KD over CE {increment(mat_kd, mat_ce):+.1f}% [{lo:.1f}, {hi:.1f}]")
# CE 2.875 ±0.009, KD 3.291 ±0.008
# KD over CE +14.4% [13.7, 15.2]
