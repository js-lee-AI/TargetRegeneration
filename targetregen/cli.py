"""The `targetregen` command. `python -m targetregen` runs the same thing."""
from __future__ import annotations

import argparse
import sys

from . import __version__


def _demo(args):
    import numpy as np

    from .metrics import geomean, increment, mat, ratio_of_ratios
    from .objectives import hard_gate, slot_weights

    print(f"MAT of two prompts, 42 tokens in 16 rounds: {mat([30, 12], [10, 6]):.3f}")
    print(f"run MAT, the geometric mean of domain MATs 2.0, 3.0 and 4.5: {geomean([2.0, 3.0, 4.5]):.3f}")
    external, regenerated = increment(3.45, 3.0), increment(4.12, 4.0)
    print(f"KD over CE: external {external:+.1f}%, regenerated {regenerated:+.1f}%, "
          f"ratio of ratios {ratio_of_ratios(regenerated, external):+.1f}%")
    print("slot weights for gamma = 7:", " ".join(f"{w:.3f}" for w in slot_weights()[:5]))
    gate = hard_gate(np.array([[5, 7, 9, 4, 4]]), np.array([[5, 7, 9, 8, 4]]))[0]
    print("gate of a block whose slot 3 disagrees with the target:", " ".join(str(int(g)) for g in gate))
    return 0


def _summary(args):
    from .evaluation import summarize

    for temperature, s in summarize(args.results).items():
        domains = "  ".join(f"{d} {v:.3f}" for d, v in s["domains"].items())
        sd = "" if s["sd"] is None else f" ±{s['sd']:.3f}"
        n = len(s["runs"])
        print(f"{temperature}: {domains}  MAT {s['mat']:.3f}{sd} ({n} run{'s' if n > 1 else ''})")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="targetregen",
        description="Draft-training objectives measured against a target-regenerated baseline.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the CPU quickstart")
    demo.set_defaults(func=_demo)

    summary = sub.add_parser("summary", help="MAT of evaluation files, one file for each training run")
    summary.add_argument("results", nargs="+", help="JSON files written by pipeline/evaluate_vision.py")
    summary.set_defaults(func=_summary)

    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    return args.func(args)
