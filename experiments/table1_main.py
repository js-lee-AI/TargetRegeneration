"""Table 1: MAT of CE, KD and KD + gate on each response version, and the two increments.

Every cell averages two training runs (anchor seeds 0 and 1); ± is their standard deviation.
Increments compare cell means, with 95% prompt-bootstrap intervals. The Agreement and Kept columns
of the paper are computed on the training corpora with the target and are not rebuilt here.
"""
from common import NAMES, OBJECTIVES, Check, find, fmt_cell, fmt_gain, gain

# corpus, responses, printed CE, KD, KD + gate, KD over CE, gate over KD
PANEL_A = [
    ("allava", "external", "2.875 ±.009", "3.291 ±.008", "3.453 ±.007", "+14.4 [13.7, 15.2]", "+4.9 [4.3, 5.6]"),
    ("allava", "full", "3.843 ±.030", "3.987 ±.002", "3.965 ±.012", "+3.7 [3.0, 4.5]", "-0.5 [-1.1, 0.0]"),
    ("allava", "truncated", "3.483 ±.014", "3.724 ±.008", "3.807 ±.031", "+6.9 [6.2, 7.7]", "+2.2 [1.6, 2.9]"),
    ("allava", "window", "3.568 ±.001", "3.789 ±.008", "3.797 ±.006", "+6.2 [5.4, 6.9]", "+0.2 [-0.4, 0.8]"),
    ("allava", "sampled_t1", "3.756 ±.043", "3.915 ±.009", "3.997 ±.035", "+4.2 [3.6, 4.8]", "+2.1 [1.5, 2.7]"),
    ("allava", "sampled_t07", "3.810 ±.020", "3.947 ±.015", "3.989 ±<.001", "+3.6 [2.9, 4.3]", "+1.1 [0.5, 1.7]"),
    ("sharegpt4v", "external", "2.802 ±.003", "3.307 ±.005", "3.507 ±.015", "+18.0 [17.2, 18.8]", "+6.1 [5.3, 6.8]"),
    ("sharegpt4v", "full", "3.872 ±.024", "4.013 ±.006", "4.051 ±.018", "+3.6 [3.0, 4.3]", "+0.9 [0.3, 1.5]"),
]
# control, settings, responses, printed values
PANEL_B = [
    ("gamma = 2", dict(gamma=2), "external", "2.890 ±.010", "3.365 ±.020", "3.526 ±.006", "+16.4 [15.6, 17.3]", "+4.8 [4.0, 5.6]"),
    ("", dict(gamma=2), "full", "3.864 ±.005", "4.033 ±.004", "4.032 ±.003", "+4.4 [3.6, 5.1]", "0.0 [-0.7, 0.6]"),
    ("Chain verifier", dict(decoder="chain"), "external", "2.040 ±.001", "2.261 ±<.001", "2.397 ±.005", "+10.8 [10.1, 11.5]", "+6.0 [5.3, 6.7]"),
    ("", dict(decoder="chain"), "full", "2.640 ±.008", "2.666 ±.004", "2.687 ±.011", "+1.0 [0.4, 1.5]", "+0.8 [0.3, 1.3]"),
    ("", dict(decoder="chain"), "truncated", "2.448 ±.018", "2.540 ±.005", "2.569 ±.002", "+3.8 [3.2, 4.3]", "+1.2 [0.6, 1.7]"),
    ("", dict(decoder="chain"), "window", "2.498 ±.011", "2.568 ±.003", "2.575 ±.006", "+2.8 [2.3, 3.3]", "+0.3 [-0.2, 0.8]"),
    ("T = 1", dict(temperature=1), "external", "2.793 ±.001", "3.180 ±.010", "3.314 ±.002", "+13.9 [13.0, 14.7]", "+4.2 [3.5, 4.9]"),
    ("", dict(temperature=1), "full", "3.576 ±.012", "3.738 ±.007", "3.740 ±.014", "+4.5 [3.8, 5.2]", "0.0 [-0.6, 0.7]"),
]
HEAD = f"{'':16s} {'CE':>12s} {'KD':>12s} {'KD + gate':>12s}   {'KD over CE':>19s}   {'gate over KD':>17s}"


def row(check, label, cells, printed):
    mats = [check(f"{label} {o}", fmt_cell(cells[o]), p) for o, p in zip(OBJECTIVES, printed[:3])]
    kd = check(f"{label} KD over CE", fmt_gain(*gain(cells["kd"], cells["ce"])), printed[3])
    gate = check(f"{label} gate over KD", fmt_gain(*gain(cells["gate"], cells["kd"])), printed[4])
    return f"{''.join(f'{m:>13s}' for m in mats)}   {kd:>19s}   {gate:>17s}"


def main():
    check = Check()
    print("(a) Vision factorial, Qwen3-VL-8B after three epochs, RTX A6000, T=0")
    print(HEAD)
    corpus = None
    for c, responses, *printed in PANEL_A:
        if c != corpus:
            print(f"-- {'ALLaVA' if c == 'allava' else 'ShareGPT4V'}")
            corpus = c
        cells = {o: find(c, responses, o) for o in OBJECTIVES}
        assert all(len(v) == 2 for v in cells.values()), (c, responses, cells)
        print(f"{NAMES[responses]:16s}" + row(check, f"{c} {responses}", cells, printed))
    print("\n(b) Controls on ALLaVA")
    print(HEAD)
    for control, settings, responses, *printed in PANEL_B:
        if control:
            print(f"-- {control}")
        cells = {o: find("allava", responses, o, **settings) for o in OBJECTIVES}
        assert all(len(v) == 2 for v in cells.values()), (control, responses, cells)
        print(f"{NAMES[responses]:16s}" + row(check, f"{control or 'control'} {responses} {settings}", cells, printed))
    return check.done()


if __name__ == "__main__":
    raise SystemExit(main())
