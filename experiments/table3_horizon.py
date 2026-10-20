"""Table 3, Figure 4 and Tables 5 and 6: CE, KD and KD + gate across training lengths.

Every schedule is trained and annealed separately (a fractional epoch stops the schedule early, it
is not a checkpoint of a longer run) and evaluated with two training runs. The script also rebuilds
the best-schedule comparisons: each objective's best measured schedule is chosen on the cell means
and held fixed inside the bootstrap. The cross-fitted variant chooses the schedules on one half of
the prompts of each domain (alternate positions) and estimates the ratio of ratios on the other
half; the printed value is the mean of the two directions.
"""
import numpy as np

from common import (DOMAINS, Check, find, fmt_bound, fmt_gain, fmt_mat, fmt_pct, gain, prompts, ratio_of_ratios,
                    run_value, cell_value, domain_values)
from targetregen.metrics import geomean, increment, mat

SCHEDULES = {
    ("allava", "external"): (0.25, 0.5, 1, 2, 3),
    ("allava", "full"): (1, 2, 3, 4, 6),
    ("sharegpt4v", "external"): (0.25, 0.5, 1, 2, 3),
    ("sharegpt4v", "full"): (0.5, 1, 3, 6),
}
GATE_SCHEDULES = {"external": (0.5, 1, 2, 3), "full": (1, 3, 6)}

# corpus, responses, epochs: printed CE, KD, gain (%)
TABLE3 = {
    ("allava", "external", 0.25): ("3.098", "3.318", "+7.1"),
    ("allava", "external", 0.5): ("3.137", "3.357", "+7.0"),
    ("allava", "external", 1): ("3.132", "3.400", "+8.5"),
    ("allava", "external", 2): ("3.078", "3.392", "+10.2"),
    ("allava", "external", 3): ("2.875", "3.291", "+14.4"),
    ("allava", "full", 1): ("3.859", "3.894", "+0.9"),
    ("allava", "full", 2): ("3.895", "3.969", "+1.9"),
    ("allava", "full", 3): ("3.843", "3.987", "+3.7"),
    ("allava", "full", 4): ("3.718", "3.927", "+5.6"),
    ("allava", "full", 6): ("3.491", "3.827", "+9.6"),
    ("sharegpt4v", "external", 0.25): ("3.140", "3.412", "+8.7"),
    ("sharegpt4v", "external", 0.5): ("3.144", "3.459", "+10.0"),
    ("sharegpt4v", "external", 1): ("3.104", "3.434", "+10.6"),
    ("sharegpt4v", "external", 2): ("3.003", "3.402", "+13.3"),
    ("sharegpt4v", "external", 3): ("2.802", "3.307", "+18.0"),
    ("sharegpt4v", "full", 0.5): ("3.895", "3.891", "-0.1"),
    ("sharegpt4v", "full", 1): ("3.984", "4.012", "+0.7"),
    ("sharegpt4v", "full", 3): ("3.872", "4.013", "+3.6"),
    ("sharegpt4v", "full", 6): ("3.499", "3.838", "+9.7"),
}
# appendix: domain MATs, each cell CE / KD, domains in the order caption, TextVQA, DocVQA
DOMAIN_TABLE = {
    ("allava", "external", 0.25): ("2.718 / 2.879", "2.887 / 3.194", "3.789 / 3.971"),
    ("allava", "external", 0.5): ("2.747 / 2.922", "2.928 / 3.247", "3.837 / 3.990"),
    ("allava", "external", 1): ("2.761 / 2.980", "2.894 / 3.240", "3.846 / 4.071"),
    ("allava", "external", 2): ("2.723 / 2.983", "2.849 / 3.224", "3.758 / 4.059"),
    ("allava", "external", 3): ("2.565 / 2.915", "2.650 / 3.136", "3.498 / 3.898"),
    ("sharegpt4v", "external", 0.25): ("2.730 / 3.053", "2.982 / 3.243", "3.804 / 4.014"),
    ("sharegpt4v", "external", 0.5): ("2.747 / 3.119", "3.007 / 3.282", "3.764 / 4.041"),
    ("sharegpt4v", "external", 1): ("2.727 / 3.151", "2.968 / 3.221", "3.694 / 3.991"),
    ("sharegpt4v", "external", 2): ("2.673 / 3.162", "2.877 / 3.202", "3.520 / 3.890"),
    ("sharegpt4v", "external", 3): ("2.542 / 3.112", "2.691 / 3.110", "3.216 / 3.735"),
    ("allava", "full", 1): ("3.362 / 3.390", "3.922 / 3.947", "4.360 / 4.412"),
    ("allava", "full", 2): ("3.417 / 3.473", "3.993 / 4.102", "4.331 / 4.390"),
    ("allava", "full", 3): ("3.377 / 3.506", "3.940 / 4.097", "4.267 / 4.411"),
    ("allava", "full", 4): ("3.303 / 3.452", "3.811 / 4.072", "4.083 / 4.308"),
    ("allava", "full", 6): ("3.091 / 3.383", "3.583 / 3.956", "3.844 / 4.188"),
    ("sharegpt4v", "full", 0.5): ("3.787 / 3.811", "3.724 / 3.688", "4.190 / 4.194"),
    ("sharegpt4v", "full", 1): ("3.961 / 3.992", "3.802 / 3.828", "4.199 / 4.227"),
    ("sharegpt4v", "full", 3): ("4.097 / 4.205", "3.630 / 3.775", "3.904 / 4.072"),
    ("sharegpt4v", "full", 6): ("3.844 / 4.148", "3.215 / 3.601", "3.467 / 3.783"),
}
# appendix: the gate schedules on ALLaVA, printed KD, KD + gate, gate gain (%)
GATE_TABLE = {
    ("external", 0.5): ("3.357", "3.454", "+2.9"),
    ("external", 1): ("3.400", "3.492", "+2.7"),
    ("external", 2): ("3.392", "3.488", "+2.8"),
    ("external", 3): ("3.291", "3.453", "+4.9"),
    ("full", 1): ("3.894", "3.881", "-0.3"),
    ("full", 3): ("3.987", "3.965", "-0.5"),
    ("full", 6): ("3.827", "3.882", "+1.5"),
}
HALVES = (np.arange(0, 99, 2), np.arange(1, 99, 2))  # alternate prompt positions within each domain


def runs(corpus, responses, objective, epochs):
    found = find(corpus, responses, objective, epochs=epochs)
    assert len(found) == 2, (corpus, responses, objective, epochs, found)
    return found


def domain_cell(run_ids):
    return np.mean([domain_values(r) for r in run_ids], axis=0)


def best(corpus, responses, objective, schedules):
    """The schedule with the highest cell MAT (the first one on a tie)."""
    return max(schedules, key=lambda e: cell_value(runs(corpus, responses, objective, e))[0])


def half_value(run_ids, idx):
    values = []
    for r in run_ids:
        c, n = prompts()[r]
        values.append(geomean(mat(c[d, idx], n[d, idx]) for d in range(len(DOMAINS))))
    return float(np.mean(values))


def crossfit(corpus, num, den, schedules):
    """Choose every schedule on one half, estimate the ratio of ratios on the other, average both directions."""
    estimates, choices = [], []
    for sel, est in ((0, 1), (1, 0)):
        pick = {(resp, obj): max(eps, key=lambda e: half_value(runs(corpus, resp, obj, e), HALVES[sel]))
                for (resp, obj), eps in schedules.items()}
        g = {resp: half_value(runs(corpus, resp, num, pick[(resp, num)]), HALVES[est])
             / half_value(runs(corpus, resp, den, pick[(resp, den)]), HALVES[est]) for resp in ("external", "full")}
        estimates.append(100.0 * (g["full"] / g["external"] - 1.0))
        choices.append(pick)
    return float(np.mean(estimates)), choices


def main():
    check = Check()
    print("Table 3: vision MAT across training lengths (two runs a cell, RTX A6000, T=0)")
    print("The interval of each gain is the one Figure 4b plots for ALLaVA; the paper prints no number for it.")
    print(f"{'corpus':11s} {'responses':10s} {'epochs':>6s} {'CE':>7s} {'KD':>7s} {'gain (%)':>9s}  95% interval")
    for (corpus, responses, epochs), printed in TABLE3.items():
        ce, kd = runs(corpus, responses, "ce", epochs), runs(corpus, responses, "kd", epochs)
        tag = f"{corpus} {responses} {epochs} ep"
        a = check(f"{tag} CE", fmt_mat(cell_value(ce)[0]), printed[0])
        b = check(f"{tag} KD", fmt_mat(cell_value(kd)[0]), printed[1])
        g = check(f"{tag} gain", fmt_pct(increment(cell_value(kd)[0], cell_value(ce)[0])), printed[2])
        _, lo, hi = gain(kd, ce)
        print(f"{corpus:11s} {responses:10s} {epochs:>6g} {a:>7s} {b:>7s} {g:>9s}  [{fmt_bound(lo)}, {fmt_bound(hi)}]")

    print("\nDomain MATs, CE / KD")
    for (corpus, responses, epochs), printed in DOMAIN_TABLE.items():
        ce, kd = (domain_cell(runs(corpus, responses, o, epochs)) for o in ("ce", "kd"))
        cells = [check(f"{corpus} {responses} {epochs} ep {d}", f"{fmt_mat(x)} / {fmt_mat(y)}", p)
                 for d, x, y, p in zip(DOMAINS, ce, kd, printed)]
        print(f"{corpus:11s} {responses:10s} {epochs:>6g}   " + "   ".join(cells))

    print("\nGate schedules on ALLaVA: KD, KD + gate, gate gain (%)")
    for (responses, epochs), printed in GATE_TABLE.items():
        kd, gate = runs("allava", responses, "kd", epochs), runs("allava", responses, "gate", epochs)
        tag = f"gate {responses} {epochs} ep"
        a = check(f"{tag} KD", fmt_mat(cell_value(kd)[0]), printed[0])
        b = check(f"{tag} KD + gate", fmt_mat(cell_value(gate)[0]), printed[1])
        g = check(f"{tag} gain", fmt_pct(increment(cell_value(gate)[0], cell_value(kd)[0])), printed[2])
        print(f"{responses:10s} {epochs:>6g} {a:>7s} {b:>7s} {g:>7s}")

    print("\nBest measured schedules (held fixed inside the bootstrap)")
    for corpus, num, den, label, sched, want in (
            ("allava", "kd", "ce", "ALLaVA, KD over CE", SCHEDULES, ("+8.4", "+2.4", "-5.6 [-6.4, -4.8]")),
            ("allava", "gate", "kd", "ALLaVA, gate over KD", None, ("+2.7", "-0.5", "-3.1 [-4.0, -2.3]")),
            ("sharegpt4v", "kd", "ce", "ShareGPT4V, KD over CE", SCHEDULES, (None, None, "-8.4 [-9.4, -7.4]"))):
        pick = {}
        for resp in ("external", "full"):
            for obj in (num, den):
                pick[(resp, obj)] = best(corpus, resp, obj, GATE_SCHEDULES[resp] if sched is None else sched[(corpus, resp)])
        cells = {k: runs(corpus, k[0], k[1], e) for k, e in pick.items()}
        inc = {resp: increment(cell_value(cells[(resp, num)])[0], cell_value(cells[(resp, den)])[0])
               for resp in ("external", "full")}
        given, reference = (cells[("full", num)], cells[("full", den)]), (cells[("external", num)], cells[("external", den)])
        ror = check(f"{label} ratio of ratios", fmt_gain(*ratio_of_ratios(given, reference)), want[2])
        if want[0]:
            check(f"{label} on external", fmt_pct(inc["external"]), want[0])
            check(f"{label} on regenerated", fmt_pct(inc["full"]), want[1])
        if corpus == "sharegpt4v":
            check("ShareGPT4V best schedules", str({f"{r}/{o}": e for (r, o), e in pick.items()}),
                  "{'external/kd': 0.5, 'external/ce': 0.5, 'full/kd': 3, 'full/ce': 1}")
        print(f"  {label:24s} schedules {dict((f'{r}/{o}', e) for (r, o), e in pick.items())}")
        print(f"  {'':24s} external {fmt_pct(inc['external'])}, regenerated {fmt_pct(inc['full'])}, ratio of ratios {ror}")

    print("\nSentences of the horizon paragraph")
    quarter = check("quarter vs half epoch, external ALLaVA CE",
                    fmt_gain(*gain(runs("allava", "external", "ce", 0.25), runs("allava", "external", "ce", 0.5))),
                    "-1.2 [-2.1, -0.4]")
    print(f"  a quarter epoch against half an epoch, external ALLaVA CE: {quarter}")
    for obj, want in (("ce", "-8.2"), ("kd", "-3.2")):
        v = check(f"external ALLaVA {obj} 3 vs 1 epochs", fmt_pct(increment(
            cell_value(runs("allava", "external", obj, 3))[0], cell_value(runs("allava", "external", obj, 1))[0])), want)
        print(f"  external ALLaVA {obj.upper()}, three epochs against one: {v}")
    kd1, kd3 = (cell_value(runs("sharegpt4v", "full", "kd", e))[0] for e in (1, 3))
    v = check("ShareGPT4V regenerated KD, 3 vs 1 epochs", f"{100 * (kd3 / kd1 - 1):.2f}", "0.03")
    print(f"  regenerated ShareGPT4V KD, three epochs against one: {v}%")

    print("\nCross-fitted best schedules (chosen on one half of the prompts, estimated on the other)")
    sv = {(r, o): SCHEDULES[("sharegpt4v", r)] for r in ("external", "full") for o in ("ce", "kd")}
    value, choices = crossfit("sharegpt4v", "kd", "ce", sv)
    a = check("ShareGPT4V cross-fitted KD ratio of ratios", fmt_pct(value), "-8.6")
    check("ShareGPT4V halves choose regenerated KD", str([c[("full", "kd")] for c in choices]), "[1, 3]")
    print(f"  ShareGPT4V, KD over CE: {a}; regenerated KD chosen by the two halves: {[c[('full', 'kd')] for c in choices]}")
    gs = {(r, o): GATE_SCHEDULES[r] for r in ("external", "full") for o in ("kd", "gate")}
    value, choices = crossfit("allava", "gate", "kd", gs)
    b = check("ALLaVA cross-fitted gate ratio of ratios", fmt_pct(value), "-3.2")
    check("ALLaVA gate halves agree", str(choices[0] == choices[1]), "True")
    print(f"  ALLaVA, gate over KD: {b}; both halves choose {dict((f'{r}/{o}', e) for (r, o), e in choices[0].items())}")
    return check.done()


if __name__ == "__main__":
    raise SystemExit(main())
