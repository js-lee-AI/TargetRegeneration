"""Tables 4 and 7: vision MAT and increments for each domain, with the label control and the released drafter.

A domain value of a cell is the mean over its two runs of the round-pooled domain MAT; the Mean
column is the cell's MAT. Greedy labels trains CE toward the target's greedy token on the corpus
prefix. The initialization row is the released drafter before any training. The script also
rebuilds the label-control increments and the comparisons with the released drafter in the text.
"""
from collections import defaultdict

import numpy as np

from common import Check, cell_value, domain_values, find, fmt_gain, fmt_mat, fmt_pct, gain, run_value, runs
from targetregen.metrics import increment

GROUPS = [("allava", "external"), ("allava", "full"), ("allava", "truncated"), ("allava", "window"),
          ("sharegpt4v", "external"), ("sharegpt4v", "full")]
LABELS = {"ce": "CE", "kd": "KD", "gate": "KD + gate", "greedy_labels": "Greedy labels"}

# Table 4: caption, TextVQA, DocVQA, mean
INITIALIZATION = ("2.495", "2.806", "3.254", "2.835")
TABLE4 = {
    ("allava", "external"): {"ce": ("2.565", "2.650", "3.498", "2.875"), "kd": ("2.915", "3.136", "3.898", "3.291"),
                             "gate": ("3.067", "3.313", "4.052", "3.453"),
                             "greedy_labels": ("2.762", "3.008", "3.736", "3.143")},
    ("allava", "full"): {"ce": ("3.377", "3.940", "4.267", "3.843"), "kd": ("3.506", "4.097", "4.411", "3.987"),
                         "gate": ("3.500", "4.085", "4.360", "3.965"),
                         "greedy_labels": ("3.382", "3.951", "4.227", "3.837")},
    ("allava", "truncated"): {"ce": ("3.141", "3.457", "3.890", "3.483"), "kd": ("3.312", "3.775", "4.131", "3.724"),
                              "gate": ("3.348", "3.881", "4.246", "3.807")},
    ("allava", "window"): {"ce": ("3.154", "3.636", "3.960", "3.568"), "kd": ("3.344", "3.880", "4.190", "3.789"),
                           "gate": ("3.344", "3.883", "4.216", "3.797")},
    ("sharegpt4v", "external"): {"ce": ("2.542", "2.691", "3.216", "2.802"), "kd": ("3.112", "3.110", "3.735", "3.307"),
                                 "gate": ("3.357", "3.269", "3.930", "3.507")},
    ("sharegpt4v", "full"): {"ce": ("4.097", "3.630", "3.904", "3.872"), "kd": ("4.205", "3.775", "4.072", "4.013"),
                             "gate": ("4.205", "3.817", "4.141", "4.051")},
}
# Table 7: overall gain with its interval, then caption, TextVQA, DocVQA
TABLE7 = {
    ("allava", "external"): {"KD": ("+14.4 [13.7, 15.2]", "+13.7", "+18.3", "+11.4"),
                             "Gate": ("+4.9 [4.3, 5.6]", "+5.2", "+5.6", "+3.9")},
    ("allava", "full"): {"KD": ("+3.7 [3.0, 4.5]", "+3.8", "+4.0", "+3.4"),
                         "Gate": ("-0.5 [-1.1, 0.0]", "-0.2", "-0.3", "-1.1")},
    ("allava", "truncated"): {"KD": ("+6.9 [6.2, 7.7]", "+5.4", "+9.2", "+6.2"),
                              "Gate": ("+2.2 [1.6, 2.9]", "+1.1", "+2.8", "+2.8")},
    ("allava", "window"): {"KD": ("+6.2 [5.4, 6.9]", "+6.0", "+6.7", "+5.8"),
                           "Gate": ("+0.2 [-0.4, 0.8]", "0.0", "+0.1", "+0.6")},
    ("sharegpt4v", "external"): {"KD": ("+18.0 [17.2, 18.8]", "+22.4", "+15.6", "+16.1"),
                                 "Gate": ("+6.1 [5.3, 6.8]", "+7.9", "+5.1", "+5.2")},
    ("sharegpt4v", "full"): {"KD": ("+3.6 [3.0, 4.3]", "+2.6", "+4.0", "+4.3"),
                             "Gate": ("+0.9 [0.3, 1.5]", "0.0", "+1.1", "+1.7")},
}
PAIRS = {"KD": ("kd", "ce"), "Gate": ("gate", "kd")}
# Table 1a cells trained on target-written responses
REGENERATED = [("allava", r) for r in ("full", "truncated", "window", "sampled_t1", "sampled_t07")] + [
    ("sharegpt4v", "full")]


def domain_cell(run_ids):
    return np.mean([domain_values(r) for r in run_ids], axis=0)


def main():
    check = Check()
    print("Table 4: vision MAT for each domain (two runs a cell, RTX A6000, T=0)")
    print(f"{'':28s} {'Caption':>8s} {'TextVQA':>8s} {'DocVQA':>8s} {'Mean':>8s}")
    init = [*domain_values("initialization"), run_value("initialization")]
    row = [check(f"initialization {i}", fmt_mat(x), p) for i, (x, p) in enumerate(zip(init, INITIALIZATION))]
    print(f"{'Initialization':28s} " + " ".join(f"{v:>8s}" for v in row))
    for (corpus, responses), cells in TABLE4.items():
        print(f"-- {corpus}, {responses}")
        for objective, printed in cells.items():
            ids = find(corpus, responses, objective)
            assert len(ids) == 2, (corpus, responses, objective)
            values = [*domain_cell(ids), cell_value(ids)[0]]
            row = [check(f"{corpus} {responses} {objective} {i}", fmt_mat(x), p)
                   for i, (x, p) in enumerate(zip(values, printed))]
            print(f"{LABELS[objective]:28s} " + " ".join(f"{v:>8s}" for v in row))

    print("\nTable 7: increments (%), overall with 95% prompt-bootstrap intervals and for each domain")
    for (corpus, responses), rows in TABLE7.items():
        print(f"-- {corpus}, {responses}")
        for name, printed in rows.items():
            num, den = (find(corpus, responses, o) for o in PAIRS[name])
            overall = check(f"{corpus} {responses} {name}", fmt_gain(*gain(num, den)), printed[0])
            by_domain = [check(f"{corpus} {responses} {name} {i}", fmt_pct(increment(a, b)), p)
                         for i, (a, b, p) in enumerate(zip(domain_cell(num), domain_cell(den), printed[1:]))]
            print(f"  {name:5s} {overall:>20s}  " + " ".join(f"{v:>6s}" for v in by_domain))

    print("\nLabel control on ALLaVA")
    ce_e, kd_e, gl_e = (find("allava", "external", o) for o in ("ce", "kd", "greedy_labels"))
    ce_f, kd_f, gl_f = (find("allava", "full", o) for o in ("ce", "kd", "greedy_labels"))
    # the text prints the point estimates; "no measurable gain" for the third comparison
    for label, num, den, printed in (("external, greedy labels over CE", gl_e, ce_e, "+9.3"),
                                     ("external, KD over greedy labels", kd_e, gl_e, "+4.7"),
                                     ("full regeneration, greedy labels over CE", gl_f, ce_f, None),
                                     ("full regeneration, KD over greedy labels", kd_f, gl_f, "+3.9")):
        est, lo, hi = gain(num, den)
        if printed:
            check(label, fmt_pct(est), printed)
        print(f"  {label:42s} {fmt_gain(est, lo, hi):>18s}")

    print("\nThe released drafter and the objective settings")
    start = run_value("initialization")
    v = check("external CE over the released drafter", fmt_pct(increment(cell_value(ce_e)[0], start)), "+1.4")
    print(f"  external ALLaVA CE over the released drafter's {fmt_mat(start)}: {v}%")
    lowest = min((increment(cell_value(find(c, r, o))[0], start), c, r, o)
                 for c, r in REGENERATED for o in ("ce", "kd", "gate"))
    v = check("lowest regenerated cell over the released drafter", fmt_pct(lowest[0]), "+22.9")
    print(f"  lowest regenerated cell of Table 1a over the released drafter: {v}% ({' '.join(lowest[1:])})")
    soft_e, soft_f = (find("allava", r, "soft", gamma=2) for r in ("external", "full"))
    a = check("soft weight, gamma 2, external", fmt_mat(cell_value(soft_e)[0]), "3.542")
    b = check("soft weight, gamma 2, full", fmt_mat(cell_value(soft_f)[0]), "4.053")
    print(f"  KD with the soft survival weight at gamma = 2: external {a}, full regeneration {b}")
    external = defaultdict(list)  # every two-run cell trained on external ALLaVA responses
    for r in runs().values():
        if (r["corpus"], r["responses"], r["decoder"], r["temperature"]) == ("allava", "external", "tree", 0):
            external[(r["objective"], r["setting"], r["gamma"], r["epochs"], r["order"])].append(r["run"])
    best = max((ids for ids in external.values() if len(ids) == 2), key=lambda ids: cell_value(ids)[0])
    assert sorted(best) == sorted(soft_e), best
    v = check("regenerated CE over the best external setting", fmt_pct(increment(cell_value(ce_f)[0],
                                                                                    cell_value(best)[0])), "+8.5")
    print(f"  fully regenerated CE over the best external cell, KD with the soft weight at gamma = 2: {v}%")
    return check.done()


if __name__ == "__main__":
    raise SystemExit(main())
