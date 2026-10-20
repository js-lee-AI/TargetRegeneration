"""Figure 5 and Table 10: mixtures that replace a share of the external ALLaVA responses with greedy regenerations.

Each mixture keeps the external examples and their order and replaces a random quarter, half or
three quarters of the responses; the 0% and 100% points are the factorial cells. Every cell
averages two runs. The figure plots the two gains with 95% prompt-bootstrap intervals, which the
paper prints only as points. The Agreement column is computed on the training corpora with the
target and is not rebuilt here.
"""
from common import Check, cell_value, find, fmt_bound, fmt_mat, fmt_pct, gain
from targetregen.metrics import increment

SHARES = {0: "external", 25: "mix25", 50: "mix50", 75: "mix75", 100: "full"}
# share (%): printed CE, KD, KD + gate, KD gain, gate gain
TABLE10 = {
    0: ("2.875", "3.291", "3.453", "+14.4", "+4.9"),
    25: ("3.468", "3.724", "3.820", "+7.4", "+2.6"),
    50: ("3.625", "3.829", "3.880", "+5.6", "+1.4"),
    75: ("3.714", "3.907", "3.932", "+5.2", "+0.6"),
    100: ("3.843", "3.987", "3.965", "+3.7", "-0.5"),
}


def main():
    check = Check()
    print("Table 10 and Figure 5: ALLaVA mixtures (two runs a cell, RTX A6000, T=0)")
    print(f"{'share':>5s} {'CE':>7s} {'KD':>7s} {'KD + gate':>9s} {'KD gain':>8s} {'Gate gain':>9s}"
          f"   plotted intervals, KD and gate")
    kd_gain = {}
    for share, printed in TABLE10.items():
        cells = {o: find("allava", SHARES[share], o) for o in ("ce", "kd", "gate")}
        assert all(len(v) == 2 for v in cells.values()), (share, cells)
        mats = [check(f"{share}% {o}", fmt_mat(cell_value(cells[o])[0]), p) for o, p in zip(cells, printed)]
        g_kd, g_gate = gain(cells["kd"], cells["ce"]), gain(cells["gate"], cells["kd"])
        kd_gain[share] = g_kd[0]
        a = check(f"{share}% KD gain", fmt_pct(g_kd[0]), printed[3])
        b = check(f"{share}% gate gain", fmt_pct(g_gate[0]), printed[4])
        intervals = "  ".join(f"[{fmt_bound(g[1])}, {fmt_bound(g[2])}]" for g in (g_kd, g_gate))
        print(f"{share:>4d}% {mats[0]:>7s} {mats[1]:>7s} {mats[2]:>9s} {a:>8s} {b:>9s}   {intervals}")
    covered = (kd_gain[0] - kd_gain[25]) / (kd_gain[0] - kd_gain[100])
    print(f"\nShare of the KD drop from external to full regeneration covered by the first quarter: {covered:.2f}"
          f" (the text: about two thirds)")
    return check.done()


if __name__ == "__main__":
    raise SystemExit(main())
