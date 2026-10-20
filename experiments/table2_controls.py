"""Table 2, vision controls: CE and KD on external and fully regenerated ALLaVA responses.

Random initialization starts the drafter's layers from random weights (three and six epochs), the
lower learning rate is one third of the default, and the L2 anchor pulls the drafter toward the
released one after every step. Each cell averages two runs; gains carry 95% prompt-bootstrap
intervals. The text rows of Table 2 come from the text study and are not rebuilt here.
"""
from common import Check, find, fmt_cell, fmt_gain, gain

# label, find() settings, responses, printed CE, KD, KD gain
ROWS = [
    ("Rand. init., 3 ep.", dict(setting="random_init"), "external", "2.066 ±.002", "2.259 ±.002", "+9.3 [8.7, 9.9]"),
    ("", dict(setting="random_init"), "full", "2.828 ±.002", "2.912 ±.029", "+3.0 [2.4, 3.5]"),
    ("Rand. init., 6 ep.", dict(setting="random_init", epochs=6), "external", "2.071 ±.022", "2.349 ±.020",
     "+13.4 [12.7, 14.2]"),
    ("", dict(setting="random_init", epochs=6), "full", "3.082 ±.014", "3.200 ±.041", "+3.8 [3.2, 4.4]"),
    ("Lower LR", dict(setting="lower_lr"), "external", "3.099 ±.007", "3.474 ±.016", "+12.1 [11.3, 12.9]"),
    ("", dict(setting="lower_lr"), "full", "3.893 ±.009", "4.000 ±.002", "+2.8 [2.2, 3.3]"),
    ("L2 anchor", dict(setting="l2_anchor"), "external", "3.152 ±.007", "3.474 ±<.001", "+10.2 [9.4, 11.0]"),
    ("", dict(setting="l2_anchor"), "full", "3.937 ±.013", "4.005 ±.014", "+1.7 [1.1, 2.3]"),
]


def main():
    check = Check()
    print("Vision controls, Qwen3-VL-8B on ALLaVA, RTX A6000, T=0")
    print(f"{'condition':20s} {'responses':10s} {'CE':>12s} {'KD':>12s}   {'KD gain (%)':>19s}")
    for label, settings, responses, *printed in ROWS:
        ce, kd = (find("allava", responses, o, **settings) for o in ("ce", "kd"))
        assert len(ce) == len(kd) == 2, (label, responses)
        a = check(f"{label or 'same'} {responses} CE", fmt_cell(ce), printed[0])
        b = check(f"{label or 'same'} {responses} KD", fmt_cell(kd), printed[1])
        g = check(f"{label or 'same'} {responses} gain", fmt_gain(*gain(kd, ce)), printed[2])
        print(f"{label:20s} {responses:10s} {a:>12s} {b:>12s}   {g:>19s}")
    return check.done()


if __name__ == "__main__":
    raise SystemExit(main())
