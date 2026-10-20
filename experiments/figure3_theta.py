"""Figure 3 and Table 8: ratios of ratios (vision rows).

A ratio of ratios compares an increment on the given responses with the same increment on external
responses: (num / den on given) / (num / den on external) - 1, in percent, with 95% prompt-bootstrap
intervals. Negative values mean that the increment shrinks. Data-order rows use one training run in
each order. table3_horizon.py rebuilds the best-schedule rows; the text rows are not rebuilt.
"""
from common import Check, find, fmt_gain, ratio_of_ratios

PAIRS = {"KD": ("kd", "ce"), "Gate": ("gate", "kd"), "Soft weight": ("soft", "kd")}

# group, label, responses, find() settings, printed {increment: value}
ROWS = [
    ("Vision, tree verifier", "Full regeneration", "full", {}, {"KD": "-9.4 [-10.2, -8.5]", "Gate": "-5.2 [-6.0, -4.4]"}),
    ("", "Full regeneration, T=1", "full", dict(temperature=1),
     {"KD": "-8.2 [-9.1, -7.3]", "Gate": "-4.0 [-4.8, -3.1]"}),
    ("", "Truncated", "truncated", {}, {"KD": "-6.6 [-7.5, -5.6]", "Gate": "-2.6 [-3.4, -1.8]"}),
    ("", "Loss window", "window", {}, {"KD": "-7.2 [-8.1, -6.3]", "Gate": "-4.5 [-5.3, -3.7]"}),
    ("", "Sampled, T=1", "sampled_t1", {}, {"KD": "-8.9 [-9.7, -8.2]", "Gate": "-2.7 [-3.5, -1.9]"}),
    ("", "Sampled, T=0.7", "sampled_t07", {}, {"KD": "-9.5 [-10.3, -8.7]", "Gate": "-3.7 [-4.5, -2.8]"}),
    ("", "ShareGPT4V, full regeneration", "full", dict(corpus="sharegpt4v"),
     {"KD": "-12.2 [-13.0, -11.4]", "Gate": "-4.8 [-5.7, -4.0]"}),
    ("Vision, chain verifier", "Full regeneration", "full", dict(decoder="chain"),
     {"KD": "-8.9 [-9.6, -8.1]", "Gate": "-4.9 [-5.7, -4.2]"}),
    ("", "Truncated", "truncated", dict(decoder="chain"), {"KD": "-6.4 [-7.2, -5.6]", "Gate": "-4.6 [-5.3, -3.8]"}),
    ("", "Loss window", "window", dict(decoder="chain"), {"KD": "-7.2 [-8.0, -6.4]", "Gate": "-5.4 [-6.2, -4.6]"}),
    ("Vision controls, full regeneration", "Random initialization, three epochs", "full", dict(setting="random_init"),
     {"KD": "-5.8 [-6.5, -5.2]"}),
    ("", "Random initialization, six epochs", "full", dict(setting="random_init", epochs=6),
     {"KD": "-8.5 [-9.3, -7.7]"}),
    ("", "gamma = 2", "full", dict(gamma=2),
     {"KD": "-10.4 [-11.2, -9.5]", "Gate": "-4.6 [-5.5, -3.8]", "Soft weight": "-4.5 [-5.4, -3.7]"}),
    ("", "Second data order", "full", dict(order=1000), {"KD": "-10.0 [-11.0, -8.9]", "Gate": "-3.9 [-5.0, -2.7]"}),
    ("", "Third data order", "full", dict(order=2000), {"KD": "-10.4 [-11.5, -9.3]", "Gate": "-3.8 [-5.0, -2.5]"}),
    ("", "Lower learning rate", "full", dict(setting="lower_lr"), {"KD": "-8.3 [-9.2, -7.5]"}),
    ("", "L2 anchor", "full", dict(setting="l2_anchor"), {"KD": "-7.7 [-8.5, -6.9]"}),
    ("", "64 distilled tokens", "full", dict(setting="kd64"), {"KD": "-9.9 [-10.8, -9.1]"}),
]


def cell(responses, objective, settings):
    """A control that trains only KD (64 distilled tokens) reuses the factorial CE cells."""
    s = dict(settings)
    if s.get("setting") == "kd64" and objective == "ce":
        s.pop("setting")
    corpus = s.pop("corpus", "allava")
    runs = find(corpus, responses, objective, **s)
    assert runs, (responses, objective, settings)
    return runs


def main():
    check = Check()
    for group, label, responses, settings, printed in ROWS:
        if group:
            print(f"\n{group}")
        for name, want in printed.items():
            num, den = PAIRS[name]
            given = (cell(responses, num, settings), cell(responses, den, settings))
            reference = (cell("external", num, settings), cell("external", den, settings))
            value = check(f"{group or 'same'} / {label} / {name}", fmt_gain(*ratio_of_ratios(given, reference)), want)
            print(f"  {label:38s} {name:12s} {value:>20s}")
    return check.done()


if __name__ == "__main__":
    raise SystemExit(main())
