import json

import pytest

from targetregen.cli import main
from targetregen.evaluation import summarize


def write_run(path, scale):
    """An evaluation file of the form pipeline/evaluate_vision.py writes, two prompts a domain."""
    cells = {}
    for d, (tau, rounds) in {"caption": ((2.0, 4.0), (10, 30)), "textvqa": ((3.0, 3.0), (5, 5)),
                             "docvqa": ((4.5, 4.5), (8, 2))}.items():
        rows = [{"id": str(i), "draft": {"tau": scale * t, "rounds": n}} for i, (t, n) in enumerate(zip(tau, rounds))]
        cells[f"{d}/T0"] = {"rows": rows}
    path.write_text(json.dumps({"cells": cells}))
    return path


def test_summary_pools_rounds_and_averages_runs(tmp_path):
    a, b = write_run(tmp_path / "a.json", 1.0), write_run(tmp_path / "b.json", 1.1)
    s = summarize([a, b])["T0"]
    # caption pools (2 x 10 + 4 x 30) / 40 = 3.5
    assert s["domains"]["caption"] == pytest.approx((3.5 + 3.85) / 2)
    run_a = (3.5 * 3.0 * 4.5) ** (1 / 3)
    assert s["runs"] == pytest.approx([run_a, 1.1 * run_a])
    assert s["mat"] == pytest.approx(1.05 * run_a)


def test_command_line(tmp_path, capsys):
    assert main(["demo"]) == 0
    assert "ratio of ratios -10.4%" in capsys.readouterr().out
    a = write_run(tmp_path / "a.json", 1.0)
    assert main(["summary", str(a)]) == 0
    assert "T0: caption 3.500  textvqa 3.000  docvqa 4.500  MAT 3.615 (1 run)" in capsys.readouterr().out
