"""Every experiment script rebuilds its printed values from data/ and finds no mismatch."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {"table1_main": 80, "table2_controls": 24, "figure3_theta": 32, "table3_horizon": 151,
           "table4_domains": 140, "figure5_mixtures": 25}


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_script_matches_the_paper(name):
    run = subprocess.run([sys.executable, str(ROOT / "experiments" / f"{name}.py")], capture_output=True, text=True,
                         cwd=ROOT)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    found = re.search(r"(\d+)/(\d+) values match the paper", run.stdout)
    assert found and found.group(1) == found.group(2) == str(SCRIPTS[name]), run.stdout[-500:]
