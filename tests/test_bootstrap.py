import numpy as np
import pytest

from targetregen.bootstrap import SEED, PromptBootstrap, ci95
from targetregen.metrics import geomean, mat


def test_draws_follow_one_pcg64_stream_one_resample_at_a_time():
    boot = PromptBootstrap(resamples=3)
    rng = np.random.Generator(np.random.PCG64(SEED))
    for b in range(3):
        np.testing.assert_array_equal(boot.index[b], rng.integers(0, 99, size=(3, 99)))


def test_a_resample_recomputes_the_pooled_run_mat():
    rng = np.random.default_rng(0)
    rounds = rng.integers(20, 60, size=(3, 99))
    committed = rounds * rng.uniform(2, 5, size=(3, 99))
    boot = PromptBootstrap(resamples=4)
    draws = boot.run(committed, rounds)
    for b in range(4):
        idx = boot.index[b]
        expected = geomean(mat(committed[d, idx[d]], rounds[d, idx[d]]) for d in range(3))
        assert draws[b] == pytest.approx(expected)
    with pytest.raises(ValueError):
        boot.run(committed[:, :98], rounds[:, :98])


def test_cell_averages_runs_inside_each_resample():
    rng = np.random.default_rng(1)
    runs = [(rng.uniform(50, 90, (3, 99)), rng.uniform(20, 30, (3, 99))) for _ in range(2)]
    boot = PromptBootstrap(resamples=5)
    np.testing.assert_allclose(boot.cell(runs), (boot.run(*runs[0]) + boot.run(*runs[1])) / 2)


def test_ci95_in_percent_change():
    assert ci95(np.full(100, 1.05)) == pytest.approx((5.0, 5.0))
    lo, hi = ci95(np.linspace(0.9, 1.1, 1001))
    assert lo == pytest.approx(-9.5) and hi == pytest.approx(9.5)
    lo, hi = ci95(np.linspace(0.9, 1.1, 1001), log_scale=True)
    assert lo < 0 < hi
