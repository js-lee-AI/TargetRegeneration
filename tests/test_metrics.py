import math

import pytest

from targetregen import cell, geomean, increment, mat, ratio_of_ratios, run_mat


def test_mat_pools_rounds_rather_than_averaging_prompts():
    # 30 tokens in 10 rounds and 12 tokens in 6 rounds: 42 / 16, not the mean of 3.0 and 2.0
    assert mat([30, 12], [10, 6]) == pytest.approx(2.625)
    with pytest.raises(ValueError):
        mat([1], [0])


def test_run_mat_is_the_geometric_mean_of_domain_mats():
    domains, value = run_mat({"a": ([20], [10]), "b": ([30], [10]), "c": ([45], [10])})
    assert domains == {"a": 2.0, "b": 3.0, "c": 4.5}
    assert value == pytest.approx(3.0)
    with pytest.raises(ValueError):
        geomean([1.0, 0.0])


def test_cell_mean_and_sample_sd():
    mean, sd = cell([2.0, 4.0])
    assert mean == 3.0 and sd == pytest.approx(math.sqrt(2.0))
    assert cell([2.5]) == (2.5, None)


def test_increment_and_ratio_of_ratios():
    assert increment(3.45, 3.0) == pytest.approx(15.0)
    # (1 + 3%) / (1 + 15%) - 1
    assert ratio_of_ratios(3.0, 15.0) == pytest.approx(100 * (1.03 / 1.15 - 1))
    assert ratio_of_ratios(7.0, 7.0) == pytest.approx(0.0)
