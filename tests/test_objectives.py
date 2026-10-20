"""The numpy forms of the objectives, and their agreement with the training code in pipeline/.

The agreement tests need torch; the hard-gate test also imports the training core, which needs
transformers. They are skipped when those are missing, unless TARGETREGEN_REQUIRE_TORCH=1.
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from targetregen.objectives import hard_gate, kd_loss, kd_target, slot_weights, soft_gate

PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
REQUIRE = os.environ.get("TARGETREGEN_REQUIRE_TORCH") == "1"


def test_slot_weights():
    w = slot_weights(16, 7.0)
    assert w.shape == (16,) and w[0] == 0.0 and w[1] == 1.0
    assert w[2] == pytest.approx(math.exp(-1 / 7))


def test_hard_gate_closes_after_the_first_disagreement():
    corpus = np.array([[5, 7, 9, 4, 4]])
    greedy = np.array([[5, 7, 9, 8, 4]])
    np.testing.assert_array_equal(hard_gate(corpus, greedy), [[1, 1, 1, 1, 0]])
    # a slot without a label does not close the gate
    valid = np.array([[True, True, True, False, True]])
    np.testing.assert_array_equal(hard_gate(corpus, greedy, valid), [[1, 1, 1, 1, 1]])


def test_soft_gate_multiplies_the_earlier_probabilities():
    p = np.array([0.9, 0.5, 0.4, 1.0])  # slot 0 is the anchor and is not scored
    np.testing.assert_allclose(soft_gate(p), [1.0, 1.0, 0.5, 0.2])


def test_kd_target_and_greedy_labels():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(20), size=6)
    idx, top, tail = kd_target(p, 8)
    np.testing.assert_allclose(top.sum(-1) + tail, 1.0)
    np.testing.assert_array_equal(idx[:, 0], p.argmax(-1))
    logits = rng.normal(size=(6, 20))
    weight = rng.uniform(0.1, 1.0, size=6)
    log_q = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    ce = -log_q[np.arange(6), p.argmax(-1)]
    assert kd_loss(logits, p, weight, greedy_labels=1.0) == pytest.approx((weight * ce).sum() / (weight.sum() + 1e-6))


def _torch():
    if REQUIRE:
        import torch
    else:
        torch = pytest.importorskip("torch")
    if str(PIPELINE) not in sys.path:
        sys.path.append(str(PIPELINE))
    return torch


def test_kd_loss_matches_the_training_code():
    torch = _torch()
    from specforge.core.dflash_targetkd import targetkd_loss

    rng = np.random.default_rng(2)
    logits = rng.normal(size=(12, 50))
    p = np.exp(rng.normal(scale=2.0, size=(12, 50)))
    p /= p.sum(-1, keepdims=True)
    weight = rng.uniform(0.0, 1.0, size=12)
    tp, ti = torch.topk(torch.tensor(p, dtype=torch.float32), 8, dim=-1)
    tail = (1.0 - tp.sum(-1)).clamp_min(0.0)
    for mix in (0.0, 1.0):
        got = targetkd_loss(torch.tensor(logits, dtype=torch.float32), ti, tp, tail,
                            torch.tensor(weight, dtype=torch.float32), hard_label_mix=mix)
        assert float(got) == pytest.approx(kd_loss(logits, p, weight, greedy_labels=mix), rel=1e-5)


def test_soft_gate_matches_the_training_code():
    torch = _torch()
    from specforge.core.dflash_targetkd import soft_survival_gate

    rng = np.random.default_rng(3)
    p = rng.uniform(size=(2, 4, 16))
    valid = rng.uniform(size=(2, 4, 16)) > 0.2
    scored = torch.tensor(valid) & (torch.arange(16) != 0)
    got = soft_survival_gate(torch.tensor(p, dtype=torch.float32), scored).numpy()
    np.testing.assert_allclose(got, soft_gate(p, valid), rtol=1e-5)


def test_hard_gate_matches_the_training_code():
    torch = _torch()
    try:
        from specforge.core.dflash import OnlineDFlashModel
    except ImportError:
        if REQUIRE:
            raise
        pytest.skip("the training core needs the pipeline requirements")

    rng = np.random.default_rng(4)
    target = rng.integers(0, 4, size=(2, 3, 16))
    greedy = np.where(rng.uniform(size=target.shape) < 0.8, target, (target + 1) % 4)
    valid = rng.uniform(size=target.shape) > 0.1
    topk = torch.tensor(greedy.reshape(-1, 1)).repeat(1, 8)
    got = OnlineDFlashModel._survival_gate(None, topk, torch.tensor(target), torch.tensor(valid)).numpy()
    np.testing.assert_array_equal(got, hard_gate(target, greedy, valid))
