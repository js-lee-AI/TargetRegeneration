"""Reference forms of the training objectives, in numpy, for reading and testing.

A block drafter predicts the 15 slots after an anchor token in one forward pass. For one block,
slot k = 1 .. 15 holds the corpus token at that offset, and the target, run on the corpus prefix,
supplies its distribution at the same position. The GPU implementation lives in
pipeline/specforge/core/dflash.py and dflash_targetkd.py; tests/test_objectives.py checks that both agree.
"""
from __future__ import annotations

import numpy as np

TAIL_SUM_CLAMP = 1.0 - 1e-6


def slot_weights(block_size: int = 16, gamma: float = 7.0) -> np.ndarray:
    """Weight of each slot: exp(-(k - 1) / gamma) for k >= 1. Slot 0 is the anchor and carries no loss."""
    k = np.arange(block_size, dtype=np.float64)
    w = np.exp(-np.clip(k - 1, 0, None) / gamma)
    w[0] = 0.0
    return w


def hard_gate(corpus: np.ndarray, target_greedy: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """KD + gate: slot k keeps its weight only if every earlier slot's corpus token is the target's greedy token.

    That is the condition under which verification reaches slot k. Inputs have shape (..., block);
    slot 0 always passes, and slots without a label (valid False) do not close the gate.
    """
    agree = np.asarray(corpus) == np.asarray(target_greedy)
    if valid is not None:
        agree |= ~np.asarray(valid, dtype=bool)
    agree[..., 0] = True
    survive = np.cumprod(agree, axis=-1).astype(np.float64)
    gate = np.ones_like(survive)
    gate[..., 1:] = survive[..., :-1]
    return gate


def soft_gate(corpus_prob: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Soft weighting (PARD-2's survival weight): the product of the target's probabilities of the
    earlier slots' corpus tokens."""
    p = np.asarray(corpus_prob, dtype=np.float64)
    scored = np.ones(p.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool).copy()
    scored[..., 0] = False
    c = np.where(scored, p, 1.0)
    shifted = np.concatenate([np.ones_like(c[..., :1]), c[..., :-1]], axis=-1)
    return np.cumprod(shifted, axis=-1)


def kd_target(target_prob: np.ndarray, k: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The target's k most probable tokens, their probabilities and the remaining (tail) mass."""
    p = np.asarray(target_prob, dtype=np.float64)
    idx = np.argsort(-p, axis=-1, kind="stable")[..., :k]
    top = np.take_along_axis(p, idx, axis=-1)
    return idx, top, np.clip(1.0 - top.sum(-1), 0.0, None)


def kd_loss(draft_logits: np.ndarray, target_prob: np.ndarray, weight: np.ndarray, k: int = 8,
            greedy_labels: float = 0.0) -> float:
    """Weighted KD loss over (k + 1) outcomes: the target's top-k tokens and one tail outcome.

    greedy_labels = 1 replaces the target distribution by its greedy token, which is CE toward the
    target's greedy token (the label control of the paper).
    """
    z = np.asarray(draft_logits, dtype=np.float64)
    log_q = z - z.max(-1, keepdims=True)
    log_q = log_q - np.log(np.exp(log_q).sum(-1, keepdims=True))
    idx, top, tail = kd_target(target_prob, k)
    log_q_top = np.take_along_axis(log_q, idx, axis=-1)
    log_q_tail = np.log1p(-np.minimum(np.exp(log_q_top).sum(-1), TAIL_SUM_CLAMP))
    loss = -(top * log_q_top).sum(-1) - tail * log_q_tail
    if greedy_labels:
        loss = (1.0 - greedy_labels) * loss + greedy_labels * (-log_q_top[..., 0])
    w = np.asarray(weight, dtype=np.float64)
    return float((w * loss).sum() / (w.sum() + 1e-6))
