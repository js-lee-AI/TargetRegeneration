from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

_TAIL_SUM_CLAMP = 1.0 - 1e-06


def draft_tail_logprob(log_q_topk: torch.Tensor) -> torch.Tensor:
    q_topk_sum = log_q_topk.exp().sum(dim=-1)
    return torch.log1p(-q_topk_sum.clamp_max(_TAIL_SUM_CLAMP))


def harvest_indices(anchor_positions: torch.Tensor, block_size: int) -> torch.Tensor:
    device = anchor_positions.device
    offsets = torch.arange(block_size, device=device).view(1, 1, -1)
    return anchor_positions.unsqueeze(-1) + offsets - 1


@torch.no_grad()
def harvest_target_topk(
    final_hidden: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_size: int,
    lm_head: nn.Module,
    topk: int,
    chunk_size: int = 4096,
    label_ids: Optional[torch.Tensor] = None,
    harvested_hidden: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    B, S, D = final_hidden.shape
    N = anchor_positions.shape[1]
    device = final_hidden.device
    if harvested_hidden is None:
        idx = harvest_indices(anchor_positions, block_size)
        safe_idx = idx.clamp(0, S - 1)
        gather_idx = safe_idx.reshape(B, N * block_size, 1).expand(-1, -1, D)
        harvested = torch.gather(final_hidden, 1, gather_idx)
        harvested = harvested.reshape(B * N * block_size, D)
    else:
        if tuple(harvested_hidden.shape) != (B, N, block_size, D):
            raise ValueError(
                f"harvested_hidden has shape {tuple(harvested_hidden.shape)}; expected {(B, N, block_size, D)}"
            )
        harvested = harvested_hidden.reshape(B * N * block_size, D).to(
            final_hidden.dtype
        )
    P = harvested.shape[0]
    topk_idx = torch.empty((P, topk), dtype=torch.long, device=device)
    topk_prob = torch.empty((P, topk), dtype=torch.float32, device=device)
    tail_mass = torch.empty((P,), dtype=torch.float32, device=device)
    label_prob = (
        None
        if label_ids is None
        else torch.empty((P,), dtype=torch.float32, device=device)
    )
    for c0 in range(0, P, chunk_size):
        c1 = min(c0 + chunk_size, P)
        logits = lm_head(harvested[c0:c1]).float()
        probs = torch.softmax(logits, dim=-1)
        tp, ti = torch.topk(probs, topk, dim=-1)
        topk_idx[c0:c1] = ti
        topk_prob[c0:c1] = tp
        tail_mass[c0:c1] = (1.0 - tp.sum(dim=-1)).clamp_min(0.0)
        if label_prob is not None:
            label_prob[c0:c1] = probs.gather(1, label_ids[c0:c1].unsqueeze(-1)).squeeze(
                -1
            )
    if label_prob is None:
        return (topk_idx, topk_prob, tail_mass)
    return (topk_idx, topk_prob, tail_mass, label_prob)


@torch.no_grad()
def soft_survival_gate(label_prob: torch.Tensor, scored: torch.Tensor) -> torch.Tensor:
    # Eq. 2: slot k is weighted by the product of the target's probabilities of
    # the tokens at slots 1 .. k-1. Unscored slots contribute a factor of one.
    c = torch.where(scored > 0, label_prob.float(), torch.ones_like(label_prob))
    ones = torch.ones_like(c[..., :1])
    shifted = torch.cat([ones, c[..., :-1]], dim=-1)
    return shifted.cumprod(dim=-1)


def targetkd_loss(
    draft_logits: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_prob: torch.Tensor,
    tail_mass: torch.Tensor,
    weight: torch.Tensor,
    hard_label_mix: float = 0.0,
    tail_bucket: bool = True,
    lambda_scale: float = 1.0,
) -> torch.Tensor:
    log_q = F.log_softmax(draft_logits, dim=-1, dtype=torch.float32)
    log_q_topk = log_q.gather(1, topk_idx)
    kd = -(topk_prob * log_q_topk).sum(dim=-1)
    if tail_bucket:
        kd = kd - tail_mass * draft_tail_logprob(log_q_topk)
    if hard_label_mix > 0.0:
        hard_ce = -log_q_topk[:, 0]
        kd = (1.0 - hard_label_mix) * kd + hard_label_mix * hard_ce
    denom = weight.sum() + 1e-06
    return lambda_scale * (weight * kd).sum() / denom


@torch.no_grad()
def targetkd_accuracy(
    draft_logits: torch.Tensor, topk_idx: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    draft_argmax = draft_logits.argmax(dim=-1)
    y_star = topk_idx[:, 0]
    valid = weight > 0
    correct = (draft_argmax == y_star) & valid
    return correct.sum().float() / (valid.sum().float() + 1e-06)
