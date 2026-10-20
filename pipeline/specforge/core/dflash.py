from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from specforge.core.dflash_targetkd import (
    harvest_target_topk,
    soft_survival_gate,
    targetkd_accuracy,
    targetkd_loss,
)
from specforge.modeling.draft.dflash import DFlashDraftModel

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None
    create_block_mask = None
if hasattr(torch, "npu") and torch.npu.is_available():
    FLEX_ATTENTION_AVAILABLE = False
_VALID_LOSS_TYPES = {"dflash", "dflash_targetkd"}
FINAL_HIDDEN_LOSS_TYPES = ("dflash_targetkd",)


def create_dflash_sdpa_mask(anchor_positions, block_keep_mask, S, block_size, device):
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size
    q_indices = torch.arange(Q_LEN, device=device).view(1, 1, -1, 1)
    kv_indices = torch.arange(KV_LEN, device=device).view(1, 1, 1, -1)
    q_block_ids = q_indices // block_size
    anchor_expanded = anchor_positions.view(B, 1, N, 1).repeat_interleave(
        block_size, dim=2
    )
    mask_context = (kv_indices < S) & (kv_indices < anchor_expanded)
    is_draft = kv_indices >= S
    kv_block_ids = (kv_indices - S) // block_size
    mask_draft = is_draft & (q_block_ids == kv_block_ids)
    valid_block = block_keep_mask.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)
    final_mask = (mask_context | mask_draft) & valid_block
    return final_mask


def create_dflash_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
):
    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        safe_q_block_id = q_block_id.clamp(max=N - 1)
        anchor_pos = anchor_positions[b, safe_q_block_id]
        is_context = kv_idx < S
        mask_context = is_context & (kv_idx < anchor_pos)
        is_draft = kv_idx >= S
        kv_block_id = (kv_idx - S) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)
        is_valid_block = block_keep_mask[b, safe_q_block_id]
        in_bounds = q_block_id < N
        return (mask_context | mask_draft) & is_valid_block & in_bounds

    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size
    return create_block_mask(
        dflash_mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
    )


def create_ira_sdpa_mask(
    ctx_end, br_start, br_end, block_keep_mask, S, S_ext, block_size, device
):
    B, N = ctx_end.shape
    Q_LEN = N * block_size
    KV_LEN = S_ext + N * block_size
    q_block_ids = (torch.arange(Q_LEN, device=device) // block_size).view(1, 1, -1, 1)
    kv_indices = torch.arange(KV_LEN, device=device).view(1, 1, 1, -1)

    def per_query(t):
        return t.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)

    mask_context = (kv_indices < S) & (kv_indices < per_query(ctx_end))
    mask_branch = (kv_indices >= per_query(br_start)) & (kv_indices < per_query(br_end))
    is_draft = kv_indices >= S_ext
    mask_draft = is_draft & ((kv_indices - S_ext) // block_size == q_block_ids)
    return (mask_context | mask_branch | mask_draft) & per_query(block_keep_mask)


def create_ira_block_mask(
    ctx_end: torch.Tensor,
    br_start: torch.Tensor,
    br_end: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    S_ext: int,
    block_size: int,
    device: torch.device,
):
    def ira_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        safe_q_block_id = q_block_id.clamp(max=N - 1)
        mask_context = (kv_idx < S) & (kv_idx < ctx_end[b, safe_q_block_id])
        mask_branch = (kv_idx >= br_start[b, safe_q_block_id]) & (
            kv_idx < br_end[b, safe_q_block_id]
        )
        is_draft = kv_idx >= S_ext
        kv_block_id = (kv_idx - S_ext) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)
        is_valid_block = block_keep_mask[b, safe_q_block_id]
        in_bounds = q_block_id < N
        return (mask_context | mask_branch | mask_draft) & is_valid_block & in_bounds

    B, N = ctx_end.shape
    Q_LEN = N * block_size
    KV_LEN = S_ext + N * block_size
    return create_block_mask(
        ira_mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
    )


class OnlineDFlashModel(nn.Module):
    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        loss_type: str = "dflash",
        kd_topk: int = 8,
        kd_hard_label_mix: float = 0.0,
        kd_tail_bucket: bool = True,
        kd_lambda_scale: float = 1.0,
        kd_chunk_size: int = 4096,
        kd_survival_mask: bool = False,
        kd_survival_soft: bool = False,
        kd_anchor_rollout: bool = False,
        kd_rollout_stop_ids: Optional[Tuple[int, ...]] = None,
        kd_rollout_steps: int = 0,
        kd_inrollout_anchors: bool = False,
        kd_inrollout_seed: int = 0,
        kd_slotmatch_control: bool = False,
        kd_rollout_survival_soft: bool = False,
    ):
        super().__init__()
        if loss_type not in _VALID_LOSS_TYPES:
            raise ValueError(
                f"loss_type={loss_type!r}; must be one of {sorted(_VALID_LOSS_TYPES)}"
            )
        if kd_topk < 1:
            raise ValueError(f"kd_topk must be >= 1, got {kd_topk}")
        if not 0.0 <= kd_hard_label_mix <= 1.0:
            raise ValueError(
                f"kd_hard_label_mix must be in [0, 1], got {kd_hard_label_mix}"
            )
        if kd_lambda_scale <= 0.0:
            raise ValueError(f"kd_lambda_scale must be positive, got {kd_lambda_scale}")
        if kd_chunk_size < 1:
            raise ValueError(f"kd_chunk_size must be >= 1, got {kd_chunk_size}")
        if kd_survival_soft and loss_type != "dflash_targetkd":
            raise ValueError(
                f"kd_survival_soft is read only by the dflash_targetkd loss; got loss_type={loss_type!r}."
            )
        if kd_survival_soft and kd_survival_mask:
            raise ValueError(
                "kd_survival_soft and kd_survival_mask are the two forms of one gate; turn on at most one."
            )
        if kd_anchor_rollout and loss_type != "dflash_targetkd":
            raise ValueError(
                f"kd_anchor_rollout relabels the KD slots; it needs loss_type='dflash_targetkd', got {loss_type!r}."
            )
        if kd_anchor_rollout and (kd_survival_mask or kd_survival_soft):
            raise ValueError(
                "kd_anchor_rollout replaces the corpus-prefix labels the survival gates weigh; a gate on top of rollout labels compares corpus tokens to a continuation they no longer condition. Turn the gates off."
            )
        if kd_rollout_survival_soft and (
            not kd_anchor_rollout
            or kd_inrollout_anchors
            or kd_slotmatch_control
        ):
            raise ValueError(
                "kd_rollout_survival_soft gates ALR's own rollout labels; it needs kd_anchor_rollout=True without IRA or the slot-matched control."
            )
        if kd_anchor_rollout and (not kd_rollout_stop_ids):
            raise ValueError(
                "kd_anchor_rollout needs kd_rollout_stop_ids (the end-of-turn token ids): slots after the rollout ends its turn have no label."
            )
        if kd_rollout_steps and (not kd_anchor_rollout):
            raise ValueError(
                "kd_rollout_steps truncates the anchor rollout; it needs kd_anchor_rollout=True."
            )
        if kd_rollout_steps and (not 1 <= kd_rollout_steps <= block_size - 2):
            raise ValueError(
                f"kd_rollout_steps must be in [1, {block_size - 2}] (0 = to the block's end), got {kd_rollout_steps}"
            )
        if kd_inrollout_anchors and (not kd_anchor_rollout):
            raise ValueError(
                "kd_inrollout_anchors puts secondary anchors inside the anchor rollout; it needs kd_anchor_rollout=True."
            )
        if kd_inrollout_anchors and kd_rollout_steps:
            raise ValueError(
                "kd_inrollout_anchors labels secondary slots with rollout slots up to the block's end; it excludes kd_rollout_steps."
            )
        if kd_inrollout_anchors and block_size < 4:
            raise ValueError(
                f"kd_inrollout_anchors needs block_size >= 4 (offsets 2 .. block_size - 2), got {block_size}"
            )
        if kd_slotmatch_control and (not kd_anchor_rollout):
            raise ValueError(
                "kd_slotmatch_control is IRA's slot-matched control on ALR blocks; it needs kd_anchor_rollout=True."
            )
        if kd_slotmatch_control and (kd_inrollout_anchors or kd_rollout_steps):
            raise ValueError(
                "kd_slotmatch_control keeps ALR's full-depth, corpus-anchored blocks; it excludes kd_inrollout_anchors and kd_rollout_steps."
            )
        if kd_slotmatch_control and block_size < 4:
            raise ValueError(
                f"kd_slotmatch_control draws IRA's offsets 2 .. block_size - 2; it needs block_size >= 4, got {block_size}"
            )
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma
        self.loss_type = loss_type
        self.kd_topk = kd_topk
        self.kd_hard_label_mix = kd_hard_label_mix
        self.kd_tail_bucket = kd_tail_bucket
        self.kd_lambda_scale = kd_lambda_scale
        self.kd_chunk_size = kd_chunk_size
        self.kd_survival_mask = kd_survival_mask
        self.kd_survival_soft = kd_survival_soft
        self.kd_anchor_rollout = kd_anchor_rollout
        self.kd_rollout_stop_ids = tuple(kd_rollout_stop_ids or ())
        self.kd_rollout_steps = int(kd_rollout_steps or 0)
        self.kd_inrollout_anchors = kd_inrollout_anchors
        self.kd_slotmatch_control = kd_slotmatch_control
        self.kd_rollout_survival_soft = kd_rollout_survival_soft
        self._ira_gen: Optional[torch.Generator] = None
        self._ira_gen_device: Optional[torch.device] = None
        self._ira_seed = 9012 + int(kd_inrollout_seed)
        self._cached_block_mask: Optional[BlockMask] = None
        self._cached_seq_len: Optional[int] = None
        self._cached_bsz: Optional[int] = None

    def _inrollout_generator(self, device: torch.device) -> torch.Generator:
        if self._ira_gen is None or self._ira_gen_device != device:
            g = torch.Generator(device=device)
            g.manual_seed(self._ira_seed)
            self._ira_gen = g
            self._ira_gen_device = device
        return self._ira_gen

    def _sample_inrollout_anchors(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> dict:
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)
        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = min(self.num_anchors, int(valid_counts.max().item()) - 1)
        if max_n <= 0:
            raise ValueError("should preprocess the data.")
        indices = (
            torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        )
        masked_indices = torch.where(
            valid, indices, torch.tensor(seq_len + 1, device=device)
        )
        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))
        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        m = valid_counts.clamp(max=max_n)
        n1 = (max_n + 1) // 2
        n2 = max_n // 2
        in_order = torch.arange(n1, device=device).unsqueeze(0) < (
            (m + 1) // 2
        ).unsqueeze(1)
        first = torch.where(
            in_order, gathered[:, :n1], torch.tensor(seq_len + 1, device=device)
        )
        primary, by_position = first.sort(dim=1)
        keep1 = torch.gather(in_order, 1, by_position)
        primary = torch.where(
            keep1, primary, torch.tensor(0, dtype=torch.long, device=device)
        )
        parent = by_position.argsort(dim=1)[:, :n2]
        keep2 = torch.arange(n2, device=device).unsqueeze(0) < (m // 2).unsqueeze(1)
        j = torch.randint(
            2,
            bs - 1,
            (bsz, n2),
            generator=self._inrollout_generator(device),
            device=device,
        )
        return dict(
            primary=primary,
            keep1=keep1,
            parent=parent,
            j=j,
            keep2=keep2,
            alr_blocks=int(m.sum().item()),
        )

    def _sample_slotmatched_anchors(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)
        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = min(self.num_anchors, int(valid_counts.max().item()) - 1)
        if max_n <= 0:
            raise ValueError("should preprocess the data.")
        indices = (
            torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        )
        masked_indices = torch.where(
            valid, indices, torch.tensor(seq_len + 1, device=device)
        )
        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))
        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        m = valid_counts.clamp(max=max_n)
        n2 = max_n // 2
        j = torch.randint(
            2,
            bs - 1,
            (bsz, n2),
            generator=self._inrollout_generator(device),
            device=device,
        )
        full = torch.full((bsz, max_n), bs - 1, dtype=torch.long, device=device)
        if n2 > 0:
            i = torch.arange(max_n, device=device).unsqueeze(0) - (
                (m + 1) // 2
            ).unsqueeze(1)
            capped = (i >= 0) & (i < (m // 2).unsqueeze(1))
            j_at = torch.gather(j, 1, i.clamp(min=0, max=n2 - 1))
            cap_order = torch.where(capped, bs - 1 - j_at, full)
        else:
            cap_order = full
        anchors, by_position = gathered[:, :max_n].sort(dim=1)
        slot_cap = torch.gather(cap_order, 1, by_position)
        keep_mask = torch.arange(max_n, device=device).unsqueeze(0) < m.unsqueeze(1)
        anchors = torch.where(
            keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device)
        )
        slot_cap = torch.where(keep_mask, slot_cap, full)
        return (anchors, keep_mask, slot_cap)

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)
        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = min(self.num_anchors, int(valid_counts.max().item()) - 1)
        if max_n <= 0:
            raise ValueError("should preprocess the data.")
        indices = (
            torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        )
        masked_indices = torch.where(
            valid, indices, torch.tensor(seq_len + 1, device=device)
        )
        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))
        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values
        keep_mask = torch.arange(max_n, device=device).unsqueeze(
            0
        ) < valid_counts.unsqueeze(1).clamp(max=max_n)
        anchors = torch.where(
            keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device)
        )
        return (anchors, keep_mask)

    def _create_position_ids(self, anchor_positions: torch.Tensor) -> torch.Tensor:
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
        return pos_ids.view(bsz, -1)

    def _create_noise_embed(self, input_ids, anchor_positions, block_keep_mask):
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device
        noise_ids = torch.full(
            (bsz, n * bs), self.mask_token_id, dtype=torch.long, device=device
        )
        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)
        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)
        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )
        return self.embed_tokens(noise_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        final_hidden_states: Optional[torch.Tensor] = None,
        anchor_positions: Optional[torch.Tensor] = None,
        block_keep_mask: Optional[torch.Tensor] = None,
        rollout_hidden: Optional[torch.Tensor] = None,
        rollout_tokens: Optional[torch.Tensor] = None,
        rollout_context: Optional[torch.Tensor] = None,
        inrollout: Optional[dict] = None,
        slot_cap: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.attention_backend == "flex_attention" and (
            not FLEX_ATTENTION_AVAILABLE
        ):
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        if anchor_positions is None:
            anchor_positions, block_keep_mask = self._sample_anchor_positions(
                seq_len, loss_mask, device
            )
        elif block_keep_mask is None:
            raise ValueError("anchor_positions given without block_keep_mask")
        if self.kd_anchor_rollout and (
            rollout_hidden is None or rollout_tokens is None
        ):
            raise ValueError(
                "kd_anchor_rollout=True needs rollout_hidden and rollout_tokens from the target's generate_dflash_data(anchor_rollout=...)."
            )
        if getattr(self, "kd_inrollout_anchors", False):
            if inrollout is None or tuple(inrollout["primary"].shape) != tuple(
                anchor_positions.shape
            ):
                raise ValueError(
                    "kd_inrollout_anchors=True needs inrollout (the dict from _sample_inrollout_anchors) and anchor_positions = its primaries."
                )
            return self._inrollout_forward(
                input_ids=input_ids,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
                final_hidden_states=final_hidden_states,
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                rollout_hidden=rollout_hidden,
                rollout_tokens=rollout_tokens,
                rollout_context=rollout_context,
                inrollout=inrollout,
            )
        elif inrollout is not None or rollout_context is not None:
            raise ValueError(
                "inrollout / rollout_context need kd_inrollout_anchors=True"
            )
        if getattr(self, "kd_slotmatch_control", False):
            if slot_cap is None or tuple(slot_cap.shape) != tuple(
                anchor_positions.shape
            ):
                raise ValueError(
                    "kd_slotmatch_control=True needs slot_cap and anchor_positions from _sample_slotmatched_anchors."
                )
        elif slot_cap is not None:
            raise ValueError("slot_cap needs kd_slotmatch_control=True")
        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )
        context_position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        )
        draft_position_ids = self._create_position_ids(anchor_positions)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)
        if self.attention_backend == "flex_attention":
            dflash_attn_mask = create_dflash_block_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
            )
        else:
            dflash_attn_mask = create_dflash_sdpa_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
            )
        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attn_mask,
        )
        logits = self.lm_head(output_hidden)
        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()
        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()
        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered
        binary_eval_mask = weight_mask.view(-1)
        if self.loss_type == "dflash_targetkd":
            return self._targetkd_loss(
                logits=logits,
                anchor_positions=anchor_positions,
                weight_mask=weight_mask,
                final_hidden_states=final_hidden_states,
                target_ids=target_ids,
                valid_label_mask=valid_label_mask,
                device=device,
                rollout_hidden=rollout_hidden,
                rollout_tokens=rollout_tokens,
                inrollout=inrollout,
                slot_cap=slot_cap,
            )
        if slot_cap is not None:
            raise ValueError("kd_slotmatch_control needs loss_type='dflash_targetkd'")
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = target_ids.view(-1)
        loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        if self.loss_type == "dflash":
            loss_weights = weight_mask
            if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
                k = torch.arange(self.block_size, device=device).view(1, 1, -1)
                decay_weights = torch.exp(
                    -(k - 1).clamp(min=0).float() / self.loss_decay_gamma
                )
                loss_weights = loss_weights * decay_weights
            flat_weights = loss_weights.view(-1)
            valid_token_count = flat_weights.sum() + 1e-06
            loss = (loss_per_token * flat_weights).sum() / valid_token_count
        else:
            raise ValueError(f"unknown loss_type {self.loss_type!r}")
        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            actual_token_count = binary_eval_mask.sum() + 1e-06
            accuracy = correct.sum().float() / actual_token_count
        return (loss, accuracy)

    def _inrollout_forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        final_hidden_states: Optional[torch.Tensor],
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        rollout_hidden: torch.Tensor,
        rollout_tokens: torch.Tensor,
        rollout_context: Optional[torch.Tensor],
        inrollout: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        bs = self.block_size
        C = bs - 3
        a1, keep1 = (anchor_positions, block_keep_mask)
        n1 = a1.shape[1]
        parent, j, keep2 = (inrollout["parent"], inrollout["j"], inrollout["keep2"])
        n2 = parent.shape[1]
        feat = hidden_states.shape[-1]
        if rollout_context is None or tuple(rollout_context.shape) != (
            bsz,
            n1,
            C,
            feat,
        ):
            raise ValueError(
                f"kd_inrollout_anchors needs rollout_context [B, N1, block_size - 3, F] = {(bsz, n1, C, feat)} from generate_dflash_data(rollout_context_steps={C}); got {(None if rollout_context is None else tuple(rollout_context.shape))}"
            )
        n_blocks = n1 + n2
        S_ext = seq_len + n1 * C
        pos_in_block = torch.arange(bs, device=device).view(1, 1, -1)
        bidx = torch.arange(bsz, device=device).view(bsz, 1)
        parent_anchor = torch.gather(a1, 1, parent)
        target_hidden = torch.cat(
            [
                hidden_states,
                rollout_context.reshape(bsz, n1 * C, feat).to(hidden_states.dtype),
            ],
            dim=1,
        )
        mask_token = torch.tensor(self.mask_token_id, dtype=torch.long, device=device)
        noise_ids = torch.full(
            (bsz, n_blocks, bs), self.mask_token_id, dtype=torch.long, device=device
        )
        anchor_tokens = torch.gather(input_ids, 1, a1.clamp(0, seq_len - 1))
        noise_ids[:, :n1, 0] = torch.where(keep1, anchor_tokens, mask_token)
        noise_ids[:, n1:, 0] = torch.where(
            keep2, rollout_tokens[bidx, parent, j], mask_token
        )
        noise_embedding = self.embed_tokens(noise_ids.view(bsz, n_blocks * bs))
        full_position_ids = torch.cat(
            [
                torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1),
                (a1.unsqueeze(-1) + 1 + torch.arange(C, device=device)).reshape(
                    bsz, n1 * C
                ),
                (a1.unsqueeze(-1) + pos_in_block).reshape(bsz, n1 * bs),
                ((parent_anchor + j).unsqueeze(-1) + pos_in_block).reshape(
                    bsz, n2 * bs
                ),
            ],
            dim=1,
        )
        no_branch = torch.zeros_like(a1)
        branch_start = seq_len + parent * C
        mask_kwargs = dict(
            ctx_end=torch.cat([a1, parent_anchor + 1], dim=1),
            br_start=torch.cat([no_branch, branch_start], dim=1),
            br_end=torch.cat([no_branch, branch_start + j - 1], dim=1),
            block_keep_mask=torch.cat([keep1, keep2], dim=1),
            S=seq_len,
            S_ext=S_ext,
            block_size=bs,
            device=device,
        )
        if self.attention_backend == "flex_attention":
            dflash_attn_mask = create_ira_block_mask(**mask_kwargs)
        else:
            dflash_attn_mask = create_ira_sdpa_mask(**mask_kwargs)
        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            attention_mask=dflash_attn_mask,
        )
        logits = self.lm_head(output_hidden)
        label_indices = a1.unsqueeze(-1) + pos_in_block
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n1, -1), 2, safe_label_indices
        )
        weight_mask = keep1.unsqueeze(-1).expand(-1, -1, bs).float()
        weight_mask = weight_mask * valid_label_mask.float()
        weight_mask = weight_mask * (pos_in_block > 0).float()
        weight_mask = weight_mask * torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n1, -1), 2, safe_label_indices
        )
        alive = self._rollout_alive(rollout_tokens)
        src = j.unsqueeze(-1) + pos_in_block
        inside = src <= bs - 1
        src = src.clamp(max=bs - 1)
        sb, sp = (bidx.unsqueeze(-1), parent.unsqueeze(-1))
        weight2 = (
            weight_mask[sb, sp, src]
            * (inside & keep2.unsqueeze(-1) & (pos_in_block > 0)).float()
        )
        tokens2 = torch.where(
            inside, rollout_tokens[sb, sp, src], torch.full_like(src, -1)
        )
        return self._targetkd_loss(
            logits=logits,
            anchor_positions=torch.cat([a1, parent_anchor + j], dim=1),
            weight_mask=torch.cat([weight_mask, weight2], dim=1),
            final_hidden_states=final_hidden_states,
            device=device,
            target_ids=torch.cat([target_ids, target_ids[sb, sp, src]], dim=1),
            valid_label_mask=torch.cat(
                [valid_label_mask, valid_label_mask[sb, sp, src] & inside], dim=1
            ),
            rollout_hidden=torch.cat(
                [rollout_hidden, rollout_hidden[sb, sp, src]], dim=1
            ),
            rollout_tokens=torch.cat([rollout_tokens, tokens2], dim=1),
            rollout_alive=torch.cat(
                [alive, alive[sb, sp, src] * inside.to(alive.dtype)], dim=1
            ),
            inrollout=inrollout,
        )

    def _survival_gate(
        self,
        topk_idx: torch.Tensor,
        target_ids: torch.Tensor,
        valid_label_mask: torch.Tensor,
    ) -> torch.Tensor:
        y_star = topk_idx[:, 0].view_as(target_ids)
        agree = (target_ids == y_star) | ~valid_label_mask
        pos_in_block = torch.arange(agree.shape[-1], device=agree.device)
        agree = agree | (pos_in_block == 0)
        surv = torch.cumprod(agree.to(torch.float32), dim=-1)
        gate = torch.ones_like(surv)
        gate[..., 1:] = surv[..., :-1]
        return gate.to(target_ids.device)

    def _survival_gate_soft(
        self, label_prob: torch.Tensor, valid_label_mask: torch.Tensor
    ) -> torch.Tensor:
        pos_in_block = torch.arange(label_prob.shape[-1], device=label_prob.device)
        votes = valid_label_mask & (pos_in_block != 0)
        return soft_survival_gate(label_prob=label_prob, scored=votes)

    def _rollout_survival_gate_soft(self, label_prob, rollout_tokens):
        # Same soft gate, scored on the rollout's own tokens. Slots past the
        # rollout's end of turn do not vote.
        valid = (rollout_tokens >= 0) & self._rollout_alive(rollout_tokens).bool()
        return self._survival_gate_soft(label_prob, valid)

    def _rollout_alive(self, rollout_tokens: torch.Tensor) -> torch.Tensor:
        stop = torch.zeros_like(rollout_tokens, dtype=torch.bool)
        for sid in self.kd_rollout_stop_ids:
            stop |= rollout_tokens == int(sid)
        stop[..., 0] = False
        cont = torch.cumprod((~stop).to(torch.float32), dim=-1)
        alive = torch.ones_like(cont)
        alive[..., 1:] = cont[..., :-1]
        return alive

    def _rollout_depth(self, device, dtype) -> Optional[torch.Tensor]:
        steps = getattr(self, "kd_rollout_steps", 0)
        if not steps or steps >= self.block_size - 2:
            return None
        k = torch.arange(self.block_size, device=device)
        return (k <= steps + 1).to(dtype)

    def _report_rollout(
        self, rollout_tokens, target_ids, weight_mask, loss_weights, alive, depth=None
    ):
        with torch.no_grad():
            trainable = (weight_mask > 0).to(torch.float32)
            if depth is not None:
                trainable = trainable * depth.to(torch.float32)
            denom = trainable.sum(dim=(0, 1)).clamp(min=1.0)
            agree = (rollout_tokens == target_ids).to(torch.float32)
            agree[..., 0] = 1.0
            same_prefix = torch.ones_like(agree)
            same_prefix[..., 1:] = torch.cumprod(agree, dim=-1)[..., :-1]
            a_tr = (agree * trainable).sum(dim=(0, 1)) / denom
            relab = ((1.0 - same_prefix) * trainable).sum(dim=(0, 1)) / denom
            pos = ((1.0 - same_prefix) * trainable).sum() / trainable.sum().clamp(
                min=1.0
            )
            w = loss_weights.to(torch.float32) * trainable
            s_off = (w * (1.0 - same_prefix)).sum() / w.sum().clamp(min=1e-09)
            eot = (
                (1.0 - alive.to(torch.float32)) * trainable
            ).sum() / trainable.sum().clamp(min=1.0)
            print(
                "[ALR] rollout agreement | trainable "
                + " ".join((f"{v:.3f}" for v in a_tr.tolist())),
                flush=True,
            )
            print(
                "[ALR] relabelled | per-slot "
                + " ".join((f"{v:.3f}" for v in relab.tolist()))
                + f" | positions relabelled {pos.item():.4f}"
                + f" | weight relabelled {s_off.item():.4f}"
                + f" | end-of-turn masked {eot.item():.4f}"
                + (
                    "" if depth is None else f" | rollout steps {self.kd_rollout_steps}"
                ),
                flush=True,
            )

    def _report_inrollout(self, inrollout, loss_weights):
        with torch.no_grad():
            n1 = int(inrollout["primary"].shape[1])
            kept2 = inrollout["keep2"].to(torch.float32)
            q = int(kept2.sum().item())
            w = loss_weights.to(torch.float32)
            if q:
                mean_j = (inrollout["j"].to(torch.float32) * kept2).sum() / kept2.sum()
                slots = (w[:, n1:] > 0).to(torch.float32).sum(dim=-1)
                labelled = (slots * kept2).sum() / kept2.sum()
                tail = f" | mean j {mean_j.item():.3f} | labelled secondary slots {labelled.item():.3f}"
            else:
                tail = " | mean j n/a | labelled secondary slots n/a"
            share = w[:, n1:].sum() / w.sum().clamp(min=1e-09)
            print(
                f"[IRA] blocks primary {int(inrollout['keep1'].sum().item())} secondary {q}"
                + f" | ALR blocks {inrollout['alr_blocks']}"
                + tail
                + f" | secondary weight share {share.item():.4f}",
                flush=True,
            )

    def _report_slotmatch(self, slot_cap, weight_mask, loss_weights):
        with torch.no_grad():
            weighted = int((weight_mask.sum(dim=-1) > 0).sum().item())
            capped = (slot_cap < self.block_size - 1).to(torch.float32)
            q = int(capped.sum().item())
            w = loss_weights.to(torch.float32)
            share = (w * capped.unsqueeze(-1)).sum() / w.sum().clamp(min=1e-09)
            if q:
                mean_j = (
                    (self.block_size - 1 - slot_cap).to(torch.float32) * capped
                ).sum() / capped.sum()
                slots = (w > 0).to(torch.float32).sum(dim=-1)
                labelled = (slots * capped).sum() / capped.sum()
                tail = f" | mean j {mean_j.item():.3f} | labelled capped slots {labelled.item():.3f}"
            else:
                tail = " | mean j n/a | labelled capped slots n/a"
            print(
                f"[slot control] capped {q} | blocks with weight {weighted}"
                + tail
                + f" | capped weight share {share.item():.4f}",
                flush=True,
            )

    def _targetkd_loss(
        self,
        logits: torch.Tensor,
        anchor_positions: torch.Tensor,
        weight_mask: torch.Tensor,
        final_hidden_states: Optional[torch.Tensor],
        device: torch.device,
        target_ids: Optional[torch.Tensor] = None,
        valid_label_mask: Optional[torch.Tensor] = None,
        rollout_hidden: Optional[torch.Tensor] = None,
        rollout_tokens: Optional[torch.Tensor] = None,
        rollout_alive: Optional[torch.Tensor] = None,
        inrollout: Optional[dict] = None,
        slot_cap: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if final_hidden_states is None:
            raise ValueError(
                "loss_type='dflash_targetkd' requires final_hidden_states (the target post-final-norm hidden state); got None."
            )
        loss_weights = weight_mask
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            decay_weights = torch.exp(
                -(k - 1).clamp(min=0).float() / self.loss_decay_gamma
            )
            loss_weights = loss_weights * decay_weights
        soft = getattr(self, "kd_survival_soft", False)
        if soft and (target_ids is None or valid_label_mask is None):
            raise ValueError(
                "kd_survival_soft=True requires target_ids and valid_label_mask; got None."
            )
        rollout = getattr(self, "kd_anchor_rollout", False)
        rollout_soft = getattr(self, "kd_rollout_survival_soft", False)
        if rollout and (rollout_hidden is None or rollout_tokens is None):
            raise ValueError(
                "kd_anchor_rollout=True requires rollout_hidden and rollout_tokens; got None."
            )
        harvested = harvest_target_topk(
            final_hidden=final_hidden_states,
            anchor_positions=anchor_positions,
            block_size=self.block_size,
            lm_head=self.lm_head,
            topk=self.kd_topk,
            chunk_size=self.kd_chunk_size,
            label_ids=(
                rollout_tokens.clamp_min(0).reshape(-1)
                if rollout_soft
                else target_ids.reshape(-1)
                if soft
                else None
            ),
            harvested_hidden=rollout_hidden if rollout else None,
        )
        if soft or rollout_soft:
            topk_idx, topk_prob, tail_mass, label_prob = harvested
        else:
            topk_idx, topk_prob, tail_mass = harvested
        if rollout:
            if rollout_alive is None:
                alive = self._rollout_alive(rollout_tokens).to(loss_weights.dtype)
            else:
                alive = rollout_alive.to(loss_weights.dtype)
            depth = self._rollout_depth(loss_weights.device, loss_weights.dtype)
            if not getattr(self, "_alr_reported", False):
                self._alr_reported = True
                n1 = (
                    weight_mask.shape[1]
                    if inrollout is None
                    else inrollout["primary"].shape[1]
                )
                self._report_rollout(
                    rollout_tokens=rollout_tokens[:, :n1],
                    target_ids=target_ids[:, :n1],
                    weight_mask=weight_mask[:, :n1],
                    loss_weights=loss_weights[:, :n1],
                    alive=alive[:, :n1],
                    depth=depth,
                )
            loss_weights = loss_weights * alive
            if depth is not None:
                loss_weights = loss_weights * depth
            if inrollout is not None and (not getattr(self, "_ira_reported", False)):
                self._ira_reported = True
                self._report_inrollout(inrollout, loss_weights)
            if slot_cap is not None:
                k = torch.arange(self.block_size, device=loss_weights.device).view(
                    1, 1, -1
                )
                loss_weights = loss_weights * (k <= slot_cap.unsqueeze(-1)).to(
                    loss_weights.dtype
                )
                if not getattr(self, "_smc_reported", False):
                    self._smc_reported = True
                    self._report_slotmatch(slot_cap, weight_mask, loss_weights)
        elif slot_cap is not None:
            raise ValueError("slot_cap needs kd_anchor_rollout=True")
        if rollout_soft:
            gate = self._rollout_survival_gate_soft(
                label_prob.view_as(weight_mask), rollout_tokens
            )
            if not getattr(self, "_rollout_soft_reported", False):
                self._rollout_soft_reported = True
                with torch.no_grad():
                    scored = (loss_weights > 0).to(gate.dtype)
                    by_slot = (gate * scored).sum((0, 1)) / scored.sum((0, 1)).clamp_min(1)
                    kept = (loss_weights * gate).sum() / loss_weights.sum().clamp_min(1e-6)
                    print(
                        "[rollout gate] per-slot "
                        + " ".join((f"{v:.3f}" for v in by_slot.tolist()))
                        + f" | weight kept {kept.item():.4f}",
                        flush=True,
                    )
            loss_weights = loss_weights * gate
        if soft:
            gate = self._survival_gate_soft(
                label_prob=label_prob.view_as(weight_mask),
                valid_label_mask=valid_label_mask,
            )
            if not getattr(self, "_soft_gate_reported", False):
                self._soft_gate_reported = True
                with torch.no_grad():
                    trainable = (weight_mask > 0).to(gate.dtype)
                    denom = trainable.sum(dim=(0, 1)).clamp(min=1.0)
                    per_slot = (gate * trainable).sum(dim=(0, 1)) / denom
                    pos_keep = (gate * trainable).sum() / trainable.sum().clamp(min=1.0)
                    w_keep = (loss_weights * gate).sum() / loss_weights.sum().clamp(
                        min=1.0
                    )
                    c = label_prob.view_as(weight_mask).to(gate.dtype)
                    y_star_dbg = topk_idx[:, 0].view_as(target_ids)
                    on = (target_ids == y_star_dbg).to(gate.dtype) * trainable
                    off = (1.0 - (target_ids == y_star_dbg).to(gate.dtype)) * trainable
                    print(
                        "[erase] realised gate | per-slot "
                        + " ".join((f"{v:.3f}" for v in per_slot.tolist()))
                        + f" | mean weight {pos_keep.item():.4f}"
                        + f" | weight kept {w_keep.item():.4f}",
                        flush=True,
                    )
                    print(
                        "[erase] corpus-token prob | trainable "
                        + f"{((c * trainable).sum() / trainable.sum().clamp(min=1.0)).item():.4f}"
                        + f" | when argmax {((c * on).sum() / on.sum().clamp(min=1.0)).item():.4f}"
                        + f" | when not {((c * off).sum() / off.sum().clamp(min=1.0)).item():.4f}"
                        + f" | argmax agreement {(on.sum() / trainable.sum().clamp(min=1.0)).item():.4f}",
                        flush=True,
                    )
            loss_weights = loss_weights * gate
        if self.kd_survival_mask:
            if target_ids is None or valid_label_mask is None:
                raise ValueError(
                    "kd_survival_mask=True requires target_ids and valid_label_mask; got None."
                )
            gate = self._survival_gate(
                topk_idx=topk_idx,
                target_ids=target_ids,
                valid_label_mask=valid_label_mask,
            )
            if not getattr(self, "_hard_gate_reported", False):
                self._hard_gate_reported = True
                with torch.no_grad():
                    trainable = (weight_mask > 0).to(gate.dtype)
                    denom = trainable.sum(dim=(0, 1)).clamp(min=1.0)
                    per_slot = (gate * trainable).sum(dim=(0, 1)) / denom
                    pos_keep = (gate * trainable).sum() / trainable.sum().clamp(min=1.0)
                    w_keep = (loss_weights * gate).sum() / loss_weights.sum().clamp(
                        min=1.0
                    )
                    y_star_dbg = topk_idx[:, 0].view_as(target_ids)
                    agree_dbg = (target_ids == y_star_dbg).to(gate.dtype)
                    ib = valid_label_mask.to(gate.dtype)
                    a_tr = (agree_dbg * trainable).sum(dim=(0, 1)) / denom
                    a_ib = (agree_dbg * ib).sum(dim=(0, 1)) / ib.sum(dim=(0, 1)).clamp(
                        min=1.0
                    )
                    print(
                        "[hard gate] realised gate | per-slot "
                        + " ".join((f"{v:.3f}" for v in per_slot.tolist()))
                        + f" | positions kept {pos_keep.item():.4f}"
                        + f" | weight kept {w_keep.item():.4f}",
                        flush=True,
                    )
                    print(
                        "[hard gate] raw agreement | trainable "
                        + " ".join((f"{v:.3f}" for v in a_tr.tolist()))
                        + " | in_bounds "
                        + " ".join((f"{v:.3f}" for v in a_ib.tolist()))
                        + f" | trainable frac {(trainable.sum() / ib.sum().clamp(min=1.0)).item():.4f}",
                        flush=True,
                    )
            loss_weights = loss_weights * gate
        draft_logits = logits.view(-1, logits.size(-1))
        flat_weights = loss_weights.view(-1)
        loss = targetkd_loss(
            draft_logits=draft_logits,
            topk_idx=topk_idx,
            topk_prob=topk_prob,
            tail_mass=tail_mass,
            weight=flat_weights,
            hard_label_mix=self.kd_hard_label_mix,
            tail_bucket=self.kd_tail_bucket,
            lambda_scale=self.kd_lambda_scale,
        )
        accuracy = targetkd_accuracy(draft_logits, topk_idx, flat_weights)
        return (loss, accuracy)
