import inspect
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoModelForCausalLM
from specforge.distributed import get_tp_group

try:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import RadixCache
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
    )
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.utils import require_mlp_sync, require_mlp_tp_gather
    from .sglang_backend import SGLangRunner

    _SGLANG_IMPORT_ERROR = None
except Exception as _exc:
    _SGLANG_IMPORT_ERROR = _exc
    ModelConfig = Req = ScheduleBatch = Scheduler = None
    CacheInitParams = RadixCache = CaptureHiddenMode = ForwardBatch = None
    SamplingParams = ServerArgs = SpeculativeAlgorithm = None
    require_mlp_sync = require_mlp_tp_gather = None
    SGLangRunner = None


@dataclass
class DFlashTargetOutput:
    hidden_states: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    final_hidden_states: Optional[torch.Tensor] = None
    rollout_hidden: Optional[torch.Tensor] = None
    rollout_tokens: Optional[torch.Tensor] = None
    rollout_context: Optional[torch.Tensor] = None


class DFlashTargetModel(ABC):
    def __init__(self):
        self.capture_layer_ids = None

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "DFlashTargetModel":
        pass

    @abstractmethod
    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> DFlashTargetOutput:
        pass

    def set_capture_layers(self, layer_ids: List[int]) -> None:
        self.capture_layer_ids = layer_ids


class SGLangDFlashTargetModel(DFlashTargetModel):
    def __init__(self, model_runner: SGLangRunner):
        super().__init__()
        self.model_runner = model_runner

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "SGLangDFlashTargetModel":
        tp_size = dist.get_world_size(get_tp_group())
        server_args = ServerArgs(
            model_path=pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype,
            enable_return_hidden_states=True,
            disable_cuda_graph=True,
            tp_size=tp_size,
            pp_size=1,
            **kwargs,
        )
        tp_rank = dist.get_rank(get_tp_group())
        moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
        model_config = ModelConfig.from_server_args(server_args)
        model_runner = SGLangRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=torch.cuda.current_device(),
            tp_rank=dist.get_rank(get_tp_group()),
            tp_size=server_args.tp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=server_args.ep_size,
            pp_rank=0,
            pp_size=1,
            server_args=server_args,
            nccl_port=None,
        )
        return cls(model_runner)

    def set_capture_layers(self, layer_ids: List[int]) -> None:
        super().set_capture_layers(layer_ids)
        if hasattr(self.model_runner.model, "set_eagle3_layers_to_capture"):
            self.model_runner.model.set_eagle3_layers_to_capture(layer_ids)
            print(self.model_runner.model.model.layers_to_capture)

    @torch.no_grad
    def _extend(self, reqs):
        cache_params = CacheInitParams(
            disable=False,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            page_size=self.model_runner.server_args.page_size,
        )
        tree_cache = RadixCache(cache_params)
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        if require_mlp_sync(self.model_runner.server_args):
            Scheduler.prepare_mlp_sync_batch_raw(
                batch,
                dp_size=self.model_runner.server_args.dp_size,
                attn_tp_size=1,
                tp_group=self.model_runner.tp_group,
                get_idle_batch=None,
                disable_cuda_graph=self.model_runner.server_args.disable_cuda_graph,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                speculative_num_draft_tokens=None,
                require_mlp_tp_gather=require_mlp_tp_gather(
                    self.model_runner.server_args
                ),
                disable_overlap_schedule=self.model_runner.server_args.disable_overlap_schedule,
                offload_tags=set(),
            )
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        output = self.model_runner.forward(forward_batch)
        if hasattr(output, "logits_output"):
            output = output.logits_output
        input_lens = [len(req.origin_input_ids) for req in reqs]
        if (
            hasattr(output, "aux_hidden_states")
            and output.aux_hidden_states is not None
        ):
            hidden_states_list = torch.split(
                output.aux_hidden_states, input_lens, dim=0
            )
        elif hasattr(output, "hidden_states") and output.hidden_states is not None:
            hidden_states_list = torch.split(output.hidden_states, input_lens, dim=0)
        else:
            raise ValueError("SGLang output does not contain hidden states.")
        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()
        return hidden_states_list

    @torch.no_grad()
    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> DFlashTargetOutput:
        sampling_params = SamplingParams(temperature=0, max_new_tokens=1)
        reqs, data_cache = ([], [])
        if isinstance(input_ids, torch.Tensor):
            input_ids_list = torch.split(input_ids, 1, dim=0)
            attn_mask_list = torch.split(attention_mask, 1, dim=0)
            loss_mask_list = torch.split(loss_mask, 1, dim=0)
        for idx, (curr_ids, curr_attn, curr_loss) in enumerate(
            zip(input_ids_list, attn_mask_list, loss_mask_list)
        ):
            req = Req(
                rid=str(idx),
                origin_input_text="",
                origin_input_ids=curr_ids.view(-1).tolist(),
                sampling_params=sampling_params,
            )
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
            data_cache.append((curr_ids, curr_attn, curr_loss))
            reqs.append(req)
        hidden_states_list = self._extend(reqs)
        hidden_states = torch.cat([h.unsqueeze(0) for h in hidden_states_list], dim=0)
        input_ids = torch.cat([d[0] for d in data_cache], dim=0)
        attention_mask = torch.cat([d[1] for d in data_cache], dim=0)
        loss_mask = torch.cat([d[2] for d in data_cache], dim=0)
        return DFlashTargetOutput(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )


class HFDFlashTargetModel(DFlashTargetModel):
    def __init__(self, model: nn.Module, is_vlm: bool = False):
        super().__init__()
        self.model = model
        self.is_vlm = is_vlm

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        is_vlm: bool = False,
        **kwargs,
    ) -> "HFDFlashTargetModel":
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code
        )
        is_vlm = is_vlm or hasattr(config, "vision_config")
        if is_vlm:
            model_type = getattr(config, "model_type", "")
            if model_type == "qwen3_vl":
                from transformers import Qwen3VLForConditionalGeneration as _VLModelCls
            elif model_type in ("qwen2_5_vl", "qwen2_vl"):
                from transformers import (
                    Qwen2_5_VLForConditionalGeneration as _VLModelCls,
                )
            else:
                from transformers import AutoModelForImageTextToText as _VLModelCls
            target_model = _VLModelCls.from_pretrained(
                pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                cache_dir=cache_dir,
                **kwargs,
            )
        else:
            target_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                cache_dir=cache_dir,
                output_hidden_states=True,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        target_model.train(False)
        if device:
            target_model = target_model.to(device)
        if is_vlm and os.environ.get("SF_VISION_FASTPATH", "1") != "0":
            from specforge.modeling.target.qwen3vl_vision_fastpath import (
                apply_qwen3vl_vision_fastpath,
            )

            applied = apply_qwen3vl_vision_fastpath(target_model)
            print(f"Qwen3-VL vision fast path: {applied}")
        return cls(target_model, is_vlm=is_vlm)

    _FINAL_NORM_PATHS = ("norm", "language_model.norm", "text_model.norm", "model.norm")

    def _resolve_final_norm(self):
        cached = getattr(self, "_targetkd_final_norm", "unset")
        if cached != "unset":
            return cached
        found = None
        root = self.model.model
        for path in self._FINAL_NORM_PATHS:
            obj = root
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
            except AttributeError:
                continue
            if callable(obj):
                found = obj
                break
        self._targetkd_final_norm = found
        return found

    def _targetkd_final_hidden(self, outputs):
        raw = outputs.hidden_states[-1]
        mode = getattr(self, "_targetkd_norm_mode", None)
        if mode is None:
            lm_head = self.model.get_output_embeddings()
            ref_last = outputs.logits[:, -1, :].float()
            tol = max(0.01, 0.0001 * float(ref_last.abs().max()))
            d_raw = float((lm_head(raw[:, -1, :]).float() - ref_last).abs().max())
            norm = self._resolve_final_norm()
            d_norm = float("inf")
            if norm is not None:
                d_norm = float(
                    (lm_head(norm(raw[:, -1:, :])[:, -1, :]).float() - ref_last)
                    .abs()
                    .max()
                )
            if d_raw <= tol and d_raw <= d_norm:
                mode = "raw"
            elif d_norm <= tol:
                mode = "norm"
            else:
                raise RuntimeError(
                    f"[KD self-check] neither hidden_states[-1] nor final_norm(hidden_states[-1]) reproduces outputs.logits (max|delta| raw={d_raw:.3e}, norm={d_norm:.3e}, tol={tol:.3e}). The harvested KD labels would not be the target's own distribution. Aborting."
                )
            self._targetkd_norm_mode = mode
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(
                    f"[KD] final-hidden mode={mode} (max|delta| raw={d_raw:.3e}, norm={d_norm:.3e})",
                    flush=True,
                )
        if mode == "raw":
            return raw
        return self._resolve_final_norm()(raw)

    @torch.no_grad()
    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        return_final_hidden: bool = False,
        anchor_rollout: Optional[tuple] = None,
        rollout_steps: Optional[int] = None,
        rollout_context_steps: Optional[int] = None,
    ) -> DFlashTargetOutput:
        if anchor_rollout is not None and (not return_final_hidden):
            raise ValueError("anchor_rollout needs return_final_hidden=True")
        if rollout_steps is not None and anchor_rollout is None:
            raise ValueError(
                "rollout_steps truncates the anchor rollout; it needs anchor_rollout"
            )
        if rollout_context_steps is not None:
            if anchor_rollout is None:
                raise ValueError(
                    "rollout_context_steps reads the anchor rollout; it needs anchor_rollout"
                )
            if rollout_steps is not None:
                raise ValueError(
                    "rollout_context_steps needs the full rollout; drop rollout_steps"
                )
            if not 1 <= rollout_context_steps <= anchor_rollout[1] - 3:
                raise ValueError(
                    f"rollout_context_steps must be in [1, {anchor_rollout[1] - 3}], got {rollout_context_steps}"
                )
            if self.capture_layer_ids is None:
                raise ValueError("rollout_context_steps needs set_capture_layers(...)")
        forward_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=anchor_rollout is not None,
            logits_to_keep=1,
        )
        position_ids = None
        if self.is_vlm and pixel_values is not None:
            forward_kwargs["pixel_values"] = pixel_values
            forward_kwargs["image_grid_thw"] = image_grid_thw
            rope_index_fn = self.model.model.get_rope_index
            rope_index_kwargs = dict(
                image_grid_thw=image_grid_thw, attention_mask=attention_mask
            )
            if "second_per_grid_ts" in inspect.signature(rope_index_fn).parameters:
                rope_index_kwargs["second_per_grid_ts"] = None
            position_ids, _ = rope_index_fn(input_ids, **rope_index_kwargs)
            forward_kwargs["position_ids"] = position_ids
        outputs = self.model(**forward_kwargs)
        offset = 1
        selected = []
        if self.capture_layer_ids is not None:
            for idx in self.capture_layer_ids:
                selected.append(outputs.hidden_states[idx + offset])
            hidden_states = torch.cat(selected, dim=-1)
        else:
            hidden_states = outputs.hidden_states[-1]
        final_hidden = (
            self._targetkd_final_hidden(outputs) if return_final_hidden else None
        )
        rollout_hidden = rollout_tokens = rollout_context = None
        if anchor_rollout is not None:
            anchors, block_size = anchor_rollout
            cache = outputs.past_key_values
            outputs.past_key_values = None
            outputs.hidden_states = None
            rolled = self._anchor_rollout(
                cache=cache,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                final_hidden=final_hidden,
                anchor_positions=anchors,
                block_size=block_size,
                steps=rollout_steps,
                capture_steps=rollout_context_steps or 0,
            )
            if rollout_context_steps:
                rollout_hidden, rollout_tokens, rollout_context = rolled
            else:
                rollout_hidden, rollout_tokens = rolled
            del cache
        return DFlashTargetOutput(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            final_hidden_states=final_hidden,
            rollout_hidden=rollout_hidden,
            rollout_tokens=rollout_tokens,
            rollout_context=rollout_context,
        )

    def _text_stack(self):
        root = self.model.model
        lm = getattr(root, "language_model", root)
        return (lm.layers, lm.norm, lm.rotary_emb, lm.embed_tokens)

    @staticmethod
    def _cache_layer_kv(cache, idx):
        layers = getattr(cache, "layers", None)
        if layers is not None:
            return (layers[idx].keys, layers[idx].values)
        return (cache.key_cache[idx], cache.value_cache[idx])

    @staticmethod
    def _release_cache_layer(cache, idx):
        layers = getattr(cache, "layers", None)
        if layers is not None:
            layers[idx].keys = None
            layers[idx].values = None
        else:
            cache.key_cache[idx] = None
            cache.value_cache[idx] = None

    @torch.no_grad()
    def _greedy(self, hidden: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        lm_head = self.model.get_output_embeddings()
        flat = hidden.reshape(-1, hidden.shape[-1])
        out = torch.empty(flat.shape[0], dtype=torch.long, device=flat.device)
        for c0 in range(0, flat.shape[0], chunk):
            out[c0 : c0 + chunk] = lm_head(flat[c0 : c0 + chunk]).float().argmax(dim=-1)
        return out.view(hidden.shape[:-1])

    @torch.no_grad()
    def _anchor_rollout(
        self,
        cache,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        final_hidden: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_size: int,
        steps: Optional[int] = None,
        capture_steps: int = 0,
    ):
        layers, final_norm, rotary_emb, embed_tokens = self._text_stack()
        B, S, D = final_hidden.shape
        N = anchor_positions.shape[1]
        if steps is None:
            steps = block_size - 2
        elif not 0 <= steps <= block_size - 2:
            raise ValueError(
                f"rollout steps must be in [0, {block_size - 2}], got {steps}"
            )
        device = final_hidden.device
        cap_pos = {}
        ctx_feat = None
        if capture_steps:
            ids = list(self.capture_layer_ids or [])
            if not ids or max(ids) >= len(layers) - 1 or min(ids) < 0:
                raise ValueError(
                    f"capture_steps needs capture layers in [0, {len(layers) - 2}], got {ids}"
                )
            if not 1 <= capture_steps <= steps:
                raise ValueError(
                    f"capture_steps must be in [1, {steps}], got {capture_steps}"
                )
            cap_pos = {layer_id: j for j, layer_id in enumerate(ids)}
            ctx_feat = torch.empty(
                B,
                N,
                capture_steps,
                len(ids) * D,
                dtype=final_hidden.dtype,
                device=device,
            )
        bidx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
        hid = torch.empty(B, N, block_size, D, dtype=final_hidden.dtype, device=device)
        tok = torch.empty(B, N, block_size, dtype=torch.long, device=device)
        hid[:, :, 0] = final_hidden[bidx, (anchor_positions - 1).clamp(min=0)]
        hid[:, :, 1] = final_hidden[bidx, anchor_positions]
        tok[:, :, 0] = input_ids[bidx, anchor_positions]
        y = self._greedy(hid[:, :, 1])
        tok[:, :, 1] = y
        if steps <= 0:
            return self._blank_truncated_slots(hid, tok, steps)
        kc, vc, kr, vr = ([], [], [], [])
        for i in range(len(layers)):
            k, v = self._cache_layer_kv(cache, i)
            kc.append(k.contiguous())
            vc.append(v.contiguous())
            self._release_cache_layer(cache, i)
            kr.append(k.new_empty(B, k.shape[1], N, steps, k.shape[3]))
            vr.append(v.new_empty(B, v.shape[1], N, steps, v.shape[3]))
        ctx = torch.arange(S, device=device).view(
            1, 1, S
        ) <= anchor_positions.unsqueeze(-1)
        ctx = ctx & attention_mask.bool().unsqueeze(1)
        blocked = (~ctx).view(B, 1, 1, N, S)
        if position_ids is not None and position_ids.dim() == 3:
            base = position_ids.gather(
                2, anchor_positions.unsqueeze(0).expand(3, -1, -1)
            )
        else:
            base = anchor_positions
        attn0 = layers[0].self_attn
        mod = inspect.getmodule(type(attn0))
        apply_rope = getattr(mod, "apply_rotary_pos_emb")
        for t in range(1, steps + 1):
            x = embed_tokens(y)
            cos, sin = rotary_emb(x, base + t)
            for i, layer in enumerate(layers):
                attn = layer.self_attn
                residual = x
                h = layer.input_layernorm(x)
                q = attn.q_norm(attn.q_proj(h).view(B, N, -1, attn.head_dim)).transpose(
                    1, 2
                )
                k = attn.k_norm(attn.k_proj(h).view(B, N, -1, attn.head_dim)).transpose(
                    1, 2
                )
                v = attn.v_proj(h).view(B, N, -1, attn.head_dim).transpose(1, 2)
                q, k = apply_rope(q, k, cos, sin)
                kr[i][:, :, :, t - 1] = k
                vr[i][:, :, :, t - 1] = v
                o = self._split_attention(
                    q,
                    kc[i],
                    vc[i],
                    kr[i][:, :, :, :t],
                    vr[i][:, :, :, :t],
                    blocked,
                    attn.scaling,
                )
                x = residual + attn.o_proj(o)
                residual = x
                x = residual + layer.mlp(layer.post_attention_layernorm(x))
                if ctx_feat is not None and t <= capture_steps and (i in cap_pos):
                    j = cap_pos[i]
                    ctx_feat[:, :, t - 1, j * D : (j + 1) * D] = x
            h = final_norm(x)
            hid[:, :, t + 1] = h
            y = self._greedy(h)
            tok[:, :, t + 1] = y
        hid, tok = self._blank_truncated_slots(hid, tok, steps)
        return (hid, tok) if ctx_feat is None else (hid, tok, ctx_feat)

    @staticmethod
    def _blank_truncated_slots(hid, tok, steps):
        last = steps + 1
        if last + 1 < hid.shape[2]:
            hid[:, :, last + 1 :] = hid[:, :, last : last + 1]
            tok[:, :, last + 1 :] = -1
        return (hid, tok)

    @staticmethod
    def _split_attention(q, kc, vc, kr, vr, blocked, scaling):
        B, Hq, N, d = q.shape
        Hkv, S = (kc.shape[1], kc.shape[2])
        t = kr.shape[3]
        g = Hq // Hkv
        qg = q.reshape(B, Hkv, g, N, d)
        sc = torch.bmm(
            qg.reshape(B * Hkv, g * N, d), kc.view(B * Hkv, S, d).transpose(1, 2)
        )
        qn = qg.permute(0, 1, 3, 2, 4).reshape(B * Hkv * N, g, d)
        sr = torch.bmm(qn, kr.reshape(B * Hkv * N, t, d).transpose(1, 2))
        sr = sr.view(B, Hkv, N, g, t).permute(0, 1, 3, 2, 4)
        scores = torch.cat([sc.view(B, Hkv, g, N, S), sr], dim=-1).float().mul_(scaling)
        scores[..., :S].masked_fill_(blocked, float("-inf"))
        probs = torch.softmax(scores, dim=-1).to(vc.dtype)
        oc = torch.bmm(
            probs[..., :S].reshape(B * Hkv, g * N, S), vc.view(B * Hkv, S, d)
        )
        pr = probs[..., S:].permute(0, 1, 3, 2, 4).reshape(B * Hkv * N, g, t)
        orr = torch.bmm(pr, vr.reshape(B * Hkv * N, t, d))
        out = oc.view(B, Hkv, g, N, d) + orr.view(B, Hkv, N, g, d).permute(
            0, 1, 3, 2, 4
        )
        return out.reshape(B, Hq, N, d).transpose(1, 2).reshape(B, N, Hq * d)


def get_dflash_target_model(
    pretrained_model_name_or_path: str,
    backend: str = "sglang",
    torch_dtype: torch.dtype = None,
    device: str = None,
    cache_dir: Optional[str] = None,
    is_vlm: bool = False,
    **kwargs,
) -> DFlashTargetModel:
    if backend == "sglang":
        return SGLangDFlashTargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    elif backend == "hf":
        return HFDFlashTargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            is_vlm=is_vlm,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")
