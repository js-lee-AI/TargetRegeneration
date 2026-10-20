from __future__ import annotations
import torch
import torch.nn.functional as F

_MAX_CACHE = 64
_ACCURACY_SLACK = 1.05


def _grid_key(grid_thw: torch.Tensor):
    return tuple(map(tuple, grid_thw.detach().cpu().tolist()))


def _patch_embed_error(pe, x: torch.Tensor, fn) -> float:
    w2 = pe.proj.weight.reshape(pe.proj.weight.shape[0], -1)
    with torch.no_grad():
        exact = x.double() @ w2.double().T + pe.proj.bias.double()
        return (fn(x).double() - exact).abs().mean().item()


def _install_patch_embed(visual) -> bool:
    pe = getattr(visual, "patch_embed", None)
    proj = getattr(pe, "proj", None)
    if pe is None or not isinstance(proj, torch.nn.Conv3d):
        return False
    k = tuple(proj.kernel_size)
    if (
        tuple(proj.stride) != k
        or any((p != 0 for p in proj.padding))
        or any((d != 1 for d in proj.dilation))
    ):
        return False
    if proj.groups != 1:
        return False
    embed_dim = proj.weight.shape[0]
    weight2d = proj.weight.reshape(embed_dim, -1)
    bias = proj.bias
    orig_forward = pe.forward

    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states.to(weight2d.dtype), weight2d, bias)

    dev = weight2d.device
    probe = torch.randn(256, weight2d.shape[1], device=dev, dtype=weight2d.dtype)
    err_old = _patch_embed_error(pe, probe, orig_forward)
    err_new = _patch_embed_error(pe, probe, forward)
    if not err_new <= err_old * _ACCURACY_SLACK:
        return False
    pe.forward = forward
    pe._sf_patch_embed_error = (err_old, err_new)
    return True


def _install_pos_embed_cache(visual) -> bool:
    if not (
        hasattr(visual, "rot_pos_emb") and hasattr(visual, "fast_pos_embed_interpolate")
    ):
        return False
    orig_rot = visual.rot_pos_emb
    orig_fast = visual.fast_pos_embed_interpolate
    cache_rot: dict = {}
    cache_fast: dict = {}

    def rot_pos_emb(grid_thw: torch.Tensor):
        key = _grid_key(grid_thw)
        hit = cache_rot.get(key)
        if hit is None:
            hit = orig_rot(grid_thw.cpu())
            if len(cache_rot) < _MAX_CACHE:
                cache_rot[key] = hit
        return hit

    def fast_pos_embed_interpolate(grid_thw: torch.Tensor):
        key = _grid_key(grid_thw)
        hit = cache_fast.get(key)
        if hit is None:
            hit = orig_fast(grid_thw.cpu())
            if len(cache_fast) < _MAX_CACHE:
                cache_fast[key] = hit
        return hit

    probe = torch.tensor(
        [[1, 28, 28], [1, 22, 32]],
        dtype=torch.long,
        device=next(visual.parameters()).device,
    )
    for new_fn, old_fn in (
        (rot_pos_emb, orig_rot),
        (fast_pos_embed_interpolate, orig_fast),
    ):
        with torch.no_grad():
            a, b = (old_fn(probe), new_fn(probe))
        a = a[0] if isinstance(a, tuple) else a
        b = b[0] if isinstance(b, tuple) else b
        if a.shape != b.shape or not torch.equal(a, b):
            return False
    visual.rot_pos_emb = rot_pos_emb
    visual.fast_pos_embed_interpolate = fast_pos_embed_interpolate
    return True


def apply_qwen3vl_vision_fastpath(model) -> dict:
    visual = getattr(getattr(model, "model", model), "visual", None)
    if visual is None:
        return {}
    if getattr(visual, "_sf_vision_fastpath", None) is not None:
        return visual._sf_vision_fastpath
    applied = {
        "patch_embed_matmul": _install_patch_embed(visual),
        "pos_embed_cache": _install_pos_embed_cache(visual),
    }
    pe = getattr(visual, "patch_embed", None)
    if applied["patch_embed_matmul"] and pe is not None:
        old, new = pe._sf_patch_embed_error
        applied["patch_embed_err_conv"] = old
        applied["patch_embed_err_matmul"] = new
    visual._sf_vision_fastpath = applied
    return applied


def _fp32_conv_forward(pe):
    proj = pe.proj
    w32, b32 = (proj.weight.float(), proj.bias.float())
    stride = proj.stride
    out_dtype = proj.weight.dtype
    ic, tp, ps, embed = (
        pe.in_channels,
        pe.temporal_patch_size,
        pe.patch_size,
        pe.embed_dim,
    )

    def forward(hidden_states):
        v = hidden_states.float().view(-1, ic, tp, ps, ps)
        return F.conv3d(v, w32, b32, stride=stride).view(-1, embed).to(out_dtype)

    return forward


def _matmul_forward(pe):
    w2 = pe.proj.weight.reshape(pe.proj.weight.shape[0], -1)
    bias = pe.proj.bias

    def forward(hidden_states):
        return F.linear(hidden_states.to(w2.dtype), w2, bias)

    return forward


_MODES = {"fp32": _fp32_conv_forward, "matmul": _matmul_forward}


def _eligible(pe) -> bool:
    proj = getattr(pe, "proj", None)
    if not isinstance(proj, torch.nn.Conv3d):
        return False
    return (
        tuple(proj.stride) == tuple(proj.kernel_size)
        and all((p == 0 for p in proj.padding))
        and all((d == 1 for d in proj.dilation))
        and (proj.groups == 1)
    )


def install_patch_embed_classwide(mode: str = "matmul") -> bool:
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl as mod
    except Exception:
        return False
    cls = getattr(mod, "Qwen3VLVisionPatchEmbed", None)
    if cls is None or getattr(cls, "_sf_patched", False):
        return cls is not None
    make = _MODES.get(mode)
    if make is None:
        return False
    original = cls.forward

    def forward(self, hidden_states):
        chosen = getattr(self, "_sf_forward", None)
        if chosen is None:
            chosen = False
            if _eligible(self):
                cand = make(self)
                dev = self.proj.weight.device
                probe = torch.randn(
                    256,
                    self.proj.weight.reshape(self.proj.weight.shape[0], -1).shape[1],
                    device=dev,
                    dtype=self.proj.weight.dtype,
                )
                err_old = _patch_embed_error(self, probe, lambda x: original(self, x))
                err_new = _patch_embed_error(self, probe, cand)
                if err_new <= err_old * _ACCURACY_SLACK:
                    chosen = cand
                    self._sf_patch_embed_error = (err_old, err_new)
            self._sf_forward = chosen
        if chosen is False:
            return original(self, hidden_states)
        return chosen(hidden_states)

    cls.forward = forward
    cls._sf_patched = True
    return True
