import time
import zlib
import torch
import torch.nn.functional as F
from transformers import DynamicCache
from .tree import build_budget_tree, make_tree_mask, gather_cache


class Noise:
    def __init__(self, temperature, seed, rows, vocab, device):
        self.T = float(temperature)
        self.G = None
        if self.T >= 1e-05:
            g = torch.Generator().manual_seed(int(seed))
            u = torch.rand((rows, vocab), generator=g, dtype=torch.float32)
            u.clamp_(min=torch.finfo(torch.float32).tiny)
            self.G = u.log_().neg_().log_().neg_().to(device)

    def pick(self, logits, k):
        if self.G is None:
            return int(logits.argmax())
        return int((logits.float() / self.T + self.G[k]).argmax())

    def scores(self, logits, k):
        if self.G is None:
            return logits.float()
        return logits.float() / self.T + self.G[k]


def prompt_seed(sample_seed, domain, i):
    return (
        int(sample_seed) * 1000003
        + zlib.crc32(domain.encode()) % 100000 * 10007
        + int(i)
    ) % 2**62


@torch.inference_mode()
def ar_generate(
    target,
    input_ids,
    pixel_values,
    image_grid_thw,
    max_new_tokens,
    stop_token_ids,
    noise,
    return_stats=False,
):
    device = input_ids.device
    kv = DynamicCache()
    n_in = input_ids.shape[1]
    target.model.rope_deltas = None
    if return_stats:
        torch.cuda.synchronize()
        t_pre = time.perf_counter()
    out = target(
        input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        past_key_values=kv,
        use_cache=True,
        logits_to_keep=1,
        cache_position=torch.arange(n_in, device=device),
    )
    toks = [noise.pick(out.logits[0, -1], 0)]
    if return_stats:
        torch.cuda.synchronize()
        t_dec = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        out = target(
            torch.tensor([[toks[-1]]], device=device),
            past_key_values=kv,
            use_cache=True,
            cache_position=torch.tensor([kv.get_seq_length()], device=device),
        )
        toks.append(noise.pick(out.logits[0, -1], len(toks)))
        if stop_token_ids is not None and toks[-1] in stop_token_ids:
            break
    if return_stats:
        torch.cuda.synchronize()
        t_end = time.perf_counter()
    final = torch.cat([input_ids, torch.tensor([toks], device=device)], dim=1)
    if not return_stats:
        return final
    n_out = len(toks)
    return (
        final,
        {
            "n_tokens": n_out,
            "ms_per_token": (t_end - t_dec) * 1000.0 / max(n_out, 1),
            "prefill_ms": (t_dec - t_pre) * 1000.0,
            "total_ms": (t_end - t_pre) * 1000.0,
        },
    )


@torch.inference_mode()
def tree_generate(
    draft,
    target,
    tap,
    input_ids,
    pixel_values,
    image_grid_thw,
    max_new_tokens,
    stop_token_ids,
    noise,
    budget=31,
    branch_k=8,
    return_stats=False,
):
    device = input_ids.device
    B = draft.block_size
    mask_id = draft.mask_token_id
    n_in = input_ids.shape[1]
    max_len = n_in + max_new_tokens
    dt = next(target.parameters()).dtype
    out_ids = torch.full(
        (1, max_len + 2 * B + 4), mask_id, dtype=torch.long, device=device
    )
    pos_ids = torch.arange(out_ids.shape[1], device=device).unsqueeze(0)
    kv_t = DynamicCache()
    kv_d = DynamicCache()
    target.model.rope_deltas = None
    pos3_prefill, _deltas = target.model.get_rope_index(
        input_ids, image_grid_thw, None, None
    )
    mrope_delta = int(_deltas[0, 0])

    def _vlpos(sp):
        return (sp + mrope_delta).unsqueeze(0).expand(3, -1, -1)

    decode_t0 = None
    tap.enabled = True
    out = target(
        input_ids,
        position_ids=pos3_prefill,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        past_key_values=kv_t,
        use_cache=True,
        logits_to_keep=1,
    )
    out_ids[:, :n_in] = input_ids
    out_ids[:, n_in] = noise.pick(out.logits[0, -1], 0)
    th = tap.gather()
    emb = target.get_input_embeddings()
    start = n_in
    taus, pack_sizes = ([], [])
    while start < max_len:
        blk = out_ids[:, start : start + B].clone()
        blk[:, 1:] = mask_id
        noise_emb = emb(blk)
        d_hidden = draft(
            target_hidden=th,
            noise_embedding=noise_emb,
            position_ids=pos_ids[:, kv_d.get_seq_length() : start + B],
            past_key_values=kv_d,
            use_cache=True,
            is_causal=False,
        )
        d_logits = target.lm_head(d_hidden[:, 1 - B :, :])[0]
        kv_d.crop(start)
        lp_full = F.log_softmax(d_logits.float(), dim=-1)
        topk = torch.topk(lp_full, branch_k, dim=-1)
        topk_idx, topk_lp = (topk.indices.cpu(), topk.values.cpu())
        nodes = build_budget_tree(d_logits, budget, branch_k, topk_idx, topk_lp)
        if decode_t0 is None:
            torch.cuda.synchronize()
            decode_t0 = time.perf_counter()
        pack_tokens = torch.tensor(
            [[int(out_ids[0, start])] + [nd.token for nd in nodes]], device=device
        )
        pack_pos = torch.tensor(
            [[start] + [start + nd.depth for nd in nodes]], device=device
        )
        n_prefix = kv_t.get_seq_length()
        amask = make_tree_mask(n_prefix, nodes, device, dt)
        out = target(
            pack_tokens,
            position_ids=_vlpos(pack_pos),
            past_key_values=kv_t,
            use_cache=True,
            attention_mask=amask,
        )
        logits_pack = out.logits[0]
        feats_all = tap.gather()[0]
        children = {i: [] for i in range(-1, len(nodes))}
        for i, nd in enumerate(nodes):
            children[nd.parent].append(i)
        path = []
        cur = -1
        while True:
            packed_cur = 0 if cur == -1 else cur + 1
            depth = 0 if cur == -1 else nodes[cur].depth
            nxt_tok = noise.pick(logits_pack[packed_cur], start + depth + 1 - n_in)
            hit = None
            for c in children.get(cur, []):
                if nodes[c].token == nxt_tok:
                    hit = c
                    break
            if hit is None:
                bonus = nxt_tok
                break
            path.append(hit)
            cur = hit
        a = len(path)
        taus.append(a + 1)
        pack_sizes.append(1 + len(nodes))
        for d, ni in enumerate(path):
            out_ids[0, start + 1 + d] = nodes[ni].token
        out_ids[0, start + a + 1] = bonus
        keep = [0] + [i + 1 for i in path]
        gather_cache(kv_t, n_prefix, keep)
        th = feats_all[[k for k in keep], :][None, :, :]
        start += a + 1
        if stop_token_ids is not None and any(
            (t in out_ids[0, n_in:start].tolist() for t in stop_token_ids)
        ):
            break
    final = out_ids[:, : min(start + 1, max_len)]
    if stop_token_ids is not None:
        st = torch.tensor(stop_token_ids, device=device)
        idx = torch.isin(final[0, n_in:], st).nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            final = final[:, : n_in + idx[0] + 1]
    tap.enabled = False
    decode_elapsed = None
    if decode_t0 is not None:
        torch.cuda.synchronize()
        decode_elapsed = time.perf_counter() - decode_t0
    if not return_stats:
        return final
    n_out = final.shape[1] - n_in
    elapsed = decode_elapsed if decode_elapsed is not None else 0.0
    return (
        final,
        {
            "tau_mean": sum(taus) / max(len(taus), 1),
            "n_rounds": len(taus),
            "n_out": n_out,
            "ms_per_token": elapsed * 1000.0 / max(n_out, 1),
            "pack_mean": sum(pack_sizes) / max(len(pack_sizes), 1),
        },
    )
