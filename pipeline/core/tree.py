import heapq
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from transformers import DynamicCache


@dataclass
class TreeNode:
    token: int
    depth: int
    parent: int
    rank: int


def build_budget_tree(
    d_logits: torch.Tensor, budget: int, branch_k: int, topk_idx=None, topk_lp=None
):
    if topk_idx is None:
        lp_full = F.log_softmax(d_logits.float(), dim=-1)
        topk = torch.topk(lp_full, branch_k, dim=-1)
        topk_idx, topk_lp = (topk.indices.cpu(), topk.values.cpu())
    m = topk_idx.shape[0]
    nodes = []
    cnt = 0
    heap = []
    for r in range(min(branch_k, topk_idx.shape[1])):
        heapq.heappush(heap, (-float(topk_lp[0, r]), 1, -1, r, cnt))
        cnt += 1
    while heap and len(nodes) < budget:
        neg, d, parent, r, _ = heapq.heappop(heap)
        my_idx = len(nodes)
        nodes.append(TreeNode(int(topk_idx[d - 1, r]), d, parent, r))
        if d < m:
            for r2 in range(min(branch_k, topk_idx.shape[1])):
                heapq.heappush(
                    heap, (neg - float(topk_lp[d, r2]), d + 1, my_idx, r2, cnt)
                )
                cnt += 1
    return nodes


def make_tree_mask(n_prefix: int, nodes, device, dtype):
    import numpy as np

    n_pack = 1 + len(nodes)
    vis = np.zeros((n_pack, n_pack), dtype=bool)
    vis[0, 0] = True
    for i, nd in enumerate(nodes):
        pi = i + 1
        vis[pi] = vis[nd.parent + 1]
        vis[pi, pi] = True
    neg = torch.finfo(dtype).min
    pack_mask = torch.from_numpy(~vis).to(device)
    mask = torch.zeros((1, 1, n_pack, n_prefix + n_pack), device=device, dtype=dtype)
    mask[0, 0, :, n_prefix:] = pack_mask * neg
    return mask


def gather_cache(cache: DynamicCache, n_prefix: int, keep_packed: list[int]):
    if hasattr(cache, "layers") and cache.layers and hasattr(cache.layers[0], "keys"):
        dev = cache.layers[0].keys.device
        sel = torch.arange(n_prefix, device=dev)
        if keep_packed:
            sel = torch.cat(
                [sel, torch.tensor([n_prefix + i for i in keep_packed], device=dev)]
            )
        for layer in cache.layers:
            layer.keys = layer.keys.index_select(2, sel)
            layer.values = layer.values.index_select(2, sel)
        if hasattr(cache, "_seen_tokens"):
            cache._seen_tokens = sel.numel()
        return cache
    dev = cache.key_cache[0].device
    sel = torch.arange(n_prefix, device=dev)
    if keep_packed:
        sel = torch.cat(
            [sel, torch.tensor([n_prefix + i for i in keep_packed], device=dev)]
        )
    for l in range(len(cache.key_cache)):
        cache.key_cache[l] = cache.key_cache[l].index_select(2, sel)
        cache.value_cache[l] = cache.value_cache[l].index_select(2, sel)
    if hasattr(cache, "_seen_tokens"):
        cache._seen_tokens = sel.numel()
    return cache
