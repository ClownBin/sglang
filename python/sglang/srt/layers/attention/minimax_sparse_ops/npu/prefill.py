# Copyright 2025 XunhaoLai. All rights reserved.

from typing import List, Optional, Tuple

import torch

from .common import (
    _gather_kv_from_req_to_token,
    _sparse_attention_one_seq,
)


def npu_minimax_sparse_prefill(
    q: torch.Tensor,  # [total_extend_tokens, num_q_heads, qk_head_dim]
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged main)
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged main)
    sink: Optional[torch.Tensor],  # [num_q_heads, qk_head_dim]
    idx_q: torch.Tensor,  # [total_extend_tokens, num_idx_heads, idx_head_dim]
    idx_k_cache: torch.Tensor,  # [max_slots, 1, idx_head_dim] (paged index)
    idx_v_cache: Optional[
        torch.Tensor
    ],  # [max_slots, 1, idx_head_dim] (paged index); None when disable_index_value
    idx_sink: Optional[torch.Tensor],  # [num_idx_heads, idx_head_dim]
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    slot_ids: torch.Tensor,  # [batch_size, ]
    cu_seqlens: torch.Tensor,  # [batch_size + 1, ] (Q-side cumulative)
    seq_lens: torch.Tensor,  # [batch_size, ] total K length (prefix + chunk)
    prefix_lens: torch.Tensor,  # [batch_size, ]
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size_q: int,
    block_size_k: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
    use_msa: bool = False,
    cu_seqblocks_q: Optional[torch.Tensor] = None,
    max_seqblock_q: Optional[int] = None,
    all_seqblock_q: Optional[int] = None,
    seqlens_cpu: Optional[List[int]] = None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Run MiniMax-M3 sparse prefill on NPU.

    NPU counterpart of ``minimax_sparse_prefill``. It reuses the same
    sglang paging primitives (``req_to_token`` + paged KV cache) and the
    same argument contract, but implements the sparse selection and the
    final attention in pure PyTorch (no Triton/flash kernels), mirroring
    the vLLM-Ascend block-table based implementation.

    ``sink`` / ``idx_sink`` are accepted for interface parity but are not
    yet consumed by this NPU path (attention sink support is TODO).
    ``use_msa`` / ``cu_seqblocks_q`` / ``max_seqblock_q`` / ``all_seqblock_q``
    / ``seqlens_cpu`` are likewise accepted for parity and currently unused.
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    if idx_sm_scale is None:
        idx_sm_scale = idx_q.shape[-1] ** -0.5

    total_tokens = q.shape[0]
    o = q.new_zeros(q.shape)
    idx_o = None if disable_index_value else idx_q.new_zeros(idx_q.shape)

    batch_size = seq_lens.shape[0]
    if batch_size == 0:
        return idx_o, o

    # Flatten paged caches to 3D [total_slots, num_heads, head_dim].
    # sglang's memory pool may use a 4D layout (e.g. page dimension);
    # index_select over the flattened first dimension works with the
    # flat slot indices from req_to_token.
    def _flatten(cache: torch.Tensor) -> torch.Tensor:
        if cache.dim() <= 3:
            return cache
        return cache.reshape(-1, cache.shape[-2], cache.shape[-1])

    k_cache = _flatten(k_cache)
    v_cache = _flatten(v_cache)
    idx_k_cache = _flatten(idx_k_cache)
    if idx_v_cache is not None:
        idx_v_cache = _flatten(idx_v_cache)

    # cu_seqlens is int32 on device; read offsets on host to avoid per-seq
    # syncs inside the loop (mirrors the GPU path's seqlens_cpu rationale).
    cu = cu_seqlens.tolist()
    seq_lens_list = seq_lens.tolist()
    prefix_lens_list = prefix_lens.tolist()
    slot_ids_list = slot_ids.tolist()

    for batch_id in range(batch_size):
        q_start = cu[batch_id]
        q_end = cu[batch_id + 1]
        if q_end <= q_start:
            continue

        seq_len = int(seq_lens_list[batch_id])
        prefix_len = int(prefix_lens_list[batch_id])
        req_idx = int(slot_ids_list[batch_id])

        # Gather this sequence's contiguous K/V/index_K/index_V from the
        # paged pools via req_to_token.
        k_seq = _gather_kv_from_req_to_token(k_cache, req_to_token, req_idx, seq_len)
        v_seq = _gather_kv_from_req_to_token(v_cache, req_to_token, req_idx, seq_len)
        idx_k_seq = _gather_kv_from_req_to_token(
            idx_k_cache, req_to_token, req_idx, seq_len
        )[:, 0, :]
        if idx_v_cache is not None:
            idx_v_seq = _gather_kv_from_req_to_token(
                idx_v_cache, req_to_token, req_idx, seq_len
            )[:, 0, :]
        else:
            idx_v_seq = None

        # Query positions: prefix_len .. prefix_len + q_len - 1 (absolute).
        q_len = q_end - q_start
        query_positions = torch.arange(
            prefix_len,
            prefix_len + q_len,
            device=q.device,
            dtype=torch.long,
        )

        seq_idx_o, seq_o = _sparse_attention_one_seq(
            q[q_start:q_end],
            k_seq,
            v_seq,
            idx_q[q_start:q_end],
            idx_k_seq,
            idx_v_seq,
            query_positions,
            seq_len,
            block_size_k,
            topk,
            init_blocks,
            local_blocks,
            score_type,
            sm_scale,
            idx_sm_scale,
        )
        o[q_start:q_end] = seq_o
        if idx_o is not None and seq_idx_o is not None:
            idx_o[q_start:q_end] = seq_idx_o

    return idx_o, o
