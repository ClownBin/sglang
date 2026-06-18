# Copyright 2025 XunhaoLai. All rights reserved.

from typing import List, Optional, Tuple

import torch

from ..common.index import topk_index_reduce


def _gather_kv_from_req_to_token(
    cache: torch.Tensor,
    req_to_token: torch.Tensor,
    req_idx: int,
    seq_len: int,
) -> torch.Tensor:
    """Gather a single sequence's contiguous cache slice from a paged pool.

    Mirrors vLLM-Ascend's ``_logical_cache_slice`` but adapted to sglang's
    ``req_to_token`` paging: row ``req_idx`` of ``req_to_token`` holds the
    physical slot index for every logical token position of the request, so
    a plain ``index_select`` over the first ``seq_len`` entries yields the
    contiguous logical view without any block-table arithmetic.

    Args:
        cache: paged KV buffer ``[max_slots, num_heads, head_dim]``.
        req_to_token: ``[max_reqs, max_kv_len]`` token->slot mapping.
        req_idx: row index in ``req_to_token`` for this request.
        seq_len: number of valid KV tokens for this request.

    Returns:
        ``[seq_len, num_heads, head_dim]`` contiguous tensor. Empty
        ``[0, num_heads, head_dim]`` when ``seq_len == 0``.
    """
    if seq_len == 0:
        return cache.new_empty((0, cache.shape[1], cache.shape[2]))
    slots = req_to_token[req_idx, :seq_len].to(torch.long)
    return cache.index_select(0, slots)


def _merge_sparse_blocks(
    topk_blocks: torch.Tensor,
    query_positions: torch.Tensor,
    num_blocks: int,
    block_size: int,
    init_blocks: int,
    local_blocks: int,
    total_blocks: int,
) -> torch.Tensor:
    """Merge top-k blocks with forced init/local blocks, deduplicated.

    Ported from vLLM-Ascend ``_merge_minimax_sparse_blocks``. The general
    branch (sort + dedup + scatter into a fixed-size buffer) already covers
    every combination of ``init_blocks`` / ``local_blocks``, including the
    ``init==0 & local==1`` case, so the dedicated fast path is dropped for
    simplicity.

    Args:
        topk_blocks: ``[q_len, num_idx_heads, topk]`` selected blocks (-1 pad).
        query_positions: ``[q_len]`` absolute position of each query token.
        num_blocks: total number of KV blocks for this sequence.
        block_size: tokens per block (== ``block_size_k``).
        init_blocks: number of leading blocks always selected.
        local_blocks: number of trailing blocks always selected.
        total_blocks: output capacity = topk + init_blocks + local_blocks.

    Returns:
        ``[q_len, num_idx_heads, total_blocks]`` merged block indices,
        ``-1`` for unused slots. All entries satisfy
        ``block * block_size <= query_position`` (causal) and
        ``0 <= block < num_blocks``.
    """
    if init_blocks <= 0 and local_blocks <= 0:
        return topk_blocks

    q_len = query_positions.shape[0]
    num_index_heads = topk_blocks.shape[1]

    forced_parts: list[torch.Tensor] = []
    if init_blocks > 0:
        forced_parts.append(
            torch.arange(
                init_blocks,
                device=topk_blocks.device,
                dtype=topk_blocks.dtype,
            )
            .view(1, 1, -1)
            .expand(q_len, num_index_heads, -1)
        )
    if local_blocks > 0:
        local_offsets = torch.arange(
            local_blocks,
            device=topk_blocks.device,
            dtype=query_positions.dtype,
        )
        block_ids = query_positions // block_size
        first_local_block = (block_ids - local_blocks + 1).clamp(min=0)
        forced_parts.append(
            (first_local_block[:, None] + local_offsets[None, :])
            .to(topk_blocks.dtype)
            .view(q_len, 1, -1)
            .expand(-1, num_index_heads, -1)
        )

    if not forced_parts:
        return topk_blocks

    forced = torch.cat(forced_parts, dim=-1)
    candidates = torch.cat([forced, topk_blocks], dim=-1)
    valid = (candidates >= 0) & (candidates < num_blocks)
    valid = valid & (candidates * block_size <= query_positions[:, None, None])

    invalid_value = torch.full_like(candidates, num_blocks)
    sorted_candidates = torch.sort(
        torch.where(valid, candidates, invalid_value), dim=-1
    ).values
    sorted_valid = sorted_candidates < num_blocks
    previous = torch.cat(
        [
            torch.full_like(sorted_candidates[..., :1], -1),
            sorted_candidates[:, :, :-1],
        ],
        dim=-1,
    )
    keep = sorted_valid & (sorted_candidates != previous)
    ranks = torch.cumsum(keep.to(torch.int32), dim=-1) - 1

    output = torch.full(
        (q_len, num_index_heads, total_blocks + 1),
        -1,
        dtype=topk_blocks.dtype,
        device=topk_blocks.device,
    )
    overflow_rank = torch.full_like(ranks, total_blocks)
    scatter_index = torch.where(
        keep & (ranks < total_blocks), ranks, overflow_rank
    ).long()
    scatter_src = torch.where(keep, sorted_candidates, -1)
    output.scatter_(2, scatter_index, scatter_src)
    return output[:, :, :total_blocks]


def _block_topk_select(
    index_query: torch.Tensor,
    index_key: torch.Tensor,
    query_positions: torch.Tensor,
    seq_len: int,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    score_type: str = "max",
) -> torch.Tensor:
    """Select top-k KV blocks per (query token, index head) and merge forced blocks.

    Ported from vLLM-Ascend ``_select_minimax_sparse_blocks``. Computes
    index-query vs index-key dot-product scores, masks invalid (future /
    padding) positions, reduces each block to a single score, takes top-k,
    then delegates to ``_merge_sparse_blocks`` to fold in init/local blocks.

    Unlike the vLLM original (max-only), ``score_type`` is honored to match
    sglang's GPU contract: ``"max"`` / ``"sum"`` / ``"mean"``.

    Args:
        index_query: ``[q_len, num_idx_heads, idx_head_dim]``.
        index_key: ``[seq_len, idx_head_dim]`` (index head dim == 1, squeezed).
        query_positions: ``[q_len]`` absolute position of each query token.
        seq_len: number of valid KV tokens.
        block_size: tokens per block (== ``block_size_k``).
        topk: number of top blocks to select.
        init_blocks / local_blocks: forced block counts (see ``_merge_sparse_blocks``).
        score_type: block score reduction (``"max"`` / ``"sum"`` / ``"mean"``).

    Returns:
        ``[q_len, num_idx_heads, topk + init_blocks + local_blocks]`` merged
        block indices (``-1`` padding), causal and range-valid.
    """
    num_blocks = (seq_len + block_size - 1) // block_size
    topk_blocks = min(topk, num_blocks)
    total_blocks = topk + init_blocks + local_blocks

    scores = torch.einsum("qhd,kd->qhk", index_query, index_key)
    scores = scores.float()

    padded_tokens = num_blocks * block_size
    if padded_tokens != seq_len:
        pad_len = padded_tokens - seq_len
        scores = torch.nn.functional.pad(scores, (0, pad_len), value=-1.0e30)

    key_positions = torch.arange(
        padded_tokens,
        device=index_query.device,
        dtype=query_positions.dtype,
    )
    valid = (key_positions[None, :] < seq_len) & (
        key_positions[None, :] <= query_positions[:, None]
    )
    scores = scores.masked_fill(~valid[:, None, :], -1.0e30)

    block_scores = scores.view(
        index_query.shape[0],
        index_query.shape[1],
        num_blocks,
        block_size,
    )
    if score_type == "max":
        block_scores = block_scores.amax(dim=-1)
    elif score_type == "sum":
        block_scores = block_scores.sum(dim=-1)
    elif score_type in ("mean", "avg"):
        block_scores = block_scores.mean(dim=-1)
    else:
        raise ValueError(f"unsupported score_type={score_type!r}")

    blocks = torch.topk(block_scores, k=topk_blocks, dim=-1).indices.to(torch.int32)
    if topk_blocks < topk:
        blocks = torch.nn.functional.pad(blocks, (0, topk - topk_blocks), value=-1)

    return _merge_sparse_blocks(
        blocks,
        query_positions,
        num_blocks,
        block_size,
        init_blocks,
        local_blocks,
        total_blocks,
    )


def _expand_blocks_to_tokens(
    block_indices: torch.Tensor,
    seq_len: int,
    block_size: int,
) -> torch.Tensor:
    """Expand block indices into concrete token indices.

    Ported from vLLM-Ascend ``_expand_sparse_blocks_to_tokens``. For each
    block id ``b`` (>= 0) emits ``block_size`` token ids
    ``b * block_size + [0, block_size)``; invalid blocks (``< 0``) and
    tokens beyond ``seq_len`` are mapped to ``-1``.

    Args:
        block_indices: ``[q_len, num_idx_heads, num_blocks]`` block ids (-1 pad).
        seq_len: valid KV length; tokens >= seq_len are masked to -1.
        block_size: tokens per block (== ``block_size_k``).

    Returns:
        ``[q_len, num_idx_heads, num_blocks * block_size]`` token indices,
        ``-1`` for invalid slots.
    """
    offsets = torch.arange(
        block_size,
        device=block_indices.device,
        dtype=block_indices.dtype,
    )
    token_indices = block_indices[..., None] * block_size + offsets
    valid_tokens = (block_indices[..., None] >= 0) & (token_indices < seq_len)
    token_indices = token_indices.flatten(start_dim=2)
    return torch.where(
        valid_tokens.flatten(start_dim=2),
        token_indices,
        torch.full_like(token_indices, -1),
    )


def _sparse_attention_one_seq(
    q_seq: torch.Tensor,
    k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    idx_q_seq: torch.Tensor,
    idx_k_seq: torch.Tensor,
    idx_v_seq: Optional[torch.Tensor],
    query_positions: torch.Tensor,
    seq_len: int,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    score_type: str,
    sm_scale: float,
    idx_sm_scale: float,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Run MiniMax-M3 sparse prefill attention for a single sequence.

    Ported from vLLM-Ascend ``_sparse_attention_from_indices`` (prefill
    branch only). Short sequences (``seq_len <= topk * block_size``) use
    full causal attention; long sequences use block-sparse attention where
    the index heads select top-k blocks and the main heads attend only to
    the union of selected tokens (GQA-shared).

    Index-head reduction follows sglang's GPU convention: when
    ``num_idx_heads > num_kv_heads`` the per-index-head token indices are
    union-reduced via ``topk_index_reduce`` so each kv head group shares
    one token set, instead of vLLM's explicit ``sparse_head_ranges`` loop.

    Args:
        q_seq: ``[q_len, num_q_heads, head_dim]``.
        k_seq: ``[seq_len, num_kv_heads, head_dim]``.
        v_seq: ``[seq_len, num_kv_heads, head_dim]``.
        idx_q_seq: ``[q_len, num_idx_heads, idx_head_dim]``.
        idx_k_seq: ``[seq_len, idx_head_dim]`` (index head dim == 1).
        idx_v_seq: ``[seq_len, idx_head_dim]`` or None (disable_index_value).
        query_positions: ``[q_len]`` absolute position of each query token.
        seq_len: number of valid KV tokens.
        block_size / topk / init_blocks / local_blocks / score_type: sparse cfg.
        sm_scale: main-head softmax scale.
        idx_sm_scale: index-head softmax scale (used for index-head output).

    Returns:
        ``(idx_o, o)``: ``idx_o`` is None when ``idx_v_seq`` is None;
        ``o`` is ``[q_len, num_q_heads, head_dim]``.
    """
    q_len = q_seq.shape[0]
    num_q_heads = q_seq.shape[1]
    num_kv_heads = k_seq.shape[1]
    num_idx_heads = idx_q_seq.shape[1]
    group_size = max(1, num_q_heads // num_kv_heads)

    # Index-head output (full attention over index head), when enabled.
    idx_o = None
    if idx_v_seq is not None:
        idx_scores = torch.einsum("qhd,kd->qhk", idx_q_seq, idx_k_seq)
        idx_scores = idx_scores.float() * idx_sm_scale
        key_positions = torch.arange(
            seq_len, device=q_seq.device, dtype=query_positions.dtype
        )
        valid = key_positions[None, :] <= query_positions[:, None]
        idx_scores = idx_scores.masked_fill(~valid[:, None, :], -1.0e30)
        idx_probs = torch.softmax(idx_scores, dim=-1)
        idx_o = torch.einsum("qhk,kd->qhd", idx_probs.to(idx_v_seq.dtype), idx_v_seq)

    if seq_len == 0:
        o = q_seq.new_zeros(q_seq.shape)
        return idx_o, o

    # Short sequence: full causal attention on the main heads.
    sparse_count = topk * block_size
    if seq_len <= sparse_count:
        key_positions = torch.arange(
            seq_len, device=q_seq.device, dtype=query_positions.dtype
        )
        valid = key_positions[None, :] <= query_positions[:, None]
        scores = torch.einsum("qhd,khd->qhk", q_seq, k_seq)
        scores = scores.float() * sm_scale
        scores = scores.masked_fill(~valid[:, None, :], -1.0e30)
        probs = torch.softmax(scores, dim=-1)
        o = torch.einsum("qhk,khd->qhd", probs.to(v_seq.dtype), v_seq)
        return idx_o, o

    # Long sequence: block-sparse attention.
    sparse_blocks = _block_topk_select(
        idx_q_seq,
        idx_k_seq,
        query_positions,
        seq_len,
        block_size,
        topk,
        init_blocks,
        local_blocks,
        score_type,
    )
    token_indices = _expand_blocks_to_tokens(sparse_blocks, seq_len, block_size)
    # [q_len, num_idx_heads, num_tokens]

    # Reduce index-head token indices to kv-head groups (GQA-shared sparse).
    idx_group_size = num_idx_heads // num_kv_heads
    if idx_group_size > 1:
        token_indices = topk_index_reduce(
            token_indices.view(q_len, num_kv_heads, idx_group_size, -1), dim=2
        )
    # Now token_indices: [q_len, num_kv_heads, num_tokens]

    o = q_seq.new_zeros(q_seq.shape)
    safe_limit = max(seq_len - 1, 0)
    for kvh in range(num_kv_heads):
        qh_start = kvh * group_size
        qh_end = qh_start + group_size
        q_group = q_seq[:, qh_start:qh_end, :]
        idx = token_indices[:, kvh, :]
        valid = (idx >= 0) & (idx < seq_len)
        valid = valid & (idx <= query_positions[:, None])
        safe_idx = idx.clamp(0, safe_limit).long()
        k_sel = k_seq.index_select(0, safe_idx.reshape(-1)).view(
            q_len, -1, num_kv_heads, q_seq.shape[2]
        )[:, :, kvh, :]
        v_sel = v_seq.index_select(0, safe_idx.reshape(-1)).view(
            q_len, -1, num_kv_heads, v_seq.shape[2]
        )[:, :, kvh, :]
        scores = torch.einsum("qhd,qkd->qhk", q_group, k_sel)
        scores = scores.float() * sm_scale
        scores = scores.masked_fill(~valid[:, None, :], -1.0e30)
        probs = torch.softmax(scores, dim=-1)
        o[:, qh_start:qh_end, :] = torch.einsum(
            "qhk,qkd->qhd", probs.to(v_sel.dtype), v_sel
        )
    return idx_o, o


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
