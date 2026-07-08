"""MiniMax-M3 dense TARGET_VERIFY paged attention for Ascend NPU.

This is the dense counterpart to the MiniMax sparse NPU verify path.  It scans
contiguous logical KV blocks directly, split into chunks, so dense attention does
not masquerade as a sparse all-block selection.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton.topk_sparse_decode import (
    _MERGE_NS,
    _MERGE_NW,
    _floor_power_of_2,
    _get_vectorcore_num_safe,
    _merge_topk_attn_out_bnsd_kernel,
)

_DENSE_VERIFY_NW = 4
_DENSE_VERIFY_NS = 2


def _choose_num_kv_chunks(
    batch_size: int,
    num_kv_heads: int,
    max_blocks: int,
    max_num_kv_chunks: int = 16,
) -> int:
    """Choose split-K chunks for dense verify using an Ascend-conservative cap."""
    if max_blocks <= 1:
        return 1

    max_num_kv_chunks = max(1, int(max_num_kv_chunks))
    max_num_kv_chunks = _floor_power_of_2(max_num_kv_chunks)

    num_vectorcore = _get_vectorcore_num_safe()
    target_grid = num_vectorcore * 4
    denom = max(1, int(batch_size) * int(num_kv_heads))
    target = max(1, target_grid // denom)
    target = min(max_blocks, max_num_kv_chunks, target)
    return _floor_power_of_2(target)


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_N": lambda args: triton.next_power_of_2(args["block_size"]),
    }
)
@triton.jit
def _dense_verify_paged_attention_kernel(
    q_ptr,  # [B, QH, D]
    k_cache_ptr,  # [NBLOCKS, BLOCK, KVH, D]
    v_cache_ptr,  # [NBLOCKS, BLOCK, KVH, D]
    block_table_ptr,  # [B, max_blocks]
    per_query_seq_lens,  # [B]
    o_ptr,  # [C, B, QH, D]
    lse_ptr,  # [C, B, QH]
    # shape
    batch_size,
    gqa_group_size,
    head_dim,
    max_blocks,
    max_kv_len,
    # block/scaling
    block_size: tl.constexpr,
    sm_scale,
    # strides
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_k_block,
    stride_k_offset,
    stride_k_h,
    stride_k_d,
    stride_v_block,
    stride_v_offset,
    stride_v_h,
    stride_v_d,
    stride_bt_b,
    stride_bt_n,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    # meta
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    NUM_KV_CHUNKS: tl.constexpr,
    CHUNK_SIZE_BLOCKS: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_N >= block_size)

    pid_bc = tl.program_id(0)
    pid_kh = tl.program_id(1)

    pid_b = pid_bc % batch_size
    pid_c = pid_bc // batch_size
    pid_h = pid_kh * gqa_group_size

    seq_len = tl.minimum(tl.load(per_query_seq_lens + pid_b).to(tl.int32), max_kv_len)
    num_blocks = tl.cdiv(seq_len, block_size)
    chunk_start_block = pid_c * CHUNK_SIZE_BLOCKS

    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_n = tl.arange(0, BLOCK_SIZE_N)
    dim_mask = off_d < head_dim

    q_offsets = (
        pid_b * stride_q_b
        + (pid_h + off_h[:, None]) * stride_q_h
        + off_d[None, :] * stride_q_d
    )
    q = tl.load(
        q_ptr + q_offsets,
        mask=(off_h[:, None] < gqa_group_size) & (off_d[None, :] < head_dim),
        other=0.0,
    )

    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    acc_o = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_D), 0.0, dtype=tl.float32)

    for step in tl.range(CHUNK_SIZE_BLOCKS):
        logical_block = chunk_start_block + step
        valid_block = logical_block < num_blocks
        safe_logical_block = tl.minimum(logical_block, max_blocks - 1)

        physical_block = tl.load(
            block_table_ptr + pid_b * stride_bt_b + safe_logical_block * stride_bt_n,
            mask=valid_block & (logical_block < max_blocks),
            other=0,
        ).to(tl.int64)

        pos = logical_block * block_size + off_n
        pos_mask = valid_block & (pos < seq_len)

        k_offsets = (
            physical_block * stride_k_block
            + off_n[None, :] * stride_k_offset
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d
        )
        k = tl.load(
            k_cache_ptr + k_offsets,
            mask=dim_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )

        v_offsets = (
            physical_block * stride_v_block
            + off_n[:, None] * stride_v_offset
            + pid_kh * stride_v_h
            + off_d[None, :] * stride_v_d
        )
        v = tl.load(
            v_cache_ptr + v_offsets,
            mask=pos_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )

        qk = tl.dot(q, k) * sm_scale
        qk = tl.where(pos_mask[None, :], qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.where(
            valid_block,
            tl.exp(qk - m_ij[:, None]),
            tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_N), dtype=tl.float32),
        )
        l_ij = tl.sum(p, axis=1)

        acc_o_scale = tl.where(
            valid_block,
            tl.exp(m_i - m_ij),
            tl.full((BLOCK_SIZE_H,), 1.0, dtype=tl.float32),
        )
        acc_o_new = acc_o * acc_o_scale[:, None] + tl.dot(p.to(v.dtype), v)
        lse_i_new = m_ij + tl.log(tl.exp(lse_i - m_ij) + l_ij)

        acc_o = tl.where(valid_block, acc_o_new, acc_o)
        m_i = tl.where(valid_block, m_ij, m_i)
        lse_i = tl.where(valid_block, lse_i_new, lse_i)

    scale = tl.where(
        lse_i > float("-inf"),
        tl.exp(m_i - lse_i),
        tl.zeros_like(lse_i),
    )
    acc_o = acc_o * scale[:, None]

    o_offsets = (
        pid_c * stride_o_c
        + pid_b * stride_o_b
        + (pid_h + off_h[:, None]) * stride_o_h
        + off_d[None, :] * stride_o_d
    )
    tl.store(
        o_ptr + o_offsets,
        acc_o.to(o_ptr.dtype.element_ty),
        mask=(off_h[:, None] < gqa_group_size) & (off_d[None, :] < head_dim),
    )

    l_offsets = pid_c * stride_l_c + pid_b * stride_l_b + (pid_h + off_h) * stride_l_h
    tl.store(
        lse_ptr + l_offsets,
        lse_i.to(lse_ptr.dtype.element_ty),
        mask=off_h < gqa_group_size,
    )


@torch.no_grad()
def dense_verify_paged_attention(
    q: torch.Tensor,  # [batch_size, num_q_heads, head_dim]
    k_cache_bnsd: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache_bnsd: torch.Tensor,  # same shape
    block_table: torch.Tensor,  # [batch_size, max_blocks]
    per_query_seq_lens: torch.Tensor,  # [batch_size]
    block_size: int,
    sm_scale: Optional[float] = None,
    num_kv_chunks: Optional[int] = None,
    max_num_kv_chunks: int = 16,
) -> torch.Tensor:
    """Dense paged causal attention for flattened MiniMax-M3 verify queries."""
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert k_cache_bnsd.dtype == q.dtype
    assert v_cache_bnsd.dtype == q.dtype
    assert k_cache_bnsd.shape == v_cache_bnsd.shape

    batch_size, num_q_heads, head_dim = q.shape
    _, block_size_from_cache, num_kv_heads, cache_head_dim = k_cache_bnsd.shape

    assert block_size_from_cache == block_size
    assert cache_head_dim == head_dim
    assert num_q_heads % num_kv_heads == 0
    assert block_table.shape[0] == batch_size
    assert per_query_seq_lens.shape[0] == batch_size

    gqa_group_size = num_q_heads // num_kv_heads
    max_blocks = block_table.shape[1]
    max_kv_len = max_blocks * block_size

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    if num_kv_chunks is None:
        num_kv_chunks = _choose_num_kv_chunks(
            batch_size,
            num_kv_heads,
            max_blocks,
            max_num_kv_chunks=max_num_kv_chunks,
        )
    else:
        num_kv_chunks = int(num_kv_chunks)

    assert num_kv_chunks >= 1
    assert (num_kv_chunks & (num_kv_chunks - 1)) == 0
    assert num_kv_chunks <= max(1, max_blocks)

    chunk_size_blocks = (max_blocks + num_kv_chunks - 1) // num_kv_chunks
    chunk_size_blocks = max(2, chunk_size_blocks)

    single_chunk = num_kv_chunks == 1
    out = torch.empty_like(q)
    if single_chunk:
        o_partial = out.view(1, batch_size, num_q_heads, head_dim)
    else:
        o_partial = torch.empty(
            (num_kv_chunks, batch_size, num_q_heads, head_dim),
            dtype=q.dtype,
            device=q.device,
        )
    lse_partial = torch.empty(
        (num_kv_chunks, batch_size, num_q_heads),
        dtype=torch.float32,
        device=q.device,
    )

    grid = (batch_size * num_kv_chunks, num_kv_heads)
    _dense_verify_paged_attention_kernel[grid](
        q,
        k_cache_bnsd,
        v_cache_bnsd,
        block_table,
        per_query_seq_lens,
        o_partial,
        lse_partial,
        batch_size,
        gqa_group_size,
        head_dim,
        max_blocks,
        max_kv_len,
        block_size,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache_bnsd.stride(0),
        k_cache_bnsd.stride(1),
        k_cache_bnsd.stride(2),
        k_cache_bnsd.stride(3),
        v_cache_bnsd.stride(0),
        v_cache_bnsd.stride(1),
        v_cache_bnsd.stride(2),
        v_cache_bnsd.stride(3),
        block_table.stride(0),
        block_table.stride(1),
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        BLOCK_SIZE_N=block_size,
        NUM_KV_CHUNKS=num_kv_chunks,
        CHUNK_SIZE_BLOCKS=chunk_size_blocks,
        num_warps=_DENSE_VERIFY_NW,
        num_stages=_DENSE_VERIFY_NS,
    )

    if not single_chunk:
        merge_grid = (batch_size, num_q_heads)
        _merge_topk_attn_out_bnsd_kernel[merge_grid](
            o_partial,
            lse_partial,
            out,
            head_dim,
            o_partial.stride(0),
            o_partial.stride(1),
            o_partial.stride(2),
            o_partial.stride(3),
            lse_partial.stride(0),
            lse_partial.stride(1),
            lse_partial.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            NUM_TOPK_CHUNKS=num_kv_chunks,
            num_warps=_MERGE_NW,
            num_stages=_MERGE_NS,
        )

    return out
