from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _merge_topk_index_kernel(
    topk_ptr,
    query_pos_ptr,
    out_ptr,
    stride_th,
    stride_tb,
    stride_tt,
    stride_oh,
    stride_ob,
    stride_ot,
    num_blocks,
    IDX_GROUP_SIZE: tl.constexpr,
    TOPK_BLOCKS: tl.constexpr,
    INIT_BLOCKS: tl.constexpr,
    LOCAL_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_kv = tl.program_id(1)
    offs = tl.arange(0, BLOCK_C)

    query_pos = tl.load(query_pos_ptr + pid_b).to(tl.int32)
    candidates = tl.full((BLOCK_C,), -1, tl.int32)

    init_mask = offs < INIT_BLOCKS
    candidates = tl.where(init_mask, offs.to(tl.int32), candidates)

    local_pos = offs - INIT_BLOCKS
    block_id = query_pos // BLOCK_SIZE
    local_first = tl.maximum(block_id - LOCAL_BLOCKS + 1, 0)
    local_mask = (local_pos >= 0) & (local_pos < LOCAL_BLOCKS)
    local_val = local_first + local_pos
    candidates = tl.where(local_mask, local_val.to(tl.int32), candidates)

    flat = offs - (INIT_BLOCKS + LOCAL_BLOCKS)
    topk_mask = (flat >= 0) & (flat < IDX_GROUP_SIZE * TOPK_BLOCKS)
    idx_group = flat // TOPK_BLOCKS
    topk_col = flat - idx_group * TOPK_BLOCKS
    idx_head = pid_kv * IDX_GROUP_SIZE + idx_group
    topk_vals = tl.load(
        topk_ptr + idx_head * stride_th + pid_b * stride_tb + topk_col * stride_tt,
        mask=topk_mask,
        other=-1,
    ).to(tl.int32)
    candidates = tl.where(topk_mask, topk_vals, candidates)

    valid = (candidates >= 0) & (candidates < num_blocks)
    valid = valid & (candidates * BLOCK_SIZE <= query_pos)
    prev = tl.full((), -1, tl.int32)

    for out_i in tl.static_range(0, TOTAL):
        next_vals = tl.where(valid & (candidates > prev), candidates, num_blocks)
        next_val = tl.min(next_vals, axis=0)
        out_val = tl.where(next_val < num_blocks, next_val, -1)
        tl.store(out_ptr + pid_kv * stride_oh + pid_b * stride_ob + out_i * stride_ot, out_val)
        prev = next_val


def _next_power_of_2(x: int) -> int:
    return 1 << (max(1, int(x)) - 1).bit_length()


def merge_topk_index_for_sparse_decode_triton(
    topk_idx: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    num_kv_heads: int,
    topk_blocks: int,
    init_blocks: int,
    local_blocks: int,
    block_size: int,
    num_blocks: int,
) -> torch.Tensor:
    num_idx_heads, batch_size, actual_topk = topk_idx.shape
    if actual_topk != topk_blocks:
        raise ValueError(f"topk_idx.shape[-1]={actual_topk} != topk_blocks={topk_blocks}")
    if num_idx_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_idx_heads={num_idx_heads} must be divisible by "
            f"num_kv_heads={num_kv_heads}"
        )
    if topk_idx.dtype != torch.int32:
        topk_idx = topk_idx.to(torch.int32)
    if not topk_idx.is_contiguous():
        topk_idx = topk_idx.contiguous()
    if not query_positions.is_contiguous():
        query_positions = query_positions.contiguous()

    idx_group_size = num_idx_heads // num_kv_heads
    num_candidates = init_blocks + local_blocks + idx_group_size * topk_blocks
    if init_blocks == 0 and local_blocks == 1:
        total = num_candidates
    else:
        total = topk_blocks + init_blocks + local_blocks
    out = torch.empty(
        (num_kv_heads, batch_size, total),
        dtype=torch.int32,
        device=topk_idx.device,
    )
    block_c = _next_power_of_2(num_candidates)
    grid = (batch_size, num_kv_heads)
    _merge_topk_index_kernel[grid](
        topk_idx,
        query_positions,
        out,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        int(num_blocks),
        IDX_GROUP_SIZE=idx_group_size,
        TOPK_BLOCKS=int(topk_blocks),
        INIT_BLOCKS=int(init_blocks),
        LOCAL_BLOCKS=int(local_blocks),
        BLOCK_SIZE=int(block_size),
        TOTAL=int(total),
        BLOCK_C=int(block_c),
        num_warps=1,
        num_stages=1,
    )
    return out
