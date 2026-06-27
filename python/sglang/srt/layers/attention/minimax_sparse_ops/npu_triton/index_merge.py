from __future__ import annotations

import os

import torch

from sglang.srt.layers.attention.minimax_sparse_ops.common.index import (
    topk_index_reduce,
)


def _env_flag_enabled(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _merge_forced_blocks_torch(
    topk_blocks_tensor: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    topk_blocks: int,
    init_blocks: int,
    local_blocks: int,
    block_size: int,
    num_blocks: int,
) -> torch.Tensor:
    total = topk_blocks + init_blocks + local_blocks
    if init_blocks <= 0 and local_blocks <= 0:
        return topk_blocks_tensor[..., :total]

    q_len = query_positions.shape[0]
    num_kv_heads = topk_blocks_tensor.shape[1]
    qcol = query_positions[:, None, None]

    if init_blocks == 0 and local_blocks == 1:
        local = (query_positions // block_size).clamp(
            min=0, max=max(num_blocks - 1, 0)
        )
        local = local.to(topk_blocks_tensor.dtype).view(q_len, 1, 1).expand(
            -1, num_kv_heads, -1
        )
        valid_topk = (topk_blocks_tensor >= 0) & (topk_blocks_tensor < num_blocks)
        valid_topk = valid_topk & (topk_blocks_tensor * block_size <= qcol)
        local_duplicate = ((topk_blocks_tensor == local) & valid_topk).any(
            dim=-1, keepdim=True
        )
        valid_local = (local >= 0) & (local < num_blocks)
        valid_local = (
            valid_local & (local * block_size <= qcol) & ~local_duplicate
        )
        return torch.cat(
            [
                torch.where(
                    valid_topk,
                    topk_blocks_tensor,
                    torch.full_like(topk_blocks_tensor, -1),
                ),
                torch.where(valid_local, local, torch.full_like(local, -1)),
            ],
            dim=-1,
        )

    forced_parts = []
    if init_blocks > 0:
        forced_parts.append(
            torch.arange(
                init_blocks,
                device=topk_blocks_tensor.device,
                dtype=topk_blocks_tensor.dtype,
            )
            .view(1, 1, -1)
            .expand(q_len, num_kv_heads, -1)
        )
    if local_blocks > 0:
        offsets = torch.arange(
            local_blocks,
            device=topk_blocks_tensor.device,
            dtype=query_positions.dtype,
        )
        block_ids = query_positions // block_size
        first = (block_ids - local_blocks + 1).clamp(min=0)
        forced_parts.append(
            (first[:, None] + offsets[None, :])
            .to(topk_blocks_tensor.dtype)
            .view(q_len, 1, -1)
            .expand(-1, num_kv_heads, -1)
        )

    forced = torch.cat(forced_parts, dim=-1)
    candidates = torch.cat([forced, topk_blocks_tensor], dim=-1)
    valid = (candidates >= 0) & (candidates < num_blocks)
    valid = valid & (candidates * block_size <= qcol)
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
        (q_len, num_kv_heads, total + 1),
        -1,
        dtype=topk_blocks_tensor.dtype,
        device=topk_blocks_tensor.device,
    )
    overflow_rank = torch.full_like(ranks, total)
    scatter_index = torch.where(keep & (ranks < total), ranks, overflow_rank).long()
    scatter_src = torch.where(keep, sorted_candidates, -1)
    output.scatter_(2, scatter_index, scatter_src)
    return output[:, :, :total]


def _merge_topk_index_for_sparse_decode_torch(
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
    num_idx_heads = topk_idx.shape[0]
    if num_idx_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_idx_heads={num_idx_heads} must be divisible by "
            f"num_kv_heads={num_kv_heads}"
        )

    if num_idx_heads > num_kv_heads:
        idx_group_size = num_idx_heads // num_kv_heads
        topk_idx = topk_index_reduce(
            topk_idx.view(num_kv_heads, idx_group_size, -1, topk_blocks),
            dim=1,
        )
    topk_2d = topk_idx.permute(1, 0, 2).contiguous()
    merged = _merge_forced_blocks_torch(
        topk_2d,
        query_positions,
        topk_blocks=topk_blocks,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        block_size=block_size,
        num_blocks=num_blocks,
    )
    return merged.permute(1, 0, 2).contiguous()


def _try_merge_topk_index_for_sparse_decode_triton(
    topk_idx: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    num_kv_heads: int,
    topk_blocks: int,
    init_blocks: int,
    local_blocks: int,
    block_size: int,
    num_blocks: int,
) -> torch.Tensor | None:
    if (
        topk_idx.device.type != "npu"
        or not _env_flag_enabled("SGLANG_MINIMAX_NPU_FUSED_INDEX_MERGE", True)
    ):
        return None
    try:
        from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton.index_merge_triton import (
            merge_topk_index_for_sparse_decode_triton,
        )
    except Exception:
        return None

    return merge_topk_index_for_sparse_decode_triton(
        topk_idx,
        query_positions,
        num_kv_heads=num_kv_heads,
        topk_blocks=topk_blocks,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        block_size=block_size,
        num_blocks=num_blocks,
    )


def merge_topk_index_for_sparse_decode(
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
    """Reduce index heads and append MiniMax forced blocks for sparse decode.

    Input is the pure indexer top-k result ``[num_idx_heads, batch, topk]``.
    Output is the main sparse-attention block list with duplicates removed.
    The common ``init=0/local=1`` fast path keeps the existing PyTorch width
    ``idx_group_size * topk + 1`` after index-head union; other paths truncate to
    ``topk + init + local``.
    """
    triton_result = _try_merge_topk_index_for_sparse_decode_triton(
        topk_idx,
        query_positions,
        num_kv_heads=num_kv_heads,
        topk_blocks=topk_blocks,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        block_size=block_size,
        num_blocks=num_blocks,
    )
    if triton_result is not None:
        return triton_result
    return _merge_topk_index_for_sparse_decode_torch(
        topk_idx,
        query_positions,
        num_kv_heads=num_kv_heads,
        topk_blocks=topk_blocks,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        block_size=block_size,
        num_blocks=num_blocks,
    )
