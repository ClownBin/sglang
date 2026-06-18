from typing import Callable, Optional, Tuple

import torch

from .common import (
    _gather_kv_from_req_to_token,
    _sparse_attention_one_seq,
)


def _sparse_decode_one_seq(
    q_seq: torch.Tensor,
    k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    idx_q_seq: torch.Tensor,
    idx_k_seq: torch.Tensor,
    idx_v_seq: Optional[torch.Tensor],
    query_position: int,
    seq_len: int,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    score_type: str,
    sm_scale: float,
    idx_sm_scale: float,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Thin decode wrapper around ``_sparse_attention_one_seq``.

    The only difference from the canonical implementation is that decode
    receives a scalar ``query_position`` instead of a ``[q_len]`` tensor.
    Convert it to a 1-element tensor and delegate.
    """
    query_positions = torch.tensor(
        [query_position], device=q_seq.device, dtype=torch.long
    )
    return _sparse_attention_one_seq(
        q_seq,
        k_seq,
        v_seq,
        idx_q_seq,
        idx_k_seq,
        idx_v_seq,
        query_positions,
        seq_len,
        block_size,
        topk,
        init_blocks,
        local_blocks,
        score_type,
        sm_scale,
        idx_sm_scale,
    )


def npu_minimax_sparse_decode(
    q: torch.Tensor,
    sink: Optional[torch.Tensor],
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx_q: torch.Tensor,
    idx_sink: Optional[torch.Tensor],
    idx_k_cache: torch.Tensor,
    idx_v_cache: Optional[torch.Tensor],
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seqlen: int,
    block_size_q: int,
    block_size_k: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
    dense_main_attn_fn: Optional[Callable] = None,
    page_size: int = 1,
    use_msa: bool = False,
    msa_kv_indices: Optional[torch.Tensor] = None,
    msa_plan=None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Run MiniMax-M3 sparse decode on NPU.

    NPU counterpart of ``minimax_sparse_decode``. It reuses the same sglang
    paging primitives (``req_to_token`` + paged KV cache) and the same argument
    contract, but implements the sparse selection and the final attention in
    pure PyTorch (no Triton/flash kernels), mirroring the vLLM-Ascend
    block-table based decode path.

    ``sink`` / ``idx_sink`` are accepted for interface parity but are not yet
    consumed by this NPU path (attention sink support is TODO).
    ``dense_main_attn_fn`` / ``use_msa`` / ``msa_kv_indices`` / ``msa_plan``
    are likewise accepted for parity and currently unused.

    Args:
        q: ``[batch_size, num_q_heads, qk_head_dim]``.
        sink: unused, for interface parity.
        k_cache: ``[max_slots, num_kv_heads, head_dim]`` (paged main).
        v_cache: ``[max_slots, num_kv_heads, head_dim]`` (paged main).
        idx_q: ``[batch_size, num_idx_heads, idx_head_dim]``.
        idx_sink: unused, for interface parity.
        idx_k_cache: ``[max_slots, 1, idx_head_dim]`` (paged index).
        idx_v_cache: ``[max_slots, 1, idx_head_dim]`` or None.
        req_to_token: ``[max_reqs, max_kv_len]`` mapping.
        slot_ids: ``[batch_size]`` request indices.
        seq_lens: ``[batch_size]`` sequence lengths.
        max_seqlen: max of seq_lens (passed from caller).
        block_size_q: unused (always 1 for decode).
        block_size_k: tokens per KV block.
        topk / init_blocks / local_blocks: sparse configuration.
        sm_scale / idx_sm_scale: softmax scales.
        score_type: ``"max"`` / ``"sum"`` / ``"mean"``.
        disable_index_value: if True, skip index-head value output.
        dense_main_attn_fn / page_size / use_msa / msa_kv_indices / msa_plan:
            unused, for interface parity.

    Returns:
        ``(idx_o, o)`` where ``idx_o`` is ``[batch_size, num_idx_heads, idx_head_dim]``
        or None, and ``o`` is ``[batch_size, num_q_heads, qk_head_dim]``.
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    if idx_sm_scale is None:
        idx_sm_scale = idx_q.shape[-1] ** -0.5

    batch_size = q.shape[0]
    o = q.new_zeros(q.shape)
    idx_o = (
        None
        if disable_index_value or idx_v_cache is None
        else idx_q.new_zeros(idx_q.shape)
    )

    if batch_size == 0:
        return idx_o, o

    def _flatten(cache: torch.Tensor) -> torch.Tensor:
        if cache.dim() <= 3:
            return cache
        return cache.reshape(-1, cache.shape[-2], cache.shape[-1])

    k_cache = _flatten(k_cache)
    v_cache = _flatten(v_cache)
    idx_k_cache = _flatten(idx_k_cache)
    if idx_v_cache is not None:
        idx_v_cache = _flatten(idx_v_cache)

    seq_lens_list = seq_lens.tolist()
    slot_ids_list = slot_ids.tolist()

    for batch_id in range(batch_size):
        seq_len = int(seq_lens_list[batch_id])
        if seq_len <= 0:
            continue

        req_idx = int(slot_ids_list[batch_id])

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

        query_position = seq_len - 1

        seq_idx_o, seq_o = _sparse_decode_one_seq(
            q[batch_id : batch_id + 1],
            k_seq,
            v_seq,
            idx_q[batch_id : batch_id + 1],
            idx_k_seq,
            idx_v_seq,
            query_position,
            seq_len,
            block_size_k,
            topk,
            init_blocks,
            local_blocks,
            score_type,
            sm_scale,
            idx_sm_scale,
        )
        o[batch_id : batch_id + 1] = seq_o
        if idx_o is not None and seq_idx_o is not None:
            idx_o[batch_id : batch_id + 1] = seq_idx_o

    return idx_o, o
