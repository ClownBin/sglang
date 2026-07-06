from __future__ import annotations

import torch


def build_extend_block_table_token_slots(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    max_len: int,
    page_size: int,
    prefix_lens: torch.Tensor | None = None,
    extend_lens: torch.Tensor | None = None,
    out_cache_loc: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return KV token slots at logical page starts.

    Prefix pages are read from req_to_token. Page starts that fall inside the
    current extend chunk are taken directly from out_cache_loc so NPU attention
    metadata does not depend on immediately re-reading freshly written
    req_to_token suffix entries.
    """

    bs = req_pool_indices.shape[0]
    max_seq_pages = (int(max_len) + page_size - 1) // page_size
    if max_seq_pages == 0:
        return torch.empty(
            (bs, 0),
            dtype=req_to_token.dtype,
            device=req_to_token.device,
        )

    page_offsets = (
        torch.arange(max_seq_pages, device=req_pool_indices.device, dtype=torch.long)
        * page_size
    )
    token_slots = req_to_token[req_pool_indices[:, None], page_offsets[None, :]]

    if (
        out_cache_loc is None
        or prefix_lens is None
        or extend_lens is None
        or out_cache_loc.numel() == 0
    ):
        return token_slots

    prefix_lens_long = prefix_lens.to(device=req_pool_indices.device, dtype=torch.long)
    extend_lens_long = extend_lens.to(device=req_pool_indices.device, dtype=torch.long)
    out_starts = torch.cumsum(extend_lens_long, dim=0) - extend_lens_long

    page_offsets_2d = page_offsets.unsqueeze(0)
    prefix_lens_2d = prefix_lens_long.unsqueeze(1)
    extend_lens_2d = extend_lens_long.unsqueeze(1)
    current_page_mask = (page_offsets_2d >= prefix_lens_2d) & (
        page_offsets_2d < prefix_lens_2d + extend_lens_2d
    )

    current_offsets = out_starts.unsqueeze(1) + page_offsets_2d - prefix_lens_2d
    safe_offsets = torch.where(
        current_page_mask, current_offsets, torch.zeros_like(current_offsets)
    ).reshape(-1)
    current_slots = out_cache_loc[safe_offsets].reshape_as(token_slots)
    if current_slots.dtype != token_slots.dtype:
        current_slots = current_slots.to(token_slots.dtype)

    return torch.where(current_page_mask, current_slots, token_slots)
