from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _store_kv_index_kernel(
    cache_k_ptr,
    cache_v_ptr,
    k_out_ptr,
    v_out_ptr,
    cache_idx_k_ptr,
    idx_k_out_ptr,
    cache_idx_v_ptr,
    idx_v_out_ptr,
    loc_ptr,
    n_tokens,
    main_cols,
    idx_cols,
    stride_ck_t,
    stride_ck_c,
    stride_cv_t,
    stride_cv_c,
    stride_ko_t,
    stride_ko_c,
    stride_vo_t,
    stride_vo_c,
    stride_cik_t,
    stride_cik_c,
    stride_iko_t,
    stride_iko_c,
    stride_civ_t,
    stride_civ_c,
    stride_ivo_t,
    stride_ivo_c,
    HAS_IDX_V: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    token = tl.program_id(0)
    offs = tl.arange(0, BLOCK_COLS)
    slot = tl.load(loc_ptr + token).to(tl.int64)

    main_mask = (token < n_tokens) & (offs < main_cols)
    k_val = tl.load(
        cache_k_ptr + token * stride_ck_t + offs * stride_ck_c,
        mask=main_mask,
        other=0.0,
    )
    v_val = tl.load(
        cache_v_ptr + token * stride_cv_t + offs * stride_cv_c,
        mask=main_mask,
        other=0.0,
    )
    tl.store(k_out_ptr + slot * stride_ko_t + offs * stride_ko_c, k_val, mask=main_mask)
    tl.store(v_out_ptr + slot * stride_vo_t + offs * stride_vo_c, v_val, mask=main_mask)

    idx_mask = (token < n_tokens) & (offs < idx_cols)
    idx_k_val = tl.load(
        cache_idx_k_ptr + token * stride_cik_t + offs * stride_cik_c,
        mask=idx_mask,
        other=0.0,
    )
    tl.store(
        idx_k_out_ptr + slot * stride_iko_t + offs * stride_iko_c,
        idx_k_val,
        mask=idx_mask,
    )

    if HAS_IDX_V:
        idx_v_val = tl.load(
            cache_idx_v_ptr + token * stride_civ_t + offs * stride_civ_c,
            mask=idx_mask,
            other=0.0,
        )
        tl.store(
            idx_v_out_ptr + slot * stride_ivo_t + offs * stride_ivo_c,
            idx_v_val,
            mask=idx_mask,
        )


def _next_power_of_2(x: int) -> int:
    return 1 << (max(1, int(x)) - 1).bit_length()


def store_kv_index_npu_triton(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    cache_idx_k: torch.Tensor,
    idx_k_out: torch.Tensor,
    cache_idx_v: torch.Tensor | None,
    idx_v_out: torch.Tensor | None,
    loc: torch.Tensor,
) -> None:
    n_tokens, main_cols = cache_k.shape
    if cache_v.shape != cache_k.shape:
        raise ValueError(f"cache_v shape {cache_v.shape} must match cache_k {cache_k.shape}")
    if k_out.shape[1] != main_cols or v_out.shape[1] != main_cols:
        raise ValueError("main output cache width must match cache_k width")
    if cache_idx_k.shape[0] != n_tokens:
        raise ValueError("cache_idx_k token count must match cache_k")
    idx_cols = cache_idx_k.shape[1]
    if idx_k_out.shape[1] != idx_cols:
        raise ValueError("index output cache width must match cache_idx_k width")
    has_idx_v = cache_idx_v is not None
    if has_idx_v:
        assert idx_v_out is not None
        if cache_idx_v.shape != cache_idx_k.shape or idx_v_out.shape[1] != idx_cols:
            raise ValueError("index V shapes must match index K shapes")
    else:
        cache_idx_v = cache_idx_k
        idx_v_out = idx_k_out

    if loc.dtype != torch.int32:
        loc = loc.to(torch.int32)
    if not loc.is_contiguous():
        loc = loc.contiguous()

    block_cols = _next_power_of_2(max(main_cols, idx_cols))
    grid = (n_tokens,)
    _store_kv_index_kernel[grid](
        cache_k,
        cache_v,
        k_out,
        v_out,
        cache_idx_k,
        idx_k_out,
        cache_idx_v,
        idx_v_out,
        loc,
        n_tokens,
        main_cols,
        idx_cols,
        cache_k.stride(0),
        cache_k.stride(1),
        cache_v.stride(0),
        cache_v.stride(1),
        k_out.stride(0),
        k_out.stride(1),
        v_out.stride(0),
        v_out.stride(1),
        cache_idx_k.stride(0),
        cache_idx_k.stride(1),
        idx_k_out.stride(0),
        idx_k_out.stride(1),
        cache_idx_v.stride(0),
        cache_idx_v.stride(1),
        idx_v_out.stride(0),
        idx_v_out.stride(1),
        HAS_IDX_V=has_idx_v,
        BLOCK_COLS=block_cols,
        num_warps=4,
        num_stages=1,
    )
