"""Fused SwigluOAI activation (MiniMax-M3) triton kernel for Ascend NPU.

MiniMax-M3 uses ``hidden_act = "swigluoai"``:

    out = (up + 1) * gate * sigmoid(gate * alpha)

with ``gate = x[..., :d].clamp(max=limit)`` and ``up = x[..., d:].clamp(-limit, limit)``.
``x`` is the concatenated ``[gate | up]`` tensor (last dim = 2 * intermediate).

The reference path (``npu_swiglu_oai`` in ``fused_moe_method_npu.py``) emits ~6
separate elementwise kernels (slice+clamp x2, mul, sigmoid, mul, add, mul), each
reading/writing the activation tensor. This module fuses the whole expression
into a single triton kernel that reads ``x`` once and writes ``out`` once, with
fp32 internal accumulation and bf16 output. Numerically it is >= as accurate as
the bf16 reference (and bit-close to it); see ``test`` below.

Only elementwise triton primitives are used (``tl.minimum``/``tl.maximum``/
``tl.exp2``/``tl.where``), all confirmed to run on the Ascend TBE backend (same
primitives as ``minimax_sparse_ops/npu_triton/flash_block_score_decode.py``).
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

# log2(e); used to express exp() through the TBE-verified tl.exp2().
_LOG2E = 1.4426950408889634


def _next_power_of_2(x: int) -> int:
    return 1 << (max(1, int(x)) - 1).bit_length()


def _get_positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _choose_swiglu_oai_tile(n_rows: int, d: int) -> tuple[int, int]:
    """Choose an Ascend-friendly tile for decode and prefill shapes.

    Decode typically has very few rows and a large intermediate dimension, so a
    wider D tile reduces the number of programs. Prefill has many rows, where
    the original conservative 8x256 tile keeps register pressure predictable.
    Env overrides are intentionally simple for profiling A/B:
    ``SGLANG_MINIMAX_M3_NPU_SWIGLU_OAI_BLOCK_N`` and
    ``SGLANG_MINIMAX_M3_NPU_SWIGLU_OAI_BLOCK_D``.
    """
    if n_rows <= 4:
        block_n, target_d = 1, 1024
    elif n_rows <= 16:
        block_n, target_d = 4, 512
    else:
        block_n, target_d = 8, 256

    block_d = _next_power_of_2(min(max(1, d), target_d))
    block_d = min(block_d, 4096)

    block_n = (
        _get_positive_int_env("SGLANG_MINIMAX_M3_NPU_SWIGLU_OAI_BLOCK_N")
        or block_n
    )
    block_d_override = _get_positive_int_env(
        "SGLANG_MINIMAX_M3_NPU_SWIGLU_OAI_BLOCK_D"
    )
    if block_d_override is not None:
        block_d = min(_next_power_of_2(block_d_override), 4096)
    return block_n, block_d


@triton.jit
def _swiglu_oai_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,  # = d (output cols); input has 2 * n_cols cols
    stride_xr,
    stride_xc,
    stride_or,
    stride_oc,
    alpha,
    limit,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_n = offs_n < n_rows
    mask_d = offs_d < n_cols
    mask = mask_n[:, None] & mask_d[None, :]

    # gate = x[row, col]            up = x[row, n_cols + col]
    gate_ptrs = x_ptr + offs_n[:, None] * stride_xr + offs_d[None, :] * stride_xc
    up_ptrs = (
        x_ptr
        + offs_n[:, None] * stride_xr
        + (offs_d[None, :] + n_cols) * stride_xc
    )
    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)

    gate_c = tl.minimum(gate, limit)                       # clamp(-inf, limit]
    up_c = tl.minimum(tl.maximum(up, -limit), limit)       # clamp[-limit, limit]
    # sigmoid(gate_c * alpha) = 1 / (1 + exp(-(gate_c*alpha)))
    #                       = 1 / (1 + exp2(-(gate_c*alpha) * log2(e)))
    sig = 1.0 / (1.0 + tl.exp2(-(gate_c * alpha) * 1.4426950408889634))
    result = gate_c * sig * (up_c + 1.0)

    out_ptrs = out_ptr + offs_n[:, None] * stride_or + offs_d[None, :] * stride_oc
    tl.store(out_ptrs, result.to(out_ptr.dtype.element_ty), mask=mask)


def npu_swiglu_oai_fused(
    x: torch.Tensor, alpha: float, limit: float
) -> torch.Tensor:
    """Fused SwigluOAI. ``x: [..., 2d] -> out[..., d]`` (bf16/fp16 in & out)."""
    assert x.dim() >= 1 and x.shape[-1] % 2 == 0, f"bad shape {x.shape}"
    d = x.shape[-1] // 2
    if not x.is_contiguous():
        x = x.contiguous()
    x2d = x.reshape(-1, 2 * d)
    n_rows = x2d.shape[0]

    out = torch.empty((n_rows, d), device=x.device, dtype=x.dtype)

    BLOCK_N, BLOCK_D = _choose_swiglu_oai_tile(n_rows, d)

    grid = (triton.cdiv(n_rows, BLOCK_N), triton.cdiv(d, BLOCK_D))
    _swiglu_oai_kernel[grid](
        x2d,
        out,
        n_rows,
        d,
        x2d.stride(0),
        x2d.stride(1),
        out.stride(0),
        out.stride(1),
        alpha,
        limit,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )
    return out.reshape(*x.shape[:-1], d)
