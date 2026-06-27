import argparse
import os
import time

import torch

from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    npu_swiglu_oai,
)


def _sync():
    torch.npu.synchronize()


def _bench(fn, x, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn(x)
    _sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn(x)
    _sync()
    return (time.perf_counter() - start) * 1e6 / iters


def main():
    parser = argparse.ArgumentParser(description="Benchmark MiniMax-M3 NPU SwiGLU-OAI")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--block-n", type=int, default=None)
    parser.add_argument("--block-d", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.702)
    parser.add_argument("--limit", type=float, default=7.0)
    args = parser.parse_args()

    if not hasattr(torch, "npu"):
        raise RuntimeError("This benchmark must be run in an Ascend NPU environment.")

    if args.block_n is not None:
        os.environ["SGLANG_MINIMAX_M3_NPU_SWIGLU_OAI_BLOCK_N"] = str(args.block_n)
    if args.block_d is not None:
        os.environ["SGLANG_MINIMAX_M3_NPU_SWIGLU_OAI_BLOCK_D"] = str(args.block_d)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    x = torch.randn(args.rows, args.dim * 2, device="npu", dtype=dtype)

    os.environ["SGLANG_MINIMAX_M3_NPU_FUSED_SWIGLU_OAI"] = "1"
    fused_us = _bench(
        lambda t: npu_swiglu_oai(t, args.alpha, args.limit), x, args.warmup, args.iters
    )

    os.environ["SGLANG_MINIMAX_M3_NPU_FUSED_SWIGLU_OAI"] = "0"
    fallback_us = _bench(
        lambda t: npu_swiglu_oai(t, args.alpha, args.limit), x, args.warmup, args.iters
    )

    print(
        "rows={rows} dim={dim} dtype={dtype} block_n={block_n} block_d={block_d} "
        "fused_us={fused:.3f} fallback_us={fallback:.3f} speedup={speedup:.3f}x".format(
            rows=args.rows,
            dim=args.dim,
            dtype=args.dtype,
            block_n=os.environ.get(
                "SGLANG_MINIMAX_M3_NPU_SWIGLU_OAI_BLOCK_N", "auto"
            ),
            block_d=os.environ.get(
                "SGLANG_MINIMAX_M3_NPU_SWIGLU_OAI_BLOCK_D", "auto"
            ),
            fused=fused_us,
            fallback=fallback_us,
            speedup=fallback_us / fused_us if fused_us > 0 else float("inf"),
        )
    )


if __name__ == "__main__":
    main()
