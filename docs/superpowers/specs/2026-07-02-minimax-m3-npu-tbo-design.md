# MiniMax-M3 NPU Two-Batch Overlap Design

## Goal

Support MiniMax-M3 on Ascend NPU with the existing explicit
`--enable-two-batch-overlap` switch, improving TTFT and TPOT through the current
SGLang TBO execution chain while preserving MiniMax-M3 sparse attention accuracy.

## Scope

In scope:

- Enable the existing TBO scheduler/model execution path for MiniMax-M3 decoder
  layers.
- Reuse existing MiniMax-M3 attention, sparse attention, NPU Triton, and DeepEP
  MoE math implementations.
- Add focused static and CPU unit tests that catch missing MiniMax-M3 TBO
  contracts.
- Keep default behavior unchanged when `--enable-two-batch-overlap` is absent.

Out of scope:

- Automatic TBO enablement for MiniMax-M3 NPU.
- New NPU sparse attention kernels or MSA math changes.
- Changes to top-k sparse block selection, index value semantics, or KV/index
  cache layout.
- A performance benchmark harness inside unit tests.

## Current State

MiniMax-M3 already calls `model_forward_maybe_tbo` from `MiniMaxM3Model.forward`
when `forward_batch.can_run_tbo` is true. The generic TBO implementation then
expects the model layer type to be registered in `OperationsStrategy` and expects
each decoder layer to expose operation methods such as `op_comm_prepare_attn`,
`self_attn.op_prepare`, `self_attn.op_core`, `mlp.op_gate`, and the DeepEP
dispatch/combine phases.

The existing strategy table supports `DeepseekV2DecoderLayer`,
`Qwen3MoeDecoderLayer`, and `MiMoV2DecoderLayer`, but not
`MiniMaxM3DecoderLayer`. MiniMax-M3 classes also do not expose the required
TBO `op_*` methods yet.

MiniMax-M3 sparse attention is accuracy-sensitive. On NPU the sparse path uses
`MiniMaxSparseAttnBackend`, the MiniMax sparse KV wrapper, and optional
`SGLANG_MINIMAX_NPU_TRITON=1` kernels. These paths already encode important
precision and layout assumptions and should remain unchanged for this feature.

## Design

### Enablement

TBO remains explicitly enabled by `--enable-two-batch-overlap`. MiniMax-M3 NPU
will not auto-enable TBO. Existing server argument validation requiring a
non-`none` `--moe-a2a-backend` remains in force.

### Decoder Layer Operation Split

Add TBO operation methods to `MiniMaxM3DecoderLayer` matching the established
Qwen3/DeepSeek/MiMo pattern:

- `op_comm_prepare_attn`: call `LayerCommunicator.prepare_attn`, store
  `positions`, `forward_batch`, and `tbo_subbatch_index` in stage state.
- `op_comm_prepare_mlp`: call `LayerCommunicator.prepare_mlp`.
- `op_comm_postprocess_layer`: call `LayerCommunicator.postprocess_layer` and
  return the stage output dictionary expected by TBO.

The existing `forward` method stays unchanged for non-TBO execution.

### Attention Operation Split

Add `op_prepare` and `op_core` to `MiniMaxM3Attention`:

- `op_prepare` calls the existing NPU-aware `forward_prepare_npu` on NPU and
  `forward_prepare` elsewhere, exactly as `forward` does today.
- `op_core` calls the existing `forward_core`.

This preserves MiniMax-M3 QK norm, RoPE, sparse index branch, fused cache-store
markers, and NPU dense/sparse routing semantics.

### MoE Operation Split

Add TBO operation methods to `MiniMaxM3MoE` for the DeepEP path:

- `op_gate`: compute router logits from the existing `_compute_router_logits`,
  or store `None` for empty/idle inputs.
- `op_select_experts`: call the existing `TopK` with
  `num_token_non_padded=state.forward_batch.num_token_non_padded` and
  `ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id)`.
- `op_dispatch_a` and `op_dispatch_b`: use `self.experts.dispatcher` with
  `tbo_subbatch_index`.
- `op_experts`: call `self.experts.run_moe_core`.
- `op_combine_a` and `op_combine_b`: call dispatcher combine phases with
  `tbo_subbatch_index`.
- `op_shared_experts`: compute MiniMax shared experts independently for the
  same sub-batch input when shared experts are configured outside fused experts.
- `op_output`: merge routed and shared outputs with the same addition semantics
  as `forward_deepep`.

The non-DeepEP MoE path remains outside TBO because server validation requires
an A2A backend when `--enable-two-batch-overlap` is set.

### Operation Strategy

Register `MiniMaxM3DecoderLayer` in `OperationsStrategy.init_new_tbo`.

Use a MiniMax-specific strategy instead of aliasing another model's function
directly, so the shared-expert stage is explicit:

- Prefill: align with the DeepSeek/Qwen3 prefill shape and set
  `deep_gemm_num_sms` like the existing CUDA path, but keep NPU behavior safe by
  letting the existing `deep_gemm_wrapper` context no-op outside CUDA where
  applicable.
- Decode and target verify: use the established decode stage spacing with
  `tbo_delta_stages=2`.
- Include `layer.mlp.op_shared_experts` only where MiniMax needs an unfused
  shared-expert output. It should be placed so it overlaps with DeepEP dispatch
  without changing output order.

### Accuracy Constraints

The implementation must not change:

- `MiniMaxSparseAttnBackend.forward_extend` or `forward_decode` math.
- NPU Triton sparse decode/prefill kernels.
- top-k sparse block scoring, merge, or reduction.
- MSA fallback and gating behavior.
- MiniMax sparse KV/index pool layout and writes.
- QK norm/RoPE formulas.

TBO child batches may receive sliced `ForwardBatch` metadata, but each child must
continue to call the same attention and MoE code paths as a normal batch.

### Tests

Add or extend CPU/static tests:

- Assert `MiniMaxM3Attention` exposes `op_prepare` and `op_core`.
- Assert `MiniMaxM3MoE` exposes the required TBO MoE op methods.
- Assert `MiniMaxM3DecoderLayer` exposes communication op methods.
- Assert `OperationsStrategy.init_new_tbo` has a `MiniMaxM3DecoderLayer`
  branch.
- Assert MiniMax-M3 TBO code does not introduce changes to
  `minimax_sparse_backend.py` sparse attention math for this feature, by keeping
  the new tests focused on model/TBO contract wiring.

Manual NPU validation should use the user's existing launch command plus
`--enable-two-batch-overlap`, then compare generation accuracy against the same
command without TBO and measure TTFT/TPOT.

## Risks

- Empty child batches can exercise edge cases in MoE dispatch/combine. The op
  methods must mirror the existing empty-input handling in `forward_deepep`.
- Shared experts must be added exactly once and only when present. Fused shared
  experts are already included in the top-k/expert path.
- TBO metadata slicing must preserve MiniMax sparse layer cache markers per
  child batch. The attention op split reuses the existing marker field on each
  child `ForwardBatch`.

## Acceptance Criteria

- Launching MiniMax-M3 NPU with `--enable-two-batch-overlap` no longer fails at
  TBO operation strategy initialization.
- Without `--enable-two-batch-overlap`, MiniMax-M3 execution remains unchanged.
- Unit/static tests for MiniMax-M3 TBO contracts pass.
- Manual NPU accuracy with TBO matches the non-TBO baseline within the normal
  MiniMax-M3 NPU tolerance for deterministic prompts.
- TTFT and TPOT can be benchmarked with the same server command by toggling only
  `--enable-two-batch-overlap`.
