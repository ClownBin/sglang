# MiniMax-M3 NPU TBO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable MiniMax-M3 decoder layers to run through the existing explicit `--enable-two-batch-overlap` execution chain on NPU without changing sparse attention math.

**Architecture:** Add the missing TBO operation contracts to MiniMax-M3 attention, MoE, and decoder layer classes, then register a MiniMax-specific operation strategy. Tests stay static/CPU-focused and prove wiring contracts without pretending to validate NPU numerical output.

**Tech Stack:** Python, PyTorch modules, SGLang TBO batch overlap executor, MiniMax-M3 model classes, unittest/pytest.

---

### Task 1: Add Failing MiniMax-M3 TBO Contract Tests

**Files:**
- Modify: `test/registered/unit/models/test_minimax_m3_npu_static.py`

- [ ] **Step 1: Write the failing tests**

Append these tests to `TestMiniMaxM3NPUStaticContracts`:

```python
    def test_minimax_m3_exposes_tbo_operation_contracts(self):
        source = _read("python/sglang/srt/models/minimax_m3.py")
        tree = ast.parse(source)
        classes = {
            node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        }

        attention_methods = {
            node.name
            for node in classes["MiniMaxM3Attention"].body
            if isinstance(node, ast.FunctionDef)
        }
        moe_methods = {
            node.name
            for node in classes["MiniMaxM3MoE"].body
            if isinstance(node, ast.FunctionDef)
        }
        decoder_methods = {
            node.name
            for node in classes["MiniMaxM3DecoderLayer"].body
            if isinstance(node, ast.FunctionDef)
        }

        self.assertIn("op_prepare", attention_methods)
        self.assertIn("op_core", attention_methods)
        self.assertTrue(
            {
                "op_gate",
                "op_select_experts",
                "op_dispatch_a",
                "op_dispatch_b",
                "op_experts",
                "op_combine_a",
                "op_combine_b",
                "op_shared_experts",
                "op_output",
            }.issubset(moe_methods)
        )
        self.assertTrue(
            {
                "op_comm_prepare_attn",
                "op_comm_prepare_mlp",
                "op_comm_postprocess_layer",
            }.issubset(decoder_methods)
        )

    def test_minimax_m3_registered_in_tbo_strategy(self):
        source = _read("python/sglang/srt/batch_overlap/operations_strategy.py")

        self.assertIn('layer_name == "MiniMaxM3DecoderLayer"', source)
        self.assertIn("_compute_moe_minimax_m3_layer_operations_strategy_tbo", source)
        self.assertIn("_compute_moe_minimax_m3_prefill", source)
        self.assertIn("_compute_moe_minimax_m3_decode", source)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test/registered/unit/models/test_minimax_m3_npu_static.py -q`

Expected: FAIL because MiniMax-M3 classes and `OperationsStrategy` do not expose these TBO contracts yet.

---

### Task 2: Add MiniMax-M3 Attention and Decoder Layer TBO Ops

**Files:**
- Modify: `python/sglang/srt/models/minimax_m3.py`

- [ ] **Step 1: Implement attention op wrappers**

Add these methods to `MiniMaxM3Attention` before `forward`:

```python
    def op_prepare(self, state):
        if _is_npu:
            state.attn_intermediate_state = self.forward_prepare_npu(
                positions=state.positions,
                hidden_states=state.pop("hidden_states_after_comm_pre_attn"),
                forward_batch=state.forward_batch,
            )
        else:
            state.attn_intermediate_state = self.forward_prepare(
                positions=state.positions,
                hidden_states=state.pop("hidden_states_after_comm_pre_attn"),
                forward_batch=state.forward_batch,
            )

    def op_core(self, state):
        state.hidden_states_after_attn = self.forward_core(
            state.pop("attn_intermediate_state")
        )
```

- [ ] **Step 2: Implement decoder communication ops**

Add these methods to `MiniMaxM3DecoderLayer` after `forward` or immediately before it:

```python
    def op_comm_prepare_attn(
        self,
        state,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tbo_subbatch_index: Optional[int] = None,
    ):
        state.hidden_states_after_comm_pre_attn, state.residual_after_input_ln = (
            self.layer_communicator.prepare_attn(hidden_states, residual, forward_batch)
        )
        state.update(
            dict(
                forward_batch=forward_batch,
                positions=positions,
                tbo_subbatch_index=tbo_subbatch_index,
            )
        )

    def op_comm_prepare_mlp(self, state):
        state.hidden_states_mlp_input, state.residual_after_comm_pre_mlp = (
            self.layer_communicator.prepare_mlp(
                state.pop("hidden_states_after_attn"),
                state.pop("residual_after_input_ln"),
                state.forward_batch,
            )
        )

    def op_comm_postprocess_layer(self, state):
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            state.pop("hidden_states_mlp_output"),
            state.pop("residual_after_comm_pre_mlp"),
            state.forward_batch,
        )
        output = dict(
            positions=state.positions,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=state.forward_batch,
            tbo_subbatch_index=state.tbo_subbatch_index,
        )
        state.clear(
            expect_keys={
                "positions",
                "forward_batch",
                "tbo_subbatch_index",
            }
        )
        return output
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest test/registered/unit/models/test_minimax_m3_npu_static.py -q`

Expected: still FAIL because MoE op methods and strategy registration are not complete.

---

### Task 3: Add MiniMax-M3 MoE TBO Ops

**Files:**
- Modify: `python/sglang/srt/models/minimax_m3.py`

- [ ] **Step 1: Implement MoE op methods**

Add these methods to `MiniMaxM3MoE` after `_forward_shared_experts`:

```python
    def op_gate(self, state):
        if state.hidden_states_mlp_input.shape[0] > 0:
            state.router_logits = self._compute_router_logits(
                state.hidden_states_mlp_input
            )
        else:
            state.router_logits = None

    def op_shared_experts(self, state):
        hidden_states_mlp_input = state.pop("hidden_states_mlp_input")
        if hidden_states_mlp_input.shape[0] > 0:
            state.shared_output = self._forward_shared_experts(
                hidden_states_mlp_input
            )
        else:
            state.shared_output = None

    def op_select_experts(self, state):
        router_logits = state.pop("router_logits")
        hidden_states = state.hidden_states_mlp_input
        if router_logits is not None:
            state.topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=state.forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        else:
            state.topk_output = self.topk.empty_topk_output(hidden_states.device)

    def op_dispatch_a(self, state):
        if self.ep_size > 1:
            self.experts.dispatcher.dispatch_a(
                hidden_states=state.hidden_states_mlp_input,
                topk_output=state.pop("topk_output"),
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_dispatch_b(self, state):
        if self.ep_size > 1:
            state.dispatch_output = self.experts.dispatcher.dispatch_b(
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_experts(self, state):
        state.combine_input = self.experts.run_moe_core(
            dispatch_output=state.dispatch_output,
        )

    def op_combine_a(self, state):
        if self.ep_size > 1:
            self.experts.dispatcher.combine_a(
                combine_input=state.pop("combine_input"),
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )
            state.pop("dispatch_output")

    def op_combine_b(self, state):
        if self.ep_size > 1:
            state.hidden_states_after_combine = self.experts.dispatcher.combine_b(
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_output(self, state):
        hidden_states = state.pop("hidden_states_after_combine")
        shared_output = state.pop("shared_output")
        if shared_output is not None:
            hidden_states = hidden_states + shared_output
        state.hidden_states_mlp_output = hidden_states
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest test/registered/unit/models/test_minimax_m3_npu_static.py -q`

Expected: still FAIL because `OperationsStrategy` is not registered yet.

---

### Task 4: Register MiniMax-M3 Operation Strategy

**Files:**
- Modify: `python/sglang/srt/batch_overlap/operations_strategy.py`

- [ ] **Step 1: Add `MiniMaxM3DecoderLayer` dispatch**

In `OperationsStrategy.init_new_tbo`, add this branch after the Qwen/MiMo branches:

```python
        elif layer_name == "MiniMaxM3DecoderLayer":
            return OperationsStrategy.concat(
                [
                    _compute_moe_minimax_m3_layer_operations_strategy_tbo(
                        layer, forward_mode
                    )
                    for layer in layers
                ]
            )
```

- [ ] **Step 2: Add MiniMax-M3 strategy helpers**

Add these helper functions after the Qwen3 section or before the MiMo section:

```python
def _compute_moe_minimax_m3_layer_operations_strategy_tbo(
    layer: torch.nn.Module,
    forward_mode: ForwardMode,
) -> OperationsStrategy:
    assert layer.is_layer_sparse, "MiniMax-M3 TBO only supports sparse MoE layers"
    if forward_mode == ForwardMode.EXTEND:
        return _compute_moe_minimax_m3_prefill(layer)
    elif (
        forward_mode == ForwardMode.DECODE or forward_mode == ForwardMode.TARGET_VERIFY
    ):
        return _compute_moe_minimax_m3_decode(layer)
    else:
        raise NotImplementedError(f"Unsupported {forward_mode=}")


def _compute_moe_minimax_m3_prefill(layer):
    deep_gemm_num_sms = None
    if not _is_hip and torch.cuda.is_available():
        device_properties = torch.cuda.get_device_properties(device="cuda")
        total_num_sms = device_properties.multi_processor_count
        deep_gemm_num_sms = total_num_sms - DeepEPConfig.get_instance().num_sms

    return OperationsStrategy(
        deep_gemm_num_sms=deep_gemm_num_sms,
        tbo_delta_stages=0,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            layer.mlp.op_dispatch_a,
            operations.YieldOperation(),
            layer.mlp.op_shared_experts,
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
        ],
    )


def _compute_moe_minimax_m3_decode(layer):
    return OperationsStrategy(
        deep_gemm_num_sms=None,
        tbo_delta_stages=2,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            operations.YieldOperation(),
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_a,
            layer.mlp.op_shared_experts,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
            operations.YieldOperation(),
        ],
    )
```

- [ ] **Step 3: Run contract tests**

Run: `python -m pytest test/registered/unit/models/test_minimax_m3_npu_static.py -q`

Expected: PASS.

---

### Task 5: Final Verification

**Files:**
- Verify: `python/sglang/srt/models/minimax_m3.py`
- Verify: `python/sglang/srt/batch_overlap/operations_strategy.py`
- Verify: `test/registered/unit/models/test_minimax_m3_npu_static.py`

- [ ] **Step 1: Run focused unit tests**

Run: `python -m pytest test/registered/unit/models/test_minimax_m3_npu_static.py test/registered/unit/batch_overlap/test_tbo_filter_batch_marker.py test/registered/unit/batch_overlap/test_tbo_cuda_graph_num_token_device.py -q`

Expected: PASS.

- [ ] **Step 2: Run syntax compilation**

Run: `python -m py_compile python/sglang/srt/models/minimax_m3.py python/sglang/srt/batch_overlap/operations_strategy.py test/registered/unit/models/test_minimax_m3_npu_static.py`

Expected: exit code 0.

- [ ] **Step 3: Inspect diff for sparse attention math changes**

Run: `git diff -- python/sglang/srt/layers/attention/minimax_sparse_backend.py python/sglang/srt/layers/attention/minimax_sparse_ops`

Expected: no diff.

- [ ] **Step 4: Commit implementation**

Run:

```bash
git add python/sglang/srt/models/minimax_m3.py python/sglang/srt/batch_overlap/operations_strategy.py test/registered/unit/models/test_minimax_m3_npu_static.py docs/superpowers/plans/2026-07-02-minimax-m3-npu-tbo.md
git commit -m "feat: support minimax m3 npu two batch overlap"
```

Expected: commit succeeds after verification.
