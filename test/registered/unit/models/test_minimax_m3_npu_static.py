import ast
import re
import unittest
from pathlib import Path

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ModuleNotFoundError:
    register_cpu_ci = None

if register_cpu_ci is not None:
    register_cpu_ci(est_time=2, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[4]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


class TestMiniMaxM3NPUStaticContracts(unittest.TestCase):
    def test_model_has_explicit_npu_prepare_path(self):
        source = _read("python/sglang/srt/models/minimax_m3.py")
        tree = ast.parse(source)
        function_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }

        self.assertIn("_is_npu = is_npu()", source)
        self.assertIn("split_qkv_tp_rmsnorm_rope", source)
        self.assertIn("get_attention_tp_group", source)
        self.assertIn("forward_prepare_npu", function_names)
        self.assertRegex(source, r"if\s+_is_npu:\s*\n\s+s = self\.forward_prepare_npu")
        self.assertNotRegex(
            source,
            r"_fuse_qkv_index_enabled\s*=.*_is_npu",
            "NPU must not enter CUDA/ROCm qkv+index fused GEMM path.",
        )

    def test_npu_memory_pool_exposes_minimax_sparse_wrapper(self):
        source = _read("python/sglang/srt/hardware_backend/npu/memory_pool_npu.py")
        tree = ast.parse(source)
        class_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        }

        self.assertIn("NPUMHATokenToKOnlyPool", class_names)
        self.assertIn("NPUMiniMaxSparseKVPool", class_names)
        self.assertIn("torch_npu.npu_scatter_nd_update_", source)

    def test_ascend_pool_selection_prefers_minimax_sparse_pool(self):
        source = _read(
            "python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py"
        )
        ascend_branch = source[
            source.index('self.server_args.attention_backend == "ascend"') :
        ]

        minimax_match = re.search(
            r"is_minimax_sparse\(self\.model_config\.hf_config\)", ascend_branch
        )
        generic_mha_match = re.search(r"NPUMHATokenToKVPool", ascend_branch)

        self.assertIsNotNone(minimax_match)
        self.assertIsNotNone(generic_mha_match)
        self.assertLess(minimax_match.start(), generic_mha_match.start())
        self.assertIn("NPUMiniMaxSparseKVPool", ascend_branch)

    def test_minimax_sparse_backend_has_npu_guardrails(self):
        source = _read("python/sglang/srt/layers/attention/minimax_sparse_backend.py")

        self.assertIn("is_npu", source)
        self.assertIn("_raise_npu_sparse_not_ready", source)
        self.assertRegex(source, r"self\.is_npu\s*=\s*is_npu\(\)")

    def test_minimax_hybrid_backend_exposes_pool_refs_for_tbo(self):
        source = _read("python/sglang/srt/layers/attention/minimax_sparse_backend.py")
        tree = ast.parse(source)
        hybrid_class = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "MiniMaxHybridAttnBackend"
        )
        init_fn = next(
            node
            for node in hybrid_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        init_source = ast.get_source_segment(source, init_fn)

        self.assertIn(
            "self.token_to_kv_pool = dense_backend.token_to_kv_pool",
            init_source,
        )
        self.assertIn(
            "self.req_to_token_pool = dense_backend.req_to_token_pool",
            init_source,
        )

    def test_npu_sparse_prefill_avoids_metadata_item_syncs(self):
        source = _read("python/sglang/srt/layers/attention/minimax_sparse_backend.py")
        tree = ast.parse(source)
        prefill = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_forward_npu_sparse_prefill"
        )
        forbidden_metadata = (
            "req_pool_indices",
            "cu_seqlens",
            "seq_lens",
            "prefix_lens",
        )

        item_sources = []
        for node in ast.walk(prefill):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "item"
            ):
                item_sources.append(ast.get_source_segment(source, node.func.value))

        offending = [
            item_source
            for item_source in item_sources
            if item_source is not None
            and any(name in item_source for name in forbidden_metadata)
        ]
        self.assertEqual(
            offending,
            [],
            "NPU sparse prefill should use CPU metadata instead of per-request "
            ".item() syncs for batch lengths/indices.",
        )

    def test_npu_sparse_prefill_callers_pass_prefill_meta(self):
        source = _read("python/sglang/srt/layers/attention/minimax_sparse_backend.py")
        tree = ast.parse(source)
        forward_extend = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "forward_extend"
        )

        prefill_calls = [
            node
            for node in ast.walk(forward_extend)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_forward_npu_sparse_prefill"
        ]

        self.assertGreaterEqual(len(prefill_calls), 1)
        self.assertTrue(
            all(len(call.args) == 11 for call in prefill_calls),
            "Every NPU sparse prefill call should pass precomputed prefill_meta.",
        )

    def test_minimax_sparse_target_verify_extends_kv_len_by_draft_tokens(self):
        source = _read("python/sglang/srt/layers/attention/minimax_sparse_backend.py")
        tree = ast.parse(source)
        init_out_graph = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "init_forward_metadata_out_graph"
        )
        body = ast.get_source_segment(source, init_out_graph)

        self.assertIn("seq_lens_max", body)
        self.assertIn("forward_batch.forward_mode.is_target_verify()", body)
        self.assertIn("self.speculative_num_draft_tokens", body)
        self.assertRegex(
            body,
            r"seq_lens_max\s*\+=\s*int\(self\.speculative_num_draft_tokens or 0\)",
            "MiniMax sparse TARGET_VERIFY block tables must include draft tokens "
            "when computing max KV length.",
        )

    def test_minimax_sparse_npu_block_tables_overlay_current_extend_slots(self):
        source = _read("python/sglang/srt/layers/attention/minimax_sparse_backend.py")
        tree = ast.parse(source)
        function_sources = {
            node.name: ast.get_source_segment(source, node)
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }

        self.assertIn("build_extend_block_table_token_slots", source)
        self.assertIn("_build_extend_block_table_token_slots", function_sources)
        for name in ("_forward_npu_triton_verify", "_forward_npu_triton_prefill"):
            with self.subTest(name=name):
                self.assertIn(
                    "_build_extend_block_table_token_slots",
                    function_sources[name],
                )

    def test_minimax_m3_dense_verify_triton_module_is_dedicated_dense_path(self):
        source = _read(
            "python/sglang/srt/hardware_backend/npu/attention/"
            "minimax_m3_dense_verify_triton.py"
        )

        self.assertIn("dense_verify_paged_attention", source)
        self.assertIn("NUM_BLOCKS", source)
        self.assertIn("per_query_seq_lens", source)
        self.assertIn("block_table", source)
        self.assertIn("grid = (batch_size, num_kv_heads)", source)
        self.assertIn("safe_logical_block", source)
        self.assertIn("logical_block < max_blocks", source)
        self.assertIn("num_pages", source)
        self.assertIn(
            "physical_block = tl.minimum(tl.maximum(physical_block, 0), num_pages - 1)",
            source,
        )
        self.assertIn("safe_off_h", source)
        self.assertIn("safe_off_d", source)
        self.assertIn("safe_off_n", source)
        self.assertIn("pid_h + safe_off_h[:, None]", source)
        self.assertIn("safe_off_d[None, :] * stride_q_d", source)
        self.assertIn("safe_off_n[None, :] * stride_k_offset", source)
        self.assertIn("safe_off_n[:, None] * stride_v_offset", source)
        self.assertNotIn("topk_idx", source)
        self.assertNotIn("flash_decode_bnsd_with_gqa_share_sparse", source)
        self.assertNotIn("_merge_topk_attn_out_bnsd_kernel", source)

    def test_minimax_m3_dense_verify_graph_uses_single_kernel_without_merge(self):
        source = _read(
            "python/sglang/srt/hardware_backend/npu/attention/"
            "minimax_m3_dense_verify_triton.py"
        )

        self.assertIn("num_kv_chunks = 1", source)
        self.assertNotIn("_merge_topk_attn_out_bnsd_kernel", source)
        self.assertNotIn("lse_partial", source)
        self.assertNotIn("o_partial", source)

    def test_minimax_m3_dense_verify_gate_avoids_fia_cpu_seq_list(self):
        source = _read("python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py")
        tree = ast.parse(source)
        function_sources = {
            node.name: ast.get_source_segment(source, node)
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }

        gate = function_sources["_can_use_minimax_m3_triton_mtp_verify"]
        impl = function_sources["_forward_minimax_m3_triton_mtp_verify"]
        forward_mtp = function_sources["forward_mtp"]

        for fragment in (
            "forward_batch.forward_mode.is_target_verify()",
            "self.graph_mode",
            "not self.use_mla",
            "sinks is None",
            "not self._is_swa_layer(layer)",
            "layer.qk_head_dim == layer.v_head_dim",
            "self._is_minimax_m3",
        ):
            self.assertIn(fragment, gate)

        self.assertIn("dense_verify_paged_attention", impl)
        self.assertIn("forward_batch.seq_lens", impl)
        self.assertIn("repeat_interleave", impl)
        self.assertNotIn(".cpu().int().tolist()", impl)

        gate_idx = forward_mtp.index("_can_use_minimax_m3_triton_mtp_verify")
        fia_idx = forward_mtp.index("seq_lens_cpu_int.cpu().int().tolist()")
        self.assertLess(gate_idx, fia_idx)

    def test_npu_graph_seq_lens_update_skip_is_backend_opt_in_only(self):
        base_source = _read("python/sglang/srt/layers/attention/base_attn_backend.py")
        ascend_source = _read(
            "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
        )
        hybrid_source = _read(
            "python/sglang/srt/layers/attention/minimax_sparse_backend.py"
        )
        runner_source = _read(
            "python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py"
        )
        ascend_tree = ast.parse(ascend_source)
        ascend_functions = {
            node.name: ast.get_source_segment(ascend_source, node)
            for node in ast.walk(ascend_tree)
            if isinstance(node, ast.FunctionDef)
        }
        ascend_skip_fn = ascend_functions["can_skip_npu_graph_seq_lens_update"]

        self.assertIn(
            "def can_skip_npu_graph_seq_lens_update(self, forward_batch: ForwardBatch)",
            base_source,
        )
        self.assertIn("return False", base_source)
        self.assertIn(
            "def can_skip_npu_graph_seq_lens_update", ascend_source
        )
        self.assertIn(
            "return self._can_use_minimax_m3_triton_mtp_verify",
            ascend_source,
        )
        self.assertIn(
            "self._minimax_m3_dense_verify_static_supported",
            ascend_skip_fn,
        )
        self.assertIn(
            "def can_skip_npu_graph_seq_lens_update", hybrid_source
        )
        self.assertIn(
            "return self.dense.can_skip_npu_graph_seq_lens_update(forward_batch)",
            hybrid_source,
        )
        self.assertIn("can_skip_npu_graph_seq_lens_update", runner_source)
        self.assertIn("self.backend.replay(graph_key, forward_batch)", runner_source)
        self.assertIn("self.backend.replay_with_input_update", runner_source)

    def test_npu_backend_has_minimax_target_verify_update_guard(self):
        source = _read(
            "python/sglang/srt/hardware_backend/npu/graph_runner/"
            "npu_cudagraph_backend.py"
        )
        tree = ast.parse(source)
        function_sources = {
            node.name: ast.get_source_segment(source, node)
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }

        guard = function_sources["_can_skip_minimax_m3_target_verify_update"]
        replay_with_update = function_sources["replay_with_input_update"]

        self.assertIn("self._cuda_graph_runner = cuda_graph_runner", source)
        self.assertIn("is_minimax_sparse", source)
        self.assertIn("SimpleNamespace", source)
        self.assertIn("capture_forward_mode.is_target_verify()", guard)
        self.assertIn("can_skip_npu_graph_seq_lens_update", guard)
        self.assertIn("spec_algorithm.is_speculative()", guard)
        self.assertIn("is_draft_worker", guard)
        self.assertIn("use_mla_backend", guard)
        self.assertIn("is_hybrid_swa", guard)
        self.assertIn("has_attention_sinks", guard)
        self.assertIn("actual_seq_kvlen", guard)
        self.assertIn("actual_seq_lengths_kv", guard)
        self.assertIn(
            "self._can_skip_minimax_m3_target_verify_update",
            replay_with_update,
        )
        self.assertLess(
            replay_with_update.index("cpu_update_input = [{attr_name: seq_lens}]"),
            replay_with_update.index(
                "self._can_skip_minimax_m3_target_verify_update"
            ),
        )
        self.assertLess(
            replay_with_update.index(
                "self._can_skip_minimax_m3_target_verify_update"
            ),
            replay_with_update.index("graph.update"),
        )

    def test_minimax_m3_npu_eagle3_disables_draft_graph_update_paths(self):
        source = _read("python/sglang/srt/speculative/eagle_worker_v2.py")
        tree = ast.parse(source)
        function_sources = {
            node.name: ast.get_source_segment(source, node)
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }

        guard = function_sources["_disable_minimax_m3_npu_eagle_draft_graphs"]
        capture = function_sources["_capture_cuda_graphs"]

        self.assertIn("is_minimax_sparse", source)
        self.assertIn("_is_npu", guard)
        self.assertIn("self.speculative_algorithm.is_eagle3()", guard)
        self.assertIn("self.target_worker", guard)
        self.assertIn("self.draft_runner", guard)
        self.assertIn("is_minimax_sparse(hf_config)", guard)
        self.assertIn(
            "if self._disable_minimax_m3_npu_eagle_draft_graphs():",
            capture,
        )
        self.assertLess(
            capture.index("if self._disable_minimax_m3_npu_eagle_draft_graphs():"),
            capture.index("EAGLEDraftNpuGraphRunner"),
        )
        self.assertLess(
            capture.index("if self._disable_minimax_m3_npu_eagle_draft_graphs():"),
            capture.index("EAGLEDraftExtendNpuGraphRunner"),
        )

    def test_minimax_m3_npu_eagle3_keeps_target_verify_graph(self):
        source = _read("python/sglang/srt/speculative/eagle_utils.py")

        self.assertNotIn(
            "_disable_minimax_m3_npu_eagle_target_verify_graph", source
        )
        self.assertNotIn("verify_forward_batch.disable_cuda_graph", source)

    def test_minimax_m3_npu_eagle3_target_verify_captures_exact_odd_bs(self):
        source = _read(
            "python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py"
        )

        self.assertIn(
            "_should_capture_minimax_m3_eagle3_target_verify_exact_bs", source
        )
        self.assertIn("is_minimax_sparse", source)
        self.assertIn("is_npu", source)
        self.assertIn("model_runner.spec_algorithm.is_eagle3()", source)
        self.assertIn("not model_runner.is_draft_worker", source)
        self.assertIn("not model_runner.server_args.enable_two_batch_overlap", source)
        self.assertIn("num_tokens_per_bs > 1", source)
        self.assertIn("range(1, min(16, num_max_requests) + 1)", source)

    def test_model_runner_does_not_have_per_batch_cuda_graph_disable(self):
        forward_batch_source = _read(
            "python/sglang/srt/model_executor/forward_batch_info.py"
        )
        model_runner_source = _read(
            "python/sglang/srt/model_executor/model_runner.py"
        )

        self.assertNotIn("disable_cuda_graph: bool = False", forward_batch_source)
        self.assertNotIn("forward_batch.disable_cuda_graph", model_runner_source)

    def test_eagle3_debug_spec_cycle_syncs_verify_boundaries(self):
        source = _read("python/sglang/srt/speculative/eagle_worker_v2.py")

        self.assertIn("def _debug_spec_cycle_sync", source)
        self.assertIn("envs.SGLANG_DEBUG_SPEC_CYCLE.get()", source)
        self.assertIn("torch.get_device_module(device).synchronize()", source)
        for label in (
            "decode.after_draft",
            "verify.before_target_forward",
            "verify.after_target_forward",
            "verify.after_eagle_sample",
            "verify.after_bonus_tokens",
            "decode.after_verify_before_draft_extend",
        ):
            self.assertIn(label, source)

    def test_tbo_backend_delegates_npu_graph_seq_lens_update_skip(self):
        source = _read("python/sglang/srt/layers/attention/tbo_backend.py")

        self.assertIn(
            "def can_skip_npu_graph_seq_lens_update", source
        )
        self.assertIn(
            "return self.primary.can_skip_npu_graph_seq_lens_update(forward_batch)",
            source,
        )

    def test_inner_fb_view_carries_extend_metadata_for_ascend_draft(self):
        source = _read("python/sglang/srt/model_executor/forward_batch_info.py")
        tree = ast.parse(source)
        build_inner_fb_view = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "build_inner_fb_view"
        )
        body = ast.get_source_segment(source, build_inner_fb_view)

        self.assertIn("extend_prefix_lens=", body)
        self.assertIn("extend_seq_lens=", body)
        self.assertIn("extend_prefix_lens_cpu=", body)
        self.assertIn("extend_seq_lens_cpu=", body)

    def test_replay_fb_view_carries_extend_metadata_for_ascend_verify(self):
        source = _read(
            "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py"
        )
        tree = ast.parse(source)
        build_replay_fb_view = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "build_replay_fb_view"
        )
        body = ast.get_source_segment(source, build_replay_fb_view)

        self.assertIn("extend_prefix_lens=", body)
        self.assertIn("extend_seq_lens=", body)
        self.assertIn("extend_prefix_lens_cpu=", body)
        self.assertIn("extend_seq_lens_cpu=", body)

    def test_eagle_draft_extend_replay_views_carry_extend_metadata_for_ascend(self):
        paths = [
            "python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py",
            "python/sglang/srt/speculative/"
            "multi_layer_eagle_draft_extend_cuda_graph_runner.py",
        ]

        for path in paths:
            with self.subTest(path=path):
                source = _read(path)
                tree = ast.parse(source)
                fb_view_assignments = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == "fb_view"
                        for target in node.targets
                    )
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "SimpleNamespace"
                ]

                self.assertGreaterEqual(len(fb_view_assignments), 1)
                for assignment in fb_view_assignments:
                    body = ast.get_source_segment(source, assignment.value)
                    for field in (
                        "extend_prefix_lens=",
                        "extend_seq_lens=",
                        "extend_prefix_lens_cpu=",
                        "extend_seq_lens_cpu=",
                    ):
                        self.assertIn(field, body)

    def test_tbo_merge_length_uses_splitter_token_ranges_after_scatter(self):
        source = _read("python/sglang/srt/batch_overlap/two_batch_overlap.py")
        tree = ast.parse(source)
        model_forward_tbo = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_model_forward_tbo"
        )
        body = ast.get_source_segment(source, model_forward_tbo)

        self.assertIn("_compute_tbo_merge_original_len(inputs_arr)", body)
        self.assertNotIn('inputs["hidden_states"].shape[0]', body)

        helper = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_compute_tbo_merge_original_len"
        )
        helper_source = ast.get_source_segment(source, helper)
        self.assertIn("tbo_parent_token_range", helper_source)

    def test_tbo_child_replay_view_carries_extend_metadata_for_ascend(self):
        source = _read("python/sglang/srt/layers/attention/tbo_backend.py")
        tree = ast.parse(source)
        helper = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_build_tbo_child_replay_fb_view"
        )
        body = ast.get_source_segment(source, helper)

        self.assertIn("extend_prefix_lens=", body)
        self.assertIn("extend_seq_lens=", body)
        self.assertIn("extend_prefix_lens_cpu=", body)
        self.assertIn("extend_seq_lens_cpu=", body)

    def test_swigluoai_has_npu_eager_path(self):
        source = _read("python/sglang/srt/models/minimax_m3.py")
        tree = ast.parse(source)
        function_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }

        self.assertIn("_swigluoai_torch", function_names)
        self.assertRegex(
            source,
            r"(?s)elif hidden_act == \"swigluoai\".*?if _is_npu:",
            "NPU must avoid the torch.compile/Triton swigluoai helper.",
        )

    def test_dense_mlp_accepts_decoder_layer_call_signature(self):
        source = _read("python/sglang/srt/models/minimax_m3.py")
        tree = ast.parse(source)
        mlp_class = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "MiniMaxM3MLP"
        )
        forward = next(
            node
            for node in mlp_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        arg_names = [arg.arg for arg in forward.args.args]

        self.assertEqual(
            arg_names[:5],
            [
                "self",
                "x",
                "forward_batch",
                "should_allreduce_fusion",
                "use_reduce_scatter",
            ],
        )

    def test_decoder_layer_passes_forward_batch_to_mlp_by_keyword(self):
        source = _read("python/sglang/srt/models/minimax_m3.py")
        tree = ast.parse(source)
        decoder_class = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "MiniMaxM3DecoderLayer"
        )
        forward = next(
            node
            for node in decoder_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        mlp_calls = [
            node
            for node in ast.walk(forward)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "mlp"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ]

        self.assertEqual(len(mlp_calls), 1)
        keyword_names = {keyword.arg for keyword in mlp_calls[0].keywords}
        self.assertIn("forward_batch", keyword_names)
        self.assertIn("should_allreduce_fusion", keyword_names)
        self.assertIn("use_reduce_scatter", keyword_names)

    def test_large_gemma_rmsnorm_residual_avoids_npu_triton_kernel(self):
        source = _read("python/sglang/srt/layers/layernorm.py")
        tree = ast.parse(source)
        class_node = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "GemmaRMSNorm"
        )
        function_names = {
            node.name for node in class_node.body if isinstance(node, ast.FunctionDef)
        }
        forward_npu = next(
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward_npu"
        )
        forward_source = ast.get_source_segment(source, forward_npu)

        self.assertIn("_forward_npu_unfused_residual", function_names)
        self.assertIn("_NPU_GEMMA_RMS_NORM_TRITON_MAX_HIDDEN_SIZE", source)
        self.assertRegex(
            forward_source,
            r"x\.shape\[-1\]\s*>\s*_NPU_GEMMA_RMS_NORM_TRITON_MAX_HIDDEN_SIZE",
            "MiniMax-M3 hidden_size=6144 must avoid the fused Triton residual kernel.",
        )

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

    def test_minimax_m3_attention_core_handles_empty_tbo_subbatch(self):
        source = _read("python/sglang/srt/models/minimax_m3.py")
        tree = ast.parse(source)
        attention_class = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "MiniMaxM3Attention"
        )
        forward_core = next(
            node
            for node in attention_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward_core"
        )
        forward_core_source = ast.get_source_segment(source, forward_core)

        self.assertRegex(
            forward_core_source,
            r"if\s+inner_state\s+is\s+None:\s*\n\s+return\s+hidden_states",
            "TBO runs attention op_core for every subbatch; empty subbatches must "
            "short-circuit before sparse attention unpacking.",
        )

    def test_minimax_m3_registered_in_tbo_strategy(self):
        source = _read("python/sglang/srt/batch_overlap/operations_strategy.py")

        self.assertIn('layer_name == "MiniMaxM3DecoderLayer"', source)
        self.assertIn("_compute_moe_minimax_m3_layer_operations_strategy_tbo", source)
        self.assertIn("_compute_moe_minimax_m3_prefill", source)
        self.assertIn("_compute_moe_minimax_m3_decode", source)

    def test_minimax_m3_tbo_runs_dense_prefix_before_sparse_suffix(self):
        source = _read("python/sglang/srt/models/minimax_m3.py")
        tree = ast.parse(source)
        model_class = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "MiniMaxM3Model"
        )
        forward = next(
            node
            for node in model_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        forward_source = ast.get_source_segment(source, forward)

        self.assertIn("normal_start_layer = self.start_layer", forward_source)
        self.assertIn("normal_end_layer = self.end_layer", forward_source)
        self.assertIn("range(normal_start_layer, normal_end_layer)", forward_source)
        self.assertIn("if normal_end_layer != self.end_layer:", forward_source)
        self.assertIn(
            "layers=self.layers[normal_end_layer : self.end_layer]",
            forward_source,
        )
        self.assertRegex(
            forward_source,
            r"self\.layers\[\s*normal_end_layer\s*-\s*1\s*\]\.layer_scatter_modes\.layer_output_mode",
        )
        self.assertNotIn("layers=self.layers,", forward_source)

    def test_minimax_m3_tbo_keeps_eagle3_capture_layers_in_normal_prefix(self):
        source = _read("python/sglang/srt/models/minimax_m3.py")
        tree = ast.parse(source)
        model_class = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "MiniMaxM3Model"
        )
        helper = next(
            (
                node
                for node in model_class.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "_compute_tbo_normal_end_layer"
            ),
            None,
        )
        self.assertIsNotNone(
            helper,
            "MiniMax-M3 TBO must compute its suffix boundary separately so "
            "EAGLE3 aux hidden-state capture keeps the mtp_tmp_okay layer-loop "
            "semantics before TBO takes over.",
        )
        helper_source = ast.get_source_segment(source, helper)

        self.assertIn("last_capture_layer", helper_source)
        self.assertIn('"_is_layer_to_capture"', helper_source)
        self.assertIn("last_capture_layer + 1", helper_source)
        self.assertRegex(
            helper_source,
            r"normal_end_layer\s*=\s*min\(\s*max\(self\.first_tbo_layer,\s*self\.start_layer\)",
        )

    def test_minimax_m3_tbo_strategy_avoids_cuda_sms_on_npu_path(self):
        source = _read("python/sglang/srt/batch_overlap/operations_strategy.py")
        tree = ast.parse(source)
        prefill = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_compute_moe_minimax_m3_prefill"
        )
        prefill_source = ast.get_source_segment(source, prefill)

        self.assertNotIn("torch.cuda", prefill_source)
        self.assertNotIn("get_device_properties", prefill_source)
        self.assertNotIn("DeepEPConfig.get_instance", prefill_source)
        self.assertRegex(prefill_source, r"deep_gemm_num_sms\s*=\s*None")

    def test_minimax_m3_tbo_strategy_preserves_semantic_op_order(self):
        source = _read("python/sglang/srt/batch_overlap/operations_strategy.py")
        tree = ast.parse(source)

        def function_source(name):
            function = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == name
            )
            return ast.get_source_segment(source, function)

        def assert_order(function_body, ordered_fragments):
            indexes = [function_body.index(fragment) for fragment in ordered_fragments]
            self.assertEqual(indexes, sorted(indexes), ordered_fragments)

        for function_name in (
            "_compute_moe_minimax_m3_prefill",
            "_compute_moe_minimax_m3_decode",
        ):
            body = function_source(function_name)
            assert_order(
                body,
                [
                    "layer.op_comm_prepare_attn",
                    "layer.self_attn.op_prepare",
                    "layer.self_attn.op_core",
                    "layer.op_comm_prepare_mlp",
                    "layer.mlp.op_gate",
                    "layer.mlp.op_select_experts",
                    "layer.mlp.op_dispatch_a",
                    "layer.mlp.op_shared_experts",
                    "layer.mlp.op_dispatch_b",
                    "layer.mlp.op_experts",
                    "layer.mlp.op_combine_a",
                    "layer.mlp.op_combine_b",
                    "layer.mlp.op_output",
                    "layer.op_comm_postprocess_layer",
                ],
            )

    def test_minimax_m3_tbo_moe_ops_reuse_forward_deepep_math(self):
        source = _read("python/sglang/srt/models/minimax_m3.py")
        tree = ast.parse(source)
        moe_class = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "MiniMaxM3MoE"
        )
        op_sources = {
            node.name: ast.get_source_segment(source, node)
            for node in moe_class.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("op_")
        }

        self.assertIn("_compute_router_logits", op_sources["op_gate"])
        self.assertIn("num_token_non_padded", op_sources["op_select_experts"])
        self.assertIn(
            "ExpertLocationDispatchInfo.init_new",
            op_sources["op_select_experts"],
        )
        self.assertIn("_forward_shared_experts", op_sources["op_shared_experts"])
        self.assertIn("run_moe_core", op_sources["op_experts"])
        self.assertIn("combine_b", op_sources["op_combine_b"])
        self.assertIn("hidden_states + shared_output", op_sources["op_output"])


if __name__ == "__main__":
    unittest.main()
