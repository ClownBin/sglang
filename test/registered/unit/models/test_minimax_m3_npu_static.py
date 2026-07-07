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
