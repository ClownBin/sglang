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
        source = _read("python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py")
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

    def test_minimax_m3_tbo_entry_is_npu_extend_only(self):
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

        self.assertIn("ForwardMode", source)
        self.assertIn("_is_npu", forward_source)
        self.assertIn("forward_batch.can_run_tbo", forward_source)
        self.assertIn(
            "forward_batch.global_forward_mode == ForwardMode.EXTEND",
            forward_source,
        )

    def test_minimax_m3_exposes_tbo_ops(self):
        source = _read("python/sglang/srt/models/minimax_m3.py")
        tree = ast.parse(source)

        def class_node(name: str) -> ast.ClassDef:
            return next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef) and node.name == name
            )

        def method_names(node: ast.ClassDef) -> set[str]:
            return {
                item.name for item in node.body if isinstance(item, ast.FunctionDef)
            }

        attention_methods = method_names(class_node("MiniMaxM3Attention"))
        dense_mlp_methods = method_names(class_node("MiniMaxM3MLP"))
        moe_methods = method_names(class_node("MiniMaxM3MoE"))
        decoder_methods = method_names(class_node("MiniMaxM3DecoderLayer"))

        self.assertIn("op_prepare", attention_methods)
        self.assertIn("op_core", attention_methods)
        self.assertIn("op_forward", dense_mlp_methods)
        for name in (
            "op_gate",
            "op_shared_experts",
            "op_select_experts",
            "op_dispatch_a",
            "op_dispatch_b",
            "op_experts",
            "op_combine_a",
            "op_combine_b",
            "op_output",
        ):
            self.assertIn(name, moe_methods)
        for name in (
            "op_comm_prepare_attn",
            "op_comm_prepare_mlp",
            "op_comm_postprocess_layer",
        ):
            self.assertIn(name, decoder_methods)

        attention_class = class_node("MiniMaxM3Attention")
        op_prepare = next(
            node
            for node in attention_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "op_prepare"
        )
        op_prepare_source = ast.get_source_segment(source, op_prepare)
        self.assertIn("forward_prepare_npu", op_prepare_source)
        self.assertIn("_is_npu", op_prepare_source)

    def test_minimax_m3_tbo_strategy_is_npu_prefill_only(self):
        source = _read("python/sglang/srt/batch_overlap/operations_strategy.py")
        tree = ast.parse(source)
        init_new_tbo = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "init_new_tbo"
        )
        init_source = ast.get_source_segment(source, init_new_tbo)

        self.assertIn("is_npu", source)
        self.assertIn("_is_npu = is_npu()", source)
        self.assertIn('"MiniMaxM3DecoderLayer"', init_source)

        strategy = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_compute_moe_minimax_m3_layer_operations_strategy_tbo"
        )
        strategy_source = ast.get_source_segment(source, strategy)
        self.assertIn("not _is_npu", strategy_source)
        self.assertIn("forward_mode != ForwardMode.EXTEND", strategy_source)
        self.assertNotIn("ForwardMode.DECODE", strategy_source)

        prefill = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_compute_moe_minimax_m3_prefill"
        )
        prefill_source = ast.get_source_segment(source, prefill)
        self.assertIn("deep_gemm_num_sms=None", prefill_source)
        self.assertNotIn("torch.cuda.get_device_properties", prefill_source)
        self.assertIn("layer.mlp.op_shared_experts", prefill_source)


if __name__ == "__main__":
    unittest.main()
