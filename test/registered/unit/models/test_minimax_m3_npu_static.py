import ast
import re
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[4]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text()


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

    def test_fuseep_prefill_uses_global_dp_extend_mode(self):
        source = _read("python/sglang/srt/models/minimax_m3.py")

        self.assertIn("get_is_extend_in_batch", source)
        self.assertRegex(
            source,
            r"forward_batch\.forward_mode\.is_extend\(\)\s+or\s+\(\s*"
            r"is_dp_attention_enabled\(\)\s+and\s+get_is_extend_in_batch\(\)\s*\)",
            "M3 FuseEP must include idle DP-attention ranks in an extend EP collective.",
        )
        self.assertIn(
            'getattr(topk_output, "expert_location_dispatch_info", None)', source
        )
        self.assertIn("m3_fuseep_num_input_tokens", source)
        self.assertIn("if is_extend_in_batch and dp_global_num_tokens is not None", source)

    def test_fuseep_normal_mode_is_extend_only(self):
        source = _read("python/sglang/srt/models/minimax_m3.py")

        self.assertRegex(
            source,
            r"(?s)use_m3_fuseep_normal\s*=\s*\(.*?"
            r"and is_extend_in_batch\s+"
            r"and getattr\(topk_output, \"expert_location_dispatch_info\", None\) is None",
        )

    def test_low_latency_fuseep_replaces_invalid_expert_ids(self):
        source = _read("python/sglang/srt/hardware_backend/npu/moe/fuseep.py")
        low_latency_source = source[
            source.index("is_idle_dp_rank = is_dp_attention_enabled()") :
        ]

        self.assertIn(
            "topk_ids = topk_ids.masked_fill(topk_ids < 0, 0)",
            low_latency_source,
        )

    def test_ascend_fuseep_uses_a2a_moe_forward(self):
        source = _read("python/sglang/srt/models/minimax_m3.py")

        self.assertRegex(
            source,
            r"get_moe_a2a_backend\(\)\.is_deepep\(\)\s+or\s+"
            r"get_moe_a2a_backend\(\)\.is_ascend_fuseep\(\)",
            "Ascend FuseEP must use M3's A2A MoE forward path.",
        )

    def test_m3_fuseep_kwargs_are_not_passed_to_deepep(self):
        source = _read("python/sglang/srt/models/minimax_m3.py")

        self.assertIn(
            "if get_moe_a2a_backend().is_ascend_fuseep():\n"
            "            final_hidden_states = self.experts(",
            source,
        )

    def test_fuseep_workspace_uses_global_dp_tokens(self):
        source = _read("python/sglang/srt/hardware_backend/npu/moe/fuseep.py")

        self.assertIn("get_dp_global_num_tokens", source)
        self.assertIn(
            "num_input_tokens = max(num_input_tokens, sum(global_num_tokens))",
            source,
        )
        self.assertIn("is_idle_dp_rank", source)
        self.assertIn("return hidden_states[:num_output_tokens]", source)
        self.assertIn("if not is_dp_attention_enabled():", source)
        self.assertIn("normal_decode and hidden_states.shape[0] < 128", source)

    def test_fuseep_scale_preserves_expert_dimension(self):
        source = _read("python/sglang/srt/hardware_backend/npu/moe/fuseep.py")

        self.assertIn(
            ").reshape(scale.shape).to(scale.device)", source
        )

    def test_fuseep_copies_dp_gathered_input(self):
        source = _read("python/sglang/srt/hardware_backend/npu/moe/fuseep.py")

        self.assertIn("hidden_states = hidden_states.clone()", source)
        self.assertIn("topk_weights = torch.ones", source)

    def test_fuseep_replaces_invalid_expert_ids(self):
        source = _read("python/sglang/srt/hardware_backend/npu/moe/fuseep.py")

        self.assertIn("topk_ids.masked_fill(topk_ids < 0, 0)", source)

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


if __name__ == "__main__":
    unittest.main()
