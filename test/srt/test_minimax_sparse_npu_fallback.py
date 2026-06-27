import importlib.util
import sys
import types
from pathlib import Path

import torch


def _install_fake_modules():
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.configs",
        "sglang.srt.configs.model_config",
        "sglang.srt.layers",
        "sglang.srt.layers.attention",
        "sglang.srt.layers.attention.base_attn_backend",
        "sglang.srt.layers.attention.minimax_sparse_ops",
        "sglang.srt.layers.attention.minimax_sparse_ops.common",
        "sglang.srt.layers.attention.minimax_sparse_ops.common.index",
        "sglang.srt.mem_cache",
        "sglang.srt.mem_cache.memory_pool",
        "sglang.srt.model_executor",
        "sglang.srt.model_executor.forward_batch_info",
        "sglang.srt.utils",
        "sglang.srt.utils.async_probe",
        "triton",
        "triton.language",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))

    model_config = sys.modules["sglang.srt.configs.model_config"]
    model_config.get_minimax_sparse_attention_config = lambda _cfg: {}
    model_config.get_minimax_sparse_disable_value_layer_ids = lambda _cfg: []
    model_config.get_minimax_sparse_layer_ids = lambda _cfg: ([], [])
    model_config.get_minimax_sparse_score_type = lambda _cfg: "max"

    base_attn = sys.modules["sglang.srt.layers.attention.base_attn_backend"]
    base_attn.AttentionBackend = type("AttentionBackend", (), {})

    common_index = sys.modules[
        "sglang.srt.layers.attention.minimax_sparse_ops.common.index"
    ]
    def topk_index_reduce(tensor, dim):
        tensor_permuted = torch.movedim(tensor, source=dim, destination=-2)
        combined = tensor_permuted.flatten(start_dim=-2)
        sorted_vals, _ = combined.sort(dim=-1)
        is_new_element = sorted_vals[..., 1:] != sorted_vals[..., :-1]
        first_col_true = torch.ones_like(sorted_vals[..., :1], dtype=torch.bool)
        non_duplicate_mask = torch.cat([first_col_true, is_new_element], dim=-1)
        valid_mask = non_duplicate_mask & (sorted_vals != -1)
        sort_idx = torch.argsort((~valid_mask).int(), dim=-1, stable=True)
        result = torch.gather(sorted_vals, -1, sort_idx)
        valid_count = valid_mask.sum(dim=-1, keepdim=True)
        idx_range = torch.arange(result.size(-1), device=tensor.device)
        return torch.where(idx_range < valid_count, result, -1)

    common_index.topk_index_reduce = topk_index_reduce

    memory_pool = sys.modules["sglang.srt.mem_cache.memory_pool"]
    memory_pool.MiniMaxSparseKVPool = type("MiniMaxSparseKVPool", (), {})

    forward_batch = sys.modules["sglang.srt.model_executor.forward_batch_info"]
    forward_batch.ForwardBatch = type("ForwardBatch", (), {})

    utils = sys.modules["sglang.srt.utils"]
    utils.get_bool_env_var = lambda _name, default: default == "True"
    utils.is_npu = lambda: True

    async_probe = sys.modules["sglang.srt.utils.async_probe"]
    async_probe.maybe_detect_oob = lambda *args, **kwargs: None

    triton_mod = sys.modules["triton"]
    triton_mod.jit = lambda fn=None, **_kwargs: (lambda f: f) if fn is None else fn
    triton_mod.heuristics = lambda _values: (lambda fn: fn)
    triton_mod.next_power_of_2 = lambda x: 1 << (int(x) - 1).bit_length()
    triton_mod.language = sys.modules["triton.language"]

    tl_mod = sys.modules["triton.language"]
    tl_mod.constexpr = int


def _load_minimax_sparse_backend_module():
    _install_fake_modules()
    module_path = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/srt/layers/attention/minimax_sparse_backend.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_minimax_sparse_backend_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_npu_index_merge_module():
    _install_fake_modules()
    module_path = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/srt/layers/attention/minimax_sparse_ops/npu_triton/index_merge.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_npu_index_merge_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_npu_sparse_block_selection_masks_future_blocks_and_dedups_local():
    module = _load_minimax_sparse_backend_module()
    backend = module.MiniMaxSparseAttnBackend.__new__(module.MiniMaxSparseAttnBackend)
    backend.block_size_k = 2
    backend.topk_blocks = 1
    backend.init_blocks = 0
    backend.local_blocks = 1
    backend.score_type = "max"

    idx_q = torch.ones((3, 1, 1), dtype=torch.bfloat16)
    idx_k = torch.tensor([[0.0], [1.0], [100.0], [100.0], [200.0], [200.0]])
    query_positions = torch.tensor([0, 1, 2], dtype=torch.long)

    blocks = backend._select_sparse_blocks(idx_q, idx_k, query_positions, seq_len=6)

    assert blocks.shape == (3, 1, 2)
    expected = torch.tensor([[[0, -1]], [[0, -1]], [[1, -1]]], dtype=torch.int32)
    torch.testing.assert_close(blocks, expected)


def test_npu_sparse_seq_matches_dense_attention_when_all_blocks_are_selected():
    module = _load_minimax_sparse_backend_module()
    backend = module.MiniMaxSparseAttnBackend.__new__(module.MiniMaxSparseAttnBackend)
    backend.block_size_k = 2
    backend.topk_blocks = 4
    backend.init_blocks = 0
    backend.local_blocks = 0
    backend.score_type = "max"

    q_seq = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.bfloat16)
    k_seq = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[2.0, 0.0]]],
        dtype=torch.bfloat16,
    )
    v_seq = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[2.0, 2.0]]],
        dtype=torch.bfloat16,
    )
    idx_q = torch.ones((2, 1, 1), dtype=torch.bfloat16)
    idx_k = torch.ones((4, 1), dtype=torch.bfloat16)
    query_positions = torch.tensor([0, 1], dtype=torch.long)

    _, out = backend._npu_sparse_seq(
        q_seq, k_seq, v_seq, idx_q, idx_k, None, query_positions, seq_len=4
    )

    scores = torch.einsum("qhd,khd->qhk", q_seq.float(), k_seq.float()) * (
        q_seq.shape[-1] ** -0.5
    )
    key_pos = torch.arange(k_seq.shape[0])
    valid = key_pos[None, :] <= query_positions[:, None]
    scores = scores.masked_fill(~valid[:, None, :], -1.0e30)
    probs = torch.softmax(scores, dim=-1)
    expected = torch.einsum("qhk,khd->qhd", probs.to(v_seq.dtype), v_seq)

    torch.testing.assert_close(out.float(), expected.float(), rtol=1e-3, atol=1e-3)


def test_npu_index_reduce_merge_matches_reference_for_msa_blocks():
    backend_module = _load_minimax_sparse_backend_module()
    index_merge = _load_npu_index_merge_module()
    backend = backend_module.MiniMaxSparseAttnBackend.__new__(
        backend_module.MiniMaxSparseAttnBackend
    )
    backend.block_size_k = 4
    backend.topk_blocks = 2
    backend.init_blocks = 1
    backend.local_blocks = 1

    topk_idx = torch.tensor(
        [
            [[2, 1], [3, -1]],
            [[1, 4], [2, 3]],
            [[0, 5], [1, 3]],
            [[5, 2], [4, -1]],
        ],
        dtype=torch.int32,
    )
    num_kv_heads = 2
    idx_group_size = topk_idx.shape[0] // num_kv_heads
    query_positions = torch.tensor([9, 15], dtype=torch.long)
    max_blocks = 4

    reduced = backend_module.topk_index_reduce(
        topk_idx.view(num_kv_heads, idx_group_size, 2, backend.topk_blocks),
        dim=1,
    )
    reference = backend._merge_sparse_blocks(
        reduced.permute(1, 0, 2).contiguous(), query_positions, max_blocks
    ).permute(1, 0, 2).contiguous()

    actual = index_merge.merge_topk_index_for_sparse_decode(
        topk_idx,
        query_positions,
        num_kv_heads=num_kv_heads,
        topk_blocks=backend.topk_blocks,
        init_blocks=backend.init_blocks,
        local_blocks=backend.local_blocks,
        block_size=backend.block_size_k,
        num_blocks=max_blocks,
    )

    torch.testing.assert_close(actual, reference)


def test_npu_index_reduce_merge_preserves_local_fast_path_width():
    backend_module = _load_minimax_sparse_backend_module()
    index_merge = _load_npu_index_merge_module()
    backend = backend_module.MiniMaxSparseAttnBackend.__new__(
        backend_module.MiniMaxSparseAttnBackend
    )
    backend.block_size_k = 4
    backend.topk_blocks = 2
    backend.init_blocks = 0
    backend.local_blocks = 1

    topk_idx = torch.tensor(
        [
            [[2, 1]],
            [[1, 4]],
            [[0, 5]],
            [[5, 2]],
        ],
        dtype=torch.int32,
    )
    num_kv_heads = 2
    idx_group_size = topk_idx.shape[0] // num_kv_heads
    query_positions = torch.tensor([9], dtype=torch.long)
    max_blocks = 4

    reduced = backend_module.topk_index_reduce(
        topk_idx.view(num_kv_heads, idx_group_size, 1, backend.topk_blocks),
        dim=1,
    )
    reference = backend._merge_sparse_blocks(
        reduced.permute(1, 0, 2).contiguous(), query_positions, max_blocks
    ).permute(1, 0, 2).contiguous()

    actual = index_merge.merge_topk_index_for_sparse_decode(
        topk_idx,
        query_positions,
        num_kv_heads=num_kv_heads,
        topk_blocks=backend.topk_blocks,
        init_blocks=backend.init_blocks,
        local_blocks=backend.local_blocks,
        block_size=backend.block_size_k,
        num_blocks=max_blocks,
    )

    assert actual.shape[-1] == idx_group_size * backend.topk_blocks + 1
    torch.testing.assert_close(actual, reference)
