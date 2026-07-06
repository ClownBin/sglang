import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


def register_cpu_ci(*_args, **_kwargs):
    return None


register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FailingTritonWriter:
    def __getitem__(self, _grid):
        def _raise_if_called(*_args, **_kwargs):
            raise AssertionError("NPU ascend must not use triton req_to_token writer")

        return _raise_if_called


def _install_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _install_package(monkeypatch, name):
    module = _install_module(monkeypatch, name)
    module.__path__ = []
    return module


def _load_mem_cache_common(monkeypatch):
    for package in (
        "sglang",
        "sglang.srt",
        "sglang.srt.hardware_backend",
        "sglang.srt.hardware_backend.npu",
        "sglang.srt.hardware_backend.npu.dsv4",
        "sglang.srt.mem_cache",
        "sglang.srt.mem_cache.allocator",
        "sglang.srt.mem_cache.triton_ops",
        "sglang.srt.utils",
    ):
        _install_package(monkeypatch, package)

    _install_module(
        monkeypatch,
        "sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks",
        maybe_evict_dsv4_state_on_swa=lambda *_args, **_kwargs: None,
        maybe_write_dsv4_decode=lambda *_args, **_kwargs: None,
        maybe_write_dsv4_extend=lambda *_args, **_kwargs: None,
    )
    _install_module(
        monkeypatch,
        "sglang.srt.mem_cache.allocator.swa",
        SWATokenToKVPoolAllocator=type("SWATokenToKVPoolAllocator", (), {}),
    )
    _install_module(
        monkeypatch,
        "sglang.srt.mem_cache.base_prefix_cache",
        BasePrefixCache=type("BasePrefixCache", (), {}),
        EvictParams=type("EvictParams", (), {}),
    )
    _install_module(
        monkeypatch,
        "sglang.srt.mem_cache.memory_pool",
        HybridReqToTokenPool=type("HybridReqToTokenPool", (), {}),
        ReqToTokenPool=type("ReqToTokenPool", (), {}),
    )
    _install_module(
        monkeypatch,
        "sglang.srt.mem_cache.triton_ops.common",
        _get_last_loc_safe_kernel=None,
        get_last_loc_kernel=None,
        get_last_loc_triton=lambda *_args, **_kwargs: None,
        get_last_loc_triton_safe=lambda *_args, **_kwargs: None,
        write_req_to_token_pool_triton=_FailingTritonWriter(),
    )
    _install_module(
        monkeypatch,
        "sglang.srt.server_args",
        ServerArgs=type("ServerArgs", (), {}),
        get_global_server_args=lambda: types.SimpleNamespace(
            attention_backend="ascend",
            dcp_size=1,
        ),
    )
    utils_module = sys.modules["sglang.srt.utils"]
    utils_module.is_cuda = lambda: False
    utils_module.is_hip = lambda: False
    utils_module.is_npu = lambda: True
    utils_module.support_triton = lambda _backend: True
    _install_module(
        monkeypatch,
        "sglang.srt.utils.common",
        ceil_align=lambda x, y: ((x + y - 1) // y) * y,
        is_pin_memory_available=lambda _device: False,
    )

    module_name = "sglang.srt.mem_cache.common"
    module_path = (
        Path(__file__).resolve().parents[4]
        / "python"
        / "sglang"
        / "srt"
        / "mem_cache"
        / "common.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _ReqToTokenPool:
    def __init__(self):
        self.device = "cpu"
        self.req_to_token = torch.full((4, 8), -1, dtype=torch.int64)

    def write(self, indices, values):
        if (
            isinstance(indices, tuple)
            and isinstance(indices[1], slice)
            and indices[1].start not in (None, 0)
        ):
            raise AssertionError(
                "NPU shape-safe writer must batch-write extend slices"
            )
        self.req_to_token[indices] = values


def test_write_cache_indices_uses_npu_shape_safe_extend_writer(monkeypatch):
    common = _load_mem_cache_common(monkeypatch)

    pool = _ReqToTokenPool()
    out_cache_loc = torch.tensor([20, 21, 30, 31], dtype=torch.int64)
    req_pool_indices = torch.tensor([1, 2, 3], dtype=torch.int64)
    prefix_lens = torch.tensor([2, 1, 0], dtype=torch.int64)
    seq_lens = torch.tensor([4, 1, 2], dtype=torch.int64)
    extend_lens = torch.tensor([2, 0, 2], dtype=torch.int64)
    prefix_tensors = [
        torch.tensor([10, 11], dtype=torch.int64),
        torch.tensor([12], dtype=torch.int64),
        torch.empty((0,), dtype=torch.int64),
    ]

    common.write_cache_indices(
        out_cache_loc,
        req_pool_indices,
        req_pool_indices,
        prefix_lens,
        prefix_lens,
        seq_lens,
        seq_lens,
        extend_lens,
        extend_lens,
        prefix_tensors,
        pool,
    )

    torch.testing.assert_close(
        pool.req_to_token[1, :4], torch.tensor([10, 11, 20, 21])
    )
    torch.testing.assert_close(pool.req_to_token[2, :1], torch.tensor([12]))
    torch.testing.assert_close(pool.req_to_token[3, :2], torch.tensor([30, 31]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
