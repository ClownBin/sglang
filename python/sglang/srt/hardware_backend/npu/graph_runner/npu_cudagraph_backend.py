"""NPUCudaGraphBackend — Ascend NPU full-graph capture (torch.npu.NPUGraph).

Mirrors FullCudaGraphBackend with two differences:
  - Captures via torch.npu.graph(...) into torch.npu.NPUGraph.
  - replay_with_input_update(shape_key, seq_lens, attr_name) rebinds
    the recorded graph's input bindings for variable seq_lens at replay
    time (NPU's NPUGraph.update(...) API).

torch.npu is imported lazily inside methods so the module loads on
non-NPU hosts.
"""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager, contextmanager
from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import numpy as np
import torch

from sglang.srt.configs.model_config import is_minimax_sparse
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.utils import empty_context, get_bool_env_var
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )


class NPUCudaGraphBackend(BaseCudaGraphBackend):
    """One torch.npu.NPUGraph per shape; attention metadata captured
    inside the graph. replay_with_input_update substitutes fresh
    seq_lens without re-recording."""

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
    ) -> None:
        self._graphs: Dict[Any, Any] = {}
        self._outputs: Dict[Any, Any] = {}
        self._pool = None
        self._cuda_graph_runner = cuda_graph_runner
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = cuda_graph_runner.model_runner.tp_group
        self._capture_stream = None
        self._memory_saver_adapter: Optional[Any] = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )
        self._enable_torch_compile = getattr(
            cuda_graph_runner, "enable_torch_compile", False
        )

    @contextmanager
    def capture_session(self, stream):
        if self._pool is None:
            self._pool = self._device_module.graph_pool_handle()
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        try:
            yield
        finally:
            self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        dummies: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        import torch_npu  # noqa: F401  (verifies NPU availability)

        # Two warmups so kernels are loaded and one-time setup is paid before capture.
        # post_warmup_hook lets the attention backend reset state that warmup mutated.
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()

        graph = torch.npu.NPUGraph()

        if self._enable_torch_compile:
            skip_guard_context = torch.compiler.set_stance(skip_guard_eval_unsafe=True)
        else:
            skip_guard_context = empty_context()

        graph_ctx: Callable[..., AbstractContextManager]
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            graph_ctx = partial(
                self._memory_saver_adapter.cuda_graph,
                tag=GPU_MEMORY_TYPE_CUDA_GRAPH,
            )
        else:
            graph_ctx = torch.npu.graph

        with skip_guard_context, graph_ctx(
            graph,
            pool=self._pool,
            stream=self._capture_stream,
            auto_dispatch_capture=True,
        ):
            out = forward_fn()

        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        return shape_key in self._graphs

    @contextmanager
    def replay_session(self):
        yield

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    @staticmethod
    def _cpu_update_attr_names(
        attr_name: str = None,
        cpu_update_input: list = None,
    ) -> set[str]:
        names = set()
        if attr_name is not None:
            names.add(attr_name)
        if cpu_update_input is not None:
            for item in cpu_update_input:
                names.update(item.keys())
        return names

    def _can_skip_minimax_m3_target_verify_update(
        self,
        cpu_update_input: list = None,
    ) -> bool:
        attr_names = self._cpu_update_attr_names(cpu_update_input=cpu_update_input)
        if attr_names not in ({"actual_seq_kvlen"}, {"actual_seq_lengths_kv"}):
            return False

        runner = self._cuda_graph_runner
        model_runner = getattr(runner, "model_runner", None)
        if model_runner is None or getattr(model_runner, "is_draft_worker", False):
            return False

        model_config = getattr(model_runner, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        if hf_config is None or not is_minimax_sparse(hf_config):
            return False

        capture_forward_mode = getattr(runner, "capture_forward_mode", None)
        attn_backend = getattr(runner, "attn_backend", None) or getattr(
            model_runner, "attn_backend", None
        )
        can_skip = getattr(attn_backend, "can_skip_npu_graph_seq_lens_update", None)
        if capture_forward_mode is not None and can_skip is not None:
            if can_skip(SimpleNamespace(forward_mode=capture_forward_mode)):
                return True

        if capture_forward_mode is not None:
            if not capture_forward_mode.is_target_verify():
                return False
        else:
            spec_algorithm = getattr(model_runner, "spec_algorithm", None)
            if not (
                spec_algorithm is not None and spec_algorithm.is_speculative()
            ):
                return False

        if getattr(model_runner, "use_mla_backend", False) or getattr(
            model_runner, "is_hybrid_swa", False
        ):
            return False
        if getattr(model_config, "has_attention_sinks", False):
            return False
        return True

    def _replay_without_update(self, shape_key: ShapeKey) -> Any:
        self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    def replay_with_input_update(
        self,
        shape_key: ShapeKey,
        seq_lens: Any,
        attr_name: str = None,
        attr_type: Any = None,
        cpu_update_input: list = None,
    ) -> Any:
        """Rebind seq_lens on the recorded NPU graph in a background
        thread, then replay. Used when the model is not deepseek-nsa.

        Two calling conventions:
        1. (legacy) seq_lens + attr_name + attr_type:
           Constructs cpu_update_input=[{attr_name: seq_lens}] internally.
        2. cpu_update_input: A list of {attr_name: seq_lens} dicts,
           one per speculative step.  Used by EAGLE draft runners.
        """
        if cpu_update_input is None:
            if isinstance(attr_type, torch.Tensor):
                seq_lens = torch.from_numpy(np.array(seq_lens).astype(np.int32))
            cpu_update_input = [{attr_name: seq_lens}]

        if self._can_skip_minimax_m3_target_verify_update(
            cpu_update_input=cpu_update_input
        ):
            return self._replay_without_update(shape_key)

        graph = self._graphs[shape_key]

        def _update():
            graph.update(cpu_update_input=cpu_update_input)

        thread = threading.Thread(target=_update)
        thread.start()
        graph.replay()
        thread.join()
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self._graphs.clear()
        self._outputs.clear()
        self._pool = None
