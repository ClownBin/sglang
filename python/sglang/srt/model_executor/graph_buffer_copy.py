# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Shape-safe graph-buffer copy planning.

This module owns device-specific copy policy for graph-resident ForwardBatch
buffers. Registries build ``GraphBufferCopyEntry`` lists; the planner decides
once per shape signature whether a group can use ``torch._foreach_copy_`` or
must use single tensor copies.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_has_foreach_copy = hasattr(torch, "_foreach_copy_")
_NPU_GRAPH_COPY_FOREACH_ENV = "SGLANG_NPU_GRAPH_COPY_FOREACH"
_npu_foreach_probe_cache: Dict[Tuple[str, Optional[int]], bool] = {}


@dataclass(frozen=True)
class GraphBufferCopyEntry:
    name: str
    dst: torch.Tensor
    src: torch.Tensor


@dataclass(frozen=True)
class _CopyPlanStep:
    use_foreach: bool
    indices: Tuple[int, ...]


@dataclass(frozen=True)
class _CopyPlan:
    steps: Tuple[_CopyPlanStep, ...]


def _tensor_device_type(tensor: torch.Tensor) -> str:
    return tensor.device.type


def _tensor_device_index(tensor: torch.Tensor) -> Optional[int]:
    return tensor.device.index


def _npu_foreach_copy_mode() -> str:
    mode = os.environ.get(_NPU_GRAPH_COPY_FOREACH_ENV, "auto").lower()
    if mode not in ("auto", "on", "off"):
        raise ValueError(
            f"{_NPU_GRAPH_COPY_FOREACH_ENV} must be one of auto/on/off, got {mode!r}."
        )
    return mode


def _probe_npu_foreach_copy(device: torch.device) -> bool:
    if not _has_foreach_copy:
        return False
    key = (device.type, device.index)
    if key not in _npu_foreach_probe_cache:
        try:
            dst = torch.empty((1,), dtype=torch.int32, device=device)
            src = torch.ones((1,), dtype=torch.int32, device=device)
            torch._foreach_copy_([dst], [src])
            _npu_foreach_probe_cache[key] = True
        except Exception as e:
            logger.warning(
                "Disabling NPU foreach graph-buffer copy because the capability "
                "probe failed: %s",
                e,
            )
            _npu_foreach_probe_cache[key] = False
    return _npu_foreach_probe_cache[key]


def _npu_foreach_copy_enabled(device: torch.device) -> bool:
    mode = _npu_foreach_copy_mode()
    if mode == "off":
        return False
    if _probe_npu_foreach_copy(device):
        return True
    if mode == "on":
        raise RuntimeError(
            f"{_NPU_GRAPH_COPY_FOREACH_ENV}=on requested NPU foreach graph-buffer "
            "copy, but the startup capability probe failed."
        )
    return False


def _npu_shape_safe_foreach_group(
    group_entries: List[GraphBufferCopyEntry],
) -> bool:
    if len(group_entries) < 2:
        return False
    first = group_entries[0]
    dst_device = (_tensor_device_type(first.dst), _tensor_device_index(first.dst))
    src_device = (_tensor_device_type(first.src), _tensor_device_index(first.src))
    dst_shape = tuple(first.dst.shape)
    src_shape = tuple(first.src.shape)
    dst_stride = tuple(first.dst.stride())
    src_stride = tuple(first.src.stride())
    dst_dtype = first.dst.dtype
    src_dtype = first.src.dtype
    if dst_device != src_device or dst_shape != src_shape or dst_dtype != src_dtype:
        return False
    if not (
        first.dst.is_contiguous()
        and first.src.is_contiguous()
        or dst_stride == src_stride
    ):
        return False

    for entry in group_entries[1:]:
        entry_dst_device = (
            _tensor_device_type(entry.dst),
            _tensor_device_index(entry.dst),
        )
        entry_src_device = (
            _tensor_device_type(entry.src),
            _tensor_device_index(entry.src),
        )
        if entry_dst_device != dst_device:
            return False
        if entry_src_device != src_device:
            return False
        if entry.dst.dtype != dst_dtype or entry.src.dtype != src_dtype:
            return False
        if tuple(entry.dst.shape) != dst_shape or tuple(entry.src.shape) != src_shape:
            return False
        if tuple(entry.dst.stride()) != dst_stride:
            return False
        if tuple(entry.src.stride()) != src_stride:
            return False
        if not (
            entry.dst.is_contiguous()
            and entry.src.is_contiguous()
            or tuple(entry.dst.stride()) == tuple(entry.src.stride())
        ):
            return False
    return True


def _copy_group_key(entry: GraphBufferCopyEntry):
    return (
        _tensor_device_type(entry.dst),
        _tensor_device_index(entry.dst),
        _tensor_device_type(entry.src),
        _tensor_device_index(entry.src),
        entry.dst.dtype,
        entry.src.dtype,
        tuple(entry.dst.shape),
        tuple(entry.src.shape),
        tuple(entry.dst.stride()),
        tuple(entry.src.stride()),
        entry.dst.is_contiguous(),
        entry.src.is_contiguous(),
    )


def _copy_entries_signature(entries: List[GraphBufferCopyEntry]):
    return tuple((entry.name, *_copy_group_key(entry)) for entry in entries)


def _copy_one(name: str, dst: torch.Tensor, src: torch.Tensor) -> None:
    try:
        dst.copy_(src)
    except RuntimeError as e:
        raise RuntimeError(
            f"copy failed for slot {name!r}: "
            f"dst shape={tuple(dst.shape)}, stride={tuple(dst.stride())}; "
            f"src shape={tuple(src.shape)}, stride={tuple(src.stride())}"
        ) from e


class GraphBufferCopyPlanner:
    def __init__(self, *, enable_cpu_foreach: bool = False) -> None:
        self.enable_cpu_foreach = enable_cpu_foreach
        self._copy_plan_cache: Dict[Tuple[Any, ...], _CopyPlan] = {}

    def copy(self, entries: List[GraphBufferCopyEntry]) -> None:
        if not entries:
            return
        signature = _copy_entries_signature(entries)
        plan = self._copy_plan_cache.get(signature)
        if plan is None:
            plan = self._build_copy_plan(entries)
            self._copy_plan_cache[signature] = plan
        self._execute_copy_plan(entries, plan)

    def _should_use_foreach_copy(
        self, group_entries: List[GraphBufferCopyEntry]
    ) -> bool:
        if not _has_foreach_copy:
            return False
        if not group_entries:
            return False
        device_type = _tensor_device_type(group_entries[0].dst)
        if device_type == "npu":
            return _npu_shape_safe_foreach_group(
                group_entries
            ) and _npu_foreach_copy_enabled(group_entries[0].dst.device)
        if device_type == "cpu":
            return self.enable_cpu_foreach
        return True

    def _build_copy_plan(self, entries: List[GraphBufferCopyEntry]) -> _CopyPlan:
        groups: Dict[Tuple[Any, ...], List[int]] = {}
        for idx, entry in enumerate(entries):
            key = _copy_group_key(entry)
            if key not in groups:
                groups[key] = []
            groups[key].append(idx)

        steps: List[_CopyPlanStep] = []
        for indices in groups.values():
            group_entries = [entries[idx] for idx in indices]
            steps.append(
                _CopyPlanStep(
                    use_foreach=self._should_use_foreach_copy(group_entries),
                    indices=tuple(indices),
                )
            )
        return _CopyPlan(steps=tuple(steps))

    def _execute_copy_plan(
        self, entries: List[GraphBufferCopyEntry], plan: _CopyPlan
    ) -> None:
        for step in plan.steps:
            if step.use_foreach:
                group_entries = [entries[idx] for idx in step.indices]
                group_dsts = [entry.dst for entry in group_entries]
                group_srcs = [entry.src for entry in group_entries]
                group_names = [entry.name for entry in group_entries]
                try:
                    torch._foreach_copy_(group_dsts, group_srcs)
                except RuntimeError as e:
                    raise RuntimeError(
                        "foreach copy failed for slots "
                        f"{group_names}: dst shapes "
                        f"{[tuple(dst.shape) for dst in group_dsts]}, src shapes "
                        f"{[tuple(src.shape) for src in group_srcs]}"
                    ) from e
            else:
                for idx in step.indices:
                    entry = entries[idx]
                    _copy_one(entry.name, entry.dst, entry.src)


def _grouped_foreach_copy_(
    dsts: List[torch.Tensor],
    srcs: List[torch.Tensor],
    names: Optional[List[str]] = None,
) -> None:
    """Compatibility helper used by graph runners outside the registry."""
    if names is None:
        names = ["<unnamed>"] * len(dsts)
    entries = [
        GraphBufferCopyEntry(name=name, dst=dst, src=src)
        for name, dst, src in zip(names, dsts, srcs)
    ]
    GraphBufferCopyPlanner(enable_cpu_foreach=True).copy(entries)
