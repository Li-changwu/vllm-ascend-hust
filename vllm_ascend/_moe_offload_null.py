# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
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
"""No-op implementations for the optional MoE offload plugin hooks."""

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum


class _NullMoeOffloadConfig:
    enabled = False
    trace_only = False
    num_slots = 0
    release_original_expert_weights = False
    phase_split_enabled = False
    max_phases = 1


class _NullMoeOffloadRuntime:
    config = _NullMoeOffloadConfig()
    should_use_fixed_slots = False
    should_use_layered_runtime = False

    def next_step_id(self) -> int:
        return 0

    def trace_routing(self, *, layer_id, topk_ids, topk_weights, num_experts, step_id=None, mode="unknown"):
        return topk_ids, topk_weights

    def should_use_fixed_slot_plan_for_layer(self, layer_id: int) -> bool:
        return False

    def register_layer_for_fixed_slots(self, layer, *, slot_device=None) -> None:
        pass

    def release_original_expert_weights_if_ready(self, layer) -> None:
        pass

    def is_layer_registered(self, layer_id: int) -> bool:
        return False


_NULL_RUNTIME = _NullMoeOffloadRuntime()


def get_moe_offload_runtime() -> _NullMoeOffloadRuntime:
    return _NULL_RUNTIME


class MoeOffloadDecisionPath(str, Enum):
    FULL_WEIGHT_PATH = "full_weight_path"
    SLOT_CACHE_PATH = "slot_cache_path"
    FAIL_CLOSED = "fail_closed"


class _NullMoePipelineProfiler:
    enabled = False

    def record(self):
        return None

    def commit(self, **kwargs) -> None:
        pass

    def commit_detail_context(self, ctx) -> None:
        pass

    @contextmanager
    def detail_context(self, **kwargs):
        yield None

    def add_detail_event(self, *args, **kwargs) -> None:
        pass

    def add_wall_detail_timing(self, *args, **kwargs) -> None:
        pass


_NULL_PROFILER = _NullMoePipelineProfiler()


def get_moe_pipeline_profiler() -> _NullMoePipelineProfiler:
    return _NULL_PROFILER
