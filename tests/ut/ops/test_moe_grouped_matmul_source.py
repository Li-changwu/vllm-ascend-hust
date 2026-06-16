#
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
#

from pathlib import Path


def test_moe_grouped_matmul_has_no_unconditional_host_debug_prints():
    repo_root = Path(__file__).parents[3]
    source_paths = [
        repo_root / "csrc/moe_grouped_matmul/op_host/moe_grouped_matmul_cpu.cpp",
        repo_root / "csrc/moe_grouped_matmul/op_host/moe_grouped_matmul_l0.cpp",
    ]

    combined = "\n".join(path.read_text(encoding="utf-8") for path in source_paths)

    assert "printf(" not in combined
    assert "GMM_tiling:" not in combined
    assert "x0_dim_num" not in combined


def test_moe_grouped_matmul_tiling_knobs_are_compile_time_configurable():
    repo_root = Path(__file__).parents[3]
    host_source = (
        repo_root / "csrc/moe_grouped_matmul/op_host/moe_grouped_matmul_cpu.cpp"
    ).read_text(encoding="utf-8")
    host_cmake = (
        repo_root / "csrc/moe_grouped_matmul/op_host/CMakeLists.txt"
    ).read_text(encoding="utf-8")

    for macro in [
        "VLLM_ASCEND_MOE_GMM_BASE_N",
        "VLLM_ASCEND_MOE_GMM_SINGLE_M",
        "VLLM_ASCEND_MOE_GMM_SINGLE_N",
        "VLLM_ASCEND_MOE_GMM_MAX_BASE_M",
    ]:
        assert macro in host_source

    assert "VLLM_ASCEND_MOE_GMM_TILING_DEFINES" in host_cmake
    assert "target_compile_options(optiling" in host_cmake
