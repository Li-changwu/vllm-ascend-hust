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
# WITHOUT WARRANTIES OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json

from tools.moe_gmm.compare_gmm_stages import render_markdown, summarize_stage_case, summarize_stage_files


def test_summarize_stage_case_adds_oracle_fanout_gates():
    microbench = {
        "mode": "gmm1",
        "results": [
            {
                "class_id": 1,
                "fanout": 39,
                "coverage_percent": 10.0,
                "variants": [
                    {"variant": "counts:s2", "backend": "torch_npu", "ms": {"avg": 1.0, "p50": 0.9, "p99": 1.5}},
                    {"variant": "counts:s2:custom", "backend": "custom", "ms": {"avg": 1.2, "p50": 1.1, "p99": 1.6}},
                ],
            },
            {
                "class_id": 2,
                "fanout": 40,
                "coverage_percent": 30.0,
                "variants": [
                    {"variant": "counts:s2", "backend": "torch_npu", "ms": {"avg": 1.0, "p50": 0.9, "p99": 1.5}},
                    {"variant": "counts:s2:custom", "backend": "custom", "ms": {"avg": 0.8, "p50": 0.7, "p99": 1.2}},
                ],
            },
            {
                "class_id": 3,
                "fanout": 45,
                "coverage_percent": 60.0,
                "variants": [
                    {"variant": "counts:s2", "backend": "torch_npu", "ms": {"avg": 2.0, "p50": 1.9, "p99": 3.0}},
                    {"variant": "counts:s2:custom", "backend": "custom", "ms": {"avg": 1.6, "p50": 1.5, "p99": 2.5}},
                ],
            },
        ],
    }

    summary = summarize_stage_case(
        "gmm1_default",
        microbench,
        thresholds=(38, 40, 41, 45),
        custom_policy_threshold=40,
    )

    assert summary["baseline_avg_ms"] == 1.6
    assert summary["custom_avg_ms"] == 1.32
    assert summary["improvement_percent"] == 17.5
    assert summary["fanout_bucket_summary"]["40-44"]["coverage_percent"] == 30.0
    assert summary["fanout_bucket_summary"]["40-44"]["win_coverage_percent"] == 30.0

    gate40 = summary["oracle_gates"]["fanout>=40"]
    assert gate40["custom_coverage_percent"] == 90.0
    assert gate40["improvement_percent"] == 18.75
    assert gate40["candidate"] == "gmm1_custom_fanout_ge_40"

    gate45 = summary["oracle_gates"]["fanout>=45"]
    assert gate45["custom_coverage_percent"] == 60.0
    assert gate45["improvement_percent"] == 15.0


def test_summarize_stage_files_keeps_gmm2_reverse_analysis_conservative(tmp_path):
    gmm2_path = tmp_path / "microbench_gmm2.json"
    gmm2_path.write_text(
        json.dumps(
            {
                "mode": "gmm2",
                "results": [
                    {
                        "class_id": 1,
                        "fanout": 40,
                        "coverage_percent": 50.0,
                        "variants": [
                            {"variant": "counts:s2", "backend": "torch_npu", "ms": {"avg": 1.0, "p50": 0.9, "p99": 1.5}},
                            {"variant": "counts:s2:custom", "backend": "custom", "ms": {"avg": 1.1, "p50": 1.0, "p99": 1.6}},
                        ],
                    },
                    {
                        "class_id": 2,
                        "fanout": 46,
                        "coverage_percent": 50.0,
                        "variants": [
                            {"variant": "counts:s2", "backend": "torch_npu", "ms": {"avg": 1.0, "p50": 0.9, "p99": 1.5}},
                            {"variant": "counts:s2:custom", "backend": "custom", "ms": {"avg": 0.995, "p50": 0.9, "p99": 1.4}},
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = summarize_stage_files({"gmm2_default": gmm2_path})

    assert report["runtime_policy"] == "do_not_enable_custom_gmm2_or_runtime_gating_in_this_phase"
    assert report["cases"]["gmm2_default"]["oracle_gates"]["fanout>=45"]["improvement_percent"] == 0.25
    assert report["gmm2_reverse_analysis"]["status"] == "needs_profiler_evidence"
    assert "no meaningful oracle gain" in report["gmm2_reverse_analysis"]["conclusion"]
    assert any("--mode gmm2" in command for command in report["gmm2_reverse_analysis"]["profiler_commands"])

    markdown = render_markdown(report)
    assert "## GMM2 Reverse Analysis" in markdown
    assert "benchmark_grouped_matmul.py" in markdown


def test_summarize_stage_files_emits_conservative_runtime_recommendation(tmp_path):
    gmm1_path = tmp_path / "microbench_gmm1.json"
    gmm1_path.write_text(
        json.dumps(
            {
                "mode": "gmm1",
                "results": [
                    {
                        "class_id": 1,
                        "fanout": 39,
                        "coverage_percent": 25.0,
                        "variants": [
                            {"variant": "counts:s2", "backend": "torch_npu", "ms": {"avg": 1.0, "p50": 0.9, "p99": 1.2}},
                            {"variant": "counts:s2:custom", "backend": "custom", "ms": {"avg": 1.2, "p50": 1.1, "p99": 1.4}},
                        ],
                    },
                    {
                        "class_id": 2,
                        "fanout": 46,
                        "coverage_percent": 75.0,
                        "variants": [
                            {"variant": "counts:s2", "backend": "torch_npu", "ms": {"avg": 1.0, "p50": 0.9, "p99": 1.2}},
                            {"variant": "counts:s2:custom", "backend": "custom", "ms": {"avg": 0.8, "p50": 0.7, "p99": 1.0}},
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    gmm2_path = tmp_path / "microbench_gmm2.json"
    gmm2_path.write_text(
        json.dumps(
            {
                "mode": "gmm2",
                "results": [
                    {
                        "class_id": 1,
                        "fanout": 46,
                        "coverage_percent": 100.0,
                        "variants": [
                            {"variant": "counts:s2", "backend": "torch_npu", "ms": {"avg": 1.0, "p50": 0.9, "p99": 1.2}},
                            {"variant": "counts:s2:custom", "backend": "custom", "ms": {"avg": 1.1, "p50": 1.0, "p99": 1.3}},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = summarize_stage_files(
        {
            "gmm1_default": gmm1_path,
            "gmm2_default": gmm2_path,
        },
        gmm1_policy_threshold=40,
    )

    recommendation = report["runtime_recommendation"]

    assert recommendation["safe_to_enable_runtime"] is False
    assert recommendation["stages"]["gmm1_default"]["backend"] == "custom"
    assert recommendation["stages"]["gmm1_default"]["fanout_threshold"] == 40
    assert recommendation["stages"]["gmm2_default"]["backend"] == "torch_npu"
    assert recommendation["stages"]["gmm2_default"]["fanout_threshold"] is None
    assert "reverse_analysis_required" in recommendation["stages"]["gmm2_default"]["reason"]

    # P3: fanout-level risk assessment
    assert recommendation["stages"]["gmm1_default"]["gmm1_risky_fanouts"] == [39]
    assert recommendation["stages"]["gmm1_default"]["gmm1_safe_fanouts"] == [46]


def test_summarize_stage_files_emits_gmm1_profiler_diagnostics(tmp_path):
    gmm1_path = tmp_path / "microbench_gmm1.json"
    gmm1_path.write_text(
        json.dumps(
            {
                "mode": "gmm1",
                "results": [
                    {
                        "class_id": 1,
                        "fanout": 35,
                        "coverage_percent": 5.0,
                        "variants": [
                            {"variant": "counts:s2", "backend": "torch_npu", "ms": {"avg": 1.0, "p50": 0.9, "p99": 1.2}},
                            {"variant": "counts:s2:custom", "backend": "custom", "ms": {"avg": 1.3, "p50": 1.2, "p99": 1.6}},
                        ],
                    },
                    {
                        "class_id": 2,
                        "fanout": 40,
                        "coverage_percent": 10.0,
                        "variants": [
                            {"variant": "counts:s2", "backend": "torch_npu", "ms": {"avg": 1.0, "p50": 0.9, "p99": 1.2}},
                            {"variant": "counts:s2:custom", "backend": "custom", "ms": {"avg": 0.7, "p50": 0.6, "p99": 0.9}},
                        ],
                    },
                    {
                        "class_id": 3,
                        "fanout": 44,
                        "coverage_percent": 7.0,
                        "variants": [
                            {"variant": "counts:s2", "backend": "torch_npu", "ms": {"avg": 1.0, "p50": 0.9, "p99": 1.2}},
                            {"variant": "counts:s2:custom", "backend": "custom", "ms": {"avg": 0.99, "p50": 0.9, "p99": 1.2}},
                        ],
                    },
                    {
                        "class_id": 4,
                        "fanout": 53,
                        "coverage_percent": 8.0,
                        "variants": [
                            {"variant": "counts:s2", "backend": "torch_npu", "ms": {"avg": 2.0, "p50": 1.8, "p99": 2.5}},
                            {"variant": "counts:s2:custom", "backend": "custom", "ms": {"avg": 1.5, "p50": 1.4, "p99": 2.0}},
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = summarize_stage_files({"gmm1_default": gmm1_path})

    diagnostics = report["gmm1_profiler_diagnostics"]

    assert diagnostics["status"] == "profile_before_kernel_changes"
    assert [item["fanout"] for item in diagnostics["winning_fanouts"][:2]] == [40, 53]
    assert diagnostics["losing_fanouts"][0]["fanout"] == 35
    assert diagnostics["neutral_fanouts"][0]["fanout"] == 44
    assert any("--mode gmm1" in command for command in diagnostics["profiler_commands"])
    assert any("--fanouts 35,40,44,53" in command for command in diagnostics["profiler_commands"])
    assert any("--variants counts:s2 " in command for command in diagnostics["profiler_commands"])
    assert any("--variants counts:s2:custom" in command for command in diagnostics["profiler_commands"])

    markdown = render_markdown(report)
    assert "## GMM1 Profiler Diagnostics" in markdown
    assert "fanout 40" in markdown
