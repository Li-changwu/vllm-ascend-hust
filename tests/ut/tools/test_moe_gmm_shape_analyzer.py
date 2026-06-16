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

import json

from tools.moe_gmm.analyze_gmm_shapes import analyze_trace_file, write_reports


def _write_trace(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_analyzer_reports_top_signatures_and_inactive_expert_overhead(tmp_path):
    trace_path = tmp_path / "gmm_trace.jsonl"
    records = [
        {
            "layer_id": 0,
            "step_id": step,
            "mode": "decode",
            "source": "grouped_dispatch",
            "num_tokens": 4,
            "fanout": 2,
            "active_experts": [0, 2],
            "expert_token_counts": {"0": 2, "2": 2},
            "group_list_type": 1,
            "group_list_signature": "counts:2,0,2,0",
            "physical_expert_count": 4,
        }
        for step in range(8)
    ]
    records.extend(
        {
            "layer_id": 1,
            "step_id": step,
            "mode": "decode",
            "source": "grouped_dispatch",
            "num_tokens": 4,
            "fanout": 2,
            "active_experts": [1, 3],
            "expert_token_counts": {"1": 1, "3": 3},
            "group_list_type": 1,
            "group_list_signature": "counts:0,1,0,3",
            "physical_expert_count": 4,
        }
        for step in range(2)
    )
    _write_trace(trace_path, records)

    report = analyze_trace_file(trace_path, top_k=20)

    assert report["record_count"] == 10
    assert report["top20_coverage_percent"] == 100.0
    assert report["classification"]["primary"] == "inactive_expert_overhead"
    assert report["fanout"]["avg_active_experts"] == 2.0
    assert report["fanout"]["avg_active_ratio"] == 0.5
    assert report["top_signatures"][0]["signature"] == "counts:2,0,2,0"
    assert report["top_signatures"][0]["sample_count"] == 8
    assert report["bucket_plan"]["buckets"][0]["active_expert_ids"] == [0, 2]
    assert report["bucket_plan"]["buckets"][0]["compact_group_list"] == [2, 2]


def test_write_reports_creates_json_markdown_and_bucket_plan(tmp_path):
    trace_path = tmp_path / "gmm_trace.jsonl"
    json_output = tmp_path / "gmm_shape_report.json"
    markdown_output = tmp_path / "gmm_shape_report.md"
    plan_output = tmp_path / "gmm_bucket_plan.json"
    _write_trace(
        trace_path,
        [
            {
                "layer_id": 0,
                "step_id": 0,
                "mode": "decode",
                "source": "grouped_dispatch",
                "num_tokens": 4,
                "fanout": 4,
                "active_experts": [0, 1, 2, 3],
                "expert_token_counts": {"0": 1, "1": 1, "2": 1, "3": 1},
                "group_list_type": 1,
                "group_list_signature": "counts:1,1,1,1",
                "physical_expert_count": 4,
            }
        ],
    )

    report = analyze_trace_file(trace_path)
    write_reports(
        report,
        json_output=json_output,
        markdown_output=markdown_output,
        bucket_plan_output=plan_output,
    )

    assert json.loads(json_output.read_text(encoding="utf-8"))["record_count"] == 1
    assert "GroupedMatmul Shape Report" in markdown_output.read_text(encoding="utf-8")
    assert json.loads(plan_output.read_text(encoding="utf-8"))["buckets"][0]["signature"] == "counts:1,1,1,1"
