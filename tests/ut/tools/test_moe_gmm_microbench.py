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
import sys
from types import SimpleNamespace

import torch

from tools.moe_gmm.benchmark_grouped_matmul import (
    _build_task_model,
    _counts_to_cumsum,
    _parse_full_chain_variant,
    _parse_shape_variant,
    load_shape_classes,
)


def _write_trace(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_load_shape_classes_groups_by_token_and_fanout(tmp_path):
    trace_path = tmp_path / "gmm_trace.jsonl"
    records = []
    records.extend(
        {
            "source": "grouped_dispatch",
            "layer_id": step % 2,
            "step_id": step,
            "num_tokens": 80,
            "fanout": 45,
            "group_list_type": 1,
            "group_list_signature": "counts:2,0,1,0,77",
        }
        for step in range(9)
    )
    records.extend(
        {
            "source": "grouped_dispatch",
            "layer_id": 3,
            "step_id": step,
            "num_tokens": 80,
            "fanout": 46,
            "group_list_type": 1,
            "group_list_signature": "counts:1,1,0,78,0",
        }
        for step in range(3)
    )
    records.append(
        {
            "source": "grouped_dispatch",
            "layer_id": 4,
            "step_id": 99,
            "num_tokens": 4096,
            "fanout": 128,
            "group_list_type": 1,
            "group_list_signature": "counts:1024,1024,1024,1024",
        }
    )
    _write_trace(trace_path, records)

    classes = load_shape_classes(trace_path, top_k=2)

    assert [case["class_id"] for case in classes] == [1, 2]
    assert classes[0]["num_tokens"] == 80
    assert classes[0]["fanout"] == 45
    assert classes[0]["sample_count"] == 9
    assert classes[0]["coverage_percent"] == 69.2308
    assert classes[0]["sample_signature"] == "counts:2,0,1,0,77"
    assert classes[0]["group_counts"] == [2, 0, 1, 0, 77]
    assert classes[0]["group_cumsum"] == [2, 2, 3, 3, 80]


def test_load_shape_classes_filters_by_fanout(tmp_path):
    trace_path = tmp_path / "gmm_trace.jsonl"
    _write_trace(
        trace_path,
        [
            {
                "source": "grouped_dispatch",
                "layer_id": 0,
                "step_id": 0,
                "num_tokens": 80,
                "fanout": 40,
                "group_list_type": 1,
                "group_list_signature": "counts:1,0,79",
            },
            {
                "source": "grouped_dispatch",
                "layer_id": 0,
                "step_id": 1,
                "num_tokens": 80,
                "fanout": 41,
                "group_list_type": 1,
                "group_list_signature": "counts:2,0,78",
            },
            {
                "source": "grouped_dispatch",
                "layer_id": 0,
                "step_id": 2,
                "num_tokens": 80,
                "fanout": 35,
                "group_list_type": 1,
                "group_list_signature": "counts:3,0,77",
            },
        ],
    )

    classes = load_shape_classes(trace_path, top_k=10, fanouts={40, 41})

    assert [case["fanout"] for case in classes] == [40, 41]

    assert load_shape_classes(trace_path, top_k=10, fanouts={54}) == []


def test_counts_to_cumsum_keeps_device_and_dtype():
    counts = torch.tensor([2, 0, 3], dtype=torch.int64)

    cumsum = _counts_to_cumsum(counts)

    assert torch.equal(cumsum, torch.tensor([2, 2, 5], dtype=torch.int64))
    assert cumsum.dtype == counts.dtype
    assert cumsum.device == counts.device


def test_run_shape_class_microbench_exposes_task_model_knobs(monkeypatch):
    from tools.moe_gmm import benchmark_grouped_matmul as bench

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(bench, "_bench_shape_class", lambda **kwargs: {"task_single_m": kwargs["task_single_m"]})

    report = bench.run_shape_class_microbench(
        shape_classes=[{"class_id": 1}],
        hidden_size=2048,
        intermediate_size=768,
        warmup=0,
        iters=0,
        mode="gmm1",
        device="npu:0",
        task_single_m=32,
        task_single_n=256,
    )

    assert report["task_model_single_m"] == 32
    assert report["task_model_single_n"] == 256
    assert report["results"] == [{"task_single_m": 32}]


def test_build_task_model_estimates_active_expert_block_work():
    model = _build_task_model(
        group_counts=[2, 0, 17, 1],
        single_m=16,
        single_n=256,
        output_n=1536,
    )

    assert model == {
        "total_tokens": 20,
        "fanout": 3,
        "physical_expert_count": 4,
        "active_token_histogram": {"1": 1, "2": 1, "17": 1},
        "max_tokens_per_expert": 17,
        "single_m": 16,
        "single_n": 256,
        "n_block_count": 6,
        "estimated_m_block_count": 4,
        "estimated_task_count": 24,
    }


def test_parse_shape_variant_describes_group_list_and_split_item():
    variant = _parse_shape_variant("cumsum:s3:custom")

    assert variant["name"] == "cumsum:s3:custom"
    assert variant["group_list"] == "cumsum"
    assert variant["group_list_type"] == 2
    assert variant["split_item"] == 3
    assert variant["backend"] == "custom"


def test_parse_shape_variant_rejects_unknown_parts():
    try:
        _parse_shape_variant("counts:s9")
    except ValueError as exc:
        assert "unsupported split_item" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_parse_full_chain_variant_allows_separate_gmm_settings():
    variant = _parse_full_chain_variant("g1=counts:s2,g2=cumsum:s3")

    assert variant["name"] == "g1=counts:s2,g2=cumsum:s3"
    assert variant["gmm1"]["group_list_type"] == 1
    assert variant["gmm1"]["split_item"] == 2
    assert variant["gmm2"]["group_list_type"] == 0
    assert variant["gmm2"]["split_item"] == 3
