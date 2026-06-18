# SPDX-License-Identifier: Apache-2.0

import json

from tools.sew_offload.moe_offload_timeline import (
    read_jsonl,
    render_chrome_trace,
    render_decode_layer_svg,
    render_markdown,
    render_svg,
    summarize_profile,
)


def test_moe_offload_timeline_summarizes_and_renders_views(tmp_path):
    profile = tmp_path / "sew_moe_profile.jsonl"
    records = [
        {
            "event": "moe_offload_timeline",
            "name": "slot_cache_lookup",
            "layer_id": 1,
            "step_id": 7,
            "start_ns": 1_000,
            "end_ns": 3_000,
            "duration_us": 2.0,
            "payload": {"expert_id": 2, "cache_hit": False},
        },
        {
            "event": "moe_offload_timeline",
            "name": "expert_h2d_load_sync",
            "layer_id": 1,
            "step_id": 7,
            "start_ns": 3_000,
            "end_ns": 103_000,
            "duration_us": 100.0,
            "payload": {"expert_id": 2, "slot_id": 0, "bytes": 4096},
        },
        {
            "event": "moe_offload_timeline",
            "name": "prepare_fixed_slot_plan",
            "layer_id": 1,
            "step_id": 7,
            "start_ns": 1_000,
            "end_ns": 120_000,
            "duration_us": 119.0,
            "payload": {"active_slot_ids": [0]},
        },
        {
            "event": "moe_pipeline_timing",
            "layer_id": 1,
            "step_id": 7,
            "stage_t_ms": 0.2,
            "stage_r_ms": 0.3,
            "stage_c_ms": 1.5,
            "stage_m_ms": 0.4,
        },
        {
            "event": "moe_pipeline_detail_timing",
            "layer_id": 1,
            "step_id": 7,
            "name": "r_init_routing",
            "duration_ms": 0.11,
            "source": "npu_event",
        },
    ]
    profile.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    loaded = read_jsonl(profile)
    summary = summarize_profile(loaded)
    markdown = render_markdown(loaded)
    svg = render_svg(loaded)
    chrome_trace = render_chrome_trace(loaded)

    assert summary["representative_step_id"] == 7
    assert summary["pipeline"]["total_ms"]["mean_ms"] == 2.4
    assert summary["pipeline_detail"]["record_count"] == 1
    assert summary["offload_timeline"]["cache"]["misses"] == 1
    assert summary["offload_timeline"]["h2d_bytes"] == 4096
    assert "```mermaid" in markdown
    assert "expert H2D load e2" in markdown
    assert "<svg" in svg
    assert "SEW-MoE Offload Timing Overview" in svg
    assert "expert H2D load" in svg
    assert any(event.get("name") == "expert_h2d_load_sync" for event in chrome_trace["traceEvents"])
    assert any(event.get("name") == "C expert MLP" for event in chrome_trace["traceEvents"])
    assert any(event.get("name") == "R init routing op" for event in chrome_trace["traceEvents"])


def test_moe_offload_timeline_renders_decode_layer_svg():
    records = []
    for step_id, layer_id in enumerate([0, 1, 0, 1]):
        records.append(
            {
                "event": "moe_pipeline_timing",
                "layer_id": layer_id,
                "step_id": step_id,
                "stage_t_ms": 1.0 + step_id,
                "stage_r_ms": 0.2,
                "stage_c_ms": 0.3,
                "stage_m_ms": 0.1,
            }
        )
        records.append(
            {
                "event": "moe_offload_profile",
                "name": "layered_path_decision",
                "layer_id": layer_id,
                "payload": {
                    "path": "slot_cache_path",
                    "step_id": step_id,
                },
            }
        )
    records.extend(
        [
            {
                "event": "moe_offload_timeline",
                "name": "slot_cache_lookup",
                "layer_id": 0,
                "step_id": 2,
                "duration_us": 20.0,
                "payload": {"cache_hit": True},
            },
            {
                "event": "moe_offload_timeline",
                "name": "expert_h2d_load_sync",
                "layer_id": 0,
                "step_id": 2,
                "duration_us": 100.0,
                "payload": {"bytes": 4096},
            },
            {
                "event": "moe_offload_timeline",
                "name": "prepare_fixed_slot_plan",
                "layer_id": 0,
                "step_id": 2,
                "duration_us": 200.0,
            },
            {
                "event": "moe_pipeline_detail_timing",
                "layer_id": 0,
                "step_id": 2,
                "name": "r_log2phy_map",
                "duration_ms": 0.03,
            },
            {
                "event": "moe_pipeline_detail_timing",
                "layer_id": 0,
                "step_id": 2,
                "name": "r_init_routing",
                "duration_ms": 0.04,
            },
        ]
    )

    svg = render_decode_layer_svg(records, layers_per_wave=2)

    assert "Decode Wave Layer Pipeline" in svg
    assert "wave=1" in svg
    assert "steps=2..3" in svg
    assert "L00" in svg
    assert "L01" in svg
    assert "slot" in svg
    assert "H2D load detail" in svg
    assert "R init routing" in svg
    assert "R residual/wait" in svg
