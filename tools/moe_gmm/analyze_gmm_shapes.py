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

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


def analyze_trace_file(path: str | Path, *, top_k: int = 20) -> dict[str, Any]:
    records = _load_grouped_records(path)
    record_count = len(records)
    signature_counts = Counter(str(record.get("group_list_signature", "")) for record in records)
    signature_counts.pop("", None)
    top_signatures = _top_signatures(signature_counts, records, record_count, top_k=top_k)
    top20_coverage = _coverage_percent(top_signatures[:20], record_count)
    fanout = _fanout_summary(records)
    layers = _layer_summary(records)
    classification = _classify(top20_coverage_percent=top20_coverage, fanout=fanout)
    bucket_plan = _bucket_plan(top_signatures, coverage_percent=top20_coverage)

    return {
        "record_count": record_count,
        "top_k": int(top_k),
        "top_signatures": top_signatures,
        "top20_coverage_percent": top20_coverage,
        "fanout": fanout,
        "layers": layers,
        "classification": classification,
        "bucket_plan": bucket_plan,
    }


def write_reports(
    report: dict[str, Any],
    *,
    json_output: str | Path,
    markdown_output: str | Path,
    bucket_plan_output: str | Path | None = None,
) -> None:
    json_path = Path(json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    markdown_path = Path(markdown_output)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")

    if bucket_plan_output is not None:
        plan_path = Path(bucket_plan_output)
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(
            json.dumps(report["bucket_plan"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _load_grouped_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if record.get("source") != "grouped_dispatch":
                continue
            if not record.get("group_list_signature"):
                continue
            records.append(record)
    return records


def _top_signatures(
    signature_counts: Counter[str],
    records: list[dict[str, Any]],
    record_count: int,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    records_by_signature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_signature[str(record["group_list_signature"])].append(record)

    top_items: list[dict[str, Any]] = []
    for rank, (signature, sample_count) in enumerate(signature_counts.most_common(top_k), start=1):
        sample = records_by_signature[signature][0]
        active_expert_ids, compact_group_list, original_expert_count = _active_plan_from_signature(signature)
        top_items.append(
            {
                "rank": rank,
                "signature": signature,
                "sample_count": int(sample_count),
                "coverage_percent": _round_percent(sample_count, record_count),
                "group_list_type": sample.get("group_list_type"),
                "mode": sample.get("mode", "unknown"),
                "physical_expert_count": _physical_expert_count(sample, original_expert_count),
                "active_expert_count": len(active_expert_ids),
                "active_expert_ids": list(active_expert_ids),
                "compact_group_list": list(compact_group_list),
            }
        )
    return top_items


def _fanout_summary(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {
            "avg_active_experts": 0.0,
            "max_active_experts": 0.0,
            "avg_physical_experts": 0.0,
            "avg_active_ratio": 0.0,
        }
    fanouts = [int(record.get("fanout", 0) or 0) for record in records]
    physical_counts = [
        int(record.get("physical_expert_count", 0) or _signature_expert_count(record.get("group_list_signature", "")))
        for record in records
    ]
    ratios = [
        fanout / physical
        for fanout, physical in zip(fanouts, physical_counts, strict=False)
        if physical > 0
    ]
    return {
        "avg_active_experts": round(sum(fanouts) / len(fanouts), 4),
        "max_active_experts": float(max(fanouts)),
        "avg_physical_experts": round(sum(physical_counts) / len(physical_counts), 4),
        "avg_active_ratio": round(sum(ratios) / len(ratios), 4) if ratios else 0.0,
    }


def _layer_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_layer[int(record.get("layer_id", -1))].append(record)

    layers: list[dict[str, Any]] = []
    for layer_id in sorted(by_layer):
        layer_records = by_layer[layer_id]
        signatures = Counter(str(record.get("group_list_signature", "")) for record in layer_records)
        fanouts = [int(record.get("fanout", 0) or 0) for record in layer_records]
        layers.append(
            {
                "layer_id": layer_id,
                "record_count": len(layer_records),
                "unique_signature_count": len(signatures),
                "top_signature": signatures.most_common(1)[0][0] if signatures else "",
                "avg_active_experts": round(sum(fanouts) / len(fanouts), 4) if fanouts else 0.0,
            }
        )
    return layers


def _classify(*, top20_coverage_percent: float, fanout: dict[str, float]) -> dict[str, Any]:
    avg_active_ratio = float(fanout.get("avg_active_ratio", 0.0))
    if top20_coverage_percent < 60.0:
        primary = "shape_fragmentation"
        reason = "top20_signature_coverage_below_60_percent"
    elif avg_active_ratio > 0.0 and avg_active_ratio <= 0.75:
        primary = "inactive_expert_overhead"
        reason = "active_expert_count_is_much_smaller_than_physical_expert_count"
    else:
        primary = "kernel_compute_bound"
        reason = "signatures_are_stable_but_active_expert_compaction_is_not_promising"
    return {
        "primary": primary,
        "reason": reason,
        "top20_coverage_percent": top20_coverage_percent,
        "avg_active_ratio": avg_active_ratio,
    }


def _bucket_plan(top_signatures: list[dict[str, Any]], *, coverage_percent: float) -> dict[str, Any]:
    buckets: list[dict[str, Any]] = []
    for item in top_signatures:
        buckets.append(
            {
                "bucket_id": int(item["rank"]),
                "signature": item["signature"],
                "sample_count": int(item["sample_count"]),
                "coverage_percent": float(item["coverage_percent"]),
                "active_expert_ids": item["active_expert_ids"],
                "compact_group_list": item["compact_group_list"],
                "original_expert_count": int(item["physical_expert_count"]),
                "compact_expert_count": int(item["active_expert_count"]),
            }
        )
    return {
        "target": "gmm_active_expert_compaction",
        "phase": "mixed",
        "coverage_percent": float(coverage_percent),
        "fallback_percent": round(100.0 - float(coverage_percent), 4),
        "buckets": buckets,
    }


def _active_plan_from_signature(signature: str) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    prefix, separator, payload = str(signature).partition(":")
    if separator != ":":
        return (), (), 0
    values = tuple(_safe_int(value) for value in payload.split(",") if value.strip())
    if prefix == "counts":
        return (
            tuple(index for index, count in enumerate(values) if count > 0),
            tuple(count for count in values if count > 0),
            len(values),
        )
    if prefix == "cumsum":
        active_ids: list[int] = []
        compact: list[int] = []
        previous = 0
        for index, cumulative in enumerate(values):
            if cumulative > previous:
                active_ids.append(index)
                compact.append(cumulative)
            previous = cumulative
        return tuple(active_ids), tuple(compact), len(values)
    return (), (), len(values)


def _physical_expert_count(record: dict[str, Any], original_expert_count: int) -> int:
    physical = int(record.get("physical_expert_count", 0) or 0)
    if physical > 0:
        return physical
    return int(original_expert_count)


def _signature_expert_count(signature: str) -> int:
    _, separator, payload = str(signature).partition(":")
    if separator != ":":
        return 0
    return len([value for value in payload.split(",") if value.strip()])


def _coverage_percent(items: list[dict[str, Any]], record_count: int) -> float:
    if record_count <= 0:
        return 0.0
    return round(sum(int(item["sample_count"]) for item in items) * 100.0 / record_count, 4)


def _round_percent(part: int, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return round(float(part) * 100.0 / float(whole), 4)


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# GroupedMatmul Shape Report",
        "",
        f"- records: {report['record_count']}",
        f"- top20 coverage: {report['top20_coverage_percent']:.4f}%",
        f"- primary classification: {report['classification']['primary']}",
        f"- reason: {report['classification']['reason']}",
        f"- avg active experts: {report['fanout']['avg_active_experts']}",
        f"- avg active ratio: {report['fanout']['avg_active_ratio']}",
        "",
        "## Top Signatures",
        "",
        "| rank | samples | coverage | active/physical | signature |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for item in report["top_signatures"][:20]:
        lines.append(
            "| {rank} | {sample_count} | {coverage_percent:.4f}% | {active_expert_count}/{physical_expert_count} | `{signature}` |".format(
                **item
            )
        )
    lines.append("")
    lines.append("## Per Layer")
    lines.append("")
    lines.append("| layer | records | unique signatures | avg active | top signature |")
    lines.append("| ---: | ---: | ---: | ---: | --- |")
    for item in report["layers"]:
        lines.append(
            "| {layer_id} | {record_count} | {unique_signature_count} | {avg_active_experts} | `{top_signature}` |".format(
                **item
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MoE GroupedMatmul grouped-dispatch trace shapes.")
    parser.add_argument("--trace", required=True, help="Path to VLLM_ASCEND_MOE_GMM_TRACE_PATH JSONL.")
    parser.add_argument("--json-output", required=True, help="Path to write gmm_shape_report.json.")
    parser.add_argument("--markdown-output", required=True, help="Path to write gmm_shape_report.md.")
    parser.add_argument("--bucket-plan-output", default="", help="Optional path to write gmm_bucket_plan.json.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of signatures to include.")
    args = parser.parse_args()

    report = analyze_trace_file(args.trace, top_k=args.top_k)
    write_reports(
        report,
        json_output=args.json_output,
        markdown_output=args.markdown_output,
        bucket_plan_output=args.bucket_plan_output or None,
    )


if __name__ == "__main__":
    main()
