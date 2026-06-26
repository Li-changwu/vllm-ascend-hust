# SPDX-License-Identifier: Apache-2.0
"""Compare two SEW-MoE offload timing profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.sew_offload.moe_offload_timeline import read_jsonl, summarize_profile


PIPELINE_KEYS = ("stage_t_ms", "stage_r_ms", "stage_c_ms", "stage_m_ms")


def compare_profiles(
    *,
    baseline_path: str | Path,
    candidate_path: str | Path,
    baseline_label: str,
    candidate_label: str,
) -> dict[str, Any]:
    baseline_records = read_jsonl(baseline_path)
    candidate_records = read_jsonl(candidate_path)
    baseline = _profile_summary(baseline_records, baseline_label, baseline_path)
    candidate = _profile_summary(candidate_records, candidate_label, candidate_path)
    return {
        "baseline": baseline,
        "candidate": candidate,
        "deltas": _build_deltas(baseline, candidate),
    }


def render_markdown(comparison: dict[str, Any]) -> str:
    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    deltas = comparison["deltas"]

    lines = [
        "# MoE Offload Timeline Comparison",
        "",
        f"Baseline: `{baseline['label']}`",
        "",
        f"Candidate: `{candidate['label']}`",
        "",
        "## Profile Scope",
        "",
        "| Metric | Baseline | Candidate |",
        "|---|---:|---:|",
        f"| Pipeline records | {baseline['record_counts']['pipeline']} | {candidate['record_counts']['pipeline']} |",
        f"| Offload events | {baseline['record_counts']['offload_timeline']} | {candidate['record_counts']['offload_timeline']} |",
        f"| Cache hit rate | {_pct(baseline['cache']['hit_rate'])} | {_pct(candidate['cache']['hit_rate'])} |",
        f"| H2D bytes | {_human_bytes(baseline['h2d_bytes'])} | {_human_bytes(candidate['h2d_bytes'])} |",
        "",
        "## Pipeline Mean",
        "",
        "| Stage | Baseline ms | Candidate ms | Delta |",
        "|---|---:|---:|---:|",
    ]

    for key in PIPELINE_KEYS:
        baseline_ms = baseline["pipeline_mean_ms"].get(key, 0.0)
        candidate_ms = candidate["pipeline_mean_ms"].get(key, 0.0)
        lines.append(
            f"| {key} | {baseline_ms:.4f} | {candidate_ms:.4f} | {_delta_pct(baseline_ms, candidate_ms)} |"
        )

    lines.extend([
        "",
        "## H2D Transfer",
        "",
        "| Metric | Baseline | Candidate |",
        "|---|---:|---:|",
        f"| Transfer mode | {baseline['h2d']['mode']} | {candidate['h2d']['mode']} |",
        f"| Event count | {baseline['h2d']['event_count']} | {candidate['h2d']['event_count']} |",
        f"| Expert count | {baseline['h2d']['expert_count']} | {candidate['h2d']['expert_count']} |",
        f"| Mean batch size | {baseline['h2d']['mean_batch_size']:.2f} | {candidate['h2d']['mean_batch_size']:.2f} |",
        f"| Total transfer event ms | {baseline['h2d']['total_ms']:.4f} | {candidate['h2d']['total_ms']:.4f} |",
        f"| Mean ms / event | {baseline['h2d']['mean_event_ms']:.4f} | {candidate['h2d']['mean_event_ms']:.4f} |",
        f"| Mean ms / expert | {baseline['h2d']['mean_ms_per_expert']:.4f} | {candidate['h2d']['mean_ms_per_expert']:.4f} |",
        f"| Async batch events | {baseline['h2d']['async_event_count']} | {candidate['h2d']['async_event_count']} |",
        "",
        "## Main Deltas",
        "",
        f"- H2D mean ms / expert: {deltas['h2d_mean_ms_per_expert_pct']}",
        f"- H2D total event ms: {deltas['h2d_total_ms_pct']}",
        f"- Pipeline T mean: {deltas['pipeline_stage_t_ms_pct']}",
    ])
    note = candidate.get("note")
    if note:
        lines.extend(["", "## Note", "", note])
    return "\n".join(lines) + "\n"


def _profile_summary(
    records: list[dict[str, Any]],
    label: str,
    path: str | Path,
) -> dict[str, Any]:
    summary = summarize_profile(records)
    return {
        "label": label,
        "path": str(path),
        "record_counts": summary["record_counts"],
        "pipeline_mean_ms": _pipeline_means(summary),
        "cache": summary["offload_timeline"].get("cache", {}),
        "h2d_bytes": int(summary["offload_timeline"].get("h2d_bytes") or 0),
        "h2d": _h2d_summary(records),
    }


def _pipeline_means(summary: dict[str, Any]) -> dict[str, float]:
    means: dict[str, float] = {}
    for row in summary.get("pipeline", {}).get("stages", []):
        key = str(row.get("key") or "")
        if key:
            means[key] = _float(row.get("mean_ms"))
    return means


def _h2d_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    batch_events: list[dict[str, Any]] = []
    per_expert_events: list[dict[str, Any]] = []
    for record in records:
        if record.get("event") != "moe_offload_timeline":
            continue
        name = record.get("name")
        payload = record.get("payload") or {}
        if name == "expert_h2d_batch_load_sync":
            batch_events.append(record)
        elif name == "expert_h2d_load_sync" and not payload.get("batched"):
            per_expert_events.append(record)

    source_events = batch_events or per_expert_events
    durations_ms = [_duration_ms(record) for record in source_events]
    batch_sizes = [_batch_size(record) for record in source_events]
    expert_count = sum(batch_sizes)
    total_ms = sum(durations_ms)
    total_bytes = sum(_payload_int(record, "bytes") for record in source_events)
    async_event_count = sum(1 for record in source_events if (record.get("payload") or {}).get("async"))
    mean_event_ms = total_ms / len(source_events) if source_events else 0.0
    mean_batch_size = expert_count / len(source_events) if source_events else 0.0
    return {
        "mode": "batch" if batch_events else "per_expert",
        "event_count": len(source_events),
        "expert_count": expert_count,
        "mean_batch_size": mean_batch_size,
        "total_ms": total_ms,
        "mean_event_ms": mean_event_ms,
        "mean_ms_per_expert": total_ms / expert_count if expert_count else 0.0,
        "total_bytes": total_bytes,
        "async_event_count": async_event_count,
    }


def _build_deltas(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, str]:
    baseline_h2d = baseline["h2d"]
    candidate_h2d = candidate["h2d"]
    baseline_t = baseline["pipeline_mean_ms"].get("stage_t_ms", 0.0)
    candidate_t = candidate["pipeline_mean_ms"].get("stage_t_ms", 0.0)
    return {
        "h2d_mean_ms_per_expert_pct": _delta_pct(
            baseline_h2d["mean_ms_per_expert"],
            candidate_h2d["mean_ms_per_expert"],
        ),
        "h2d_total_ms_pct": _delta_pct(baseline_h2d["total_ms"], candidate_h2d["total_ms"]),
        "pipeline_stage_t_ms_pct": _delta_pct(baseline_t, candidate_t),
    }


def _batch_size(record: dict[str, Any]) -> int:
    payload = record.get("payload") or {}
    if "batch_size" in payload:
        return max(1, _int(payload.get("batch_size")))
    expert_ids = payload.get("expert_ids")
    if isinstance(expert_ids, list):
        return max(1, len(expert_ids))
    return 1


def _duration_ms(record: dict[str, Any]) -> float:
    if "duration_ms" in record:
        return _float(record.get("duration_ms"))
    return max(0.0, _float(record.get("end_ns")) - _float(record.get("start_ns"))) / 1_000_000.0


def _payload_int(record: dict[str, Any], key: str) -> int:
    payload = record.get("payload") or {}
    return _int(payload.get(key))


def _float(value: Any) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def _delta_pct(baseline: float, candidate: float) -> str:
    if baseline == 0:
        return "n/a"
    delta = (candidate - baseline) / baseline
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.2%}"


def _pct(value: Any) -> str:
    return f"{_float(value):.2%}"


def _human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024.0 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TiB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-profile", required=True)
    parser.add_argument("--candidate-profile", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument("--candidate-note", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = compare_profiles(
        baseline_path=args.baseline_profile,
        candidate_path=args.candidate_profile,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
    )
    if args.candidate_note:
        comparison["candidate"]["note"] = args.candidate_note
    Path(args.json_output).write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    Path(args.markdown_output).write_text(render_markdown(comparison), encoding="utf-8")
    print("MOE_OFFLOAD_TIMELINE_COMPARISON " + json.dumps(comparison["deltas"]), flush=True)


if __name__ == "__main__":
    main()
