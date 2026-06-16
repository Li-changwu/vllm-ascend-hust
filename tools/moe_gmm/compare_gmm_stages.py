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

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_THRESHOLDS = (38, 40, 41, 45)
DEFAULT_BUCKETS = (
    (0, 37, "<38"),
    (38, 39, "38-39"),
    (40, 44, "40-44"),
    (45, 49, "45-49"),
    (50, 10_000, ">=50"),
)


def summarize_stage_case(
    case_name: str,
    microbench: dict[str, Any],
    *,
    thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS,
    custom_policy_threshold: int | None = None,
) -> dict[str, Any]:
    rows = [_row_from_result(result) for result in microbench.get("results", [])]
    weighted_baseline = _weighted_average(rows, "baseline_avg_ms")
    weighted_custom = _weighted_average(rows, "custom_avg_ms")
    winning_coverage = _coverage_sum(row for row in rows if row["improvement_percent"] > 0.0)
    losing_coverage = _coverage_sum(row for row in rows if row["improvement_percent"] < 0.0)
    threshold = custom_policy_threshold if custom_policy_threshold is not None else 40

    return {
        "case": case_name,
        "mode": microbench.get("mode", ""),
        "coverage_percent": round(_coverage_sum(rows), 4),
        "baseline_avg_ms": weighted_baseline,
        "custom_avg_ms": weighted_custom,
        "improvement_percent": _improvement_percent(weighted_baseline, weighted_custom),
        "winning_coverage_percent": round(winning_coverage, 4),
        "losing_coverage_percent": round(losing_coverage, 4),
        "fanout_bucket_summary": _fanout_bucket_summary(rows),
        "oracle_gates": {
            f"fanout>={fanout_threshold}": _oracle_gate(
                rows,
                threshold=fanout_threshold,
                candidate=_candidate_name(case_name, fanout_threshold),
            )
            for fanout_threshold in thresholds
        },
        "recommended_policy": {
            "name": _candidate_name(case_name, threshold),
            "fanout_threshold": threshold,
            "use_custom_when": f"fanout >= {threshold}",
            "do_not_enable_runtime": True,
        },
        "rows": rows,
    }


def summarize_stage_files(
    case_paths: dict[str, str | Path],
    *,
    thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS,
    gmm1_policy_threshold: int = 40,
) -> dict[str, Any]:
    cases = {}
    for case_name, path in case_paths.items():
        microbench = json.loads(Path(path).read_text(encoding="utf-8"))
        policy_threshold = gmm1_policy_threshold if str(case_name).startswith("gmm1") else None
        cases[case_name] = summarize_stage_case(
            case_name,
            microbench,
            thresholds=thresholds,
            custom_policy_threshold=policy_threshold,
        )
    return {
        "target": "moe_gmm_stage_compare",
        "oracle_gate_thresholds": list(thresholds),
        "runtime_policy": "do_not_enable_custom_gmm2_or_runtime_gating_in_this_phase",
        "cases": cases,
        "runtime_recommendation": _runtime_recommendation(cases),
        "gmm1_profiler_diagnostics": _gmm1_profiler_diagnostics(cases),
        "gmm2_reverse_analysis": _gmm2_reverse_analysis(cases),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MoE GMM Stage Comparison",
        "",
        f"- runtime policy: {report['runtime_policy']}",
        f"- oracle gates: {', '.join('fanout>=' + str(value) for value in report['oracle_gate_thresholds'])}",
        "",
        "## Case Summary",
        "",
        "| case | coverage% | baseline avg ms | custom avg ms | improvement% | win coverage% | lose coverage% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case_name, case in report["cases"].items():
        lines.append(
            f"| {case_name} | {case['coverage_percent']:.4f} | {case['baseline_avg_ms']:.4f} | "
            f"{case['custom_avg_ms']:.4f} | {case['improvement_percent']:.4f} | "
            f"{case['winning_coverage_percent']:.4f} | {case['losing_coverage_percent']:.4f} |"
        )

    recommendation = report["runtime_recommendation"]
    lines.extend(
        [
            "",
            "## Runtime Recommendation",
            "",
            f"- safe to enable automatically: `{recommendation['safe_to_enable_runtime']}`",
            f"- summary: {recommendation['summary']}",
            "",
            "| stage | backend | fanout threshold | candidate | reason |",
            "|---|---|---:|---|---|",
        ]
    )
    for stage_name, stage in recommendation["stages"].items():
        threshold = "" if stage["fanout_threshold"] is None else str(stage["fanout_threshold"])
        lines.append(
            f"| {stage_name} | `{stage['backend']}` | {threshold} | `{stage['candidate']}` | {stage['reason']} |"
        )

    gmm1 = report["gmm1_profiler_diagnostics"]
    lines.extend(
        [
            "",
            "## GMM1 Profiler Diagnostics",
            "",
            f"- status: {gmm1['status']}",
            f"- goal: {gmm1['goal']}",
            "",
            "| bucket | case | fanout | coverage% | improvement% | baseline avg ms | custom avg ms |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for bucket_name in ("winning_fanouts", "losing_fanouts", "neutral_fanouts"):
        for item in gmm1[bucket_name]:
            lines.append(
                f"| {bucket_name} | {item['case']} | fanout {item['fanout']} | "
                f"{item['coverage_percent']:.4f} | {item['improvement_percent']:.4f} | "
                f"{item['baseline_avg_ms']:.4f} | {item['custom_avg_ms']:.4f} |"
            )
    lines.extend(
        [
            "",
            "### Diagnostic Questions",
            "",
        ]
    )
    for question in gmm1["diagnostic_questions"]:
        lines.append(f"- {question}")
    lines.extend(["", "### Profiler Commands", "", "```bash", *gmm1["profiler_commands"], "```", ""])
    lines.extend(["", f"Note: {gmm1['profiler_note']}", ""])

    lines.extend(["", "## Oracle Gates", ""])
    for case_name, case in report["cases"].items():
        lines.extend(
            [
                f"### {case_name}",
                "",
                "| gate | custom coverage% | avg ms | improvement% | candidate |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for gate_name, gate in case["oracle_gates"].items():
            lines.append(
                "| {gate_name} | {custom_coverage_percent:.4f} | {avg_ms:.4f} | "
                "{improvement_percent:.4f} | `{candidate}` |".format(gate_name=gate_name, **gate)
            )
        lines.append("")

    gmm2 = report["gmm2_reverse_analysis"]
    lines.extend(
        [
            "## GMM2 Reverse Analysis",
            "",
            f"- status: {gmm2['status']}",
            f"- conclusion: {gmm2['conclusion']}",
            "",
            "### Profiler Commands",
            "",
            "```bash",
            *gmm2["profiler_commands"],
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(report: dict[str, Any], *, json_output: str | Path, markdown_output: str | Path) -> None:
    json_path = Path(json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = Path(markdown_output)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(report), encoding="utf-8")


def _row_from_result(result: dict[str, Any]) -> dict[str, Any]:
    variants = result.get("variants", [])
    if len(variants) < 2:
        raise ValueError(f"expected at least two variants for class_id={result.get('class_id')}")
    baseline = variants[0]
    custom = _custom_variant(variants)
    baseline_avg = float(baseline["ms"]["avg"])
    custom_avg = float(custom["ms"]["avg"])
    return {
        "class_id": int(result.get("class_id", 0)),
        "fanout": int(result.get("fanout", 0)),
        "coverage_percent": float(result.get("coverage_percent", 0.0)),
        "baseline_variant": str(baseline.get("variant", "")),
        "custom_variant": str(custom.get("variant", "")),
        "baseline_avg_ms": baseline_avg,
        "custom_avg_ms": custom_avg,
        "baseline_p50_ms": float(baseline["ms"].get("p50", 0.0)),
        "custom_p50_ms": float(custom["ms"].get("p50", 0.0)),
        "baseline_p99_ms": float(baseline["ms"].get("p99", 0.0)),
        "custom_p99_ms": float(custom["ms"].get("p99", 0.0)),
        "improvement_percent": _improvement_percent(baseline_avg, custom_avg),
        "task_model": result.get("task_model"),
    }


def _custom_variant(variants: list[dict[str, Any]]) -> dict[str, Any]:
    for variant in variants:
        if variant.get("backend") == "custom" or str(variant.get("variant", "")).endswith(":custom"):
            return variant
    return variants[1]


def _weighted_average(rows: list[dict[str, Any]], key: str) -> float:
    coverage = _coverage_sum(rows)
    if coverage <= 0.0:
        return 0.0
    return round(sum(float(row["coverage_percent"]) * float(row[key]) for row in rows) / coverage, 4)


def _coverage_sum(rows) -> float:
    return sum(float(row["coverage_percent"]) for row in rows)


def _improvement_percent(baseline: float, candidate: float) -> float:
    if baseline <= 0.0:
        return 0.0
    return round((baseline - candidate) * 100.0 / baseline, 4)


def _oracle_gate(rows: list[dict[str, Any]], *, threshold: int, candidate: str) -> dict[str, Any]:
    coverage = _coverage_sum(rows)
    if coverage <= 0.0:
        return {
            "custom_coverage_percent": 0.0,
            "avg_ms": 0.0,
            "improvement_percent": 0.0,
            "candidate": candidate,
        }
    baseline_avg = _weighted_average(rows, "baseline_avg_ms")
    gated_avg = round(
        sum(
            float(row["coverage_percent"])
            * (float(row["custom_avg_ms"]) if int(row["fanout"]) >= threshold else float(row["baseline_avg_ms"]))
            for row in rows
        )
        / coverage,
        4,
    )
    custom_coverage = _coverage_sum(row for row in rows if int(row["fanout"]) >= threshold)
    return {
        "custom_coverage_percent": round(custom_coverage, 4),
        "avg_ms": gated_avg,
        "improvement_percent": _improvement_percent(baseline_avg, gated_avg),
        "candidate": candidate,
    }


def _fanout_bucket_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for low, high, label in DEFAULT_BUCKETS:
        bucket_rows = [row for row in rows if low <= int(row["fanout"]) <= high]
        baseline = _weighted_average(bucket_rows, "baseline_avg_ms")
        custom = _weighted_average(bucket_rows, "custom_avg_ms")
        summary[label] = {
            "coverage_percent": round(_coverage_sum(bucket_rows), 4),
            "baseline_avg_ms": baseline,
            "custom_avg_ms": custom,
            "improvement_percent": _improvement_percent(baseline, custom),
            "win_coverage_percent": round(_coverage_sum(row for row in bucket_rows if row["improvement_percent"] > 0), 4),
            "lose_coverage_percent": round(_coverage_sum(row for row in bucket_rows if row["improvement_percent"] < 0), 4),
        }
    return summary


def _candidate_name(case_name: str, threshold: int) -> str:
    mode = "gmm1" if str(case_name).startswith("gmm1") else "gmm2"
    return f"{mode}_custom_fanout_ge_{threshold}"


def _runtime_recommendation(cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    stages: dict[str, dict[str, Any]] = {}
    for case_name, case in cases.items():
        recommended_policy = case["recommended_policy"]
        if str(case_name).startswith("gmm1"):
            # P3: extract fanout-level risk assessment
            losing_fanouts = _extract_risky_fanouts(case, threshold=-1.0, min_coverage=0.5)
            winning_fanouts = _extract_winning_fanouts(case, threshold=1.0, min_coverage=0.5)
            stages[case_name] = {
                "backend": "custom",
                "fanout_threshold": int(recommended_policy["fanout_threshold"]),
                "candidate": recommended_policy["name"],
                "reason": "gmm1_custom_has_positive_stage_signal_but_requires_full_chain_validation",
                "gmm1_risky_fanouts": losing_fanouts,
                "gmm1_safe_fanouts": winning_fanouts,
            }
        elif str(case_name).startswith("gmm2"):
            stages[case_name] = {
                "backend": "torch_npu",
                "fanout_threshold": None,
                "candidate": "torch_npu_grouped_matmul",
                "reason": "gmm2_custom_disabled_reverse_analysis_required",
            }
        else:
            stages[case_name] = {
                "backend": "torch_npu",
                "fanout_threshold": None,
                "candidate": "torch_npu_grouped_matmul",
                "reason": "unknown_stage_keep_baseline",
            }
    return {
        "safe_to_enable_runtime": False,
        "summary": "Keep runtime on torch_npu for GMM2; treat fanout-gated custom GMM1 as an experiment candidate only.",
        "stages": stages,
    }


def _extract_risky_fanouts(case: dict[str, Any], threshold: float = -1.0, min_coverage: float = 0.5) -> list[int]:
    """P3: Extract fanouts where custom kernel consistently loses (>|threshold|% regression)."""
    risky = []
    for row in case.get("rows", []):
        impr = float(row.get("improvement_percent", 0))
        cov = float(row.get("coverage_percent", 0))
        if impr < threshold and cov >= min_coverage:
            risky.append(int(row["fanout"]))
    return sorted(risky)


def _extract_winning_fanouts(case: dict[str, Any], threshold: float = 1.0, min_coverage: float = 0.5) -> list[int]:
    """P3: Extract fanouts where custom kernel consistently wins (>threshold% improvement)."""
    winning = []
    for row in case.get("rows", []):
        impr = float(row.get("improvement_percent", 0))
        cov = float(row.get("coverage_percent", 0))
        if impr > threshold and cov >= min_coverage:
            winning.append(int(row["fanout"]))
    return sorted(winning)


def _gmm1_profiler_diagnostics(cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for case_name, case in cases.items():
        if not str(case_name).startswith("gmm1"):
            continue
        for row in case["rows"]:
            rows.append({**row, "case": case_name})

    winning = sorted(
        (row for row in rows if row["improvement_percent"] > 1.0),
        key=lambda row: (float(row["improvement_percent"]), float(row["coverage_percent"])),
        reverse=True,
    )[:5]
    losing = sorted(
        (row for row in rows if row["improvement_percent"] < -1.0),
        key=lambda row: (float(row["improvement_percent"]), -float(row["coverage_percent"])),
    )[:5]
    neutral = sorted(
        (row for row in rows if -1.0 <= row["improvement_percent"] <= 1.0),
        key=lambda row: float(row["coverage_percent"]),
        reverse=True,
    )[:3]

    selected_fanouts = sorted({int(row["fanout"]) for row in [*winning, *losing, *neutral]})
    return {
        "status": "profile_before_kernel_changes",
        "goal": "Explain why custom GMM1 wins on high-benefit fanouts and loses on low-benefit fanouts before changing kernel scheduling.",
        "winning_fanouts": [_diagnostic_row(row) for row in winning],
        "losing_fanouts": [_diagnostic_row(row) for row in losing],
        "neutral_fanouts": [_diagnostic_row(row) for row in neutral],
        "selected_fanouts": selected_fanouts,
        "diagnostic_questions": [
            "Do winning fanouts have fewer idle AIC cores or shorter tail blocks than losing fanouts?",
            "Does custom GMM1 reduce Cube time or mainly reduce launch/task scheduling overhead?",
            "Are losing fanouts dominated by fixed per-expert overhead for 1-3 token experts?",
            "Does the estimated task count match the profiler task count for each selected fanout?",
            "Is p99 improvement coming from fewer tail tasks, better core assignment, or reduced memory movement?",
        ],
        "profiler_commands": _gmm1_profiler_commands(selected_fanouts),
        "profiler_note": (
            "benchmark_grouped_matmul.py profiles only the selected fanouts when --fanouts is provided."
        ),
    }


def _diagnostic_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": row["case"],
        "class_id": int(row["class_id"]),
        "fanout": int(row["fanout"]),
        "coverage_percent": round(float(row["coverage_percent"]), 4),
        "baseline_avg_ms": round(float(row["baseline_avg_ms"]), 4),
        "custom_avg_ms": round(float(row["custom_avg_ms"]), 4),
        "baseline_p99_ms": round(float(row["baseline_p99_ms"]), 4),
        "custom_p99_ms": round(float(row["custom_p99_ms"]), 4),
        "improvement_percent": round(float(row["improvement_percent"]), 4),
        "task_model": row.get("task_model"),
    }


def _gmm1_profiler_commands(fanouts: list[int]) -> list[str]:
    fanout_csv = ",".join(str(fanout) for fanout in fanouts) if fanouts else "40,41,46,53,54,35,36,37,38,39"
    base = (
        "python tools/moe_gmm/benchmark_grouped_matmul.py "
        "--trace \"$VLLM_ASCEND_MOE_GMM_TRACE_PATH\" --shape-classes --top-k 20 "
        "--hidden-size 2048 --intermediate-size 768 --mode gmm1 "
        f"--warmup 10 --iters 100 --device npu:0 --fanouts {fanout_csv}"
    )
    return [
        "export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-4}",
        "export VLLM_ASCEND_MOE_GMM_TRACE_PATH=benchmarks/results/qwen3_30b_a3b_gmm_fastpath_real_20260614T101101Z/gmm_trace.jsonl",
        f"# Selected fanouts for profiler comparison: {fanout_csv}",
        f"{base} --variants counts:s2 --json-output benchmarks/results/gmm1_fanout_profile/microbench_gmm1_torch.json",
        f"{base} --variants counts:s2:custom --json-output benchmarks/results/gmm1_fanout_profile/microbench_gmm1_custom.json",
    ]


def _gmm2_reverse_analysis(cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gmm2_cases = {name: case for name, case in cases.items() if str(name).startswith("gmm2")}
    best_gate = None
    for case in gmm2_cases.values():
        for gate in case["oracle_gates"].values():
            if best_gate is None or gate["improvement_percent"] > best_gate["improvement_percent"]:
                best_gate = gate
    best_improvement = 0.0 if best_gate is None else float(best_gate["improvement_percent"])
    status = "needs_profiler_evidence"
    conclusion = (
        "custom gmm2 has no meaningful oracle gain; profile torch_npu and custom on identical shape classes "
        "before designing a distinct gmm2 kernel"
        if best_improvement < 1.0
        else "custom gmm2 has a narrow oracle window, but still requires profiler evidence before runtime use"
    )
    return {
        "status": status,
        "best_oracle_improvement_percent": round(best_improvement, 4),
        "conclusion": conclusion,
        "profiler_commands": _profiler_commands(),
    }


def _profiler_commands() -> list[str]:
    return [
        "export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-4}",
        "export VLLM_ASCEND_MOE_GMM_TRACE_PATH=benchmarks/results/qwen3_30b_a3b_gmm_fastpath_real_20260614T101101Z/gmm_trace.jsonl",
        "python tools/moe_gmm/benchmark_grouped_matmul.py --trace \"$VLLM_ASCEND_MOE_GMM_TRACE_PATH\" --shape-classes --top-k 20 --hidden-size 2048 --intermediate-size 768 --mode gmm2 --variants counts:s2 --warmup 10 --iters 100 --device npu:0 --json-output benchmarks/results/gmm2_torch_profile/microbench_gmm2_torch.json",
        "python tools/moe_gmm/benchmark_grouped_matmul.py --trace \"$VLLM_ASCEND_MOE_GMM_TRACE_PATH\" --shape-classes --top-k 20 --hidden-size 2048 --intermediate-size 768 --mode gmm2 --variants counts:s2:custom --warmup 10 --iters 100 --device npu:0 --json-output benchmarks/results/gmm2_custom_profile/microbench_gmm2_custom.json",
    ]


def _parse_case_arg(values: list[str]) -> dict[str, str]:
    cases = {}
    for value in values:
        name, separator, path = value.partition("=")
        if separator != "=" or not name or not path:
            raise ValueError(f"case must be NAME=PATH, got {value}")
        cases[name] = path
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MoE GMM stage microbench results and oracle gates.")
    parser.add_argument("--case", action="append", default=[], help="Case input as NAME=PATH. May be repeated.")
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument("--thresholds", default="38,40,41,45")
    parser.add_argument("--gmm1-policy-threshold", type=int, default=40)
    args = parser.parse_args()

    thresholds = tuple(int(value.strip()) for value in args.thresholds.split(",") if value.strip())
    report = summarize_stage_files(
        _parse_case_arg(args.case),
        thresholds=thresholds,
        gmm1_policy_threshold=args.gmm1_policy_threshold,
    )
    write_reports(report, json_output=args.json_output, markdown_output=args.markdown_output)


if __name__ == "__main__":
    main()
