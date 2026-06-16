#!/usr/bin/env python3
"""Targeted GMM1 profiler: compare torch_npu vs custom backend on winning vs losing fanouts.

Key diagnostic questions from the GMM1 Profiler Diagnostic Plan:
- Do winning fanouts have fewer idle AIC cores or shorter tail blocks?
- Does custom GMM1 reduce Cube time or mainly reduce launch/task scheduling overhead?
- Are losing fanouts dominated by fixed per-expert overhead for 1-3 token experts?
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

RESULTS_DIR = Path("benchmarks/results/gmm1_fanout_profiler")
TRACE_PATH = Path("benchmarks/results/qwen3_30b_a3b_gmm_fastpath_real_20260614T101101Z/gmm_trace.jsonl")

# Fanouts to profile: winning + losing
TARGET_FANOUTS = [41, 40, 35, 36]

# Model dims for Qwen3-30B-A3B
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 768


def build_fanout_bucket_plan(fanout: int) -> dict:
    """Build a minimal bucket plan targeting a single fanout from the real trace."""
    with Path(TRACE_PATH).open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    matching = [r for r in records
                if r.get("source") == "grouped_dispatch"
                and int(r.get("fanout", 0)) == fanout]
    if not matching:
        raise ValueError(f"No trace records found for fanout={fanout}")

    signature = matching[0].get("group_list_signature", "")
    prefix, _, payload = signature.partition(":")
    values = [int(v) for v in payload.split(",") if v.strip()]
    counts = values if prefix == "counts" else []
    if not counts and prefix == "cumsum":
        prev = 0
        for v in values:
            counts.append(v - prev)
            prev = v

    return {
        "description": f"fanout_{fanout}_counts_from_trace",
        "buckets": [{
            "bucket_id": f"fanout_{fanout}",
            "signature": signature,
            "group_list_type": 2,
            "compact_group_list": [[i, c] for i, c in enumerate(counts) if c > 0],
            "active_expert_ids": [i for i, c in enumerate(counts) if c > 0],
            "original_expert_count": len(counts),
            "compact_expert_count": sum(1 for c in counts if c > 0),
            "token_count": sum(counts),
            "fanout": fanout,
            "group_counts": counts,
            "physical_expert_count": len(counts),
        }]
    }


def run_msprof(bucket_plan_path: str, backend: str) -> dict:
    """Run msprof profiling for a specific backend on the bucket plan."""
    output_dir = RESULTS_DIR / f"msprof_{backend}_fanouts"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prof_dir = output_dir / f"PROF_PIPE_{timestamp}"
    prof_dir.mkdir(parents=True, exist_ok=True)

    json_out = output_dir / f"microbench_{backend}_{timestamp}.json"

    variants = "counts:s2:custom" if backend == "custom" else "counts:s2"

    cmd = [
        sys.executable, "-u",
        "tools/moe_gmm/benchmark_grouped_matmul.py",
        "--bucket-plan", bucket_plan_path,
        "--json-output", str(json_out),
        "--top-k", str(len(TARGET_FANOUTS)),
        "--hidden-size", str(HIDDEN_SIZE),
        "--intermediate-size", str(INTERMEDIATE_SIZE),
        "--mode", "gmm1",
        "--warmup", "10",
        "--iters", "50",
        "--device", "npu:0",
        "--variants", variants,
    ]

    print(f"  Profiling {backend} GMM1 with msprof...")
    msprof_cmd = [
        "msprof",
        "--output", str(prof_dir),
        "--ai-core=on",
        "--aic-metrics=PipeUtilization",
        "--task-time=on",
        "--ascendcl=on",
    ] + cmd

    result = subprocess.run(msprof_cmd, capture_output=True, text=True, timeout=300,
                            env={"ASCEND_RT_VISIBLE_DEVICES": "4"})
    if result.returncode != 0:
        print(f"  msprof failed: {result.stderr[-500:]}")
        return {"status": "failed", "error": result.stderr[-500:]}

    # Load benchmark results
    benchmark_data = json.loads(json_out.read_text(encoding="utf-8"))

    return {
        "status": "ok",
        "prof_dir": str(prof_dir),
        "json_out": str(json_out),
        "benchmark": benchmark_data,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_lines = []
    report_lines.append("# GMM1 Fanout Profiler Results\n")
    report_lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
    report_lines.append(f"Target fanouts: {TARGET_FANOUTS}\n")
    report_lines.append(f"Hidden: {HIDDEN_SIZE}, Intermediate: {INTERMEDIATE_SIZE}\n\n")

    for fanout in TARGET_FANOUTS:
        plan = build_fanout_bucket_plan(fanout)
        plan_path = RESULTS_DIR / f"fanout_{fanout}_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        print(f"Created plan for fanout={fanout} ({plan['buckets'][0]['token_count']} tokens): {plan_path}")

    # Run profiling for both backends
    # Strategy: run torch_npu first, then custom
    backend_results = {}
    for backend in ["torch_npu", "custom"]:
        all_results = []
        for fanout in TARGET_FANOUTS:
            plan_path = str(RESULTS_DIR / f"fanout_{fanout}_plan.json")
            print(f"\n=== GMM1 fanout={fanout} backend={backend} ===")
            try:
                r = run_msprof(plan_path, backend)
                all_results.append({"fanout": fanout, **r})
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results.append({"fanout": fanout, "status": "failed", "error": str(e)})
        backend_results[backend] = all_results

    # Generate summary
    report_lines.append("## Summary\n\n")
    report_lines.append("| fanout | backend | avg_ms | p99_ms | improvement% |\n")
    report_lines.append("|---|---:|---:|---:|---:|\n")
    for backend, results in backend_results.items():
        for r in results:
            if r.get("status") == "ok" and r.get("benchmark", {}).get("results"):
                for variant in r["benchmark"]["results"]:
                    vr = variant.get("variants", [variant])[0]
                    improvement = vr.get("improvement_percent_vs_first", 0)
                    report_lines.append(
                        f"| {r['fanout']} | {backend} | {vr['ms']['avg']:.4f} | "
                        f"{vr['ms']['p99']:.4f} | {improvement:+.2f}% |\n"
                    )

    report_path = RESULTS_DIR / "gmm1_fanout_profiler_summary.md"
    report_path.write_text("".join(report_lines), encoding="utf-8")
    print(f"\nReport written to: {report_path}")

    # Save full results
    full_results_path = RESULTS_DIR / "gmm1_fanout_profiler_results.json"
    json.dump(backend_results, full_results_path.open("w", encoding="utf-8"), indent=2)
    print(f"Full results written to: {full_results_path}")


if __name__ == "__main__":
    main()
