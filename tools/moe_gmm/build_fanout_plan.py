#!/usr/bin/env python3
"""Build a combined bucket plan for targeted GMM1 fanout profiling.
Parses the real trace's group_list_signature and active_experts to produce
the compact_group_list format expected by benchmark_grouped_matmul."""
import json
import sys
from pathlib import Path

TRACE_PATH = Path("benchmarks/results/qwen3_30b_a3b_gmm_fastpath_real_20260614T101101Z/gmm_trace.jsonl")
TARGET_FANOUTS = [41, 40, 35, 36]
OUTPUT = Path("benchmarks/results/gmm1_fanout_profiler/fanout_bucket_plan.json")


def counts_from_signature(sig: str) -> list[int]:
    prefix, _, payload = sig.partition(":")
    vals = [int(v) for v in payload.split(",") if v.strip()]
    if prefix == "counts":
        return vals
    if prefix == "cumsum":
        result = []
        prev = 0
        for v in vals:
            result.append(v - prev)
            prev = v
        return result
    raise ValueError(f"bad signature: {sig}")


def build():
    records = [json.loads(line) for line in TRACE_PATH.open() if line.strip()]
    dispatch = [r for r in records if r.get("source") == "grouped_dispatch"]

    buckets = []
    seen = set()
    for fanout in TARGET_FANOUTS:
        for r in dispatch:
            if int(r.get("fanout", 0)) != fanout:
                continue
            sig = r.get("group_list_signature", "")
            if sig in seen:
                continue
            seen.add(sig)

            counts = counts_from_signature(sig)
            active_experts = r.get("active_experts", [])
            if not isinstance(active_experts, list) or not active_experts:
                # Fallback: iterate counts to find non-zero positions
                active_experts = [i for i, c in enumerate(counts) if c > 0]

            # compact_group_list = flat token counts (NOT pairs!)
            counts_from_sig = [c for c in counts if c > 0]
            active_expert_ids_from_counts = [i for i, c in enumerate(counts) if c > 0]

            buckets.append({
                "bucket_id": str(fanout),
                "signature": sig,
                "compact_group_list": counts_from_sig,
                "active_expert_ids": active_expert_ids_from_counts,
                "original_expert_count": len(counts),
                "compact_expert_count": len(counts_from_sig),
                "sample_count": sum(1 for rr in dispatch if int(rr.get("fanout", 0)) == fanout and rr.get("group_list_signature") == sig),
                "coverage_percent": 0.0,
            })
            break  # One bucket per fanout

    plan = {
        "description": f"Targeted GMM1 fanout profiling: {TARGET_FANOUTS}",
        "fanouts": TARGET_FANOUTS,
        "buckets": buckets,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"Plan written: {OUTPUT} ({len(buckets)} buckets)")
    for b in buckets:
        print(f"  fanout={b['bucket_id']}: {b['compact_expert_count']} experts, "
              f"{b['original_expert_count']} total, sig={b['signature'][:50]}...")


if __name__ == "__main__":
    build()
