#!/usr/bin/env python3
"""Direct torch_npu vs custom GMM1 comparison using REAL trace data."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
import torch_npu


HIDDEN = 2048
INTER = 1536
NUM_EXPERTS = 128
BF16 = torch.bfloat16
DEVICE = "npu:0"
WARMUP = 10
ITERS = 100
TRACE = "benchmarks/results/qwen3_30b_a3b_gmm_fastpath_real_20260614T101101Z/gmm_trace.jsonl"


def counts_from_sig(sig):
    _, _, payload = sig.partition(":")
    return [int(v) for v in payload.split(",") if v.strip()]


def load_fanout_counts(fanout: int) -> list[int]:
    records = [json.loads(line) for line in Path(TRACE).open() if line.strip()]
    for r in records:
        if r.get("source") != "grouped_dispatch":
            continue
        if int(r.get("fanout", 0)) == fanout:
            return counts_from_sig(r["group_list_signature"])
    raise ValueError(f"fanout {fanout} not found in trace")


def bench(label, counts, use_custom):
    backend = "custom" if use_custom else "torch_npu"
    assert len(counts) == NUM_EXPERTS
    n_active = sum(1 for c in counts if c > 0)
    n_tokens = sum(counts)
    print(f"  [{label}] {backend} ({n_active} active/{n_tokens} tokens)...", end=" ", flush=True)

    torch.manual_seed(42)
    x = torch.randn(n_tokens, HIDDEN, dtype=BF16, device=DEVICE)
    torch.manual_seed(43)
    w = torch.randn(NUM_EXPERTS, HIDDEN, INTER, dtype=BF16, device=DEVICE)

    if use_custom:
        import vllm_ascend.vllm_ascend_C  # noqa: F401
        from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ
        # Custom kernel requires NZ format (matching real vLLM path)
        w_nz = torch_npu.npu_format_cast(w, ACL_FORMAT_FRACTAL_NZ)
        active_pairs = [[i, c] for i, c in enumerate(counts) if c > 0]
        gl = torch.tensor(active_pairs, dtype=torch.int64, device=DEVICE)

        def run():
            return torch.ops._C_ascend.moe_grouped_matmul(x, w_nz, gl, 2, 0, 2)[0]
    else:
        gl = torch.tensor(counts, dtype=torch.int64, device=DEVICE)

        def run():
            return torch_npu.npu_grouped_matmul(
                x=[x], weight=[w], split_item=2, group_list_type=1,
                group_type=0, group_list=gl)[0]

    for _ in range(WARMUP):
        run()
    torch_npu.npu.synchronize()

    times = []
    for _ in range(ITERS):
        start = perf_counter()
        run()
        torch_npu.npu.synchronize()
        times.append(perf_counter() - start)

    avg_ms = mean(times) * 1000
    p99_ms = sorted(times)[int(len(times) * 0.99)] * 1000
    print(f"avg={avg_ms:.4f}ms p99={p99_ms:.4f}ms")
    return avg_ms, p99_ms


def bench_correctness(label, counts):
    """Verify custom kernel produces identical results to torch_npu.
    
    NOTE: The custom kernel (aclnnMoeGroupedMatmulWeightNz) requires 
    FRACTAL_NZ format weights, matching vLLM's real inference path 
    where weights are cast to NZ in process_weights_after_loading().
    """
    from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

    torch.manual_seed(42)
    x = torch.randn(sum(counts), HIDDEN, dtype=BF16, device=DEVICE)
    torch.manual_seed(43)
    w = torch.randn(NUM_EXPERTS, HIDDEN, INTER, dtype=BF16, device=DEVICE)
    w_nz = torch_npu.npu_format_cast(w, ACL_FORMAT_FRACTAL_NZ)
    gl = torch.tensor(counts, dtype=torch.int64, device=DEVICE)

    # torch_npu baseline
    out_tn = torch_npu.npu_grouped_matmul(
        x=[x], weight=[w], split_item=2, group_list_type=1,
        group_type=0, group_list=gl)[0]

    # custom with NZ weight
    import vllm_ascend.vllm_ascend_C  # noqa: F401
    active_pairs = [[i, c] for i, c in enumerate(counts) if c > 0]
    gl_custom = torch.tensor(active_pairs, dtype=torch.int64, device=DEVICE)
    out_cu = torch.ops._C_ascend.moe_grouped_matmul(x, w_nz, gl_custom, 2, 0, 2)[0]

    diff = (out_tn.float() - out_cu.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    status = "PASS" if max_diff < 5e-3 else "FAIL"
    print(f"  [{label}] correctness: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} → {status}")
    return max_diff, mean_diff, status


def main():
    print("=== GMM1 torch_npu vs custom (REAL trace data) ===\n")

    # --- Correctness check (uses trace data from real vLLM server) ---
    print("--- Correctness Verification (same inputs → same outputs?) ---")
    all_pass = True
    for fanout in [41, 40, 35, 36]:
        counts = load_fanout_counts(fanout)
        _, _, status = bench_correctness(f"fanout_{fanout}", counts)
        if status != "PASS":
            all_pass = False
    print(f"  Overall: {'ALL PASS ✅' if all_pass else 'SOME FAILURES ❌'}\n")

    # --- Performance comparison ---
    print("--- Performance Comparison ---")
    results = {}
    for fanout in [41, 40, 35, 36]:
        counts = load_fanout_counts(fanout)
        n_active = sum(1 for c in counts if c > 0)
        n_tokens = sum(counts)

        label = f"fanout_{fanout}"
        print(f"--- {label} ({n_active} active, {n_tokens} tokens) ---")
        tn_avg, tn_p99 = bench(label, counts, use_custom=False)
        cu_avg, cu_p99 = bench(label, counts, use_custom=True)
        impr = (tn_avg - cu_avg) / tn_avg * 100
        print(f"  improvement: {impr:+.2f}%\n")

        results[label] = {
            "n_active": n_active,
            "n_tokens": n_tokens,
            "torch_npu_avg_ms": round(tn_avg, 4),
            "torch_npu_p99_ms": round(tn_p99, 4),
            "custom_avg_ms": round(cu_avg, 4),
            "custom_p99_ms": round(cu_p99, 4),
            "improvement_pct": round(impr, 2),
        }

    # Overall stats
    avg_impr = sum(r["improvement_pct"] for r in results.values()) / len(results)
    print(f"\n=== Overall average improvement: {avg_impr:+.2f}% ===")

    out = Path("benchmarks/results/gmm1_fanout_profiler/direct_compare_real.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out_json = {
        "correctness_all_pass": all_pass,
        "overall_avg_improvement_pct": round(avg_impr, 2),
        "fanouts": results,
    }
    out.write_text(json.dumps(out_json, indent=2) + "\n", encoding="utf-8")
    print(f"Results: {out}")
    print("\n=== Summary ===")
    for label, r in results.items():
        print(f"  {label}: {r['n_active']}a/{r['n_tokens']}t | "
              f"torch_npu={r['torch_npu_avg_ms']}ms custom={r['custom_avg_ms']}ms "
              f"Δ={r['improvement_pct']:+.2f}%")


if __name__ == "__main__":
    main()
