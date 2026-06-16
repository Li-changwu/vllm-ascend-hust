#!/usr/bin/env python3
"""Run a single GMM1 fanout under torch_npu and custom backend for msprof."""
import json
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
import torch_npu
import vllm_ascend.vllm_ascend_C  # noqa: F401

HIDDEN = 2048
INTER = 1536
NUM_EXPERTS = 128
DEVICE = "npu:0"
WARMUP = 3
ITERS = 20
TRACE = "benchmarks/results/qwen3_30b_a3b_gmm_fastpath_real_20260614T101101Z/gmm_trace.jsonl"


def counts_from_sig(sig):
    _, _, payload = sig.partition(":")
    return [int(v) for v in payload.split(",") if v.strip()]


def load_fanout_counts(fanout: int) -> list[int]:
    records = [json.loads(line) for line in Path(TRACE).open() if line.strip()]
    for r in records:
        if r.get("source") == "grouped_dispatch" and int(r.get("fanout", 0)) == fanout:
            return counts_from_sig(r["group_list_signature"])
    raise ValueError(f"fanout {fanout} not found")


def run(fanout, use_custom):
    backend = "custom" if use_custom else "torch_npu"
    counts = load_fanout_counts(fanout)
    n_tokens = sum(counts)

    x = torch.randn(n_tokens, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    w = torch.randn(NUM_EXPERTS, HIDDEN, INTER, dtype=torch.bfloat16, device=DEVICE)

    if use_custom:
        active_pairs = [[i, c] for i, c in enumerate(counts) if c > 0]
        gl = torch.tensor(active_pairs, dtype=torch.int64, device=DEVICE)

        def run_op():
            return torch.ops._C_ascend.moe_grouped_matmul(x, w, gl, 2, 0, 2)[0]
    else:
        gl = torch.tensor(counts, dtype=torch.int64, device=DEVICE)

        def run_op():
            return torch_npu.npu_grouped_matmul(x=[x], weight=[w], split_item=2,
                                                 group_list_type=1, group_type=0,
                                                 group_list=gl)[0]

    # Warmup
    for _ in range(WARMUP):
        run_op()
    torch_npu.npu.synchronize()

    # Timed iters
    times = []
    for _ in range(ITERS):
        start = perf_counter()
        run_op()
        torch_npu.npu.synchronize()
        times.append(perf_counter() - start)

    avg_ms = mean(times) * 1000
    print(f"  [fanout_{fanout}] {backend}: avg={avg_ms:.4f}ms ({ITERS} iters)")
    return avg_ms


if __name__ == "__main__":
    fanout = int(sys.argv[1])
    use_custom = sys.argv[2] == "custom"
    run(fanout, use_custom)
