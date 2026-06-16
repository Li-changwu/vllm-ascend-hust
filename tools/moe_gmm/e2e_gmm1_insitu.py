#!/usr/bin/env python3
"""In-situ e2e GMM1 validation via env-gated in-module probe (V1-safe).

vLLM V1 runs EngineCore in a subprocess, so a parent-process monkey-patch never
reaches the real MoE forward. Instead we gate the probe inside
vllm_ascend/ops/fused_moe/moe_mlp.py with env vars that the subprocess reads at
import time:
  VLLM_ASCEND_GMM1_PROBE_BACKEND = torch_npu | custom
  VLLM_ASCEND_GMM1_PROBE_PATH    = file to append per-call GMM1 latency (ms)

We time ONLY the GMM1 grouped matmul (undiluted), accumulate across all real
decode steps, and capture output token ids so correctness can be verified.

Run each backend in a SEPARATE process:
  python tools/moe_gmm/e2e_gmm1_insitu.py --backend torch_npu --out <dir>/torch.json
  python tools/moe_gmm/e2e_gmm1_insitu.py --backend custom    --out <dir>/custom.json
Then compare:
  python tools/moe_gmm/e2e_gmm1_insitu.py --compare <dir>/torch.json <dir>/custom.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import median
from typing import Any


def run_once(backend: str, out_path: str) -> dict[str, Any]:
    probe_file = Path(out_path).with_suffix(".gmm1_ms.txt")
    if probe_file.exists():
        probe_file.unlink()
    # Must be set BEFORE importing vllm/vllm_ascend so the subprocess inherits.
    os.environ["VLLM_ASCEND_GMM1_PROBE_BACKEND"] = backend
    os.environ["VLLM_ASCEND_GMM1_PROBE_PATH"] = str(probe_file)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model="/data/shared-models/Qwen3-30B-A3B",
        dtype="bfloat16", max_model_len=512, max_num_seqs=4,
        max_num_batched_tokens=512, tensor_parallel_size=1,
        trust_remote_code=True, enforce_eager=True,
        gpu_memory_utilization=0.97,
    )
    prompts = [
        "The capital of France is",
        "Explain how a transformer neural network works in detail.",
        "Write a short story about a robot learning to paint.",
        "你好，请详细解释一下量子计算的基本原理。",
    ]
    sp = SamplingParams(max_tokens=128, temperature=0.0)

    llm.generate(prompts, sp)  # warmup (probe file gets truncated below)
    probe_file.write_text("")  # discard warmup samples

    outputs = llm.generate(prompts, sp)
    token_ids = [list(o.outputs[0].token_ids) for o in outputs]

    samples = [float(x) for x in probe_file.read_text().split() if x.strip()]
    total_ms = sum(samples)
    return {
        "backend": backend,
        "gmm1_calls": len(samples),
        "gmm1_total_ms": round(total_ms, 2),
        "gmm1_per_call_ms_median": round(median(samples), 4) if samples else None,
        "output_token_ids": token_ids,
    }


def compare(torch_path: str, custom_path: str) -> None:
    t = json.loads(Path(torch_path).read_text())
    c = json.loads(Path(custom_path).read_text())
    tok_ok = t["output_token_ids"] == c["output_token_ids"]
    tm, cm = t["gmm1_total_ms"], c["gmm1_total_ms"]
    speedup = tm / cm if cm else 0.0
    print(f"token_id_match    : {tok_ok}")
    print(f"GMM1 calls        : torch={t['gmm1_calls']}  custom={c['gmm1_calls']}")
    print(f"GMM1 total ms      : torch={tm}  custom={cm}")
    print(f"GMM1 per-call ms   : torch={t['gmm1_per_call_ms_median']}  "
          f"custom={c['gmm1_per_call_ms_median']}")
    print(f"GMM1 speedup       : {speedup:.3f}x ({(speedup-1)*100:+.1f}%)")
    if not tok_ok:
        print("REJECT: custom output diverges from torch_npu -> not e2e-correct.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["torch_npu", "custom"])
    ap.add_argument("--out", type=str)
    ap.add_argument("--compare", nargs=2, metavar=("TORCH_JSON", "CUSTOM_JSON"))
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    res = run_once(args.backend, args.out)
    print(json.dumps({k: v for k, v in res.items() if k != "output_token_ids"}, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
