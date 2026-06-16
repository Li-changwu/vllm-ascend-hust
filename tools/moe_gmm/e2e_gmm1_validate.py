#!/usr/bin/env python3
"""End-to-end validation: compare baseline (torch_npu) vs custom GMM1.

Monkey-patches vllm_ascend.ops.fused_moe.moe_mlp.unquant_apply_mlp
to swap torch_npu.npu_grouped_matmul with our optimized custom kernel
for GMM1 (gate/up projection) only. GMM2 stays on torch_npu.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import torch
import torch_npu


def _custom_gmm1(x, weight, group_list, *, split_item=2, group_list_type=1):
    """Drop-in replacement for npu_grouped_matmul using our custom kernel.

    Handles both group_list_type=1 (counts) and group_list_type=0 (cumsum).
    """
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    if group_list_type == 0:  # cumsum → counts
        gl_int = group_list.to(torch.int64)
        diffs = gl_int[1:] - gl_int[:-1]
        counts = torch.cat([gl_int[:1], diffs])
    elif group_list_type == 1:  # already counts
        counts = group_list.to(torch.int64)
    else:
        raise ValueError(f"unsupported group_list_type={group_list_type}")

    # Build active pair group_list for custom kernel (group_list_type=2)
    active_mask = counts > 0
    active_ids = torch.nonzero(active_mask, as_tuple=False).reshape(-1).to(torch.int64)
    active_counts = counts.index_select(0, active_ids.to(torch.long))
    gl_pairs = torch.stack([active_ids, active_counts], dim=1)

    return torch.ops._C_ascend.moe_grouped_matmul(
        x, weight, gl_pairs, split_item, 0, 2,
    )[0]


def make_patched_unquant_apply_mlp(original_fn, *, use_custom_gmm1: bool):
    """Return a patched version of unquant_apply_mlp."""
    import vllm_ascend.ops.fused_moe.moe_mlp as mlp_mod

    def patched(
        hidden_states, w1, w2, group_list,
        w1_bias=None, w2_bias=None, activation=None,
        group_list_type=1, topk_scales=None, need_trans=True,
    ):
        if need_trans:
            w1 = w1.transpose(1, 2)
            w2 = w2.transpose(1, 2)

        act_name = getattr(activation, "value", activation)

        # GMM1: gate/up projection
        if use_custom_gmm1:
            gate_up_out = _custom_gmm1(
                hidden_states, w1, group_list,
                split_item=2, group_list_type=group_list_type,
            )
        else:
            gate_up_out = torch_npu.npu_grouped_matmul(
                x=[hidden_states], weight=[w1],
                bias=[w1_bias.to(dtype=torch.float32)] if w1_bias is not None else None,
                split_item=2, group_list_type=group_list_type,
                group_type=0, group_list=group_list,
            )[0]

        # SwiGLU activation
        if act_name == "swigluoai":
            gate_up_out = mlp_mod.AscendSwigluOAIAndMul.swiglu_oai_forward(
                gate_up_out.view(-1, w1.shape[2])
            )
        else:
            gate_up_out = torch_npu.npu_swiglu(gate_up_out)

        if topk_scales is not None:
            gate_up_out *= topk_scales

        # GMM2: down projection (always torch_npu)
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[gate_up_out], weight=[w2],
            bias=[w2_bias.to(dtype=torch.float32)] if w2_bias is not None else None,
            split_item=2, group_list_type=group_list_type,
            group_type=0, group_list=group_list,
        )[0]
        return hidden_states

    return patched


def bench_vllm(label: str, use_custom_gmm1: bool) -> dict[str, Any]:
    import vllm_ascend.ops.fused_moe.moe_mlp as moe_mlp
    from vllm import LLM, SamplingParams

    original = moe_mlp.unquant_apply_mlp
    patched = make_patched_unquant_apply_mlp(original, use_custom_gmm1=use_custom_gmm1)
    moe_mlp.unquant_apply_mlp = patched

    print(f"\n{'='*60}")
    print(f"  {label} (GMM1: {'custom' if use_custom_gmm1 else 'torch_npu'})")
    print(f"{'='*60}")

    llm = LLM(
        model="/data/shared-models/Qwen3-30B-A3B",
        dtype="bfloat16",
        max_model_len=512,
        max_num_seqs=4,
        max_num_batched_tokens=512,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )

    prompts = [
        "The capital of France is",
        "Python is a programming language",
        "Machine learning is",
        "你好，请解释一下量子计算",
    ]
    sampling_params = SamplingParams(max_tokens=128, temperature=0.0)

    # Warmup
    print("  Warming up...")
    llm.generate(prompts[:2], sampling_params)
    torch_npu.npu.synchronize()

    # Benchmark
    print("  Benchmarking...")
    torch_npu.npu.synchronize()
    t0 = perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    torch_npu.npu.synchronize()
    elapsed = perf_counter() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs if o.outputs)
    tok_s = total_tokens / elapsed if elapsed > 0 else 0

    print(f"  Tokens generated: {total_tokens}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {tok_s:.1f} tok/s")

    # Restore original
    moe_mlp.unquant_apply_mlp = original
    del llm
    torch.npu.empty_cache()

    return {
        "label": label,
        "use_custom_gmm1": use_custom_gmm1,
        "total_tokens": total_tokens,
        "elapsed_s": round(elapsed, 2),
        "throughput_tok_s": round(tok_s, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Run both baseline and custom")
    args = parser.parse_args()

    results = []
    results.append(bench_vllm("BASELINE (torch_npu GMM1)", use_custom_gmm1=False))

    if args.compare:
        results.append(bench_vllm("CUSTOM GMM1 (optimized)", use_custom_gmm1=True))

    baseline = results[0]
    if len(results) > 1:
        custom = results[1]
        speedup = custom["throughput_tok_s"] / baseline["throughput_tok_s"]
        print(f"\n{'='*60}")
        print(f"  Speedup: {speedup:.2f}x ({(speedup - 1) * 100:+.1f}%)")
        print(f"{'='*60}")

    out = Path("benchmarks/results/gmm1_e2e_validation/compare_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
