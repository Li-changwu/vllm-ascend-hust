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
from statistics import mean, median
from time import perf_counter
from typing import Any

import torch

from tools.moe_gmm.analyze_gmm_shapes import analyze_trace_file


def load_shape_classes(path: str | Path, *, top_k: int, fanouts: set[int] | None = None) -> list[dict[str, Any]]:
    """Group real trace records by coarse GMM shape, not exact expert signature."""
    records_by_class: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if record.get("source") != "grouped_dispatch":
                continue
            signature = str(record.get("group_list_signature", ""))
            if not signature:
                continue
            if fanouts is not None and int(record.get("fanout", 0) or 0) not in fanouts:
                continue
            key = (int(record.get("num_tokens", 0) or 0), int(record.get("fanout", 0) or 0))
            records_by_class[key].append(record)

    total = sum(len(records) for records in records_by_class.values())
    classes: list[dict[str, Any]] = []
    for class_id, ((num_tokens, fanout), records) in enumerate(
        sorted(records_by_class.items(), key=lambda item: len(item[1]), reverse=True)[:top_k],
        start=1,
    ):
        signature_counts = Counter(str(record.get("group_list_signature", "")) for record in records)
        sample_signature = signature_counts.most_common(1)[0][0]
        group_counts = _counts_from_signature(sample_signature)
        classes.append(
            {
                "class_id": class_id,
                "num_tokens": num_tokens,
                "fanout": fanout,
                "sample_count": len(records),
                "coverage_percent": _round_percent(len(records), total),
                "sample_signature": sample_signature,
                "group_list_type": records[0].get("group_list_type", 1),
                "physical_expert_count": len(group_counts),
                "group_counts": group_counts,
                "group_cumsum": _prefix_sum(group_counts),
                "layers": sorted({int(record.get("layer_id", -1)) for record in records}),
            }
        )
    return classes


def load_signatures(*, trace: str | None, plan: str | None, top_k: int) -> list[dict[str, Any]]:
    if plan:
        payload = json.loads(Path(plan).read_text(encoding="utf-8"))
        return payload.get("buckets", [])[:top_k]
    if not trace:
        raise ValueError("either --trace or --bucket-plan is required")
    report = analyze_trace_file(trace, top_k=top_k)
    return report["bucket_plan"]["buckets"][:top_k]


def run_microbench(
    *,
    signatures: list[dict[str, Any]],
    hidden_size: int,
    intermediate_size: int,
    warmup: int,
    iters: int,
    mode: str,
    device: str,
) -> dict[str, Any]:
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("torch_npu is required for real GroupedMatmul microbench") from exc

    if not device.startswith("npu"):
        raise RuntimeError("GroupedMatmul microbench must run on an NPU device")

    results = []
    for signature in signatures:
        results.append(
            _bench_signature(
                signature=signature,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                warmup=warmup,
                iters=iters,
                mode=mode,
                device=device,
            )
        )
    return {
        "mode": mode,
        "device": device,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "warmup": warmup,
        "iters": iters,
        "results": results,
    }


def run_shape_class_microbench(
    *,
    shape_classes: list[dict[str, Any]],
    hidden_size: int,
    intermediate_size: int,
    warmup: int,
    iters: int,
    mode: str,
    device: str,
    variants: tuple[str, ...] = ("counts:s2", "cumsum:s2"),
    full_chain_variants: tuple[str, ...] = (),
    task_single_m: int = 128,
    task_single_n: int = 256,
) -> dict[str, Any]:
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("torch_npu is required for real GroupedMatmul microbench") from exc

    if not device.startswith("npu"):
        raise RuntimeError("GroupedMatmul microbench must run on an NPU device")

    results = []
    for shape_class in shape_classes:
        results.append(
            _bench_shape_class(
                shape_class=shape_class,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                warmup=warmup,
                iters=iters,
                mode=mode,
                device=device,
                variants=variants,
                full_chain_variants=full_chain_variants,
                task_single_m=task_single_m,
                task_single_n=task_single_n,
            )
        )
    return {
        "target": "shape_class_operator_sweep",
        "mode": mode,
        "device": device,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "warmup": warmup,
        "iters": iters,
        "variants": list(variants),
        "full_chain_variants": list(full_chain_variants),
        "task_model_single_m": int(task_single_m),
        "task_model_single_n": int(task_single_n),
        "results": results,
    }


def _bench_signature(
    *,
    signature: dict[str, Any],
    hidden_size: int,
    intermediate_size: int,
    warmup: int,
    iters: int,
    mode: str,
    device: str,
) -> dict[str, Any]:
    import torch_npu

    full_group_list = _group_list_from_signature(str(signature["signature"]), device=device)
    compact_group_list = torch.tensor(signature["compact_group_list"], dtype=torch.int64, device=device)
    token_count = int(full_group_list.sum().item())
    original_experts = int(signature.get("original_expert_count", full_group_list.numel()))
    compact_experts = int(signature.get("compact_expert_count", len(signature["compact_group_list"])))

    hidden = torch.randn(token_count, hidden_size, dtype=torch.bfloat16, device=device)
    hidden_gmm2 = torch.randn(token_count, intermediate_size, dtype=torch.bfloat16, device=device)
    w1_full = torch.randn(original_experts, hidden_size, intermediate_size * 2, dtype=torch.bfloat16, device=device)
    w2_full = torch.randn(original_experts, intermediate_size, hidden_size, dtype=torch.bfloat16, device=device)
    active_ids = torch.tensor(signature["active_expert_ids"], dtype=torch.long, device=device)
    w1_compact = w1_full.index_select(0, active_ids)
    w2_compact = w2_full.index_select(0, active_ids)

    full_times = _time_case(
        lambda: _run_case(hidden, hidden_gmm2, w1_full, w2_full, full_group_list, mode=mode),
        warmup=warmup,
        iters=iters,
        torch_npu=torch_npu,
    )
    compact_times = _time_case(
        lambda: _run_case(hidden, hidden_gmm2, w1_compact, w2_compact, compact_group_list, mode=mode),
        warmup=warmup,
        iters=iters,
        torch_npu=torch_npu,
    )
    full_avg = mean(full_times)
    compact_avg = mean(compact_times)
    return {
        "bucket_id": signature.get("bucket_id"),
        "signature": signature["signature"],
        "token_count": token_count,
        "original_expert_count": original_experts,
        "compact_expert_count": compact_experts,
        "full_ms": _summary_ms(full_times),
        "compact_ms": _summary_ms(compact_times),
        "speedup": round(full_avg / compact_avg, 4) if compact_avg > 0 else 0.0,
        "improvement_percent": round((full_avg - compact_avg) * 100.0 / full_avg, 4) if full_avg > 0 else 0.0,
    }


def _bench_shape_class(
    *,
    shape_class: dict[str, Any],
    hidden_size: int,
    intermediate_size: int,
    warmup: int,
    iters: int,
    mode: str,
    device: str,
    variants: tuple[str, ...],
    full_chain_variants: tuple[str, ...],
    task_single_m: int,
    task_single_n: int,
) -> dict[str, Any]:
    import torch_npu

    counts = torch.tensor(shape_class["group_counts"], dtype=torch.int64, device=device)
    active_group_list = _active_pair_group_list_from_counts(counts)
    token_count = int(sum(int(value) for value in shape_class["group_counts"]))
    expert_count = int(shape_class["physical_expert_count"])

    hidden = torch.randn(token_count, hidden_size, dtype=torch.bfloat16, device=device)
    hidden_gmm2 = torch.randn(token_count, intermediate_size, dtype=torch.bfloat16, device=device)
    w1 = torch.randn(expert_count, hidden_size, intermediate_size * 2, dtype=torch.bfloat16, device=device)
    w2 = torch.randn(expert_count, intermediate_size, hidden_size, dtype=torch.bfloat16, device=device)

    variant_results: list[dict[str, Any]] = []
    active_variants: list[dict[str, Any]]
    if mode == "full" and full_chain_variants:
        active_variants = [_parse_full_chain_variant(name) for name in full_chain_variants]
    else:
        active_variants = [{"name": name, "gmm1": _parse_shape_variant(name), "gmm2": _parse_shape_variant(name)}
                           for name in variants]

    for variant in active_variants:
        gmm1_group_list, gmm1_group_list_type = _variant_group_list(
            counts,
            active_group_list=active_group_list,
            variant=variant["gmm1"]["group_list"],
            backend=variant["gmm1"]["backend"],
        )
        gmm2_group_list, gmm2_group_list_type = _variant_group_list(
            counts,
            active_group_list=active_group_list,
            variant=variant["gmm2"]["group_list"],
            backend=variant["gmm2"]["backend"],
        )
        times = _time_case(
            lambda: _run_case(
                hidden,
                hidden_gmm2,
                w1,
                w2,
                gmm1_group_list,
                mode=mode,
                group_list_type=int(gmm1_group_list_type),
                split_item=int(variant["gmm1"]["split_item"]),
                backend=variant["gmm1"]["backend"],
                gmm2_group_list=gmm2_group_list,
                gmm2_group_list_type=int(gmm2_group_list_type),
                gmm2_split_item=int(variant["gmm2"]["split_item"]),
                gmm2_backend=variant["gmm2"]["backend"],
            ),
            warmup=warmup,
            iters=iters,
            torch_npu=torch_npu,
        )
        variant_results.append(
            {
                "variant": variant["name"],
                "group_list_type": int(gmm1_group_list_type),
                "split_item": int(variant["gmm1"]["split_item"]),
                "backend": variant["gmm1"]["backend"],
                "gmm2_group_list_type": int(gmm2_group_list_type),
                "gmm2_split_item": int(variant["gmm2"]["split_item"]),
                "gmm2_backend": variant["gmm2"]["backend"],
                "ms": _summary_ms(times),
            }
        )
    baseline = variant_results[0]["ms"]["avg"] if variant_results else 0.0
    for result in variant_results:
        avg = result["ms"]["avg"]
        result["improvement_percent_vs_first"] = round((baseline - avg) * 100.0 / baseline, 4) if baseline > 0 else 0.0

    return {
        "class_id": int(shape_class["class_id"]),
        "num_tokens": int(shape_class["num_tokens"]),
        "fanout": int(shape_class["fanout"]),
        "sample_count": int(shape_class["sample_count"]),
        "coverage_percent": float(shape_class["coverage_percent"]),
        "physical_expert_count": expert_count,
        "sample_signature": shape_class["sample_signature"],
        "task_model": _build_task_model(
            shape_class["group_counts"],
            single_m=task_single_m,
            single_n=task_single_n,
            output_n=intermediate_size * 2 if mode == "gmm1" else hidden_size,
        ),
        "variants": variant_results,
    }


def _time_case(fn, *, warmup: int, iters: int, torch_npu) -> list[float]:
    for _ in range(warmup):
        fn()
    torch_npu.npu.synchronize()
    times = []
    for _ in range(iters):
        start = perf_counter()
        fn()
        torch_npu.npu.synchronize()
        times.append(perf_counter() - start)
    return times


def _summary_ms(times: list[float]) -> dict[str, float]:
    sorted_times = sorted(times)
    p99_index = min(len(sorted_times) - 1, int(len(sorted_times) * 0.99))
    return {
        "avg": round(mean(times) * 1000.0, 4),
        "p50": round(median(times) * 1000.0, 4),
        "p99": round(sorted_times[p99_index] * 1000.0, 4),
    }


def _group_list_from_signature(signature: str, *, device: str) -> torch.Tensor:
    prefix, separator, payload = signature.partition(":")
    if separator != ":":
        raise ValueError(f"invalid signature: {signature}")
    values = [int(value) for value in payload.split(",") if value.strip()]
    if prefix == "counts":
        return torch.tensor(values, dtype=torch.int64, device=device)
    if prefix == "cumsum":
        counts = []
        previous = 0
        for cumulative in values:
            counts.append(int(cumulative) - previous)
            previous = int(cumulative)
        return torch.tensor(counts, dtype=torch.int64, device=device)
    raise ValueError(f"unsupported signature type: {signature}")


def _run_case(
    hidden,
    hidden_gmm2,
    w1,
    w2,
    group_list,
    *,
    mode: str,
    group_list_type: int = 1,
    split_item: int = 2,
    backend: str = "torch_npu",
    gmm2_group_list=None,
    gmm2_group_list_type: int | None = None,
    gmm2_split_item: int | None = None,
    gmm2_backend: str | None = None,
):
    import torch_npu

    out1 = hidden
    if mode in ("gmm1", "full"):
        out1 = _grouped_matmul(
            hidden,
            w1,
            group_list,
            split_item=split_item,
            group_list_type=group_list_type,
            backend=backend,
        )
    if mode == "gmm1":
        return out1
    activated = torch_npu.npu_swiglu(out1)
    if gmm2_group_list is None:
        gmm2_group_list = group_list
    if gmm2_group_list_type is None:
        gmm2_group_list_type = group_list_type
    if gmm2_split_item is None:
        gmm2_split_item = split_item
    if gmm2_backend is None:
        gmm2_backend = backend
    if mode == "full":
        return _grouped_matmul(
            activated,
            w2,
            gmm2_group_list,
            split_item=gmm2_split_item,
            group_list_type=gmm2_group_list_type,
            backend=gmm2_backend,
        )
    return _grouped_matmul(
        hidden_gmm2,
        w2,
        gmm2_group_list,
        split_item=gmm2_split_item,
        group_list_type=gmm2_group_list_type,
        backend=gmm2_backend,
    )


def _grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_list: torch.Tensor,
    *,
    split_item: int,
    group_list_type: int,
    backend: str,
) -> torch.Tensor:
    if backend == "custom":
        import vllm_ascend.vllm_ascend_C  # noqa: F401

        return torch.ops._C_ascend.moe_grouped_matmul(
            x,
            weight,
            group_list,
            split_item,
            0,
            group_list_type,
        )[0]

    if backend != "torch_npu":
        raise ValueError(f"unsupported GMM backend: {backend}")

    import torch_npu

    return torch_npu.npu_grouped_matmul(
        x=[x],
        weight=[weight],
        split_item=split_item,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]


def _variant_group_list(
    counts: torch.Tensor,
    *,
    active_group_list: torch.Tensor | None = None,
    variant: str,
    backend: str = "torch_npu",
) -> tuple[torch.Tensor, int]:
    if backend == "custom":
        if active_group_list is None:
            active_group_list = _active_pair_group_list_from_counts(counts)
        return active_group_list, 2
    if variant == "counts":
        return counts, 1
    if variant == "cumsum":
        return _counts_to_cumsum(counts), 0
    raise ValueError(f"unsupported benchmark variant: {variant}")


def _parse_shape_variant(name: str) -> dict[str, Any]:
    parts = [part.strip() for part in str(name).split(":") if part.strip()]
    if not parts:
        raise ValueError("empty benchmark variant")
    group_list = parts[0]
    if group_list == "counts":
        group_list_type = 1
    elif group_list == "cumsum":
        group_list_type = 0
    else:
        raise ValueError(f"unsupported group_list variant: {group_list}")

    split_item = 2
    backend = "torch_npu"
    for part in parts[1:]:
        if part.startswith("s"):
            split_item = int(part[1:])
            if split_item not in (0, 1, 2, 3):
                raise ValueError(f"unsupported split_item: {split_item}")
        elif part in ("torch", "torch_npu"):
            backend = "torch_npu"
        elif part == "custom":
            backend = "custom"
        else:
            raise ValueError(f"unsupported benchmark variant part: {part}")
    backend_suffix = "" if backend == "torch_npu" else f":{backend}"
    return {
        "name": f"{group_list}:s{split_item}{backend_suffix}",
        "group_list": group_list,
        "group_list_type": 2 if backend == "custom" else group_list_type,
        "split_item": split_item,
        "backend": backend,
    }


def _parse_full_chain_variant(name: str) -> dict[str, Any]:
    parts = [part.strip() for part in str(name).split(",") if part.strip()]
    parsed: dict[str, Any] = {}
    for part in parts:
        key, separator, value = part.partition("=")
        if separator != "=":
            raise ValueError(f"invalid full chain variant part: {part}")
        if key not in ("g1", "gmm1", "g2", "gmm2"):
            raise ValueError(f"unsupported full chain variant key: {key}")
        parsed["gmm1" if key in ("g1", "gmm1") else "gmm2"] = _parse_shape_variant(value)
    parsed.setdefault("gmm1", _parse_shape_variant("counts:s2"))
    parsed.setdefault("gmm2", _parse_shape_variant("counts:s2"))
    return {
        "name": f"g1={parsed['gmm1']['name']},g2={parsed['gmm2']['name']}",
        "gmm1": parsed["gmm1"],
        "gmm2": parsed["gmm2"],
    }


def _counts_to_cumsum(counts: torch.Tensor) -> torch.Tensor:
    return counts.cumsum(dim=0)


def _build_task_model(
    group_counts: list[int],
    *,
    single_m: int,
    single_n: int,
    output_n: int,
) -> dict[str, Any]:
    active_counts = [int(count) for count in group_counts if int(count) > 0]
    histogram = Counter(str(count) for count in active_counts)
    n_block_count = _ceil_div(int(output_n), int(single_n))
    estimated_m_blocks = sum(_ceil_div(count, int(single_m)) for count in active_counts)
    return {
        "total_tokens": sum(int(count) for count in group_counts),
        "fanout": len(active_counts),
        "physical_expert_count": len(group_counts),
        "active_token_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
        "max_tokens_per_expert": max(active_counts) if active_counts else 0,
        "single_m": int(single_m),
        "single_n": int(single_n),
        "n_block_count": n_block_count,
        "estimated_m_block_count": estimated_m_blocks,
        "estimated_task_count": estimated_m_blocks * n_block_count,
    }


def _active_pair_group_list_from_counts(counts: torch.Tensor) -> torch.Tensor:
    expert_ids = torch.nonzero(counts > 0, as_tuple=False).reshape(-1).to(counts.dtype)
    token_counts = counts.index_select(0, expert_ids.to(torch.long))
    return torch.stack((expert_ids, token_counts), dim=1).contiguous()


def _counts_from_signature(signature: str) -> list[int]:
    prefix, separator, payload = signature.partition(":")
    if separator != ":":
        raise ValueError(f"invalid signature: {signature}")
    values = [int(value) for value in payload.split(",") if value.strip()]
    if prefix == "counts":
        return values
    if prefix == "cumsum":
        counts: list[int] = []
        previous = 0
        for cumulative in values:
            counts.append(int(cumulative) - previous)
            previous = int(cumulative)
        return counts
    raise ValueError(f"unsupported signature type: {signature}")


def _prefix_sum(values: list[int]) -> list[int]:
    total = 0
    cumsum: list[int] = []
    for value in values:
        total += int(value)
        cumsum.append(total)
    return cumsum


def _round_percent(part: int, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return round(float(part) * 100.0 / float(whole), 4)


def _ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")
    return (int(value) + int(divisor) - 1) // int(divisor)


def _parse_fanouts(value: str) -> set[int] | None:
    if not value.strip():
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark top MoE GroupedMatmul signatures from a real trace.")
    parser.add_argument("--trace", default="", help="Path to GMM trace JSONL.")
    parser.add_argument("--bucket-plan", default="", help="Path to gmm_bucket_plan.json.")
    parser.add_argument("--json-output", required=True, help="Path to write microbench report.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--shape-classes", action="store_true", help="Benchmark coarse real shape classes instead of exact signatures.")
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--intermediate-size", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--mode", choices=("gmm1", "gmm2", "full"), default="full")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--variants",
        default="counts:s2,cumsum:s2,counts:s3,cumsum:s3",
        help="Comma-separated shape-class variants. Format: counts:s2,cumsum:s3.",
    )
    parser.add_argument(
        "--full-chain-variants",
        default="",
        help="Semicolon-separated full-chain variants, e.g. 'g1=counts:s2,g2=cumsum:s3;g1=counts:s2,g2=counts:s3'.",
    )
    parser.add_argument("--task-single-m", type=int, default=128, help="single_m used for task-model estimation.")
    parser.add_argument("--task-single-n", type=int, default=256, help="single_n used for task-model estimation.")
    parser.add_argument("--fanouts", default="", help="Comma-separated fanout values to keep when using --shape-classes.")
    args = parser.parse_args()

    if args.shape_classes:
        if not args.trace:
            raise ValueError("--trace is required with --shape-classes")
        shape_classes = load_shape_classes(args.trace, top_k=args.top_k, fanouts=_parse_fanouts(args.fanouts))
        report = run_shape_class_microbench(
            shape_classes=shape_classes,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            warmup=args.warmup,
            iters=args.iters,
            mode=args.mode,
            device=args.device,
            variants=tuple(value.strip() for value in args.variants.split(",") if value.strip()),
            full_chain_variants=tuple(value.strip() for value in args.full_chain_variants.split(";") if value.strip()),
            task_single_m=args.task_single_m,
            task_single_n=args.task_single_n,
        )
    else:
        signatures = load_signatures(trace=args.trace or None, plan=args.bucket_plan or None, top_k=args.top_k)
        report = run_microbench(
            signatures=signatures,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            warmup=args.warmup,
            iters=args.iters,
            mode=args.mode,
            device=args.device,
        )
    output_path = Path(args.json_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
