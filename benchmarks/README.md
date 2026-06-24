# vLLM Ascend Benchmarks

## Introduction

This document outlines the benchmarking methodology for vllm-ascend, aimed at evaluating the performance under a variety of workloads. The primary goal is to help developers assess whether their pull requests improve or degrade vllm-ascend's performance.

## Overview

**Benchmarking Coverage**: We measure latency, throughput, and fixed-QPS serving on the Atlas800I A2 (see [quick_start](../docs/source/quick_start.md) to learn more supported devices list), with different models(coming soon).

- Latency tests
    - Input length: 32 tokens.
    - Output length: 128 tokens.
    - Batch size: fixed (8).
    - Models: Qwen2.5-7B-Instruct, Qwen3-8B.
    - Evaluation metrics: end-to-end latency (mean, median, p99).

- Throughput tests
    - Input length: randomly sample 200 prompts from ShareGPT dataset (with fixed random seed).
    - Output length: the corresponding output length of these 200 prompts.
    - Batch size: dynamically determined by vllm to achieve maximum throughput.
    - Models: Qwen2.5-VL-7B-Instruct, Qwen2.5-7B-Instruct, Qwen3-8B.
    - Evaluation metrics: throughput.
- Serving tests
    - Input length: randomly sample 200 prompts from ShareGPT dataset (with fixed random seed).
    - Output length: the corresponding output length of these 200 prompts.
    - Batch size: dynamically determined by vllm and the arrival pattern of the requests.
    - **Average QPS (query per second)**: 1, 4, 16 and inf. QPS = inf means all requests come at once. For other QPS values, the arrival time of each query is determined using a random Poisson process (with fixed random seed).
    - Models: Qwen2.5-VL-7B-Instruct, Qwen2.5-7B-Instruct, Qwen3-8B.
    - Evaluation metrics: throughput, TTFT (time to the first token, with mean, median and p99), ITL (inter-token latency, with mean, median and p99).

**Benchmarking Duration**: about 800 seconds for single model.

## Quick Use

### Prerequisites

Before running the benchmarks, ensure the following:

- vllm and vllm-ascend are installed and properly set up in an NPU environment, as these scripts are specifically designed for NPU devices.

- Install necessary dependencies for benchmarks:
  
  ```shell
  pip install -r benchmarks/requirements-bench.txt
  ```
  
- For performance benchmark, it is recommended to set the [load-format](https://github.com/vllm-project/vllm-ascend/blob/5897dc5bbe321ca90c26225d0d70bff24061d04b/benchmarks/tests/latency-tests.json#L7) as `dummy`, It will construct random weights based on the passed model without downloading the weights from internet, which can greatly reduce the benchmark time.
- If you want to run a customized benchmark, feel free to add your own models and parameters in the [JSON](https://github.com/vllm-project/vllm-ascend/tree/main/benchmarks/tests), let's take `Qwen2.5-VL-7B-Instruct`as an example:

  ```json
  [
  {
    "test_name": "serving_qwen2_5vl_7B_tp1",
    "qps_list": [
      1,
      4,
      16,
      "inf"
    ],
    "server_parameters": {
      "model": "Qwen/Qwen2.5-VL-7B-Instruct",
      "tensor_parallel_size": 1,
      "swap_space": 16,
      "disable_log_stats": "",
      "disable_log_requests": "",
      "trust_remote_code": "",
      "max_model_len": 16384
    },
    "client_parameters": {
      "model": "Qwen/Qwen2.5-VL-7B-Instruct",
      "backend": "openai-chat",
      "dataset_name": "hf",
      "hf_split": "train",
      "endpoint": "/v1/chat/completions",
      "dataset_path": "lmarena-ai/vision-arena-bench-v0.1",
      "num_prompts": 200
    }
  }
  ]
  ```
  
this Json will be structured and parsed into server parameters and client parameters by the benchmark script. This configuration defines a test case named `serving_qwen2_5vl_7B_tp1`, designed to evaluate the performance of the `Qwen/Qwen2.5-VL-7B-Instruct` model under different request rates. The test includes both server and client parameters, for more parameters details, see vllm benchmark [cli](https://github.com/vllm-project/vllm/tree/main/vllm/benchmarks).

- **Test Overview**
    - Test Name: serving_qwen2_5vl_7B_tp1

    - Queries Per Second (QPS): The test is run at four different QPS levels: 1, 4, 16, and inf (infinite load, typically used for stress testing).

- Server Parameters
    - Model: Qwen/Qwen2.5-VL-7B-Instruct

    - Tensor Parallelism: 1 (no model parallelism is used; the model runs on a single device or node)

    - Swap Space: 16 GB (used to handle memory overflow by swapping to disk)

    - disable_log_stats: disables logging of performance statistics.

    - disable_log_requests: disables logging of individual requests.

    - Trust Remote Code: enabled (allows execution of model-specific custom code)

    - Max Model Length: 16,384 tokens (maximum context length supported by the model)

- Client Parameters

    - Model: Qwen/Qwen2.5-VL-7B-Instruct (same as the server)

    - Backend: openai-chat (suggests the client uses the OpenAI-compatible chat API format)

    - Dataset Source: Hugging Face (hf)

    - Dataset Split: train

    - Endpoint: /v1/chat/completions (the REST API endpoint to which chat requests are sent)

    - Dataset Path: lmarena-ai/vision-arena-bench-v0.1 (the benchmark dataset used for evaluation, hosted on Hugging Face)

    - Number of Prompts: 200 (the total number of prompts used during the test)

### Run benchmarks

#### Use trusted GitHub Actions workflow

For trusted Ascend runners, use the `Ascend Benchmark Leaderboard` workflow and
its `workflow_dispatch` form. The default values preserve the existing FP16
same-spec benchmark path:

| Input | Default | Notes |
| --- | --- | --- |
| `ascend_hust_target` | `vLLM-HUST/vllm-ascend-hust@main` | Target plugin repo/ref to test. |
| `vllm_hust_ref` | `main` | Paired `vllm-hust` runtime ref. |
| `benchmark_ref` | `main` | Paired `vllm-hust-benchmark` runner/export ref. Use a feature branch only for branch validation. |
| `model_name` | `Qwen/Qwen2.5-14B-Instruct` | Hugging Face model id or local model path. |
| `model_precision` | `FP16` | Leaderboard model precision metadata. |
| `model_quantization` | empty | Leave empty for unquantized models. |
| `dtype` | `float16` | Use `auto` for Ascend quantized/modelslim models. |
| `hardware_chip_model` | `910B3` | Override to `910B2` when running on 910B2 hardware. |

To run the W8A8 quantized model benchmark after the workflow changes are merged,
keep repo refs on `main` and override only the model-specific fields:

```text
model_name=aly16/Qwen2.5-14B-W8A8
model_precision=INT8
model_quantization=W8A8
dtype=auto
hardware_chip_model=910B2
```

Do not put a branch name such as `ws/quantized-model-leaderboard` in
`model_name`; use `benchmark_ref` for the benchmark runner branch when testing
unmerged benchmark changes.

#### Use benchmark script

The provided scripts automatically execute performance tests for serving, throughput, and latency. To start the benchmarking process, run command in the vllm-ascend root directory:

```shell
bash benchmarks/scripts/run-performance-benchmarks.sh
```

Once the script completes, you can find the results in the benchmarks/results folder. The output files may resemble the following:

```shell
.
|-- serving_qwen2_5_7B_tp1_qps_1.json
|-- serving_qwen2_5_7B_tp1_qps_16.json
|-- serving_qwen2_5_7B_tp1_qps_4.json
|-- serving_qwen2_5_7B_tp1_qps_inf.json
|-- latency_qwen2_5_7B_tp1.json
|-- throughput_qwen2_5_7B_tp1.json
```

These files contain detailed benchmarking results for further analysis.

#### Generate MoE offload dashboard

For research-only MoE offload comparisons, place the non-offloading upper-bound
result and the `--ascend-moe-offload-gb 14` baseline result in
`benchmarks/results/`, then generate a static HTML dashboard:

```shell
python3 benchmarks/scripts/generate_moe_offload_dashboard.py \
  --results-dir benchmarks/results \
  --upper-label non-offload \
  --baseline-label offload-14GB
```

The script reads `benchmarks/results/*.json` and writes
`benchmarks/results/moe_offload_dashboard.html` by default. It compares
throughput, TTFT, and TPOT with the non-offloading run treated as the upper
bound and the offload 14GB run treated as the baseline.

#### P0 性能剖析结论：MoE 推理的时间花在哪里

P0 剖析工作（功能一）测量 Ascend NPU 上 MoE 推理各阶段的执行开销，为下游
的 GMM 算子优化分支与专家卸载分支提供数据依据。下表是单卡非卸载
Qwen3-30B-A3B 在 mixed 阶段的核心结果（Atlas 800I A2，Ascend PyTorch
Profiler）：

| 算子类型 | 总耗时 | 平均 | 占 MoE 算子时间 | Cube 利用率 |
|---|---:|---:|---:|---:|
| **GroupedMatmul** | 2099.74 ms | 165.7 us | **62.6%** | 91.2% |
| MatMulV2 | 401.67 ms | 21.0 us | 12.0% | 68.8% |
| FusedInferAttentionScore | 179.36 ms | 28.3 us | 5.3% | 86.0% |
| RmsNorm | 158.03 ms | 12.3 us | 4.7% | 0.0% |
| MoeInitRoutingCustom | 112.13 ms | 17.7 us | 3.3% | 0.0% |

该工况下端到端指标：中位 TTFT ≈ 1169.6 ms，中位 TPOT ≈ 252.6 ms，
输出吞吐 ≈ 37.4 tok/s。

**结论一 —— GroupedMatmul 在 prefill 和 decode 两个阶段都是头号开销，且对
shape 敏感。** 把同一次运行拆成 prefill 与 decode 两个窗口后可以看到，无论
哪个阶段 GMM 都是占比最高的单一算子，但其单次调用耗时随 batch 后的 shape
大幅波动：

| 阶段 | GMM 调用数 | GMM 总耗时 | GMM 平均 | GMM 占比 | Cube 利用率 |
|---|---:|---:|---:|---:|---:|
| Prefill | 672 | 236.81 ms | **352.4 us** | 59.0% | 95.5% |
| Decode | 6240 | 725.17 ms | **116.2 us** | 55.9% | 91.0% |

**分析：** 单次调用 3 倍的差距（prefill 352 us vs decode 116 us）来自
token-per-expert 分布和 group shape 的差异，而**不是 kernel 本身的低效**
—— Cube 利用率已经达到 91–96%，计算单元基本打满。这正是 GMM 优化分支
（`feature/gmm-kernel-opt`）选择主攻 **tiling 与 stable-shape grouped-matmul
路径**、而非通用算子融合的原因：kernel 已经很忙，真正能撬动的杠杆是 shape。
也意味着 prefill（长 prompt、大 group）和 decode（小 group、高频）需要不同
的 tiling 策略，不能用一套参数覆盖两个阶段。

**结论二 —— MoE 时间约 80% 是计算约束（Cube-bound），约 20% 是
向量/访存约束（vector/MTE-bound）。** 按 Cube 利用率把 top kernel 分成两类，
可以清晰区分出两条独立的优化轨道：

| 类别 | Mixed | Prefill | Decode |
|---|---:|---:|---:|
| 计算约束（AIC：GMM、MatMul、Attention） | **80.5%** | 76.3% | 77.8% |
| 向量/访存约束（AIV，Cube=0：RmsNorm、routing、permute、slice） | 19.5% | 23.7% | 22.2% |

**分析：** 计算约束的大头由 GMM 主导，对应 GMM 算子分支与卸载分支；剩下的
向量/访存约束部分（RmsNorm、MoeInitRoutingCustom、MoeGatingTopK、
MoeTokenUnpermute、Slice —— 路由相关算子在 mixed 窗口合计约 197 ms）则是
**算子融合的候选集**：这些都是高频短 kernel，且前后常紧邻 norm / bias /
swiglu / reshape 链，具备合并的结构条件。这条占比约 20% 的轨道虽然不是最大
头，但 kernel 数量极多（routing 家族在 mixed 窗口数万次调用），launch 与
stream 调度开销不可忽视，是 TPOT 优化的次级目标。

驱动另外两个分支的核心判断：

- **GroupedMatmul 占据 62.6%** 且 Cube 利用率高（91.2%），收益来自给它喂更
  好的 shape —— 这是自定义 GMM kernel 工作（`feature/gmm-kernel-opt`）的目标。
- 两个 grouped matmul 阶段（GMM1 gate/up、GMM2 down）正是 per-expert 计算与
  HBM 常驻专家权重访问最集中的地方，也是专家卸载运行时
  （`feature/moe-offload-runtime`）在 HBM 装不下全部专家时所要重构的环节。

> 关于 wait / MTE 比率的说明：单次运行报告还会给出累计 kernel wait 比率
> （mixed 为 911.7%）和 MTE 时间比率（mixed 为 90.8%）。这两个值是跨 kernel
> 与跨 stream 累加的，会超过 100%，因此应当把它们当作**相对的 stream 压力
> 信号**，而不是绝对的 stall 时间。

可用下方的剖析套件复现该报告；每次运行的报告会落在
`benchmarks/results/<run>/ascend_moe_profile_report.md`（results 目录不纳入
版本控制）。

#### Profile Qwen3 MoE prefill and decode with Ascend PyTorch Profiler

For single-card non-offloading Qwen3-30B-A3B analysis, start vLLM with the
torch profiler enabled and do not pass any MoE offload argument. To make the
report choose a trace-backed SEW-MoE P1 target, also enable trace-only active
expert and pipeline profiling in the server process:

```shell
export VLLM_ASCEND_MOE_OFFLOAD_ENABLED=1
export VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY=1
export VLLM_ASCEND_MOE_PIPELINE_PROFILING=1
export VLLM_ASCEND_MOE_OFFLOAD_TRACE_PATH=/tmp/sew_moe_trace.jsonl
export VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH=/tmp/sew_moe_profile.jsonl
# Optional after a previous suite run: classify stable grouped signatures before
# the existing grouped matmul fallback.
export VLLM_ASCEND_MOE_COMPUTE_BUCKET_PLAN_PATH=/path/to/sew_moe_p1_plan.json

vllm serve /data/shared-models/Qwen3-30B-A3B \
  --served-model-name qwen3-30b-a3b \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "torch_profiler_with_stack": false}'
```

Then run the three-window profiling suite. It opens and closes Ascend PyTorch
Profiler around a mixed ShareGPT window, a long-prefill window, and a
long-decode window, calls `torch_npu.profiler.profiler.analyse()`, and writes
phase-specific optimization reports.

```shell
python3 benchmarks/scripts/run_ascend_moe_profile_suite.py \
  --base-url http://127.0.0.1:8005 \
  --profile-url http://127.0.0.1:8005 \
  --profiler-dir ./vllm_profile \
  --output-dir benchmarks/results/qwen3_30b_a3b_nonoffload_ascend_pt \
  --sew-moe-trace-path /tmp/sew_moe_trace.jsonl \
  --sew-moe-profile-path /tmp/sew_moe_profile.jsonl \
  --require-sew-moe-artifacts \
  --run-slot-sweep \
  --slot-sweep-range 8:64:8
```

The main outputs are:

- `profile_suite_manifest.json`: profiler windows, benchmark commands, and generated profiler directories
- `ascend_moe_profile_report.md`: TTFT/TPOT-oriented prefill and decode optimization notes
- `ascend_moe_profile_report.json`: machine-readable hotspot and recommendation data
- `sew_moe_p1_plan.json`: machine-readable P1-C compute bucket or P1-T slot sweep plan when trace data supports one
- `<phase>/moe_offload_trace.jsonl`: per-window active expert records for P1-C/RM/T/H decisions
- `<phase>/sew_moe_profile.jsonl`: per-window Stage T/R/C/M timing records
- `<phase>/slot_sweep_lru.json`: optional fixed-slot sweep summary when `--run-slot-sweep` is set

Ascend PyTorch Profiler is still the best source for CANN/runtime/operator
evidence: after `torch_npu.profiler.profiler.analyse()`, open
`<phase>/ASCEND_PROFILER_OUTPUT/trace_view.json` in MindStudio Insight,
Perfetto, or Chrome tracing to inspect NPU kernels, `aclrtMemcpy`, waits, and
stream ordering. It does not know the SEW-Offload business stages by itself
(expert cache hit/miss, slot allocation, host bundle lookup, logical-to-physical
mapping). For an offload-aware timeline, keep
`VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH` set and render the JSONL produced by the
server:

```shell
python3 tools/sew_offload/moe_offload_timeline.py \
  --profile-jsonl benchmarks/results/qwen3_30b_a3b_nonoffload_ascend_pt/decode/sew_moe_profile.jsonl \
  --summary-json /tmp/moe_offload_timeline_summary.json \
  --markdown-output /tmp/moe_offload_timeline.md \
  --chrome-trace-output /tmp/moe_offload_timeline_trace.json
```

The markdown file contains a Mermaid timing diagram for the representative MoE
step, plus stage tables. The Chrome trace contains two lanes: SEW-Offload detail
events (`slot_cache_lookup`, `expert_h2d_load_sync`, `slot_mapping_build`, etc.)
and the coarse MoE pipeline (`T` offload plan/load, `R` token dispatch, `C`
expert MLP, `M` token combine).

To turn the P1 plan into a reproducible benchmark environment matrix:

```shell
python3 tools/sew_offload/materialize_p1_experiments.py \
  --plan benchmarks/results/qwen3_30b_a3b_nonoffload_ascend_pt/sew_moe_p1_plan.json \
  --output benchmarks/results/qwen3_30b_a3b_nonoffload_ascend_pt/sew_moe_p1_experiments.json
```

Then run each experiment case against the same smoke workload:

```shell
python3 tools/sew_offload/run_p1_experiments.py \
  --matrix benchmarks/results/qwen3_30b_a3b_nonoffload_ascend_pt/sew_moe_p1_experiments.json \
  --output-dir benchmarks/results/qwen3_30b_a3b_nonoffload_ascend_pt/p1_experiment_runs \
  --inline-prompt "Hello" \
  --inline-max-output-tokens 16
```

When the plan contains both a compute bucket plan and a slot sweep result, the
matrix includes five comparable cases: `baseline`,
`p1_compute_bucket_trace_only`, `p1_compute_bucket_fast_path`,
`p1_fixed_slot_recommended`, and `p1_compute_bucket_plus_fixed_slot`.
The trace-only compute case measures classifier overhead and eligibility while
leaving math on the fallback path; the fast-path compute case allows
active-expert compaction without fixed-slot offload. The combined case keeps
the trace-backed P1-C grouped-shape fast path enabled while also running the
recommended P1-T fixed-slot offload budget, so NPU runs can measure whether the
non-offload compute path and offload residency path compose.

The runner writes `p1_experiment_summary.json` with each case's source evidence,
its smoke summary, and a `relative_to_baseline` throughput delta; the
`throughput_delta_vs_baseline` table ranks cases by measured improvement over
the baseline.

If profiler output already exists, analyze it directly:

```shell
python3 benchmarks/scripts/analyze_ascend_moe_profile.py \
  --phase mixed:/path/to/mixed/ASCEND_PROFILER_OUTPUT:/path/to/mixed.json:/path/to/mixed/moe_offload_trace.jsonl:/path/to/mixed/sew_moe_profile.jsonl \
  --phase prefill:/path/to/prefill/ASCEND_PROFILER_OUTPUT:/path/to/prefill.json:/path/to/prefill/moe_offload_trace.jsonl:/path/to/prefill/sew_moe_profile.jsonl \
  --phase decode:/path/to/decode/ASCEND_PROFILER_OUTPUT:/path/to/decode.json:/path/to/decode/moe_offload_trace.jsonl:/path/to/decode/sew_moe_profile.jsonl \
  --markdown-output /tmp/ascend_moe_profile_report.md \
  --json-output /tmp/ascend_moe_profile_report.json
```

#### Use benchmark cli

For more flexible and customized use, benchmark cli is also provided to run online/offline benchmarks
Similarly, let's take `Qwen2.5-VL-7B-Instruct` benchmark as an example:

##### Online serving

1. Launch the server:

    ```shell
    vllm serve Qwen2.5-VL-7B-Instruct --max-model-len 16789
    ```

2. Running performance tests using cli
  
    ```shell
    vllm bench serve --model Qwen2.5-VL-7B-Instruct\
    --endpoint-type "openai-chat" --dataset-name hf \
    --hf-split train --endpoint "/v1/chat/completions" \
    --dataset-path "lmarena-ai/vision-arena-bench-v0.1" \
    --num-prompts 200 \
    --request-rate 16
    ```

##### Offline

- **Throughput**

  ```shell
  vllm bench throughput --output-json results/throughput_qwen2_5_7B_tp1.json \
  --model Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 1 --load-format dummy \
  --dataset-path /github/home/.cache/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 200 --backend vllm
  ```

- **Latency**
  
  ```shell
  vllm bench latency --output-json results/latency_qwen2_5_7B_tp1.json \
  --model Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 1 \
  --load-format dummy --num-iters-warmup 5 --num-iters 15
  ```
