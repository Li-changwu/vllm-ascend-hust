<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-dark.png">
    <img alt="vllm-ascend" src="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
vLLM Ascend Plugin
</h3>

<div align="center">

[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/vllm-project/vllm-ascend)

</div>

<p align="center">
| <a href="https://www.hiascend.com/en/"><b>About Ascend</b></a> | <a href="https://docs.vllm.ai/projects/ascend/en/latest/"><b>Documentation</b></a> | <a href="https://slack.vllm.ai"><b>#SIG-Ascend</b></a> | <a href="https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support"><b>Users Forum</b></a> | <a href="https://tinyurl.com/vllm-ascend-meeting"><b>Weekly Meeting</b></a> |
</p>

<p align="center">
<a ><b>English</b></a> | <a href="README.zh.md"><b>中文</b></a>
</p>

---
*Latest News* 🔥

- [2026/05] We released the new official version [v0.18.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.18.0)! Please follow the [official guide](https://docs.vllm.ai/projects/ascend/en/v0.18.0/) to start using vLLM Ascend Plugin on Ascend.
- [2026/02] We released the new official version [v0.13.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.13.0)! Please follow the [official guide](https://docs.vllm.ai/projects/ascend/en/v0.13.0/) to start using vLLM Ascend Plugin on Ascend.
- [2025/12] We released the new official version [v0.11.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.11.0)! Please follow the [official guide](https://docs.vllm.ai/projects/ascend/en/v0.11.0/) to start using vLLM Ascend Plugin on Ascend.
- [2025/09] We released the new official version [v0.9.1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.9.1)! Please follow the [official guide](https://docs.vllm.ai/projects/ascend/en/v0.9.1/tutorials/large_scale_ep.html) to start deploying large-scale Expert Parallelism (EP) on Ascend.
- [2025/08] We hosted the [vLLM Beijing Meetup](https://mp.weixin.qq.com/s/7n8OYNrCC_I9SJaybHA_-Q) with vLLM and Tencent! Please find the meetup slides [here](https://drive.google.com/drive/folders/1Pid6NSFLU43DZRi0EaTcPgXsAzDvbBqF).
- [2025/06] [User stories](https://docs.vllm.ai/projects/ascend/en/latest/community/user_stories/index.html) page is now live! It kicks off with LLaMA-Factory/verl/TRL/GPUStack to demonstrate how vLLM Ascend assists Ascend users in enhancing their experience across fine-tuning, evaluation, reinforcement learning (RL), and deployment scenarios.
- [2025/06] [Contributors](https://docs.vllm.ai/projects/ascend/en/latest/community/contributors.html) page is now live! All contributions deserve to be recorded, thanks for all contributors.
- [2025/05] We've released the first official version [v0.7.3](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.7.3)! We collaborated with the vLLM community to publish a blog post sharing our practice: [Introducing vLLM Hardware Plugin, Best Practice from Ascend NPU](https://blog.vllm.ai/2025/05/12/hardware-plugin.html).
- [2025/03] We hosted the [vLLM Beijing Meetup](https://mp.weixin.qq.com/s/VtxO9WXa5fC-mKqlxNUJUQ) with vLLM team! Please find the meetup slides [here](https://drive.google.com/drive/folders/1Pid6NSFLU43DZRi0EaTcPgXsAzDvbBqF).
- [2025/02] vLLM community officially created [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend) repo for running vLLM seamlessly on the Ascend NPU.
- [2024/12] We are working with the vLLM community to support [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162).

---

## Overview

vLLM Ascend (`vllm-ascend`) is a community maintained hardware plugin for running vLLM seamlessly on the Ascend NPU.

It is the recommended approach for supporting the Ascend backend within the vLLM community. It adheres to the principles outlined in the [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162), providing a hardware-pluggable interface that decouples the integration of the Ascend NPU with vLLM.

By using vLLM Ascend plugin, popular open-source models, including Transformer-like, Mixture-of-Experts (MoE), Embedding, Multi-modal LLMs can run seamlessly on the Ascend NPU.

## About This Fork

This repository is maintained by [vLLM-HUST](https://github.com/vLLM-HUST), focusing on **operator-level optimization** for the vLLM Ascend backend. Our work includes:

- Custom Ascend operator development and optimization (CANN/TIK)
- Performance tuning for attention, MoE, and other critical kernels
- Deep integration with Huawei Ascend hardware features

## Prerequisites

- Hardware: Atlas 800I A2 Inference series, Atlas A2 Training series, Atlas 800I A3 Inference series, Atlas A3 Training series, Atlas 300I Duo (Experimental)
- OS: Linux
- Software:
    - Python >= 3.10, < 3.12
    - CANN == 8.5.1 (Ascend HDK version refers to [here](https://www.hiascend.com/document/detail/zh/canncommercial/83RC2/releasenote/releasenote_0000.html))
    - PyTorch == 2.9.0, torch-npu == 2.9.0
    - vLLM (the same version as vllm-ascend)

## Getting Started

Please use the following recommended versions to get started quickly:

| Version    | Release type | Doc                                  |
|------------|--------------|--------------------------------------|
| v0.19.1rc1 | Latest release candidate | See [QuickStart](https://docs.vllm.ai/projects/ascend/en/latest/quick_start.html) and [Installation](https://docs.vllm.ai/projects/ascend/en/latest/installation.html) for more details |
| v0.18.0 | Latest stable version | See [QuickStart](https://docs.vllm.ai/projects/ascend/en/v0.18.0/quick_start.html) and [Installation](https://docs.vllm.ai/projects/ascend/en/v0.18.0/installation.html) for more details |

## P0 Profiling Findings: Where MoE Inference Spends Its Time

The P0 profiling work (Feature 1) measures the per-stage execution cost of MoE
inference on Ascend NPU and provides the data basis for the downstream GMM
kernel optimization and expert offload work. The table below is the headline
result from the single-card non-offloading Qwen3-30B-A3B mixed-phase profile
(Atlas 800I A2, Ascend PyTorch Profiler):

| OP type | Total | Avg | Share of MoE op time | Cube util |
|---|---:|---:|---:|---:|
| **GroupedMatmul** | 2099.74 ms | 165.7 us | **62.6%** | 91.2% |
| MatMulV2 | 401.67 ms | 21.0 us | 12.0% | 68.8% |
| FusedInferAttentionScore | 179.36 ms | 28.3 us | 5.3% | 86.0% |
| RmsNorm | 158.03 ms | 12.3 us | 4.7% | 0.0% |
| MoeInitRoutingCustom | 112.13 ms | 17.7 us | 3.3% | 0.0% |

End-to-end at this operating point: median TTFT ≈ 1169.6 ms, median TPOT
≈ 252.6 ms, output throughput ≈ 37.4 tok/s.

**Finding 1 — GroupedMatmul is the dominant cost in both phases, and it is
shape-sensitive.** Splitting the same run into prefill and decode windows shows
GMM stays the single largest operator regardless of phase, but its per-call cost
swings with the batched shape:

| Phase | GMM calls | GMM total | GMM avg | GMM share | Cube util |
|---|---:|---:|---:|---:|---:|
| Prefill | 672 | 236.81 ms | **352.4 us** | 59.0% | 95.5% |
| Decode | 6240 | 725.17 ms | **116.2 us** | 55.9% | 91.0% |

**Analysis:** The 3x per-call gap (352 us prefill vs 116 us decode) comes from
the token-per-expert distribution and group shapes, **not kernel inefficiency**
— Cube utilization is already 91–96%, so the compute units are essentially
saturated. The real lever for performance is therefore the shape (tiling and
stable-shape grouped-matmul paths), not generic operator fusion; and prefill
(long prompt, large groups) vs decode (small groups, high frequency) need
different tiling strategies — one set of parameters cannot cover both phases.

**Finding 2 — MoE time is ~80% Cube-bound, ~20% vector/MTE-bound.** Classifying
the top kernels by Cube utilization separates the two independent optimization
tracks:

| Class | Mixed | Prefill | Decode |
|---|---:|---:|---:|
| Cube-bound (AIC: GMM, MatMul, attention) | **80.5%** | 76.3% | 77.8% |
| Vector/MTE-bound (AIV, Cube=0: RmsNorm, routing, permute, slice) | 19.5% | 23.7% | 22.2% |

**Analysis:** The Cube-bound bulk is GMM-led and is the main battlefield for GMM
kernel optimization and expert offload. The remaining vector/MTE-bound part
(RmsNorm, MoeInitRoutingCustom, MoeGatingTopK, MoeTokenUnpermute, Slice — the
routing family totals ≈ 197 ms in the mixed window) is the operator-fusion
candidate set: these are high-frequency short kernels often adjacent to
norm / bias / swiglu / reshape chains, so they meet the structural conditions
for merging. Although this ~20% track is not the largest share, the kernel count
is very high, so launch and stream-scheduling overhead cannot be ignored — it is
the secondary target for TPOT optimization.

**Finding 3 — Expert offload is dominated by host-to-device memcpy; contiguous
batching is the main transfer lever, while CPU pinned source memory mainly
stabilizes small batches.** A Qwen3-30B-A3B expert transfer micro-profile was
run with 100 independent CANN profiler windows per pattern. Each expert payload
is 9.0 MiB (bf16). The `pin` rows use PyTorch CPU `pin_memory=True` as a host
allocation control only; this is not Ascend UVA.

| Transfer pattern | Source | Total / expert | Memcpy / expert | Overhead / expert | Memcpy BW |
|---|---|---:|---:|---:|---:|
| Current two-tensor copy | no-pin | 1.1243 ms | 0.8121 ms | 0.3122 ms | 12.17 GB/s |
| Current two-tensor copy | pin | 0.7795 ms | 0.5154 ms | 0.2641 ms | 18.48 GB/s |
| 4-expert contiguous batch | no-pin | 0.5743 ms | 0.4798 ms | 0.0944 ms | 19.67 GB/s |
| 4-expert contiguous batch | pin | 0.5081 ms | 0.4055 ms | 0.1026 ms | 23.29 GB/s |
| 8-expert contiguous batch | no-pin | 0.5001 ms | 0.4408 ms | 0.0593 ms | 21.41 GB/s |
| 8-expert contiguous batch | pin | 0.4483 ms | 0.3961 ms | 0.0522 ms | 23.83 GB/s |
| 16-expert contiguous batch | no-pin | 0.4807 ms | 0.4457 ms | 0.0350 ms | 21.17 GB/s |
| 16-expert contiguous batch | pin | 0.4197 ms | 0.3879 ms | 0.0318 ms | 24.33 GB/s |

**Analysis:** The current offload miss path copies one expert as two tensors,
so per-expert overhead is still visible (0.31 ms no-pin, 0.26 ms pin). Packing
multiple experts into one contiguous transfer amortizes that overhead: no-pin
falls from 1.1243 ms/expert to 0.4807 ms/expert at 16 experts (2.34x), and pin
falls from 0.7795 ms/expert to 0.4197 ms/expert (1.86x). CPU pinned source
memory improves the small-copy path most strongly (30.7% total-time reduction
for the current path, 52.8% for 2-expert batch), but after 4–16 experts the
transfer is mostly bandwidth-bound and the remaining gain is about 10–13%.
The no-pin small-batch path also has clear long tails, while 4/8/16-expert
contiguous batches are stable; the offload design should therefore prioritize
batched contiguous expert movement before treating pinning as the primary knob.

> Note on wait/MTE ratios: the per-run report also lists a cumulative kernel
> wait ratio (911.7% mixed) and an MTE time ratio (90.8% mixed). These are summed
> across kernels and streams and can exceed 100%, so treat them as relative
> stream-pressure signals, not absolute stall time.

Reproduction steps and per-phase reports are in
[benchmarks/README.md](benchmarks/README.md); each run's report lands under
`benchmarks/results/<run>/ascend_moe_profile_report.md` (the results directory
is not version-controlled).

## Research Branch MoE Offload Service

The `research` branch contains the Ascend MoE expert offload prototype. The
validated single-NPU Qwen3-30B-A3B service command is:

```bash
MODEL_PATH=${MODEL_PATH:-/data/shared-models/Qwen3-30B-A3B}
PORT=${PORT:-8016}
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-6}

ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES} \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name qwen3-30b-a3b \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 512 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 512 \
  --kv-cache-memory-bytes 536870912 \
  --enforce-eager \
  --ascend-moe-offload-gb 14
```

MoE offload currently runs through the eager path. Keep `--enforce-eager`
enabled for this prototype because graph capture cannot record the runtime CPU
decision path used by the offload scheduler.

## Contributing

See [CONTRIBUTING](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/contribution/index.html) for more details, which is a step-by-step guide to help you set up the development environment, build and test.

We welcome and value any contributions and collaborations:

- Please let us know if you encounter a bug by [filing an issue](https://github.com/vllm-project/vllm-ascend/issues)
- Please use [User forum](https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support) for usage questions and help.

## Branch

vllm-ascend has a main branch and a dev branch.

- **main**: main branch, corresponds to the vLLM main branch, and is continuously monitored for quality through Ascend CI.
- **releases/vX.Y.Z**: development branch, created alongside new releases of vLLM. For example, `releases/v0.13.0` is the dev branch for vLLM `v0.13.0` version.

Below are the maintained branches:

| Branch           | Status       | Note                                 |
|------------------|--------------|--------------------------------------|
| main             | Maintained   | CI commitment for vLLM main branch and vLLM v0.18.0 tag |
| v0.7.1-dev       | Unmaintained | Outdated, no longer maintained. |
| v0.7.3-dev       | Unmaintained | Only bug fixes are allowed, and no new release tags anymore. |
| v0.9.1-dev       | Unmaintained | Only bug fixes are allowed, and no new release tags anymore. |
| v0.11.0-dev      | Unmaintained | Only bug fixes are allowed, and no new release tags anymore. |
| releases/v0.13.0 | Maintained   | CI commitment for vLLM 0.13.0 version |
| releases/v0.18.0 | Maintained   | CI commitment for vLLM 0.18.0 version |
| rfc/feature-name | Maintained   | [Feature branches](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html#feature-branches) for collaboration |
  
Please refer to [Versioning policy](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html) for more details.

## Weekly Meeting

- vLLM Ascend Weekly Meeting: <https://tinyurl.com/vllm-ascend-meeting>
- Wednesday, 15:00 - 16:00 (UTC+8, [Convert to your timezone](https://dateful.com/convert/gmt8?t=15))

## License

Apache License 2.0, as found in the [LICENSE](./LICENSE) file.
