<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-dark.png">
    <img alt="vllm-ascend" src="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
vLLM Ascend Plugin
</h3>

<p align="center">
| <a href="https://www.hiascend.com/en/"><b>关于昇腾</b></a> | <a href="https://docs.vllm.ai/projects/ascend/en/latest/"><b>官方文档</b></a> | <a href="https://slack.vllm.ai"><b>#sig-ascend</b></a> | <a href="https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support"><b>用户论坛</b></a> | <a href="https://tinyurl.com/vllm-ascend-meeting"><b>社区例会</b></a> |
</p>

<p align="center">
<a href="README.md"><b>English</b></a> | <a><b>中文</b></a>
</p>

---
*最新消息* 🔥

- [2026/05] 我们发布了新的正式版本 [v0.18.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.18.0)! 请按照[官方指南](https://docs.vllm.ai/projects/ascend/en/v0.18.0/)开始在Ascend上部署vLLM Ascend Plugin。
- [2026/02] 我们发布了新的正式版本 [v0.13.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.13.0)! 请按照[官方指南](https://docs.vllm.ai/projects/ascend/en/v0.13.0/)开始在Ascend上部署vLLM Ascend Plugin。
- [2025/12] 我们发布了新的正式版本 [v0.11.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.11.0)! 请按照[官方指南](https://docs.vllm.ai/projects/ascend/en/v0.11.0/)开始在Ascend上部署vLLM Ascend Plugin。
- [2025/09] 我们发布了新的正式版本 [v0.9.1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.9.1)! 请按照[官方指南](https://docs.vllm.ai/projects/ascend/en/v0.9.1/tutorials/large_scale_ep.html)开始在Ascend上部署大型专家并行 (EP)。
- [2025/08] 我们与vLLM和腾讯合作举办了[vLLM北京Meetup](https://mp.weixin.qq.com/s/7n8OYNrCC_I9SJaybHA_-Q)，！请在[这里](https://drive.google.com/drive/folders/1Pid6NSFLU43DZRi0EaTcPgXsAzDvbBqF)找到演讲材料。
- [2025/06] [用户案例](https://docs.vllm.ai/projects/ascend/en/latest/community/user_stories/index.html)现已上线！展示了LLaMA-Factory/verl/TRL/GPUStack等用户案例，展示了vLLM Ascend如何帮助昇腾用户在模型微调、评估、强化学习 (RL) 以及部署等场景中提升体验。
- [2025/06] [贡献者](https://docs.vllm.ai/projects/ascend/en/latest/community/contributors.html)页面现已上线！所有的贡献都值得被记录，感谢所有的贡献者。
- [2025/05] 我们发布了首个正式版本 [v0.7.3](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.7.3)！我们与 vLLM 社区合作发布了一篇博客文章，分享了我们的实践：[Introducing vLLM Hardware Plugin, Best Practice from Ascend NPU](https://blog.vllm.ai/2025/05/12/hardware-plugin.html)。
- [2025/03] 我们和vLLM团队举办了[vLLM Beijing Meetup](https://mp.weixin.qq.com/s/CGDuMoB301Uytnrkc2oyjg)! 你可以在[这里](https://drive.google.com/drive/folders/1Pid6NSFLU43DZRi0EaTcPgXsAzDvbBqF)找到演讲材料.
- [2025/02] vLLM社区正式创建了[vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend)仓库，让vLLM可以无缝运行在Ascend NPU。
- [2024/12] 我们正在与 vLLM 社区合作，以支持 [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162).

---

## 总览

vLLM 昇腾插件 (`vllm-ascend`) 是一个由社区维护的让vLLM在Ascend NPU无缝运行的后端插件。

此插件是 vLLM 社区中支持昇腾后端的推荐方式。它遵循[[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162)所述原则：通过解耦的方式提供了vLLM对Ascend NPU的支持。

使用 vLLM 昇腾插件，可以让类Transformer、混合专家(MOE)、嵌入、多模态等流行的大语言模型在 Ascend NPU 上无缝运行。

## 关于本分支

本仓库由 [vLLM-HUST](https://github.com/vLLM-HUST) 维护，专注于 **vLLM Ascend 后端的算子级优化**。工作内容包括：

- 自定义 Ascend 算子（CANN/TIK）的开发和优化
- Attention、MoE 等关键算子的性能调优
- 与华为昇腾硬件特性的深度集成

## P0 性能剖析结论：MoE 推理的时间花在哪里

P0 剖析工作（功能一）测量 Ascend NPU 上 MoE 推理各阶段的执行开销，为下游的
GMM 算子优化与专家卸载工作提供数据依据。下表是单卡非卸载 Qwen3-30B-A3B 在
mixed 阶段的核心结果（Atlas 800I A2，Ascend PyTorch Profiler）：

| 算子类型 | 总耗时 | 平均 | 占 MoE 算子时间 | Cube 利用率 |
|---|---:|---:|---:|---:|
| **GroupedMatmul** | 2099.74 ms | 165.7 us | **62.6%** | 91.2% |
| MatMulV2 | 401.67 ms | 21.0 us | 12.0% | 68.8% |
| FusedInferAttentionScore | 179.36 ms | 28.3 us | 5.3% | 86.0% |
| RmsNorm | 158.03 ms | 12.3 us | 4.7% | 0.0% |
| MoeInitRoutingCustom | 112.13 ms | 17.7 us | 3.3% | 0.0% |

该工况下端到端指标：中位 TTFT ≈ 1169.6 ms，中位 TPOT ≈ 252.6 ms，输出吞吐
≈ 37.4 tok/s。

**结论一 —— GroupedMatmul 在 prefill 和 decode 两个阶段都是头号开销，且对
shape 敏感。** 把同一次运行拆成 prefill 与 decode 两个窗口后可以看到，无论哪个
阶段 GMM 都是占比最高的单一算子，但其单次调用耗时随 batch 后的 shape 大幅波动：

| 阶段 | GMM 调用数 | GMM 总耗时 | GMM 平均 | GMM 占比 | Cube 利用率 |
|---|---:|---:|---:|---:|---:|
| Prefill | 672 | 236.81 ms | **352.4 us** | 59.0% | 95.5% |
| Decode | 6240 | 725.17 ms | **116.2 us** | 55.9% | 91.0% |

**分析：** 单次调用 3 倍的差距（prefill 352 us vs decode 116 us）来自
token-per-expert 分布和 group shape 的差异，而**不是 kernel 本身的低效** ——
Cube 利用率已经达到 91–96%，计算单元基本打满。这意味着真正能撬动性能的杠杆是
shape（tiling 与 stable-shape grouped-matmul 路径），而非通用算子融合；同时
prefill（长 prompt、大 group）与 decode（小 group、高频）需要不同的 tiling
策略，不能用一套参数覆盖两个阶段。

**结论二 —— MoE 时间约 80% 是计算约束（Cube-bound），约 20% 是向量/访存约束
（vector/MTE-bound）。** 按 Cube 利用率把 top kernel 分成两类，可以清晰区分出
两条独立的优化轨道：

| 类别 | Mixed | Prefill | Decode |
|---|---:|---:|---:|
| 计算约束（AIC：GMM、MatMul、Attention） | **80.5%** | 76.3% | 77.8% |
| 向量/访存约束（AIV，Cube=0：RmsNorm、routing、permute、slice） | 19.5% | 23.7% | 22.2% |

**分析：** 计算约束的大头由 GMM 主导，是 GMM 算子优化与专家卸载的主战场；剩下的
向量/访存约束部分（RmsNorm、MoeInitRoutingCustom、MoeGatingTopK、
MoeTokenUnpermute、Slice —— 路由相关算子在 mixed 窗口合计约 197 ms）则是算子
融合的候选集：这些都是高频短 kernel，前后常紧邻 norm / bias / swiglu / reshape
链，具备合并条件。这条占比约 20% 的轨道虽非最大头，但 kernel 数量极多，launch
与 stream 调度开销不可忽视，是 TPOT 优化的次级目标。

> 关于 wait / MTE 比率的说明：单次运行报告还会给出累计 kernel wait 比率（mixed
> 为 911.7%）和 MTE 时间比率（mixed 为 90.8%）。这两个值是跨 kernel 与跨 stream
> 累加的，会超过 100%，因此应当把它们当作**相对的 stream 压力信号**，而不是绝对
> 的 stall 时间。

复现方式与逐阶段报告见 [benchmarks/README.md](benchmarks/README.md)；每次运行的
报告会落在 `benchmarks/results/<run>/ascend_moe_profile_report.md`（results 目录
不纳入版本控制）。

## 准备

- 硬件：Atlas 800I A2 Inference系列、Atlas A2 Training系列、Atlas 800I A3 Inference系列、Atlas A3 Training系列、Atlas 300I Duo（实验性支持）
- 操作系统：Linux
- 软件：
    - Python >= 3.10, < 3.12
    - CANN == 8.5.1 (Ascend HDK 版本参考[这里](https://www.hiascend.com/document/detail/zh/canncommercial/83RC2/releasenote/releasenote_0000.html))
    - PyTorch == 2.9.0, torch-npu == 2.9.0
    - vLLM (与vllm-ascend版本一致)

## 开始使用

推荐您使用以下版本快速开始使用：

| Version    | Release type | Doc                                  |
|------------|--------------|--------------------------------------|
|v0.19.1rc1| 最新RC版本 |请查看[快速开始](https://docs.vllm.ai/projects/ascend/en/latest/quick_start.html)和[安装指南](https://docs.vllm.ai/projects/ascend/en/latest/installation.html)了解更多|
|v0.18.0| 最新正式/稳定版本 |[快速开始](https://docs.vllm.ai/projects/ascend/en/v0.18.0/quick_start.html) and [安装指南](https://docs.vllm.ai/projects/ascend/en/v0.18.0/installation.html)了解更多|

## 贡献

请参考[CONTRIBUTING](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/contribution/index.html)文档了解更多关于开发环境搭建、功能测试以及 PR 提交规范的信息。

我们欢迎并重视任何形式的贡献与合作：

- 请通过[Issue](https://github.com/vllm-project/vllm-ascend/issues)来告知我们您遇到的任何Bug。
- 请通过[用户论坛](https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support)来交流使用问题和寻求帮助。

## 分支策略

vllm-ascend有主干分支和开发分支。

- **main**: 主干分支，与vLLM的主干分支对应，并通过昇腾CI持续进行质量看护。
- **releases/vX.Y.Z**: 开发分支，随vLLM部分新版本发布而创建，比如`releases/v0.13.0`是vllm-ascend针对vLLM `v0.13.0` 版本的开发分支。

下面是维护中的分支：

| 分支              | 状态         | 备注                  |
|------------------|--------------|----------------------|
| main             | Maintained   | 基于vLLM main分支和vLLM最新版本（v0.18.0）CI看护   |
| v0.7.1-dev       | Unmaintained | 不再维护 |
| v0.7.3-dev       | Unmaintained | 只允许Bug修复，不会再发布新版本 |
| v0.9.1-dev       | Unmaintained | 只允许Bug修复，不会再发布新版本 |
| v0.11.0-dev      | Unmaintained | 只允许Bug修复，不会再发布新版本 |
| releases/v0.13.0 | Maintained   | 基于vLLM v0.13.0版本CI看护 |
| releases/v0.18.0 | Maintained   | 基于vLLM v0.18.0版本CI看护 |
| rfc/feature-name | Maintained   | 为协作创建的[特性分支](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html#feature-branches) |

请参阅[版本策略](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html)了解更多详细信息。

## 社区例会

- vLLM Ascend 每周社区例会: <https://tinyurl.com/vllm-ascend-meeting>
- 每周三下午，15:00 - 16:00 (UTC+8, [查看您的时区](https://dateful.com/convert/gmt8?t=15))

## 许可证

Apache 许可证 2.0，如 [LICENSE](./LICENSE) 文件中所示。
