# MoE GMM Stage Runtime Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Qwen3 MoE GMM stage benchmark conclusion into a machine-readable runtime recommendation that keeps custom GMM2 disabled while preserving a gated custom GMM1 candidate.

**Architecture:** Reuse `tools/moe_gmm/compare_gmm_stages.py` as the stage-analysis source of truth. Add a summary object that downstream scripts and reports can consume without parsing prose.

**Tech Stack:** Python, pytest, existing JSON microbench outputs.

---

### Task 1: Runtime Recommendation Summary

**Files:**
- Modify: `tools/moe_gmm/compare_gmm_stages.py`
- Modify: `tests/ut/tools/test_moe_gmm_stage_compare.py`

- [x] **Step 1: Write the failing test**

Add a test that builds one positive `gmm1_default` case and one negative `gmm2_default` case, then asserts `summarize_stage_files()` returns `runtime_recommendation` with:
- `gmm1_default.backend == "custom"` and `fanout_threshold == 40`
- `gmm2_default.backend == "torch_npu"` and `reason` mentions `reverse_analysis_required`
- `safe_to_enable_runtime == False`

- [x] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/ut/tools/test_moe_gmm_stage_compare.py::test_summarize_stage_files_emits_conservative_runtime_recommendation`

Expected: FAIL with missing `runtime_recommendation`.

- [x] **Step 3: Write minimal implementation**

Add `_runtime_recommendation(cases)` to `compare_gmm_stages.py` and include its result in `summarize_stage_files()`. Keep the policy conservative: GMM2 is always `torch_npu` in this phase; GMM1 may expose the existing threshold candidate but the top-level recommendation remains not safe for automatic runtime enablement.

- [x] **Step 4: Run targeted tests**

Run: `pytest -q tests/ut/tools/test_moe_gmm_stage_compare.py`

Expected: PASS.

- [x] **Step 5: Run supporting tool tests**

Run: `pytest -q tests/ut/tools/test_moe_gmm_microbench.py tests/ut/tools/test_moe_gmm_stage_compare.py`

Expected: PASS.

### Task 2: GMM1 Fanout Profiler Diagnostics

**Files:**
- Modify: `tools/moe_gmm/compare_gmm_stages.py`
- Modify: `tests/ut/tools/test_moe_gmm_stage_compare.py`
- Generate: `benchmarks/results/qwen3_30b_a3b_gmm_stage_analysis_20260614T141926Z/gmm1_profiler_diagnostic_plan.md`

- [x] **Step 1: Write the failing test**

Add a test that builds a `gmm1_default` case with winning, losing, and neutral fanout rows. Assert `summarize_stage_files()` returns `gmm1_profiler_diagnostics` with:
- `status == "profile_before_kernel_changes"`
- winning fanouts include the strongest positive rows
- losing fanouts include the strongest negative rows
- profiler commands are for `--mode gmm1` and include both `counts:s2` and `counts:s2:custom`

- [x] **Step 2: Run test to verify it fails**

Run: `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/tools/test_moe_gmm_stage_compare.py::test_summarize_stage_files_emits_gmm1_profiler_diagnostics`

Expected: FAIL with missing `gmm1_profiler_diagnostics`.

- [x] **Step 3: Write minimal implementation**

Add `_gmm1_profiler_diagnostics(cases)` and include it in `summarize_stage_files()`. The function should rank rows by improvement, select top winning and losing fanouts, add a neutral/mid bucket when available, and emit command templates for torch_npu and custom gmm1 microbench runs.

- [x] **Step 4: Render diagnostics in Markdown**

Add a `GMM1 Profiler Diagnostics` Markdown section with candidate fanouts, why they were selected, and command templates.

- [x] **Step 5: Generate the diagnostic plan for the current Qwen3 result**

Run `tools/moe_gmm/compare_gmm_stages.py` against the four existing stage microbench JSON files and write the updated `gmm_stage_summary.{json,md}`. Then extract the `GMM1 Profiler Diagnostics` section to `gmm1_profiler_diagnostic_plan.md`.

- [x] **Step 6: Run available verification**

Run:
- `python3 -m py_compile tools/moe_gmm/compare_gmm_stages.py tests/ut/tools/test_moe_gmm_stage_compare.py`
- a direct Python assertion that the generated summary contains `gmm1_profiler_diagnostics`

### Task 3: Fanout-Filtered GMM Microbench

**Files:**
- Modify: `tools/moe_gmm/benchmark_grouped_matmul.py`
- Modify: `tools/moe_gmm/compare_gmm_stages.py`
- Modify: `tests/ut/tools/test_moe_gmm_microbench.py`
- Modify: `tests/ut/tools/test_moe_gmm_stage_compare.py`
- Generate: `benchmarks/results/qwen3_30b_a3b_gmm_stage_analysis_20260614T141926Z/gmm1_profiler_diagnostic_plan.md`

- [x] **Step 1: Write failing fanout filter tests**

Add tests that verify `load_shape_classes(..., fanouts={40, 41})` keeps only those fanouts and that an empty filter result returns an empty class list rather than falling back to all classes.

- [x] **Step 2: Write failing command-generation test**

Update the GMM1 diagnostics test to assert profiler commands include `--fanouts`.

- [x] **Step 3: Run tests to verify failure**

Run:
- `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/tools/test_moe_gmm_microbench.py::test_load_shape_classes_filters_by_fanout`
- `/root/miniconda3/envs/vllm-hust-dev/bin/python -m pytest -q tests/ut/tools/test_moe_gmm_stage_compare.py::test_summarize_stage_files_emits_gmm1_profiler_diagnostics`

Expected: FAIL because `load_shape_classes` has no `fanouts` argument and commands do not include `--fanouts`.

- [x] **Step 4: Implement fanout filtering**

Add `fanouts: set[int] | None = None` to `load_shape_classes()`, filter grouped trace records by `record["fanout"]` when provided, and add a `--fanouts` CLI option parsed as comma-separated integers.

- [x] **Step 5: Update diagnostics commands**

Change `_gmm1_profiler_commands()` to pass `--fanouts <selected_csv>`, and update the note to say the benchmark now captures only selected fanouts.

- [x] **Step 6: Regenerate reports and verify**

Regenerate `gmm_stage_summary.{json,md}` and `gmm1_profiler_diagnostic_plan.md`, then run the two targeted test files.
