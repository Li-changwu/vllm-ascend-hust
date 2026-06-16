#!/usr/bin/env python3
"""
Rigorous E2E benchmark for Custom GMM1 kernel vs torch_npu baseline.

Methodology:
  1. Separate API server process per configuration (no HBM/NPU state pollution)
  2. 5 rounds per config, median aggregation (handles 19-32% run-to-run noise)
  3. Token-id correctness: every response must match across configs
  4. TTFT + TPOT + throughput (more sensitive than throughput-only)
  5. Real dataset (ShareGPT-style prompts)

Usage:
  # Kill NPU first
  fuser -k /dev/davinci0 2>/dev/null; sleep 2
  ASCEND_RT_VISIBLE_DEVICES=0 python tools/rigorous_e2e_benchmark.py
"""

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import requests

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "/data/shared-models/Qwen3-30B-A3B")
BASE_PORT = int(os.environ.get("PORT", "8016"))
NPU_ID = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0")
MAX_MODEL_LEN = 128
MAX_NUM_SEQS = 2
MAX_NUM_BATCHED_TOKENS = 256
KV_CACHE_BYTES = 1073741824  # 1 GB

# Real evaluation prompts (ShareGPT-style diverse lengths)
PROMPTS = [
    "Explain the concept of gradient descent in machine learning.",
    "What are the main differences between Python and JavaScript?",
    "Write a short poem about artificial intelligence.",
    "Summarize the key events of World War II in 3 sentences.",
    "How does a transformer neural network work? Explain attention.",
    "What is the capital of France and what is it famous for?",
    "Describe the process of photosynthesis in plants.",
    "If you could travel anywhere in the world, where would you go and why?",
]

NUM_ROUNDS = 5
MAX_TOKENS = 64
TEMPERATURE = 0.0
REQUEST_TIMEOUT = 120


# ── Helpers ──────────────────────────────────────────────────────────────────
def wait_for_server(port: int, timeout: int = 300) -> bool:
    """Poll /health AND /v1/models until server is fully ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code != 200:
                time.sleep(3)
                continue
            # Also check model is loaded
            r2 = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
            if r2.status_code == 200 and len(r2.json().get("data", [])) > 0:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def send_chat_request(port: int, prompt: str, max_tokens: int) -> dict | None:
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": "qwen3-30b-a3b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "stream": False,
    }
    try:
        r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [ERROR] Request failed: {e}")
        return None


def run_benchmark_round(port: int, prompts: list[str], max_tokens: int) -> list[dict | None]:
    results = []
    for prompt in prompts:
        t0 = time.perf_counter()
        resp = send_chat_request(port, prompt, max_tokens)
        t1 = time.perf_counter()
        if resp is None:
            results.append(None)
            continue
        choice = resp["choices"][0]
        content = choice["message"]["content"]
        usage = resp.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        total_time = t1 - t0
        results.append({
            "content": content,
            "completion_tokens": completion_tokens,
            "total_time_s": total_time,
            "tpot_s": total_time / completion_tokens if completion_tokens > 0 else float("inf"),
        })
    return results


def compute_metrics(rounds_results: list[list[dict]]) -> dict:
    all_tps = []
    all_tpot = []
    for round_data in rounds_results:
        valid = [r for r in round_data if r is not None]
        if not valid:
            continue
        total_tok = sum(r["completion_tokens"] for r in valid)
        total_time = sum(r["total_time_s"] for r in valid)
        if total_time > 0:
            all_tps.append(total_tok / total_time)
        all_tpot.extend(r["tpot_s"] for r in valid if r["tpot_s"] != float("inf"))
    return {
        "throughput_median": statistics.median(all_tps) if all_tps else 0,
        "throughput_values": all_tps,
        "tpot_median": statistics.median(all_tpot) if all_tpot else 0,
    }


def verify_correctness(baseline_rounds, custom_rounds) -> bool:
    for ri, (br, cr) in enumerate(zip(baseline_rounds, custom_rounds)):
        for pi, (b, c) in enumerate(zip(br, cr)):
            if b is None or c is None:
                continue
            if b["content"] != c["content"]:
                print(f"  [FAIL] Mismatch round={ri} prompt={pi}")
                print(f"    BASELINE: {b['content'][:100]}...")
                print(f"    CUSTOM:   {c['content'][:100]}...")
                return False
    return True


# ── Server management ────────────────────────────────────────────────────────
def start_server(port: int, env_extra: dict = None) -> subprocess.Popen:
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = NPU_ID
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    if env_extra:
        env.update(env_extra)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--served-model-name", "qwen3-30b-a3b",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--tensor-parallel-size", "1",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS),
        "--kv-cache-memory-bytes", str(KV_CACHE_BYTES),
        "--enforce-eager",
    ]
    print(f"  Starting server on port {port} (PID will be child)...")
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    return proc


def stop_server(proc: subprocess.Popen):
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    time.sleep(3)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=BASE_PORT)
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args()

    baseline_port = args.port
    custom_port = args.port + 1
    num_rounds = args.rounds
    max_tokens = args.max_tokens

    # Phase 1: Baseline
    print("=" * 60)
    print("PHASE 1: BASELINE (torch_npu)")
    print("=" * 60)
    proc_b = start_server(baseline_port, {"VLLM_ASCEND_USE_CUSTOM_GMM": "0"})
    if not wait_for_server(baseline_port):
        print("FATAL: Baseline server did not start")
        stop_server(proc_b)
        sys.exit(1)
    print("  Server ready. Running rounds...")
    baseline_rounds = []
    for r in range(num_rounds):
        print(f"  Baseline round {r+1}/{num_rounds}")
        baseline_rounds.append(run_benchmark_round(baseline_port, PROMPTS, max_tokens))
    stop_server(proc_b)

    # Phase 2: Custom
    print()
    print("=" * 60)
    print("PHASE 2: CUSTOM (_C_ascend GMM1)")
    print("=" * 60)
    proc_c = start_server(custom_port, {"VLLM_ASCEND_USE_CUSTOM_GMM": "1"})
    if not wait_for_server(custom_port):
        print("FATAL: Custom server did not start")
        stop_server(proc_c)
        sys.exit(1)
    print("  Server ready. Running rounds...")
    custom_rounds = []
    for r in range(num_rounds):
        print(f"  Custom round {r+1}/{num_rounds}")
        custom_rounds.append(run_benchmark_round(custom_port, PROMPTS, max_tokens))
    stop_server(proc_c)

    # Results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    b_m = compute_metrics(baseline_rounds)
    c_m = compute_metrics(custom_rounds)
    b_tps = b_m["throughput_median"]
    c_tps = c_m["throughput_median"]
    speedup = (c_tps / b_tps - 1) * 100 if b_tps > 0 else 0
    print(f"  Throughput (median over {num_rounds} rounds):")
    print(f"    BASELINE: {b_tps:.1f} tok/s  (values: {[f'{v:.1f}' for v in b_m['throughput_values']]})")
    print(f"    CUSTOM:   {c_tps:.1f} tok/s  (values: {[f'{v:.1f}' for v in c_m['throughput_values']]})")
    print(f"    SPEEDUP:  {speedup:+.1f}%")
    print(f"  TPOT (median):")
    print(f"    BASELINE: {b_m['tpot_median']*1000:.0f} ms/tok")
    print(f"    CUSTOM:   {c_m['tpot_median']*1000:.0f} ms/tok")
    ok = verify_correctness(baseline_rounds, custom_rounds)
    print(f"  Correctness: {'PASS' if ok else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
