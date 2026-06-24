"""
Benchmark: TTFT, TPOT, throughput on a running vLLM OpenAI server.
Supports concurrent requests via CONCURRENCY setting.
Usage: python bench_sharegpt.py
"""
import asyncio, json, time, random, statistics, sys
import aiohttp

# ── config ──────────────────────────────────────────────────────────────────
BASE_URL         = "http://localhost:8016"
MODEL            = "qwen3-30b-a3b"
DATASET          = "/data/shared_datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
N_PROMPTS        = 100
MAX_TOKENS       = 20
MAX_PROMPT_CHARS = 800
CONCURRENCY      = 1      # number of requests in-flight simultaneously
SEED             = 42
# ────────────────────────────────────────────────────────────────────────────

def load_prompts(path, n, max_chars, seed):
    with open(path) as f:
        data = json.load(f)
    prompts = []
    for item in data:
        convs = item.get("conversations") or item.get("conversation") or []
        for turn in convs:
            role = turn.get("from", turn.get("role", ""))
            val  = turn.get("value", turn.get("content", ""))
            if role in ("human", "user") and 10 < len(val) <= max_chars:
                prompts.append(val)
    random.seed(seed)
    random.shuffle(prompts)
    return prompts[:n]

async def stream_request(session, prompt, idx):
    """Send one streaming chat request; return (idx, ttft_s, output_tokens, total_s)."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }
    ttft = None
    output_tokens = 0
    t0 = time.perf_counter()
    async with session.post(f"{BASE_URL}/v1/chat/completions", json=payload) as resp:
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            delta = obj["choices"][0]["delta"].get("content", "")
            if delta:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                output_tokens += 1
    total = time.perf_counter() - t0
    return idx, ttft, output_tokens, total

async def main():
    prompts = load_prompts(DATASET, N_PROMPTS, MAX_PROMPT_CHARS, SEED)
    print(f"Loaded {len(prompts)} prompts. Concurrency={CONCURRENCY}. Starting benchmark …\n")

    results = [None] * len(prompts)
    sem = asyncio.Semaphore(CONCURRENCY)

    async def bounded(session, prompt, idx):
        async with sem:
            return await stream_request(session, prompt, idx)

    wall_start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = [bounded(session, p, i) for i, p in enumerate(prompts)]
        for coro in asyncio.as_completed(tasks):
            idx, ttft, out_tok, e2el = await coro
            results[idx] = (ttft, out_tok, e2el)
            print(f"  [{idx+1:3d}] TTFT={ttft*1000:6.1f}ms  "
                  f"TPOT={(e2el-ttft)/out_tok*1000:6.2f}ms/tok  out={out_tok}tok"
                  if ttft else f"  [{idx+1:3d}] empty response")
            sys.stdout.flush()

    wall = time.perf_counter() - wall_start

    ttfts, tpots, e2els, total_out = [], [], [], 0
    for r in results:
        if r is None:
            continue
        ttft, out_tok, e2el = r
        if ttft is None or out_tok == 0:
            continue
        ttfts.append(ttft * 1000)
        tpots.append((e2el - ttft) / out_tok * 1000 if out_tok > 1 else 0.0)
        e2els.append(e2el)
        total_out += out_tok

    def p(data, pct): return statistics.quantiles(data, n=100)[pct-1]

    print(f"""
{'='*55}
Results ({len(ttfts)} successful requests, concurrency={CONCURRENCY})
{'='*55}
TTFT  (ms)   mean={statistics.mean(ttfts):.1f}  p50={p(ttfts,50):.1f}  p90={p(ttfts,90):.1f}  p99={p(ttfts,99):.1f}
TPOT  (ms)   mean={statistics.mean(tpots):.2f}  p50={p(tpots,50):.2f}  p90={p(tpots,90):.2f}  p99={p(tpots,99):.2f}
E2EL  (s)    mean={statistics.mean(e2els):.2f}
Throughput   {total_out/wall:.1f} output-tokens/s  (wall {wall:.1f}s, {total_out} tok total)
{'='*55}
""")

if __name__ == "__main__":
    asyncio.run(main())
