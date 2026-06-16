"""E2E ablation: isolate batch vs seq_len effects on GMM1 speedup."""
import time, torch_npu, statistics
from vllm import LLM, SamplingParams

llm = LLM(model='/data/shared-models/Qwen3-30B-A3B', dtype='bfloat16',
          max_model_len=128, max_num_seqs=8, max_num_batched_tokens=256,
          tensor_parallel_size=1, trust_remote_code=True, enforce_eager=True,
          gpu_memory_utilization=0.98)

p4 = ['The capital of France is', 'Machine learning is a',
      'Python is a programming', 'Hello, my name is']
p8 = p4 + ['The theory of relativity', 'Deep neural networks can',
           'In the year 2050,', 'Quantum computing promises']

def run_bench(label, prompts, max_tok, rounds=3):
    sp = SamplingParams(max_tokens=max_tok, temperature=0)
    results = []
    for r in range(rounds):
        torch_npu.npu.synchronize()
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp)
        torch_npu.npu.synchronize()
        t1 = time.perf_counter()
        tok = sum(len(o.outputs[0].token_ids) for o in outs if o.outputs)
        tps = tok / (t1 - t0)
        results.append(tps)
        print(f'  {label} R{r+1}: {tok}tok in {t1-t0:.1f}s = {tps:.1f} tok/s', flush=True)
    return results

import vllm_ascend.ops.fused_moe.moe_mlp as moe_mlp
import vllm_ascend.vllm_ascend_C

_orig = moe_mlp.unquant_apply_mlp

def _custom_fn(hidden_states, w1, w2, group_list, w1_bias=None, w2_bias=None,
               activation=None, group_list_type=1, topk_scales=None, need_trans=True):
    if need_trans:
        w1 = w1.transpose(1, 2); w2 = w2.transpose(1, 2)
    act_name = getattr(activation, 'value', activation)
    counts = group_list.to(torch_npu.int64) if group_list_type == 1 else group_list
    if group_list_type == 0:
        diffs = counts[1:] - counts[:-1]
        counts = torch_npu.cat([counts[:1], diffs])
    active_ids = torch_npu.nonzero(counts > 0).reshape(-1).to(torch_npu.int64)
    active_counts = counts.index_select(0, active_ids.long())
    gl2 = torch_npu.stack([active_ids, active_counts], dim=1)
    gate_up_out = torch_npu.ops._C_ascend.moe_grouped_matmul(hidden_states, w1, gl2, 2, 0, 2)[0]
    if act_name == 'swigluoai':
        gate_up_out = moe_mlp.AscendSwigluOAIAndMul.swiglu_oai_forward(
            gate_up_out.view(-1, w1.shape[2]))
    else:
        gate_up_out = torch_npu.npu_swiglu(gate_up_out)
    if topk_scales is not None:
        gate_up_out *= topk_scales
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[gate_up_out], weight=[w2],
        bias=[w2_bias.to(dtype=torch_npu.float32)] if w2_bias is not None else None,
        split_item=2, group_list_type=group_list_type, group_type=0,
        group_list=group_list)[0]
    return hidden_states

def set_baseline():
    moe_mlp.unquant_apply_mlp = _orig

def set_custom():
    moe_mlp.unquant_apply_mlp = _custom_fn

# === Warmup ===
print('Warming up...', flush=True)
for _ in range(2):
    llm.generate(p4, SamplingParams(max_tokens=16, temperature=0))
torch_npu.npu.synchronize()

all_results = {}

# === TEST A: 4p 32t (reproduce) ===
print('\n' + '='*60)
print('TEST A: 4 prompts, 32 tokens (reproduce baseline)')
print('='*60)
set_baseline()
bA = run_bench('BASELINE', p4, 32)
set_custom()
cA = run_bench('CUSTOM', p4, 32)
all_results['A:4p32t'] = (bA, cA)

# === TEST B: 8p 32t (isolate BATCH) ===
print('\n' + '='*60)
print('TEST B: 8 prompts, 32 tokens (isolate BATCH increase)')
print('='*60)
set_baseline()
bB = run_bench('BASELINE', p8, 32)
set_custom()
cB = run_bench('CUSTOM', p8, 32)
all_results['B:8p32t'] = (bB, cB)

# === TEST C: 4p 64t (isolate SEQ_LEN) ===
print('\n' + '='*60)
print('TEST C: 4 prompts, 64 tokens (isolate SEQ_LEN increase)')
print('='*60)
set_baseline()
bC = run_bench('BASELINE', p4, 64)
set_custom()
cC = run_bench('CUSTOM', p4, 64)
all_results['C:4p64t'] = (bC, cC)

# === TEST D: Custom FIRST (thermal check) ===
print('\n' + '='*60)
print('TEST D: 8p 64t CUSTOM first, BASELINE second (thermal check)')
print('='*60)
set_custom()
cD = run_bench('CUSTOM', p8, 64)
set_baseline()
bD = run_bench('BASELINE', p8, 64)
all_results['D:8p64t(rev)'] = (bD, cD)

# === Summary ===
print('\n' + '#'*60)
print('ABLATION SUMMARY')
print('#'*60)
for name, (b, c) in all_results.items():
    ba, ca = statistics.mean(b), statistics.mean(c)
    spd = (ca / ba - 1) * 100
    b_str = ','.join(f'{x:.1f}' for x in b)
    c_str = ','.join(f'{x:.1f}' for x in c)
    print(f'{name}: BASE=[{b_str}] CUST=[{c_str}] -> +{spd:.1f}%')
print('#'*60)
