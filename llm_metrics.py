"""
Collect on-device metrics for the Qwen3-0.6B W4A16 export from its AI Hub profile jobs.

The export submits one profile job per (part, graph). Graph names look like
  prompt_ar128_cl4096_1_of_2   -> prompt processor, 128-token chunk, 4096 context, part 1 of 2
  token_ar1_cl4096_2_of_2      -> token generator, 1 token/step,   4096 context, part 2 of 2
Decode tok/s = 1e6 / sum(token-graph latency over parts, microseconds).
Prefill tok/s = 128 / sum(prompt-graph latency over parts, seconds).
TTFT(128-token prompt) ~= one prompt chunk + one decode step.
"""
import json, re, sys, time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import qai_hub as hub

MODEL_TAG = "qwen3_0_6b"
SINCE = datetime.now() - timedelta(hours=float(sys.argv[1]) if len(sys.argv) > 1 else 12)  # job dates are local naive
PAT = re.compile(r"(prompt|token)_ar(\d+)_cl(\d+)_(\d+)_of_(\d+)")

jobs = []
offset, done = 0, False
while not done:
    page = hub.get_job_summaries(offset=offset, limit=100, type=hub.JobType.PROFILE)
    if not page: break
    for sm in page:
        if sm.date < SINCE: done = True; break
        if MODEL_TAG in (sm.name or "") and PAT.search(sm.name):
            j = hub.get_job(sm.job_id)
            if hasattr(j, "download_profile"): jobs.append(j)
    offset += 100
print(f"found {len(jobs)} profile jobs since {SINCE:%Y-%m-%d %H:%M} local")

graphs = {}  # (kind, ar, cl, part) -> metrics
for j in jobs:
    m = PAT.search(j.name); kind, ar, cl, part, n = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
    st = j.wait()
    if not st.success:
        print(f"  FAILED {j.name} {j.url}"); continue
    p = j.download_profile(); es = p["execution_summary"]
    units = defaultdict(int)
    for layer in p.get("execution_detail", []): units[layer.get("compute_unit", "?")] += 1
    graphs[(kind, ar, cl, part)] = dict(
        job=j.url, device=j.device.name, n_parts=n,
        inference_us=es["estimated_inference_time"], peak_mem_bytes=es.get("estimated_inference_peak_memory"),
        first_load_us=es.get("first_load_time"), warm_load_us=es.get("warm_load_time"), units=dict(units))
    print(f"  {j.name:70s} {es['estimated_inference_time']/1000:8.2f} ms  peak {es.get('estimated_inference_peak_memory',0)/1e6:7.1f} MB  {dict(units)}")

out = {"model": "Qwen3-0.6B W4A16 (SpinQuant+AdaScale, int4 weights / int16 act / int8 KV)", "runtime": "QNN/QAIRT context binaries (geniex_qairt)",
       "device": next(iter(graphs.values()))["device"] if graphs else None, "graphs": {"_".join(map(str, k)): v for k, v in graphs.items()}, "per_context": {}}
for cl in sorted({k[2] for k in graphs}):
    tok = [v for k, v in graphs.items() if k[0] == "token" and k[2] == cl]
    pro = [v for k, v in graphs.items() if k[0] == "prompt" and k[2] == cl]
    ar = next((k[1] for k in graphs if k[0] == "prompt" and k[2] == cl), 128)
    if not tok or not pro: continue
    dec_us = sum(v["inference_us"] for v in tok); pp_us = sum(v["inference_us"] for v in pro)
    units = defaultdict(int)
    for v in tok + pro:
        for u, c in v["units"].items(): units[u] += c
    npu_share = units.get("NPU", 0) / max(1, sum(units.values()))
    out["per_context"][cl] = dict(
        decode_step_ms=dec_us / 1000, decode_tokens_per_s=1e6 / dec_us,
        prefill_chunk_ms=pp_us / 1000, prefill_tokens_per_s=ar * 1e6 / pp_us,
        ttft_ms_prompt_le_128=(pp_us + dec_us) / 1000,
        peak_mem_MB_sum_of_parts_decode=sum(v["peak_mem_bytes"] or 0 for v in tok) / 1e6,
        peak_mem_MB_max_graph=max(v["peak_mem_bytes"] or 0 for v in tok + pro) / 1e6,
        layers_by_compute_unit=dict(units), npu_residency=npu_share)
json.dump(out, open("llm_results.json", "w"), indent=1)

print("\n==================== Qwen3-0.6B W4A16 on", out["device"], "====================")
for cl, r in out["per_context"].items():
    print(f"context {cl}: decode {r['decode_tokens_per_s']:.1f} tok/s ({r['decode_step_ms']:.2f} ms/step) | "
          f"prefill {r['prefill_tokens_per_s']:.0f} tok/s | TTFT(<=128 tok) {r['ttft_ms_prompt_le_128']:.0f} ms | "
          f"peak mem {r['peak_mem_MB_sum_of_parts_decode']:.0f} MB | NPU residency {100*r['npu_residency']:.1f}% {r['layers_by_compute_unit']}")
print("Note: tok/s here is derived from per-graph NPU latency (profile jobs); runtime overhead (sampling, tokenizer, IPC) will lower end-to-end tok/s.")
