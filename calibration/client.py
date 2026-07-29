import asyncio
import csv
import json
import math
import random
import time
import httpx

VOCAB_LO, VOCAB_HI = 1000, 100000


def random_prompt(n, rng):
    return [rng.randrange(VOCAB_LO, VOCAB_HI) for _ in range(n)]


def per_step_median(streams):
    # streams: list of per-stream absolute token timestamps (index 0 = first token time).
    # returns [(k, median_dt)] where k=0 is the first observed token (ramp), k>=1 are decode gaps.
    max_len = min(len(s) for s in streams)
    out = []
    for k in range(max_len):
        if k == 0:
            out.append((0, float("nan")))  # first-token gap is ramp; kept for indexing
            continue
        dts = sorted(s[k] - s[k - 1] for s in streams)
        m = len(dts)
        med = dts[m // 2] if m % 2 else (dts[m // 2 - 1] + dts[m // 2]) / 2.0
        out.append((k, med))
    return out


def trim_ramp(steps, warmup):
    return [(k, v) for (k, v) in steps if k >= warmup]


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


async def _stream_tokens(client, base_url, model, prompt, max_tokens):
    # returns list of absolute timestamps, one per streamed token
    ts = []
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "stream": True, "ignore_eos": True, "temperature": 0.0}
    async with client.stream("POST", f"{base_url}/v1/completions", json=payload) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("choices") and obj["choices"][0].get("text") is not None:
                ts.append(time.perf_counter())
    return ts


async def run_decode(base_url, model, B, n, max_tokens, rng, warmup=4):
    async with httpx.AsyncClient(timeout=None) as client:
        prompts = [random_prompt(n, rng) for _ in range(B)]
        streams = await asyncio.gather(*[
            _stream_tokens(client, base_url, model, p, max_tokens) for p in prompts])
    steps = trim_ramp(per_step_median(streams), warmup)
    return [(B, n, k, dt) for (k, dt) in steps if not math.isnan(dt)]


async def run_prefill(base_url, model, n, rng):
    async with httpx.AsyncClient(timeout=None) as client:
        prompt = random_prompt(n, rng)
        t0 = time.perf_counter()
        ts = await _stream_tokens(client, base_url, model, prompt, max_tokens=1)
    if not ts:
        return None  # no token returned; caller skips this cell rather than crashing the sweep
    ttft = ts[0] - t0
    return (n, ttft)


def align_injection_window(streams, t_fire, t_first, n_inject, chunk_bud, B, n_decode):
    # streams: list of per-stream absolute token timestamps.
    # A decode step (gap between consecutive tokens) is co-resident with the injected
    # prefill if its [start, end] interval overlaps [t_fire, t_first]. Co-resident steps
    # are aligned by order index j to prefill chunk j (P_k = j*chunk_bud). Returns one
    # mixed row per co-resident step: (B, resident_context_sum, kappa, P_k, median_dt).
    n_chunks = math.ceil(n_inject / chunk_bud)
    per_j_dts = {}
    per_j_ctx = {}
    for s in streams:
        j = 0
        for k in range(1, len(s)):
            start, end = s[k - 1], s[k]
            if end >= t_fire and start <= t_first:  # gap overlaps the prefill window
                per_j_dts.setdefault(j, []).append(end - start)
                per_j_ctx.setdefault(j, []).append(n_decode + k)
                j += 1
    rows = []
    for j in range(min(n_chunks, len(per_j_dts))):
        dts = sorted(per_j_dts[j])
        m = len(dts)
        med = dts[m // 2] if m % 2 else (dts[m // 2 - 1] + dts[m // 2]) / 2.0
        resident_ctx_sum = float(sum(per_j_ctx[j]))
        kappa = min(chunk_bud, n_inject - j * chunk_bud)
        P_k = j * chunk_bud
        rows.append((B, resident_ctx_sum, kappa, P_k, med))
    return rows


async def run_mixed(base_url, model, B, n_decode, n_inject, max_tokens, chunk_bud, rng, warmup_s=1.0):
    # Hold B decode streams steady, then inject one long-prompt request. Its prefill spans
    # ceil(n_inject/chunk_bud) engine iterations; during that window the resident decode
    # streams' inter-token latency inflates. We timestamp the injection window
    # [t_fire, t_first_token] and keep only the decode steps overlapping it, aligned by
    # order to prefill chunks. MAPE is therefore computed only over the injection window.
    async with httpx.AsyncClient(timeout=None) as client:
        decode_prompts = [random_prompt(n_decode, rng) for _ in range(B)]
        decode_tasks = [asyncio.create_task(
            _stream_tokens(client, base_url, model, p, max_tokens)) for p in decode_prompts]
        await asyncio.sleep(warmup_s)  # let decode streams reach steady state
        inject_prompt = random_prompt(n_inject, rng)
        t_fire = time.perf_counter()
        inject_ts = await _stream_tokens(client, base_url, model, inject_prompt, max_tokens=1)
        streams = await asyncio.gather(*decode_tasks)
    if not inject_ts:
        return []  # injected request produced no token; cell invalid
    t_first = inject_ts[0]
    return align_injection_window(streams, t_fire, t_first, n_inject, chunk_bud, B, n_decode)


async def main():
    import os
    base = os.environ.get("VLLM_BASE", "http://vllm-cal:8000")
    model = "meta-llama/Llama-3.3-70B-Instruct"
    chunk_bud = int(os.environ.get("CHUNK_BUD", "8192"))
    rng = random.Random(0)

    decode_rows = []
    for B in [1, 2, 4, 8, 16, 32, 64]:
        for n in [64, 256, 1024, 4096]:
            decode_rows += await run_decode(base, model, B, n, max_tokens=256, rng=rng)
    write_csv("/results/decode.csv", ["B", "n", "k", "t_iter"], decode_rows)

    prefill_rows = []
    for n in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        r = await run_prefill(base, model, n, rng)
        if r is not None:
            prefill_rows.append(r)
    write_csv("/results/prefill.csv", ["n", "ttft"], prefill_rows)

    mixed_rows = []
    for B in [8, 16, 32]:
        mixed_rows += await run_mixed(base, model, B, n_decode=1024,
                                      n_inject=32768, max_tokens=256,
                                      chunk_bud=chunk_bud, rng=rng)
    write_csv("/results/mixed.csv",
              ["B", "resident_context_sum", "kappa", "P_k", "t_iter_observed"], mixed_rows)


if __name__ == "__main__":
    asyncio.run(main())
