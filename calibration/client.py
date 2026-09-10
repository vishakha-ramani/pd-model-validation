import asyncio
import csv
import json
import math
import os
import random
import time
import httpx

VOCAB_LO, VOCAB_HI = 1000, 100000


def env_int_list(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return list(default)
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive comma-separated integers")
    return values


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


def steady_decode_steps(streams, warmup):
    """Align decode gaps only after every submitted stream is decoding.

    Large prompts enter a chunked-prefill scheduler over several iterations.
    Treating submitted concurrency as resident decode batch size during that
    admission interval contaminates a decode-only fit with prefill work.  The
    barrier below starts after every stream has emitted ``warmup`` tokens, then
    aligns each stream at its first token at or after that wall-clock time.

    The returned ``k`` is the mean generated-token index across the resident
    streams.  Therefore ``B * (n + k)`` is the exact aggregate-context regressor
    when every prompt has the same initial length ``n``.
    """
    if not streams or any(len(stream) <= warmup for stream in streams):
        return []
    steady_start = max(stream[warmup - 1] for stream in streams)
    aligned = [
        [(index, timestamp) for index, timestamp in enumerate(stream)
         if timestamp >= steady_start]
        for stream in streams
    ]
    usable = min(len(stream) for stream in aligned)
    if usable < 2:
        return []

    rows = []
    for offset in range(1, usable):
        dts = sorted(
            stream[offset][1] - stream[offset - 1][1]
            for stream in aligned
        )
        midpoint = len(dts) // 2
        median_dt = (
            dts[midpoint]
            if len(dts) % 2
            else (dts[midpoint - 1] + dts[midpoint]) / 2.0
        )
        mean_k = sum(stream[offset][0] for stream in aligned) / len(aligned)
        rows.append((mean_k, median_dt))
    return rows


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def streamed_token_count(choice):
    """Return generated tokens represented by one completion SSE choice.

    vLLM can attach ``finish_reason`` to the final token-bearing frame, so the
    finish reason alone cannot distinguish a token from terminal metadata.
    ``return_token_ids`` makes the distinction exact.  The text fallback keeps
    the client compatible with servers that do not return token IDs and avoids
    counting an empty terminal frame as a token.
    """
    token_ids = choice.get("token_ids")
    if token_ids is not None:
        return len(token_ids)
    return int(bool(choice.get("text")))


async def _stream_tokens(client, base_url, model, prompt, max_tokens, request_id=None):
    # returns list of absolute timestamps, one per streamed token
    ts = []
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "stream": True, "ignore_eos": True, "temperature": 0.0,
               "return_token_ids": True}
    if request_id is not None:
        payload["request_id"] = request_id
    async with client.stream("POST", f"{base_url}/v1/completions", json=payload) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("choices"):
                count = streamed_token_count(obj["choices"][0])
                if count:
                    now = time.perf_counter()
                    ts.extend([now] * count)
    return ts


async def run_decode(base_url, model, B, n, max_tokens, rng, warmup=4,
                     request_prefix=None):
    async with httpx.AsyncClient(timeout=None) as client:
        prompts = [random_prompt(n, rng) for _ in range(B)]
        streams = await asyncio.gather(*[
            _stream_tokens(
                client,
                base_url,
                model,
                prompt,
                max_tokens,
                request_id=(f"{request_prefix}:r{index}" if request_prefix else None),
            )
            for index, prompt in enumerate(prompts)
        ])
    steps = steady_decode_steps(streams, warmup)
    return [(B, n, k, dt) for (k, dt) in steps if not math.isnan(dt)]


async def run_prefill(base_url, model, n, rng, request_id=None):
    async with httpx.AsyncClient(timeout=None) as client:
        prompt = random_prompt(n, rng)
        t0 = time.perf_counter()
        ts = await _stream_tokens(
            client, base_url, model, prompt, max_tokens=1, request_id=request_id
        )
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
        overlaps = []
        for k in range(1, len(s)):
            start, end = s[k - 1], s[k]
            if end >= t_fire and start <= t_first:  # gap overlaps the prefill window
                overlaps.append((k, start, end))

        # t_fire is measured before HTTP admission, and t_first after the
        # injected request's sampled token is delivered. The overlap window
        # consequently contains ordinary decode iterations at both edges.
        # Prefill-bearing iterations are the n_chunks longest gaps; restore
        # chronological order before assigning their P_k labels.
        selected = sorted(
            sorted(overlaps, key=lambda item: item[2] - item[1], reverse=True)[:n_chunks],
            key=lambda item: item[1],
        )
        for j, (k, start, end) in enumerate(selected):
            per_j_dts.setdefault(j, []).append(end - start)
            per_j_ctx.setdefault(j, []).append(n_decode + k)
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


async def run_mixed(base_url, model, B, n_decode, n_inject, max_tokens, chunk_bud, rng,
                    warmup_s=1.0, debug_rows=None, repeat=0, request_prefix=None):
    # Hold B decode streams steady, then inject one long-prompt request. Its prefill spans
    # ceil(n_inject/chunk_bud) engine iterations; during that window the resident decode
    # streams' inter-token latency inflates. We timestamp the injection window
    # [t_fire, t_first_token] and keep only the decode steps overlapping it, aligned by
    # order to prefill chunks. MAPE is therefore computed only over the injection window.
    async with httpx.AsyncClient(timeout=None) as client:
        decode_prompts = [random_prompt(n_decode, rng) for _ in range(B)]
        decode_tasks = [asyncio.create_task(_stream_tokens(
            client,
            base_url,
            model,
            prompt,
            max_tokens,
            request_id=(f"{request_prefix}:decode:r{index}" if request_prefix else None),
        )) for index, prompt in enumerate(decode_prompts)]
        await asyncio.sleep(warmup_s)  # let decode streams reach steady state
        inject_prompt = random_prompt(n_inject, rng)
        t_fire = time.perf_counter()
        inject_ts = await _stream_tokens(
            client,
            base_url,
            model,
            inject_prompt,
            max_tokens=1,
            request_id=(f"{request_prefix}:prefill" if request_prefix else None),
        )
        streams = await asyncio.gather(*decode_tasks)
    if not inject_ts:
        return []  # injected request produced no token; cell invalid
    t_first = inject_ts[0]
    if debug_rows is not None:
        for stream_index, stream in enumerate(streams):
            for k in range(1, len(stream)):
                start, end = stream[k - 1], stream[k]
                if end >= t_fire and start <= t_first:
                    debug_rows.append((
                        B,
                        repeat,
                        stream_index,
                        k,
                        start - t_fire,
                        end - t_fire,
                        end - start,
                        t_first - t_fire,
                    ))
    return align_injection_window(streams, t_fire, t_first, n_inject, chunk_bud, B, n_decode)


async def main():
    base = os.environ.get("VLLM_BASE", "http://vllm-cal:8000")
    model = os.environ.get("MODEL", "meta-llama/Llama-3.3-70B-Instruct")
    chunk_bud = int(os.environ.get("CHUNK_BUD", "8192"))
    result_dir = os.environ.get("RESULT_DIR", "/results")
    os.makedirs(result_dir, exist_ok=True)
    rng = random.Random(int(os.environ.get("SEED", "0")))
    seed = int(os.environ.get("SEED", "0"))
    request_prefix = os.environ.get("REQUEST_PREFIX", f"cal:s{seed}")

    decode_batches = env_int_list("DECODE_BATCHES", [1, 2, 4, 8, 16, 32, 64])
    decode_contexts = env_int_list("DECODE_CONTEXTS", [64, 256, 1024, 4096])
    decode_max_tokens = int(os.environ.get("DECODE_MAX_TOKENS", "256"))
    prefill_lengths = env_int_list(
        "PREFILL_LENGTHS", [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    )
    prefill_repeats = int(os.environ.get("PREFILL_REPEATS", "1"))
    mixed_batches = env_int_list("MIXED_BATCHES", [8, 16, 32])
    mixed_context = int(os.environ.get("MIXED_CONTEXT", "1024"))
    mixed_inject = int(os.environ.get("MIXED_INJECT", "32768"))
    mixed_max_tokens = int(os.environ.get("MIXED_MAX_TOKENS", "256"))
    mixed_repeats = int(os.environ.get("MIXED_REPEATS", "1"))
    if prefill_repeats <= 0 or mixed_repeats <= 0:
        raise ValueError("PREFILL_REPEATS and MIXED_REPEATS must be positive")

    metadata = {
        "model": model,
        "seed": seed,
        "request_prefix": request_prefix,
        "chunk_bud": chunk_bud,
        "decode_batches": decode_batches,
        "decode_contexts": decode_contexts,
        "decode_max_tokens": decode_max_tokens,
        "decode_warmup_tokens": 4,
        "decode_steady_state_barrier": True,
        "prefill_lengths": prefill_lengths,
        "prefill_repeats": prefill_repeats,
        "mixed_batches": mixed_batches,
        "mixed_context": mixed_context,
        "mixed_inject": mixed_inject,
        "mixed_max_tokens": mixed_max_tokens,
        "mixed_repeats": mixed_repeats,
        "mixed_warmup_s": 1.0,
    }
    with open(os.path.join(result_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    decode_rows = []
    for B in decode_batches:
        for n in decode_contexts:
            decode_rows += await run_decode(
                base,
                model,
                B,
                n,
                max_tokens=decode_max_tokens,
                rng=rng,
                request_prefix=f"{request_prefix}:decode:B{B}:N{n}",
            )
    write_csv(
        os.path.join(result_dir, "decode.csv"),
        ["B", "n", "k", "t_iter"],
        decode_rows,
    )

    prefill_rows = []
    for n in prefill_lengths:
        for repeat in range(prefill_repeats):
            r = await run_prefill(
                base,
                model,
                n,
                rng,
                request_id=f"{request_prefix}:prefill:N{n}:r{repeat}",
            )
            if r is not None:
                prefill_rows.append(r)
    write_csv(
        os.path.join(result_dir, "prefill.csv"), ["n", "ttft"], prefill_rows
    )

    mixed_rows = []
    mixed_debug_rows = []
    for B in mixed_batches:
        for repeat in range(mixed_repeats):
            mixed_rows += await run_mixed(
                base,
                model,
                B,
                n_decode=mixed_context,
                n_inject=mixed_inject,
                max_tokens=mixed_max_tokens,
                chunk_bud=chunk_bud,
                rng=rng,
                debug_rows=mixed_debug_rows,
                repeat=repeat,
                request_prefix=f"{request_prefix}:mixed:B{B}:r{repeat}",
            )
    write_csv(
        os.path.join(result_dir, "mixed.csv"),
        ["B", "resident_context_sum", "kappa", "P_k", "t_iter_observed"],
        mixed_rows,
    )
    write_csv(
        os.path.join(result_dir, "mixed_debug.csv"),
        [
            "B",
            "repeat",
            "stream_index",
            "k",
            "start_minus_fire",
            "end_minus_fire",
            "t_iter_observed",
            "injection_ttft",
        ],
        mixed_debug_rows,
    )


if __name__ == "__main__":
    asyncio.run(main())
