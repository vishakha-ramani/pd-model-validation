"""Small open-loop HTTP load generator for the admission schema gate.

This deliberately uses only the Python standard library so it can run inside
the already-started vLLM container when pulling the GuideLLM client image is
not practical. It is a validation smoke tool, not the full benchmark harness.
"""

import argparse
import concurrent.futures
import json
import math
import os
import time
import urllib.request


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = min(int(math.ceil(q * len(ordered))) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def send_one(target, body, timeout):
    started = time.monotonic()
    request = urllib.request.Request(
        target, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        usage = result.get("usage") or {}
        return {
            "ok": True,
            "latency": time.monotonic() - started,
            "id": result.get("id"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }
    except Exception as exc:
        return {
            "ok": False,
            "latency": time.monotonic() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--workers", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--target", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Token 9906 is an ordinary in-vocabulary Llama-3 token. Passing token IDs
    # avoids tokenizer-dependent drift from the intended fixed prompt length.
    payload = {
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "prompt": [9906] * args.prompt_tokens,
        "max_tokens": args.output_tokens,
        "temperature": 0,
        "ignore_eos": True,
    }
    body = json.dumps(payload).encode()

    futures = []
    started = time.monotonic()
    next_arrival = started
    stop_arrivals = started + args.seconds
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        while next_arrival < stop_arrivals:
            delay = next_arrival - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            futures.append(pool.submit(send_one, args.target, body, args.timeout))
            next_arrival += 1.0 / args.rate
        rows = [future.result() for future in concurrent.futures.as_completed(futures)]

    elapsed = time.monotonic() - started
    successful = [row for row in rows if row["ok"]]
    latencies = [row["latency"] for row in successful]
    errors = {}
    for row in rows:
        if not row["ok"]:
            errors[row["error"]] = errors.get(row["error"], 0) + 1
    report = {
        "rate": args.rate,
        "arrival_seconds": args.seconds,
        "elapsed_seconds_including_drain": elapsed,
        "submitted": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "latency_seconds": {
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p90": percentile(latencies, 0.90),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies) if latencies else None,
        },
        "observed_prompt_tokens": sorted({
            row["prompt_tokens"] for row in successful}),
        "observed_completion_tokens": sorted({
            row["completion_tokens"] for row in successful}),
        "errors": errors,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
