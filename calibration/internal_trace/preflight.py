"""Fail closed unless the live vLLM 0.26 hook emits internally consistent rows."""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


BASE_URL = os.environ.get(
    "VLLM_BASE", "http://vllm-cal-llama70b-v026-internal:8000"
)
MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.3-70B-Instruct")
TRACE_DIR = Path(
    os.environ.get(
        "ITERTRACE_OUT_DIR",
        "/results/llama3.3-70b-tp4-vllm-0.26.0/internal-v1",
    )
)


def wait_for_health(timeout_s=1800):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(10)
    raise TimeoutError(f"vLLM did not become healthy within {timeout_s} seconds")


def issue_probe():
    payload = json.dumps(
        {
            "model": MODEL,
            "prompt": [1234] * 64,
            "max_tokens": 128,
            "stream": False,
            "ignore_eos": True,
            "temperature": 0.0,
            "request_id": "internal-v1:preflight",
        }
    ).encode()
    request = urllib.request.Request(
        f"{BASE_URL}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        if response.status != 200:
            raise RuntimeError(f"preflight completion returned HTTP {response.status}")


def wait_for_trace(timeout_s=120):
    trajectory = TRACE_DIR / "trajectory.jsonl"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if trajectory.exists() and trajectory.stat().st_size:
            rows = [json.loads(line) for line in trajectory.read_text().splitlines()]
            if len(rows) >= 100:
                return rows
        time.sleep(2)
    raise TimeoutError("instrumentation did not flush at least 100 trace rows")


def validate(rows):
    meta = json.loads((TRACE_DIR / "meta.json").read_text())
    if meta["vllm_version"] != "0.26.0":
        raise RuntimeError(f"unexpected vLLM version: {meta['vllm_version']}")
    if meta["tensor_parallel_size"] != 4:
        raise RuntimeError(f"unexpected tensor parallelism: {meta['tensor_parallel_size']}")
    if meta["prefix_caching_enabled"]:
        raise RuntimeError("prefix caching must be disabled for this calibration")
    if not any(row["regime"] == "pure_prefill" for row in rows):
        raise RuntimeError("trace contains no prefill step")
    if not any(row["regime"] == "pure_decode" for row in rows):
        raise RuntimeError("trace contains no decode step")
    for row in rows:
        details = row["vllm_iteration_details"]
        if row["B_decode"] != details["generation_requests"]:
            raise RuntimeError(f"decode classification mismatch at step {row['step']}")
        if row["S_prefill"] != details["context_tokens"]:
            raise RuntimeError(f"prefill classification mismatch at step {row['step']}")
        if row["engine_step_s"] <= 0:
            raise RuntimeError(f"non-positive duration at step {row['step']}")
    return {
        "passed": True,
        "rows_checked": len(rows),
        "vllm_version": meta["vllm_version"],
        "tensor_parallel_size": meta["tensor_parallel_size"],
    }


def main():
    wait_for_health()
    issue_probe()
    result = validate(wait_for_trace())
    (TRACE_DIR / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
