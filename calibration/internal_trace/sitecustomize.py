"""vLLM 0.26.0 exact scheduler-state and iteration-duration hook.

Mount this directory on ``PYTHONPATH``.  CPython imports ``sitecustomize`` in
the API process and the spawned EngineCore process.  The hook observes, but
does not modify, scheduling decisions or model inputs.
"""
import atexit
import json
import os
import sys
import time

try:
    from trace_capture import build_step_record, extract_requests
except ImportError:
    from calibration.internal_trace.capture import build_step_record, extract_requests


OUT_DIR = os.environ.get("ITERTRACE_OUT_DIR", "/results/internal-trace")
_PATCHED = False


def _log(message):
    print(f"[ITERTRACE] {message}", file=sys.stderr, flush=True)


class JsonlWriter:
    """Append-only writer whose periodic flush happens outside measured time."""

    def __init__(self, path, flush_every=100):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._file = open(path, "a", buffering=1024 * 1024)
        self._flush_every = flush_every
        self._pending = 0

    def write(self, record):
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._pending += 1
        if self._pending >= self._flush_every:
            self.flush()

    def flush(self):
        if self._pending:
            self._file.flush()
            self._pending = 0

    def close(self):
        self.flush()
        self._file.close()


def _prompt_length(new_request):
    if new_request.prompt_token_ids is not None:
        return len(new_request.prompt_token_ids)
    embeds = getattr(new_request, "prompt_embeds", None)
    if embeds is not None:
        return int(embeds.shape[0])
    raise RuntimeError(f"cannot determine prompt length for {new_request.req_id}")


def capture_meta(vllm_config):
    scheduler = vllm_config.scheduler_config
    cache = vllm_config.cache_config
    model = vllm_config.model_config
    return {
        "schema_version": 1,
        "vllm_version": _vllm_version(),
        "model": model.model,
        "revision": getattr(model, "revision", None),
        "tensor_parallel_size": vllm_config.parallel_config.tensor_parallel_size,
        "max_num_batched_tokens": scheduler.max_num_batched_tokens,
        "max_num_seqs": scheduler.max_num_seqs,
        "max_model_len": scheduler.max_model_len,
        "chunked_prefill_enabled": scheduler.enable_chunked_prefill,
        "async_scheduling": getattr(scheduler, "async_scheduling", False),
        "block_size": getattr(cache, "block_size", None),
        "num_gpu_blocks": getattr(cache, "num_gpu_blocks", None),
        "prefix_caching_enabled": getattr(cache, "enable_prefix_caching", None),
        "timing_primary": "engine_step_s = wall time around EngineCore.step",
    }


def _vllm_version():
    try:
        import vllm

        return vllm.__version__
    except Exception:
        return None


def _scheduler_state(scheduler):
    return {
        "running_after_schedule": len(getattr(scheduler, "running", ())),
        "waiting_after_schedule": len(getattr(scheduler, "waiting", ())),
        "skipped_waiting_after_schedule": len(
            getattr(scheduler, "skipped_waiting", ())
        ),
    }


def install():
    """Install an idempotent observation-only hook; return whether it installed."""
    global _PATCHED
    if _PATCHED:
        return False
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.engine.core import EngineCore
        from vllm.v1.utils import compute_iteration_details
    except Exception:
        return False

    state = {
        "last_output": None,
        "prompt_length_cache": {},
        "writer": None,
        "step": 0,
        "meta_written": False,
    }
    original_schedule = Scheduler.schedule
    original_step = EngineCore.step

    def schedule(self, *args, **kwargs):
        output = original_schedule(self, *args, **kwargs)
        state["last_output"] = output
        return output

    def step(self, *args, **kwargs):
        state["last_output"] = None
        start_ns = time.perf_counter_ns()
        result = original_step(self, *args, **kwargs)
        end_ns = time.perf_counter_ns()
        output = state["last_output"]
        if output is None or output.total_num_scheduled_tokens <= 0:
            return result

        if state["writer"] is None:
            os.makedirs(OUT_DIR, exist_ok=True)
            state["writer"] = JsonlWriter(
                os.path.join(OUT_DIR, "trajectory.jsonl"),
                flush_every=int(os.environ.get("ITERTRACE_FLUSH_EVERY", "100")),
            )
            atexit.register(state["writer"].close)
            with open(os.path.join(OUT_DIR, "meta.json"), "w") as meta_file:
                json.dump(capture_meta(self.vllm_config), meta_file, indent=2)
            state["meta_written"] = True
            _log(f"active in pid {os.getpid()}, writing {OUT_DIR}")

        new_requests = [
            (request.req_id, _prompt_length(request), request.num_computed_tokens)
            for request in output.scheduled_new_reqs
        ]
        cached = output.scheduled_cached_reqs
        requests = extract_requests(
            output.num_scheduled_tokens,
            new_requests,
            list(cached.req_ids),
            list(cached.num_computed_tokens),
            list(cached.num_output_tokens),
            state["prompt_length_cache"],
        )
        record = build_step_record(
            state["step"],
            start_ns,
            end_ns,
            requests,
            _scheduler_state(self.scheduler),
        )
        details = compute_iteration_details(output)
        record["vllm_iteration_details"] = {
            "context_requests": details.num_ctx_requests,
            "context_tokens": details.num_ctx_tokens,
            "generation_requests": details.num_generation_requests,
            "generation_tokens": details.num_generation_tokens,
        }
        record["pid"] = os.getpid()
        state["writer"].write(record)
        state["step"] += 1
        return result

    Scheduler.schedule = schedule
    EngineCore.step = step
    _PATCHED = True
    _log("patched vLLM Scheduler.schedule and EngineCore.step")
    return True


install()
