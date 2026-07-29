"""Mode-B vLLM 0.11.0 instrumentation hook.

Auto-imported by CPython at interpreter startup for every process on PYTHONPATH,
including the spawned EngineCoreProc child that runs the scheduler loop. It
monkeypatches Scheduler.schedule (to capture the real batch composition) and
EngineCore.step (to time iterations), writing one JSONL row per step plus a
one-time meta.json to MODEB_OUT_DIR.

Ships in a ConfigMap alongside modeb_capture.py; stdlib + modeb_capture only.
"""
import json
import os
import sys
import time

try:
    # In the container both files sit at PYTHONPATH=/opt/modeb as top-level modules.
    from modeb_capture import extract_reqs, build_step_record
except ImportError:
    # In the offline test env this file is imported as calibration.modeb.sitecustomize.
    from calibration.modeb.modeb_capture import extract_reqs, build_step_record

_PATCHED = False
OUT_DIR = os.environ.get("MODEB_OUT_DIR", "/results/modeb")


def _log(msg):
    # sentinel to prove the hook loaded inside the (spawned) child process
    print(f"[MODEB] {msg}", file=sys.stderr, flush=True)


class JsonlWriter:
    def __init__(self, path, flush_every=50):
        self.path = path
        self.flush_every = flush_every
        self._buf = []
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def write(self, record):
        self._buf.append(json.dumps(record))
        if len(self._buf) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self._buf:
            return
        with open(self.path, "a") as f:
            f.write("\n".join(self._buf) + "\n")
        self._buf = []


def capture_meta(vllm_config):
    sc = vllm_config.scheduler_config
    return {
        "max_num_batched_tokens": sc.max_num_batched_tokens,
        "chunked_prefill_enabled": sc.chunked_prefill_enabled,
        "async_scheduling": getattr(sc, "async_scheduling", False),
        "max_num_partial_prefills": getattr(sc, "max_num_partial_prefills", 1),
        "long_prefill_token_threshold": getattr(sc, "long_prefill_token_threshold", 0),
        "max_model_len": getattr(sc, "max_model_len", None),
        "num_gpu_blocks": getattr(vllm_config.cache_config, "num_gpu_blocks", None),
        "model": vllm_config.model_config.model,
        "tensor_parallel_size": vllm_config.parallel_config.tensor_parallel_size,
    }


def install():
    """Idempotently patch vllm. Returns True if patched, False if vllm absent/already patched."""
    global _PATCHED
    if _PATCHED:
        return False
    try:
        from vllm.v1.engine.core import EngineCore
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception:
        return False  # vllm not present (e.g. offline test env) -> no-op

    state = {"writer": None, "prompt_len_cache": {}, "step": 0,
             "last_output": None, "meta_written": False}

    _orig_schedule = Scheduler.schedule
    _orig_step = EngineCore.step

    def schedule(self):
        out = _orig_schedule(self)
        state["last_output"] = out
        return out

    def step(self):
        # schedule() runs INSIDE _orig_step, so we must call it first and then read the
        # composition it produced. Reading last_output before _orig_step would pair this
        # step's t_start with the PREVIOUS step's composition (off-by-one).
        state["last_output"] = None
        t_start = time.perf_counter()
        result = _orig_step(self)
        t_end = time.perf_counter()
        out = state["last_output"]
        if out is not None:
            if state["writer"] is None:
                os.makedirs(OUT_DIR, exist_ok=True)
                cfg = getattr(self, "vllm_config", None)
                if cfg is not None and not state["meta_written"]:
                    with open(os.path.join(OUT_DIR, "meta.json"), "w") as mf:
                        json.dump(capture_meta(cfg), mf, indent=2)
                    state["meta_written"] = True
                state["writer"] = JsonlWriter(os.path.join(OUT_DIR, "trajectory.jsonl"))
                _log(f"hook active in pid {os.getpid()}, writing {OUT_DIR}")
            new = [(r.req_id, len(r.prompt_token_ids), r.num_computed_tokens)
                   for r in out.scheduled_new_reqs]
            cached = out.scheduled_cached_reqs
            reqs = extract_reqs(out.num_scheduled_tokens, new,
                                list(cached.req_ids), list(cached.num_computed_tokens),
                                state["prompt_len_cache"])
            rec = build_step_record(state["step"], t_start, t_end,
                                    out.total_num_scheduled_tokens,
                                    sum(1 for r in reqs if r["computed"] >= r["prompt_len"]),
                                    reqs)
            state["writer"].write(rec)
            state["step"] += 1
        return result

    Scheduler.schedule = schedule
    EngineCore.step = step
    _PATCHED = True
    _log("patched Scheduler.schedule + EngineCore.step")
    return True


install()
