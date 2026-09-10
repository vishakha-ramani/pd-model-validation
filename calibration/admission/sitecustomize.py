"""Admission-validation vLLM 0.11.0 instrumentation hook.

Sibling of the frozen calibration/modeb/sitecustomize.py hook. Reuses that
module's JsonlWriter/_log/_PATCHED structure and its extract_reqs() call on
the scheduler output, and additionally captures:
  - EngineCore input arrival and Scheduler.add_request timestamps, measuring
    the wait for an in-flight iteration rather than starting after it;
  - full queued-work aggregates, capped per-request queue detail, the complete
    running set, and free-KV state at schedule time;
  - an extended per-step trajectory record (trajectory.jsonl) that adds the
    waiting-queue/free-KV fields to the Mode-B running-batch fields.

Auto-imported by CPython at interpreter startup for every process on
PYTHONPATH, including the spawned EngineCoreProc child that runs the
scheduler loop. Ships in a ConfigMap alongside admission_capture.py and the
frozen modeb_capture.py; stdlib + those two capture modules only.
"""
import json
import os
import sys
import time

try:
    # In the container both this file and its sibling capture module sit at
    # PYTHONPATH=/opt/admission as top-level modules, alongside the frozen
    # Mode-B capture module at /opt/modeb.
    from admission_capture import (
        build_enqueue_record, build_request_work_record,
        cap_waiting, summarize_waiting, build_admission_step_record,
    )
except ImportError:
    # In the offline test env this file is imported as calibration.admission.sitecustomize.
    from calibration.admission.admission_capture import (
        build_enqueue_record, build_request_work_record,
        cap_waiting, summarize_waiting, build_admission_step_record,
    )

try:
    from modeb_capture import extract_reqs
except ImportError:
    from calibration.modeb.modeb_capture import extract_reqs

_PATCHED = False


def _int_env(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


MAX_WAITING_IDS = _int_env("ADMISSION_MAX_WAITING_IDS", 512)
OUT_DIR = os.environ.get("ADMISSION_OUT_DIR", "/results/admission-v2")


def _log(msg):
    # sentinel to prove the hook loaded inside the (spawned) child process
    print(f"[ADMISSION] {msg}", file=sys.stderr, flush=True)


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
        "block_size": vllm_config.cache_config.block_size,
        "max_num_seqs": sc.max_num_seqs,
        "waiting_ids_cap": MAX_WAITING_IDS,
        "admission_capture_schema": 2,
        "arrival_clock": "perf_counter",
        "arrival_probe": "EngineCore.preprocess_add_request",
        "enq_clock": "perf_counter",
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

    state = {"writer": None, "enq_writer": None, "prompt_len_cache": {}, "step": 0,
             "last_output": None, "last_waiting": [], "last_waiting_reqs": [],
             "last_waiting_work": None, "last_running_reqs": [],
             "last_free_kv": None, "arrivals": {}, "meta_written": False}

    _orig_preprocess_add_request = EngineCore.preprocess_add_request
    _orig_add_request = Scheduler.add_request
    _orig_schedule = Scheduler.schedule
    _orig_step = EngineCore.step

    def preprocess_add_request(self, request, *a, **kw):
        # This runs in EngineCoreProc's input-socket thread immediately after
        # the ADD frame is decoded and before it is placed on input_queue.
        # perf_counter is system-wide on the supported Linux host, so this is
        # directly comparable with the busy-loop's Scheduler timestamp.
        t_engine_arrive = time.perf_counter()
        result = _orig_preprocess_add_request(self, request, *a, **kw)
        try:
            converted = result[0] if isinstance(result, tuple) else result
            rid = getattr(converted, "request_id", None) or getattr(converted, "req_id", None)
            if rid is not None:
                state["arrivals"][rid] = t_engine_arrive
        except Exception:
            pass
        return result

    def add_request(self, request, *a, **kw):
        # Call the original first so a raising original is never masked by
        # our instrumentation, and always return its result even if the
        # logging block below raises.
        result = _orig_add_request(self, request, *a, **kw)
        try:
            t_enq = time.perf_counter()
            rid = getattr(request, "request_id", None) or getattr(request, "req_id", None)
            ptoks = getattr(request, "prompt_token_ids", None)
            try:
                plen = len(ptoks) if ptoks is not None else getattr(request, "num_prompt_tokens", 0)
            except Exception:
                plen = 0
            if state["enq_writer"] is None:
                os.makedirs(OUT_DIR, exist_ok=True)
                # Enqueue events are sparse relative to scheduler steps. Persist
                # each one immediately so a one-request schema gate is observable
                # and process shutdown cannot discard the last 49 arrivals.
                state["enq_writer"] = JsonlWriter(
                    os.path.join(OUT_DIR, "admission_events.jsonl"), flush_every=1)
            t_engine_arrive = state["arrivals"].pop(rid, None)
            state["enq_writer"].write(
                build_enqueue_record(rid, t_enq, plen, t_engine_arrive))
        except Exception:
            pass
        return result

    def schedule(self):
        out = _orig_schedule(self)
        scheduled = getattr(out, "num_scheduled_tokens", {}) or {}
        block_size = getattr(self, "block_size", 1)
        try:
            waiting_all = [build_request_work_record(r, 0, block_size) for r in self.waiting]
        except Exception:
            waiting_all = []
        try:
            running_all = [build_request_work_record(
                r, scheduled.get(getattr(r, "request_id", None), 0), block_size)
                for r in self.running]
        except Exception:
            running_all = []
        try:
            free_kv = self.kv_cache_manager.block_pool.get_num_free_blocks()
        except Exception:
            free_kv = None
        state["last_output"] = out
        state["last_waiting"] = [r["id"] for r in waiting_all]
        state["last_waiting_reqs"] = waiting_all[:MAX_WAITING_IDS]
        state["last_waiting_work"] = summarize_waiting(waiting_all, block_size)
        state["last_running_reqs"] = running_all
        state["last_free_kv"] = free_kv
        return out

    def step(self):
        # schedule() runs INSIDE _orig_step, so we must call it first and then read the
        # composition it produced. Reading last_output before _orig_step would pair this
        # step's t_start with the PREVIOUS step's composition (off-by-one). Mirrors Mode B.
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
            count, ids, trunc = cap_waiting(state["last_waiting"], MAX_WAITING_IDS)
            rec = build_admission_step_record(
                state["step"], t_start, t_end, out.total_num_scheduled_tokens,
                sum(1 for r in reqs if r["computed"] >= r["prompt_len"]),
                reqs, count, ids, trunc, state["last_free_kv"],
                running_reqs=state["last_running_reqs"],
                waiting_reqs=state["last_waiting_reqs"],
                waiting_work=state["last_waiting_work"])
            state["writer"].write(rec)
            state["step"] += 1
        return result

    EngineCore.preprocess_add_request = preprocess_add_request
    Scheduler.add_request = add_request
    Scheduler.schedule = schedule
    EngineCore.step = step
    _PATCHED = True
    _log("patched EngineCore input arrival + Scheduler.add_request + "
         "Scheduler.schedule + EngineCore.step")
    return True


install()
