"""Pure per-step and per-request extraction for the admission-validation hook.

Self-contained (stdlib only, no vllm) so it is unit-testable offline and safe
to ship in a ConfigMap. Reuses the frozen Mode-B extract_reqs for the running
batch; adds enqueue-event records and scheduler-work snapshots.  This module is
kept independent of vLLM so every attribute fallback can be tested offline.
"""

import math


def build_enqueue_record(req_id, t_enq, prompt_len, t_engine_arrive=None):
    """One row per request, retaining both sides of the EngineCore input wait.

    ``t_engine_arrive`` is stamped by ``EngineCore.preprocess_add_request`` in
    the input-socket thread. ``t_enq`` is stamped later by
    ``Scheduler.add_request`` in the busy-loop thread. Their difference is the
    wait that the original capture accidentally omitted.
    """
    rec = {"req_id": req_id, "t_enq": t_enq, "prompt_len": int(prompt_len)}
    if t_engine_arrive is not None:
        rec["t_engine_arrive"] = t_engine_arrive
        rec["engine_input_wait"] = max(0.0, t_enq - t_engine_arrive)
    return rec


def _request_id(request):
    return getattr(request, "request_id", None) or getattr(request, "req_id", None)


def _prompt_len(request):
    value = getattr(request, "num_prompt_tokens", None)
    if value is not None:
        return int(value)
    tokens = getattr(request, "prompt_token_ids", None)
    return len(tokens) if tokens is not None else 0


def build_request_work_record(request, scheduled_tokens=0, block_size=1):
    """Capture the scheduler-visible work of one running or waiting request.

    Prefix-cache discovery happens inside vLLM's scheduling call.  When vLLM
    exposes a cached-token count we retain it; otherwise ``None`` says it was
    not observable without mutating scheduler state.  The validation deployment
    disables prefix caching, so ``computed`` is sufficient for that run.
    """
    prompt_len = _prompt_len(request)
    computed = int(getattr(request, "num_computed_tokens", 0) or 0)
    scheduled = int(scheduled_tokens or 0)
    cached = getattr(request, "num_cached_tokens", None)
    if cached is not None:
        cached = int(cached)
        # vLLM initializes Request.num_cached_tokens to -1 and only replaces
        # it after prefix-cache discovery.  A negative value therefore means
        # "not observed yet", not negative cached work.
        if cached < 0:
            cached = None
    block_size = max(int(block_size or 1), 1)
    allocated_tokens = max(computed + scheduled, 0)
    return {
        "id": _request_id(request),
        "prompt_len": prompt_len,
        "computed": computed,
        "scheduled_tokens": scheduled,
        "cached_tokens": cached,
        "remaining_prefill_tokens": max(prompt_len - computed, 0),
        "kv_blocks_est": int(math.ceil(allocated_tokens / block_size)),
    }


def summarize_waiting(records, block_size=1):
    """Exact aggregate work for the full queue, even when detail is capped."""
    block_size = max(int(block_size or 1), 1)
    return {
        "count": len(records),
        "prompt_tokens": sum(r["prompt_len"] for r in records),
        "computed_tokens": sum(r["computed"] for r in records),
        "cached_tokens_observed": sum((r["cached_tokens"] or 0) for r in records),
        "cached_tokens_missing": sum(r["cached_tokens"] is None for r in records),
        "remaining_prefill_tokens": sum(r["remaining_prefill_tokens"] for r in records),
        "full_prompt_kv_blocks": sum(
            int(math.ceil(r["prompt_len"] / block_size)) for r in records),
    }


def cap_waiting(waiting_ids, max_ids):
    """Return (exact count, leading ids capped at max_ids, truncated flag).

    The count is always exact so queue depth stays correct; only the id list caps.
    """
    ids = list(waiting_ids)
    count = len(ids)
    return count, ids[:max_ids], count > max_ids


def build_admission_step_record(step, t_start, t_end, total_scheduled, num_running,
                                reqs, waiting_count, waiting_ids, waiting_truncated,
                                free_kv_blocks, running_reqs=None,
                                waiting_reqs=None, waiting_work=None):
    """Assemble one JSONL row: Mode-B running-batch fields plus the queue-and-KV snapshot."""
    rec = {
        "step": step,
        "t_start": t_start,
        "t_end": t_end,
        "total_scheduled": total_scheduled,
        "num_running": num_running,
        "reqs": reqs,
        "waiting_count": waiting_count,
        "waiting_ids": waiting_ids,
        "waiting_truncated": waiting_truncated,
        "free_kv_blocks": free_kv_blocks,
    }
    if running_reqs is not None:
        rec["running_reqs"] = running_reqs
    if waiting_reqs is not None:
        rec["waiting_reqs"] = waiting_reqs
    if waiting_work is not None:
        rec["waiting_work"] = waiting_work
    return rec
