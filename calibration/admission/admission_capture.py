"""Pure per-step and per-request extraction for the admission-validation hook.

Self-contained (stdlib only, no vllm) so it is unit-testable offline and safe
to ship in a ConfigMap. Reuses the frozen Mode-B extract_reqs for the running
batch; adds enqueue-event records and the waiting-queue / free-KV per-step fields.
"""


def build_enqueue_record(req_id, t_enq, prompt_len):
    """One row per request at its enqueue instant. prompt_len drives ReqKVNeed offline."""
    return {"req_id": req_id, "t_enq": t_enq, "prompt_len": int(prompt_len)}


def cap_waiting(waiting_ids, max_ids):
    """Return (exact count, leading ids capped at max_ids, truncated flag).

    The count is always exact so queue depth stays correct; only the id list caps.
    """
    ids = list(waiting_ids)
    count = len(ids)
    return count, ids[:max_ids], count > max_ids


def build_admission_step_record(step, t_start, t_end, total_scheduled, num_running,
                                reqs, waiting_count, waiting_ids, waiting_truncated,
                                free_kv_blocks):
    """Assemble one JSONL row: Mode-B running-batch fields plus the queue-and-KV snapshot."""
    return {
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
