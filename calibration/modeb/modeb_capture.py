"""Pure per-step extraction for Mode-B vLLM instrumentation.

Self-contained (stdlib only, no vllm, no repo imports) so it is unit-testable
offline and safe to ship in a ConfigMap into the vLLM container.
"""


def extract_reqs(num_scheduled_tokens, new, cached_ids, cached_computed, prompt_len_cache):
    """Return one raw record per scheduled request this step.

    num_scheduled_tokens: dict[req_id -> kappa] (SchedulerOutput.num_scheduled_tokens)
    new:            list of (req_id, prompt_len, num_computed) for first-time reqs
    cached_ids:     list of req_ids for continuing reqs (struct-of-arrays)
    cached_computed:list of num_computed_tokens, parallel to cached_ids
    prompt_len_cache: dict mutated in place; prompt length is immutable so we cache
                      it from a req's first (new) appearance and reuse it thereafter.
    """
    for rid, plen, _ in new:
        prompt_len_cache[rid] = plen

    computed_by_id = {}
    for rid, _plen, ncomp in new:
        computed_by_id[rid] = ncomp
    for rid, ncomp in zip(cached_ids, cached_computed):
        computed_by_id[rid] = ncomp

    reqs = []
    for rid, kappa in num_scheduled_tokens.items():
        reqs.append({
            "id": rid,
            "kappa": int(kappa),
            "computed": int(computed_by_id[rid]),
            "prompt_len": int(prompt_len_cache[rid]),
        })
    return reqs


def build_step_record(step, t_start, t_end, total_scheduled, num_running, reqs):
    """Assemble one JSONL row. Stores only raw fields; phase/regime are derived offline.

    t_start / t_end bracket the full step (schedule+execute+update); t_end - t_start is
    a secondary in-step-duration diagnostic. The authoritative T_iter is the inter-step
    delta of t_start, computed offline in parse_trajectory.
    """
    return {
        "step": step,
        "t_start": t_start,
        "t_end": t_end,
        "total_scheduled": total_scheduled,
        "num_running": num_running,
        "reqs": reqs,
    }
