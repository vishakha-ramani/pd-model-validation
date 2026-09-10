"""Byte-faithful Python ports of inference-sim/sim/admission_estimator.go.

Unit-agnostic. The admission harness runs them in seconds (coeffs.json units),
so t_iter and the returned T_adm are seconds.
"""
import math


def floored_t_adm(est, ctx):
    t_iter = ctx["t_iter"]
    if t_iter > est:
        return t_iter
    return est


def _slot_and_kv_fit(ctx):
    return ctx["batch_size"] < ctx["max_batch_size"] and ctx["free_kv_blocks"] >= ctx["req_kv_need"]


def estimate_waiting(ctx):
    mu = ctx.get("mu", 0.0)
    if mu <= 0:
        return 0.0
    return ctx.get("qwork", 0.0) / mu


def estimate_fluid(ctx):
    if _slot_and_kv_fit(ctx):
        return floored_t_adm(0.0, ctx)
    if ctx["batch_size"] <= 0 or ctx["remaining_steps_est"] <= 0 or ctx["t_iter"] <= 0:
        return floored_t_adm(0.0, ctx)
    waves = math.ceil((ctx["queue_depth"] + 1) / ctx["batch_size"])
    return floored_t_adm(waves * ctx["remaining_steps_est"] * ctx["t_iter"], ctx)


def estimate_rollforward(ctx):
    if _slot_and_kv_fit(ctx):
        return floored_t_adm(0.0, ctx)
    deps = []
    for r in ctx["running"]:
        rem = r["true_remaining"]
        if rem < 0:
            rem = int(ctx["remaining_steps_est"])
            if rem < 1:
                rem = 1
        deps.append((rem, r["kv_blocks"]))
    deps.sort(key=lambda d: d[0])  # stable ascending, matches sort.SliceStable
    need_slots = ctx["queue_depth"] + 1
    free_slots = ctx["max_batch_size"] - ctx["batch_size"]
    free_kv = ctx["free_kv_blocks"]
    for rem, kv in deps:
        free_slots += 1
        free_kv += kv
        if free_slots >= need_slots and free_kv >= ctx["req_kv_need"]:
            return floored_t_adm(rem * ctx["t_iter"], ctx)
    if ctx["batch_size"] > 0:
        waves = math.ceil((ctx["queue_depth"] + 1) / ctx["batch_size"])
        return floored_t_adm(waves * ctx["remaining_steps_est"] * ctx["t_iter"], ctx)
    if deps:
        return floored_t_adm(deps[-1][0] * ctx["t_iter"], ctx)
    return floored_t_adm(0.0, ctx)


def _ceil_blocks(tokens, block_size):
    return int(math.ceil(max(tokens, 0) / max(block_size, 1)))


def _rich_rollout_context(ctx):
    """Whether the capture contains enough state for scheduler-step replay."""
    required = ("max_num_batched_tokens", "coeffs", "waiting", "block_size")
    return bool(ctx.get("queue_work_observed")) and all(k in ctx for k in required) and all(
        "prompt_len" in r and "computed" in r for r in ctx.get("running", []))


def _remaining_for(r, default):
    rem = r.get("true_remaining", -1)
    if rem is None or rem < 0:
        rem = r.get("remaining_est", default)
    return max(int(math.ceil(rem or default or 1)), 1)


def _request_copy(r, block_size, default_remaining):
    computed = int(r.get("computed", 0) or 0)
    scheduled = int(r.get("scheduled_tokens", 0) or 0)
    return {
        "id": r.get("id"),
        "prompt_len": int(r.get("prompt_len", 0) or 0),
        "computed": computed,
        "scheduled_tokens": scheduled,
        "kv_blocks": int(r.get(
            "kv_blocks", r.get("kv_blocks_est", _ceil_blocks(computed + scheduled, block_size))) or 0),
        "remaining": _remaining_for(r, default_remaining),
        "is_target": bool(r.get("is_target", False)),
    }


def _expand_waiting(ctx, block_size, output_steps, prefill_chunk_cap):
    """Return the FIFO work ahead, filling capped detail from exact aggregates."""
    waiting = [_request_copy(r, block_size, output_steps)
               for r in ctx.get("waiting", [])]
    for req in waiting:
        chunks = int(math.ceil(
            max(req["prompt_len"] - req["computed"], 0)
            / max(prefill_chunk_cap, 1)))
        req["remaining"] = max(int(math.ceil(output_steps)) + chunks, 1)
    queue_depth = max(int(ctx.get("queue_depth", len(waiting)) or 0), 0)
    if len(waiting) >= queue_depth:
        return waiting[:queue_depth]

    missing = queue_depth - len(waiting)
    summary = ctx.get("waiting_work") or {}
    detailed_remaining = sum(max(r["prompt_len"] - r["computed"], 0) for r in waiting)
    remaining_total = max(int(summary.get("remaining_prefill_tokens", 0) or 0)
                          - detailed_remaining, 0)
    avg_remaining = int(math.ceil(remaining_total / missing)) if remaining_total else int(
        ctx.get("target_prompt_len", 1) or 1)
    for i in range(missing):
        waiting.append({
            "id": f"__uncaptured_waiting_{i}",
            "prompt_len": avg_remaining,
            "computed": 0,
            "scheduled_tokens": 0,
            "kv_blocks": 0,
            "remaining": max(int(math.ceil(output_steps or 1))
                             + int(math.ceil(avg_remaining / max(prefill_chunk_cap, 1))), 1),
            "is_target": False,
        })
    return waiting


def _step_time(scheduled, coeffs):
    total = float(coeffs["c_base"])
    for req, before, grant, was_prefill in scheduled:
        if was_prefill:
            total += (float(coeffs["c_pf"]) * grant
                      + float(coeffs["c_attn"]) * grant * (before + grant / 2.0))
        else:
            total += float(coeffs["c_dec"]) + float(coeffs["c_kv"]) * before
    return max(total, 0.0)


def _advance_finished_step(running, free_kv, block_size, default_remaining):
    """Advance the already in-flight iteration that contains the arrival."""
    kept = []
    for req in running:
        grant = req.get("scheduled_tokens", 0)
        if grant <= 0:
            kept.append(req)
            continue
        was_prefill = req["computed"] < req["prompt_len"]
        req["computed"] += grant
        req["scheduled_tokens"] = 0
        if was_prefill:
            # remaining includes the in-flight chunk and all future decode
            # records, so advancing any prefill chunk consumes one step too.
            req["remaining"] -= 1
            kept.append(req)
        else:
            req["remaining"] -= 1
            if req["remaining"] <= 0:
                free_kv += req["kv_blocks"]
            else:
                kept.append(req)
    return kept, free_kv


def estimate_token_rollforward_times(ctx):
    """Return ``(admission, first_token)`` from a scheduler-step rollout.

    Admission is the iteration boundary where the target first receives a token
    grant.  First-token time is the end of the step that processes the target's
    final prompt chunk (or its first decode grant for an already-computed
    prompt).  Thus the same event rollout can drive both the admission-only
    report and the deployable TTFT estimate.

    Legacy captures without structured queue work return the frozen admission
    estimate and ``None`` for first-token time rather than inventing queue work.
    """
    if not _rich_rollout_context(ctx):
        return estimate_rollforward(ctx), None

    token_cap = int(ctx["max_num_batched_tokens"] or 0)
    if token_cap <= 0:
        return estimate_rollforward(ctx), None
    block_size = max(int(ctx.get("block_size", 1) or 1), 1)
    max_batch = max(int(ctx.get("max_batch_size", 0) or 0), 1)
    default_remaining = max(float(ctx.get("remaining_steps_est", 1.0) or 1.0), 1.0)
    output_steps = max(float(ctx.get("output_steps_est", default_remaining) or
                             default_remaining), 1.0)
    long_threshold = int(ctx.get("long_prefill_token_threshold", 0) or 0)
    prefill_chunk_cap = min(token_cap, long_threshold) if long_threshold > 0 else token_cap
    coeffs = ctx["coeffs"]

    full_iter = max(float(ctx.get("t_iter", 0.0) or 0.0), 0.0)
    if "current_iter_remaining" in ctx:
        elapsed = max(float(ctx["current_iter_remaining"] or 0.0), 0.0)
    else:
        phase = max(float(ctx.get("current_iter_elapsed", 0.0) or 0.0), 0.0)
        elapsed = max(full_iter - phase, 0.0)

    running = [_request_copy(r, block_size, default_remaining)
               for r in ctx.get("running", [])]
    free_kv = max(int(ctx.get("free_kv_blocks", 0) or 0), 0)
    running, free_kv = _advance_finished_step(
        running, free_kv, block_size, default_remaining)

    waiting = _expand_waiting(ctx, block_size, output_steps, prefill_chunk_cap)
    target_id = ctx.get("target_id", "__target__")
    waiting = [r for r in waiting if r.get("id") != target_id]
    target_computed = max(int(ctx.get("target_cached_tokens", 0) or 0), 0)
    waiting.append({
        "id": target_id,
        "prompt_len": int(ctx.get("target_prompt_len", 0) or 0),
        "computed": target_computed,
        "scheduled_tokens": 0,
        "kv_blocks": _ceil_blocks(target_computed, block_size),
        "remaining": max(
            int(math.ceil(output_steps))
            + int(math.ceil(max(int(ctx.get("target_prompt_len", 0) or 0)
                                - target_computed, 0) / max(prefill_chunk_cap, 1))),
            1),
        "is_target": True,
    })

    target_admission = None
    max_steps = max(int(ctx.get("max_rollout_steps", 100000) or 100000), 1)
    for _ in range(max_steps):
        budget = token_cap
        scheduled = []
        preempted = []

        # vLLM 0.11 schedules RUNNING requests first. Under FCFS, an allocation
        # failure repeatedly preempts the last running request, resets its
        # computed prefix, and prepends it to WAITING. If anything is preempted,
        # vLLM deliberately skips new admissions for this scheduler pass.
        req_index = 0
        while req_index < len(running) and budget > 0:
            req = running[req_index]
            before = req["computed"]
            was_prefill = before < req["prompt_len"]
            demand = max(req["prompt_len"] - before, 0) if was_prefill else 1
            if long_threshold > 0:
                demand = min(demand, long_threshold)
            grant = min(demand, budget)
            if grant <= 0:
                req_index += 1
                continue
            new_blocks = _ceil_blocks(before + grant, block_size)
            delta_blocks = max(new_blocks - req["kv_blocks"], 0)
            can_schedule = True
            while delta_blocks > free_kv:
                victim = running.pop()
                free_kv += victim["kv_blocks"]
                victim["kv_blocks"] = 0
                resume_tokens = (victim["prompt_len"]
                                 if victim["computed"] < victim["prompt_len"]
                                 else victim["computed"] + 1)
                recompute_chunks = int(math.ceil(
                    resume_tokens / max(prefill_chunk_cap, 1)))
                victim["prompt_len"] = resume_tokens
                victim["computed"] = 0
                victim["scheduled_tokens"] = 0
                victim["remaining"] += recompute_chunks
                preempted.insert(0, victim)
                if victim is req:
                    can_schedule = False
                    break
            if not can_schedule:
                break
            free_kv -= delta_blocks
            req["kv_blocks"] = new_blocks
            scheduled.append((req, before, grant, was_prefill))
            budget -= grant
            req_index += 1

        # Then vLLM admits FIFO waiting requests while tokens, slots, and KV fit.
        if preempted:
            waiting = preempted + waiting
        while not preempted and waiting and budget > 0 and len(running) < max_batch:
            req = waiting[0]
            before = req["computed"]
            was_prefill = before < req["prompt_len"]
            demand = max(req["prompt_len"] - before, 0) if was_prefill else 1
            if long_threshold > 0:
                demand = min(demand, long_threshold)
            grant = min(demand, budget)
            if grant <= 0:
                break
            new_blocks = _ceil_blocks(before + grant, block_size)
            delta_blocks = max(new_blocks - req["kv_blocks"], 0)
            if delta_blocks > free_kv:
                break
            if req["is_target"]:
                target_admission = elapsed
            waiting.pop(0)
            free_kv -= delta_blocks
            req["kv_blocks"] = new_blocks
            running.append(req)
            scheduled.append((req, before, grant, was_prefill))
            budget -= grant

        if not scheduled and preempted:
            # A pass that preempts its only running request executes no GPU
            # work. The next pass can resume it from WAITING with the freed KV.
            continue
        if not scheduled:
            # Remote-KV/FSM/encoder blockers are not present in this validation
            # deployment. Fall back rather than spin if a corrupt capture still
            # presents an unexplained no-progress state.
            fallback = elapsed + estimate_rollforward(ctx)
            return (target_admission if target_admission is not None else fallback,
                    None)

        step_elapsed = _step_time(scheduled, coeffs)
        target_first_token = any(
            req.get("is_target")
            and ((was_prefill and before + grant >= req["prompt_len"])
                 or not was_prefill)
            for req, before, grant, was_prefill in scheduled)
        elapsed += step_elapsed

        kept = []
        for req in running:
            item = next((x for x in scheduled if x[0] is req), None)
            if item is None:
                kept.append(req)
                continue
            _, before, grant, was_prefill = item
            req["computed"] = before + grant
            if was_prefill:
                req["remaining"] -= 1
                kept.append(req)
            else:
                req["remaining"] -= 1
                if req["remaining"] <= 0:
                    free_kv += req["kv_blocks"]
                else:
                    kept.append(req)
        running = kept

        if target_first_token:
            return target_admission, elapsed

    # A finite guard is necessary for corrupt captures. Preserve monotonicity
    # by returning the simulated lower bound plus the legacy tail estimate.
    fallback = elapsed + estimate_rollforward(ctx)
    return (target_admission if target_admission is not None else fallback,
            None)


def estimate_token_rollforward(ctx):
    """Admission component of :func:`estimate_token_rollforward_times`."""
    return estimate_token_rollforward_times(ctx)[0]


ESTIMATORS = {
    "waiting": estimate_waiting,
    "fluid": estimate_fluid,
    "rollforward": estimate_rollforward,
    "token_rollforward": estimate_token_rollforward,
}
