"""Offline reconstruction: parse the captured trajectory + enqueue-event files
into per-request traces, and assemble the AdmissionContext dict that the
ported estimators in calibration.admission.estimators consume.

Pure stdlib + the frozen Mode-B predict_step/load_coeffs helpers (reused, not
reimplemented). vllm-absent testable: this module never imports vllm.

A later stage appends the replay + report on top of this file, so keep it
cleanly append-friendly (no module-level state, no side effects on import).
"""
import json
import math
import os
import statistics

try:
    # Container-style: the frozen Mode-B analysis module sits alongside this
    # deployment as a top-level "modeb" package on PYTHONPATH.
    from modeb.analysis import predict_step, load_coeffs
except ImportError:
    # Offline/repo-package style: this file is imported as
    # calibration.admission.analysis, and modeb sits at calibration.modeb.
    from calibration.modeb.analysis import predict_step, load_coeffs

# estimators.py always sits in this same package (both deployment styles),
# so a package-relative import is unambiguous here (unlike modeb above).
from .estimators import ESTIMATORS
from .estimators import estimate_token_rollforward_times

__all__ = [
    "load_meta", "parse_admission_trajectory", "parse_enqueue_events",
    "request_traces", "build_context", "block_accounting_diag",
    "predict_step", "load_coeffs",
    "RunningMean", "deployable_rem_steps_est", "enqueue_bucket", "replay",
    "admission_censored_rows",
    "admission_report", "write_admission_report",
]


def _block_size(meta):
    """meta always carries block_size; raise a clear error if it doesn't."""
    bs = meta.get("block_size")
    if bs is None or bs <= 0:
        raise ValueError(f"meta['block_size'] must be a positive int, got {bs!r}")
    return bs


def load_meta(path):
    with open(path) as f:
        return json.load(f)


def parse_admission_trajectory(path):
    """Read trajectory.jsonl, sort ascending by step, and add per-step
    t_iter/t_end_next deltas. The last step (no successor) is dropped."""
    with open(path) as f:
        steps = [json.loads(line) for line in f if line.strip()]
    steps.sort(key=lambda s: s["step"])
    out = []
    for i in range(len(steps) - 1):
        s = dict(steps[i])
        s["t_iter"] = steps[i + 1]["t_start"] - s["t_start"]
        s["t_end_next"] = steps[i + 1]["t_start"]
        out.append(s)
    return out


def parse_enqueue_events(path):
    """Read admission_events.jsonl into {req_id -> {t_enq, prompt_len}}.
    Last write per req_id wins if duplicated."""
    events = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            event = dict(row)
            event.pop("req_id", None)
            events[row["req_id"]] = event
    return events


def event_time(enq):
    """Earliest engine-side arrival time available in this capture version."""
    return enq.get("t_engine_arrive", enq["t_enq"])


def request_traces(steps):
    """Build {req_id -> {first_step_idx, last_step_idx, t_sched,
    oracle_output_len, censored}} over the FULL captured step list (censoring
    is defined against whatever steps list is passed in)."""
    steps = sorted(steps, key=lambda s: s["step"])
    n = len(steps)
    traces = {}
    for idx, step in enumerate(steps):
        for r in step["reqs"]:
            rid = r["id"]
            tr = traces.get(rid)
            if tr is None:
                tr = {"first_step_idx": idx, "last_step_idx": idx,
                      "t_sched": step["t_start"], "oracle_output_len": 0}
                traces[rid] = tr
            tr["last_step_idx"] = idx
            if r["computed"] >= r["prompt_len"]:
                tr["oracle_output_len"] += 1
    for tr in traces.values():
        tr["censored"] = tr["last_step_idx"] == n - 1
    return traces


def build_context(step, step_idx, req_id, enq, meta, coeffs, traces, use_oracle,
                  nout_est, enq_events=None, output_len_est=None):
    """Assemble the AdmissionContext dict for req_id evaluated at `step`."""
    block_size = _block_size(meta)
    reqs = step["reqs"]
    running_snapshot = step.get("running_reqs") or reqs

    # batch_size = TOTAL occupied sequence slots (every scheduled request,
    # prefill or decode, holds a slot) -- not a decode-only count.
    batch_size = len(running_snapshot)
    max_batch_size = meta["max_num_seqs"]

    free_kv_blocks = step.get("free_kv_blocks")
    ctx_extra = {}
    if free_kv_blocks is None:
        free_kv_blocks = meta["num_gpu_blocks"] - sum(
            math.ceil((r["computed"] + r.get("scheduled_tokens", r.get("kappa", 0)))
                      / block_size) for r in running_snapshot)
        ctx_extra["free_kv_reconstructed"] = True

    req_kv_need = math.ceil(enq["prompt_len"] / block_size)

    waiting_ids = step.get("waiting_ids") or []

    t_iter = predict_step(reqs, coeffs)

    # ``nout_est`` is already the mean *remaining* lifetime of the current
    # decode batch.  When the learned total output length is available, retain
    # it separately so per-request lifetimes do not subtract steps_done twice
    # and newly admitted requests receive a full output lifetime.
    output_steps_est = (None if output_len_est is None
                        else max(float(output_len_est), 1.0))
    max_decode_steps_done = max((
        max(int(r.get("computed", 0)) - int(r.get("prompt_len", 0)), 0)
        for r in running_snapshot
        if int(r.get("computed", 0)) >= int(r.get("prompt_len", 0))
    ), default=0)
    censored_output_steps_est = (
        max(output_steps_est, max_decode_steps_done)
        if output_steps_est is not None else None)

    running = []
    for r in running_snapshot:
        rid = r["id"]
        tr = traces.get(rid)
        computed = int(r.get("computed", 0))
        prompt_len = int(r.get("prompt_len", 0))
        scheduled_tokens = int(r.get("scheduled_tokens", r.get("kappa", 0)) or 0)
        steps_done = max(computed - prompt_len, 0)
        kv_blocks = int(r.get("kv_blocks_est") or math.ceil(
            (computed + scheduled_tokens) / block_size))
        # Include the already-scheduled current iteration.  The rollout first
        # advances that in-flight work, then decrements this count.
        true_remaining = ((tr["last_step_idx"] - step_idx + 1)
                          if use_oracle and tr is not None else -1)
        if computed >= prompt_len:
            if censored_output_steps_est is not None:
                remaining_est = max(censored_output_steps_est - steps_done, 1.0)
            else:
                remaining_est = max(float(nout_est) - steps_done, 1.0)
        else:
            token_cap = max(int(meta.get("max_num_batched_tokens", prompt_len or 1)), 1)
            chunks_left = math.ceil(max(prompt_len - computed, 0) / token_cap)
            # nout_est is learned from captured decode records, which begin
            # after final prefill.  Keep those future decode steps in addition
            # to every remaining prefill chunk, including the in-flight one.
            future_decode = (output_steps_est if output_steps_est is not None
                             else float(nout_est))
            remaining_est = max(chunks_left + future_decode, 1.0)
        running.append({
            "id": rid, "prompt_len": prompt_len, "computed": computed,
            "scheduled_tokens": scheduled_tokens, "steps_done": steps_done,
            "kv_blocks": kv_blocks, "true_remaining": true_remaining,
            "remaining_est": remaining_est,
        })

    waiting = []
    for r in step.get("waiting_reqs") or []:
        waiting.append(dict(r))
    if not waiting and enq_events is not None:
        for rid in waiting_ids:
            ev = enq_events.get(rid)
            if ev is None:
                continue
            waiting.append({
                "id": rid, "prompt_len": int(ev["prompt_len"]),
                "computed": int(ev.get("cached_tokens", 0) or 0),
                "scheduled_tokens": 0, "cached_tokens": ev.get("cached_tokens"),
                "remaining_prefill_tokens": max(
                    int(ev["prompt_len"]) - int(ev.get("cached_tokens", 0) or 0), 0),
                "kv_blocks_est": 0,
            })

    arrival = event_time(enq)
    # The scheduler snapshot is taken near the start of the in-flight step.
    # Requests arriving later in that same step are already ahead of a still
    # later target in EngineCore's input queue, even though none appears in the
    # scheduler's waiting deque yet. Recover that FIFO work from event order.
    known_ids = {r.get("id") for r in running_snapshot}
    known_ids.update(waiting_ids)
    pending_input = []
    if enq_events is not None and req_id not in waiting_ids:
        for rid, event in enq_events.items():
            if rid == req_id or rid in known_ids:
                continue
            candidate_arrival = event_time(event)
            if step["t_start"] < candidate_arrival < arrival:
                cached = int(event.get("cached_tokens", 0) or 0)
                prompt_len = int(event["prompt_len"])
                pending_input.append((candidate_arrival, {
                    "id": rid, "prompt_len": prompt_len,
                    "computed": cached, "scheduled_tokens": 0,
                    "cached_tokens": event.get("cached_tokens"),
                    "remaining_prefill_tokens": max(prompt_len - cached, 0),
                    "kv_blocks_est": math.ceil(cached / block_size),
                }))
        pending_input.sort(key=lambda item: item[0])

    pending_reqs = [record for _, record in pending_input]
    if req_id in waiting_ids:
        queue_depth = waiting_ids.index(req_id)
    else:
        queue_depth = int(step.get("waiting_count", len(waiting))) + len(pending_reqs)
        ctx_extra["queue_pos_from_count"] = True
    ctx_extra["pending_input_count"] = len(pending_reqs)

    waiting_work = dict(step.get("waiting_work") or {
        "count": step.get("waiting_count", len(waiting)),
        "remaining_prefill_tokens": sum(
            max(r.get("prompt_len", 0) - r.get("computed", 0), 0)
            for r in waiting),
    })
    if pending_reqs:
        waiting_work["count"] = int(waiting_work.get("count", 0)) + len(pending_reqs)
        waiting_work["prompt_tokens"] = int(waiting_work.get("prompt_tokens", 0)) + sum(
            r["prompt_len"] for r in pending_reqs)
        waiting_work["computed_tokens"] = int(
            waiting_work.get("computed_tokens", 0)) + sum(
                r["computed"] for r in pending_reqs)
        waiting_work["cached_tokens_observed"] = int(
            waiting_work.get("cached_tokens_observed", 0)) + sum(
                (r["cached_tokens"] or 0) for r in pending_reqs)
        waiting_work["cached_tokens_missing"] = int(
            waiting_work.get("cached_tokens_missing", 0)) + sum(
                r["cached_tokens"] is None for r in pending_reqs)
        waiting_work["remaining_prefill_tokens"] = int(
            waiting_work.get("remaining_prefill_tokens", 0)) + sum(
                r["remaining_prefill_tokens"] for r in pending_reqs)
        waiting_work["full_prompt_kv_blocks"] = int(
            waiting_work.get("full_prompt_kv_blocks", 0)) + sum(
                math.ceil(r["prompt_len"] / block_size) for r in pending_reqs)
        # Preserve FIFO order exactly when the captured prefix is complete. If
        # it was capped, _expand_waiting synthesizes the missing older work
        # ahead of these arrivals from the exact aggregate instead.
        if not step.get("waiting_truncated", False):
            waiting.extend(pending_reqs)

    current_iter_elapsed = max(arrival - step["t_start"], 0.0)

    ctx = {
        "batch_size": batch_size,
        "max_batch_size": max_batch_size,
        "free_kv_blocks": free_kv_blocks,
        "req_kv_need": req_kv_need,
        "t_iter": t_iter,
        "queue_depth": queue_depth,
        "remaining_steps_est": nout_est,
        "output_steps_est": (output_steps_est if output_steps_est is not None
                             else nout_est),
        "running": running,
        "waiting": waiting,
        "waiting_work": waiting_work,
        "max_num_batched_tokens": meta.get("max_num_batched_tokens", 0),
        "long_prefill_token_threshold": meta.get("long_prefill_token_threshold", 0),
        "block_size": block_size,
        "coeffs": coeffs,
        "current_iter_elapsed": current_iter_elapsed,
        "current_iter_remaining": max(t_iter - current_iter_elapsed, 0.0),
        "target_id": req_id,
        "target_prompt_len": int(enq["prompt_len"]),
        "target_cached_tokens": int(enq.get("cached_tokens", 0) or 0),
        # Only schema-v2 snapshots have the complete running state plus exact
        # aggregate queued work required by token-level scheduler replay.
        # Legacy captures retain their bit-for-bit rollforward behavior.
        "queue_work_observed": (
            "running_reqs" in step and "waiting_reqs" in step
            and "waiting_work" in step),
    }
    ctx.update(ctx_extra)
    return ctx


def block_accounting_diag(step, meta):
    """Compare the captured free-KV-block count against the value
    reconstructed from per-request `computed` tokens, for drift diagnosis."""
    block_size = _block_size(meta)
    captured_free = step.get("free_kv_blocks")
    reconstructed_free = meta["num_gpu_blocks"] - sum(
        math.ceil((r["computed"] + r.get("scheduled_tokens", r.get("kappa", 0)))
                  / block_size) for r in (step.get("running_reqs") or step["reqs"]))
    delta = None if captured_free is None else captured_free - reconstructed_free
    return {"captured_free": captured_free, "reconstructed_free": reconstructed_free,
            "delta": delta}


# ---------------------------------------------------------------------------
# Part 2 (Task 5): teacher-forced replay + report.
#
# Appends on top of the Task-4 reconstruction helpers above. Nothing above
# this line is modified.
# ---------------------------------------------------------------------------


class RunningMean:
    """Per-class running mean of realized output lengths, floored at 1.

    Mirrors `edppRunningMean` in inference-sim/sim/edpp.go: empty -> 1.0
    (conservative 1-token seed); otherwise the plain mean, floored at 1.0
    so it can never license a negative or zero remaining-steps estimate.
    """

    def __init__(self):
        self._n = 0
        self._sum = 0.0

    def add(self, x):
        self._n += 1
        self._sum += x

    def value(self):
        if self._n == 0:
            return 1.0
        return max(self._sum / self._n, 1.0)


def deployable_rem_steps_est(step, nhat_out_mean):
    """Faithful port of `decodeRemStepsEst` (inference-sim/sim/edpp.go):
    the deployable, censored remaining-decode-steps estimate for `step`.

    decode occupants only (computed >= prompt_len); no decode occupants ->
    1.0. Otherwise nhat_eff = max(nhat_out_mean, max steps_done) is the
    censored class output estimate (o_r >= steps_done for every in-flight
    occupant), and the per-request remaining estimate is
    max(nhat_eff - steps_done_r, 1); the returned value is the mean of
    those per-request estimates over decode occupants.
    """
    decode = [r for r in step["reqs"] if r["computed"] >= r["prompt_len"]]
    if not decode:
        return 1.0
    steps_done = [r["computed"] - r["prompt_len"] for r in decode]
    max_steps = max(steps_done)
    nhat_eff = max(nhat_out_mean, max_steps)
    per_req = [max(nhat_eff - sd, 1) for sd in steps_done]
    return sum(per_req) / len(per_req)


def enqueue_bucket(steps, enq_events):
    """Map each enqueued request to the step whose [t_start, t_end_next)
    bracket contains its t_enq. `steps` must already carry `t_end_next`
    (i.e. be the output of parse_admission_trajectory).

    Returns (dict[req_id -> step], dropped_count). A request enqueued
    before the first step's t_start, or at/after the last step's
    t_end_next, falls outside every bracket and is dropped (counted, not
    included in the dict).
    """
    bucket = {}
    dropped = 0
    for req_id, enq in enq_events.items():
        t_enq = event_time(enq)
        found = None
        for step in steps:
            if step["t_start"] <= t_enq < step["t_end_next"]:
                found = step
                break
        if found is None:
            dropped += 1
        else:
            bucket[req_id] = found
    return bucket, dropped


def _is_prefill_req(r):
    return r["computed"] < r["prompt_len"]


def _regime_at_enq(reqs):
    """Classify a batch composition as pure_prefill/pure_decode/mixed.
    Mirrors infocom/figures/plot_titer_modeb.py's regime()."""
    has_prefill = any(_is_prefill_req(r) for r in reqs)
    has_decode = any(not _is_prefill_req(r) for r in reqs)
    if has_prefill and has_decode:
        return "mixed"
    if has_prefill:
        return "pure_prefill"
    return "pure_decode"


def replay(steps, enq_events, traces, meta, coeffs, estimator_name, use_oracle,
           load_of, include_censored=False):
    """Run estimator `estimator_name` on every non-censored, bucketed,
    ACTUALLY-SCHEDULED request's reconstructed enqueue-time context, and
    compare its prediction against the realized admission delay.

    Requests are processed in ascending t_enq order. For the deployable
    variant (use_oracle=False), the RunningMean of completed-request
    output lengths is advanced, in that same time order, over every
    non-censored request whose OWN completion time (the t_end_next of its
    last captured step) falls before the current request's t_enq -- i.e.
    it only ever sees requests that had actually finished by that instant,
    independent of which request is being scored. This mirrors
    edppRunningMean being updated at each request's real departure.

    Returns (rows, never_scheduled_count). A request that was enqueued
    (has an enq_events entry, and a bucket) but NEVER appears in any
    step's `reqs` -- i.e. it sat in waiting_ids for the entire capture
    window, the defining symptom of the overload/backlog regime this
    harness targets -- has no `traces` entry and therefore no `t_sched`,
    so its realized T_adm is undefined. Such requests are excluded from
    `rows` and counted in `never_scheduled_count` instead of raising.
    Requests that ARE in `traces` but run past capture end (`censored`) are
    normally excluded because their output length is incomplete.  If
    ``include_censored`` is true for a deployable replay, they are included:
    admission is already exactly observed, and callers may also have observed
    their first-token instant even though departure is absent.  The deployable
    estimate never uses the target's eventual output length.  Never-scheduled
    requests are also predicted and emitted with ``right_censored=True`` and a
    realized lower bound; they carry no exact realized value and remain
    ineligible for MAPE.
    """
    estimator = ESTIMATORS[estimator_name]
    bucket, _dropped = enqueue_bucket(steps, enq_events)
    idx_of = {id(step): i for i, step in enumerate(steps)}

    completions = []
    if not use_oracle:
        for rid, tr in traces.items():
            if tr["censored"]:
                continue
            last_idx = tr["last_step_idx"]
            if last_idx < len(steps):
                completions.append((steps[last_idx]["t_end_next"], tr["oracle_output_len"]))
        completions.sort(key=lambda c: c[0])

    never_scheduled = 0
    req_ids = []
    for rid in bucket:
        if rid not in traces:
            never_scheduled += 1  # enqueued + bucketed but never scheduled (still waiting)
            if include_censored and not use_oracle:
                req_ids.append(rid)
            continue
        if (not traces[rid]["censored"]
                or (include_censored and not use_oracle)):
            req_ids.append(rid)
    req_ids.sort(key=lambda rid: event_time(enq_events[rid]))

    running_mean = RunningMean()
    comp_i = 0
    rows = []
    for req_id in req_ids:
        enq = enq_events[req_id]
        step = bucket[req_id]
        step_idx = idx_of[id(step)]
        t_enq = event_time(enq)
        reqs = step["reqs"]

        output_len_est = None
        if use_oracle:
            if reqs:
                nout_est = max(1.0, sum(
                    traces[r["id"]]["last_step_idx"] - step_idx for r in reqs) / len(reqs))
            else:
                nout_est = 1.0
        else:
            while comp_i < len(completions) and completions[comp_i][0] < t_enq:
                running_mean.add(completions[comp_i][1])
                comp_i += 1
            output_len_est = running_mean.value()
            nout_est = deployable_rem_steps_est(step, output_len_est)

        ctx = build_context(step, step_idx, req_id, enq, meta, coeffs, traces,
                            use_oracle, nout_est, enq_events=enq_events,
                            output_len_est=output_len_est)
        if estimator_name == "token_rollforward":
            predicted, predicted_first_token = estimate_token_rollforward_times(ctx)
        else:
            predicted = estimator(ctx)
            predicted_first_token = None
        right_censored = req_id not in traces
        realized = (None if right_censored
                    else traces[req_id]["t_sched"] - t_enq)
        ratio = (realized / predicted
                 if realized is not None and predicted > 0 else None)
        row = {
            "req_id": req_id,
            "predicted": predicted,
            "realized": realized,
            "ratio": ratio,
            "load_bin": load_of(req_id),
            "regime_at_enq": _regime_at_enq(reqs),
            "t_engine_arrive": enq.get("t_engine_arrive"),
            "t_enq": enq["t_enq"],
            "engine_input_wait": enq.get("engine_input_wait"),
            "right_censored": right_censored,
        }
        if predicted_first_token is not None:
            row["predicted_first_token"] = predicted_first_token
        if right_censored:
            lower_bound = max(steps[-1]["t_end_next"] - t_enq, 0.0)
            row["lower_bound"] = lower_bound
            row["prediction_below_lower_bound"] = predicted < lower_bound
            row["known_underprediction_lower_bound"] = max(
                lower_bound - predicted, 0.0)
        rows.append(row)
    return rows, never_scheduled


def admission_censored_rows(steps, enq_events, traces, load_of,
                            predictions=None):
    """Right-censored admission observations for requests never scheduled.

    Each row says only that ``T_adm`` exceeded ``lower_bound`` at capture end;
    it must not be folded into MAPE as though the bound were the realized wait.
    """
    bucket, _ = enqueue_bucket(steps, enq_events)
    if not steps:
        return []
    capture_end = steps[-1]["t_end_next"]
    rows = []
    for rid, step in bucket.items():
        if rid in traces:
            continue
        arrival = event_time(enq_events[rid])
        row = {
            "req_id": rid,
            "lower_bound": max(capture_end - arrival, 0.0),
            "arrival": arrival,
            "capture_end": capture_end,
            "load_bin": load_of(rid),
            "regime_at_enq": _regime_at_enq(step["reqs"]),
        }
        if predictions is not None and rid in predictions:
            predicted = predictions[rid]
            row["predicted"] = predicted
            row["prediction_below_lower_bound"] = predicted < row["lower_bound"]
            row["known_underprediction_lower_bound"] = max(
                row["lower_bound"] - predicted, 0.0)
        rows.append(row)
    return rows


def _load_section(load_bin):
    return "overload" if "overload" in str(load_bin) else "sub_capacity"


def admission_report(rows_by_key, diag_rows=None, censored_rows=None):
    """Aggregate replay rows into a report dict.

    rows_by_key: {(estimator, variant): [row, ...]} -- the `rows` element of
    replay()'s (rows, never_scheduled_count) return value.
    Per (estimator, variant, load_bin): {n, median_ratio, mape, bias_pct}.
      n = total rows in the bin (NOT the count contributing to mape/
      median_ratio, which silently skip rows with a None ratio or a
      falsy/zero `realized`).
      mape = mean(|predicted - realized| / realized) * 100.
      bias_pct = (median_ratio - 1) * 100 (chosen definition; the
      alternative, mean signed relative error, is not used here).
    Rows are split into "sub_capacity" and "overload" top-level sections by
    whether their load_bin contains the substring "overload".

    diag_rows, if given, is a list of block_accounting_diag(...) outputs;
    their |delta| (skipping None) is aggregated into the top-level
    "block_accounting" key (mean/max absolute delta over sampled steps).
    """
    report = {"sub_capacity": {}, "overload": {}}
    for (estimator, variant), rows in rows_by_key.items():
        by_bin = {}
        for row in rows:
            by_bin.setdefault(row["load_bin"], []).append(row)
        for load_bin, bin_rows in by_bin.items():
            n = len(bin_rows)
            ratios = [r["ratio"] for r in bin_rows if r["ratio"] is not None]
            median_ratio = statistics.median(ratios) if ratios else None
            errs = [abs(r["predicted"] - r["realized"]) / r["realized"]
                    for r in bin_rows if r.get("realized")]
            abs_errs = [abs(r["predicted"] - r["realized"])
                        for r in bin_rows if r.get("realized")]
            sorted_errs = sorted(errs)
            realized_sum = sum(r["realized"] for r in bin_rows
                               if r.get("realized"))
            mape = (sum(errs) / len(errs) * 100.0) if errs else None
            bias_pct = ((median_ratio - 1) * 100.0) if median_ratio is not None else None
            section = _load_section(load_bin)
            section_dict = report[section]
            section_dict.setdefault(estimator, {}).setdefault(variant, {})[load_bin] = {
                "n": n, "median_ratio": median_ratio, "mape": mape,
                "median_ape": (statistics.median(errs) * 100.0 if errs else None),
                "p90_ape": (sorted_errs[min(int(0.9 * len(sorted_errs)),
                                             len(sorted_errs) - 1)] * 100.0
                            if sorted_errs else None),
                "mae_ms": (sum(abs_errs) / len(abs_errs) * 1000.0
                           if abs_errs else None),
                "wape": (sum(abs_errs) / realized_sum * 100.0
                         if realized_sum > 0 else None),
                "bias_pct": bias_pct,
            }

    if diag_rows:
        deltas = [abs(d["delta"]) for d in diag_rows if d.get("delta") is not None]
    else:
        deltas = []
    report["block_accounting"] = {
        "n": len(deltas),
        "mean_abs_delta": (sum(deltas) / len(deltas)) if deltas else None,
        "max_abs_delta": max(deltas) if deltas else None,
    }
    censored_rows = censored_rows or []
    lower_bounds = sorted(r["lower_bound"] for r in censored_rows)
    predicted_censored = [r for r in censored_rows if r.get("predicted") is not None]
    known_shortfalls = sorted(
        r.get("known_underprediction_lower_bound", 0.0)
        for r in predicted_censored)
    violations = sum(bool(r.get("prediction_below_lower_bound"))
                     for r in predicted_censored)
    report["right_censored"] = {
        "n": len(censored_rows),
        "lower_bound_s_p50": (statistics.median(lower_bounds)
                              if lower_bounds else None),
        "lower_bound_s_p90": (lower_bounds[min(
            int(0.9 * len(lower_bounds)), len(lower_bounds) - 1)]
                              if lower_bounds else None),
        "by_load_bin": {
            name: sum(1 for r in censored_rows if r["load_bin"] == name)
            for name in sorted({r["load_bin"] for r in censored_rows})
        },
        "prediction_n": len(predicted_censored),
        "prediction_below_lower_bound_n": violations,
        "prediction_below_lower_bound_frac": (
            violations / len(predicted_censored) if predicted_censored else None),
        "known_underprediction_s_p50": (
            statistics.median(known_shortfalls) if known_shortfalls else None),
        "known_underprediction_s_p90": (
            known_shortfalls[min(int(0.9 * len(known_shortfalls)),
                                 len(known_shortfalls) - 1)]
            if known_shortfalls else None),
        "note": ("right-censored; lower bounds are not included in MAPE. "
                 "A prediction below its lower bound is a proven underprediction."),
    }
    return report


def write_admission_report(report, rows_by_key, out_dir):
    """Write admission_report.json (the report dict) and, if matplotlib is
    importable, a log-log predicted-vs-realized scatter colored by regime
    (admission_pred_vs_realized.png), mirroring
    infocom/figures/plot_titer_modeb.py. The JSON always writes; the PNG
    is skipped gracefully (matplotlib imported lazily, guarded) when
    matplotlib is absent.
    """
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "admission_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"json_path": json_path, "png_path": None}

    colors = {"pure_decode": "#1f77b4", "pure_prefill": "#2ca02c", "mixed": "#d62728"}
    all_rows = [row for rows in rows_by_key.values() for row in rows]
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    for name, color in colors.items():
        sub = [r for r in all_rows if r["regime_at_enq"] == name and r["predicted"] > 0
               and r["realized"] > 0]
        if not sub:
            continue
        xs = [r["realized"] for r in sub]
        ys = [r["predicted"] for r in sub]
        ax.scatter(xs, ys, s=3, alpha=0.3, color=color, edgecolors="none", label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("realized T_adm (s)")
    ax.set_ylabel("predicted T_adm (s)")
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    fig.tight_layout(pad=0.3)
    png_path = os.path.join(out_dir, "admission_pred_vs_realized.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"json_path": json_path, "png_path": png_path}
