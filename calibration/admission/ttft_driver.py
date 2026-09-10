"""Reconstruct realized and predicted first-token time from the admission capture.

This is the offline driver behind figures/ttft_parity.png. It reads the raw
per-step trajectory and per-request enqueue events written by
calibration/admission/sitecustomize.py, reconstructs each request's realized
first-token time, composes the estimator's prediction from the frozen
coefficients, and writes both a pooled per-request row file and a per-segment
summary report.

Equations, tagged as in README.md
--------------------------------

(E1) Per-iteration latency law. For a scheduled batch B at one engine step,

         T_iter(B) = c_base
                   + c_dec * B_dec
                   + c_kv  * sum_{r in decode(B)} computed_r
                   + sum_{r in prefill(B)} [ c_pf * kappa_r
                                           + c_attn * kappa_r * (computed_r + kappa_r / 2) ]

     where decode(B) = {r : computed_r >= prompt_len_r}, prefill(B) is the
     complement, B_dec = |decode(B)|, and kappa_r is the token count the
     scheduler granted r this step. This is evaluated by the frozen
     calibration.modeb.analysis.predict_step, which this module imports rather
     than reimplements.

(E2) Chunk count. With prefix caching off the uncached suffix is the whole
     prompt, so the number of prefill chunks is deterministic,

         n_c = ceil(prompt_len / kappa),      kappa = meta["max_num_batched_tokens"].

(E3) Own-prefill work. Chunk k (k = 0 .. n_c-1) carries t_k tokens against an
     already-computed causal prefix P_k,

         t_k = min(kappa, prompt_len - k * kappa),      P_k = k * kappa,

         W_p = sum_{k=0}^{n_c-1} [ c_pf * t_k + c_attn * t_k * (P_k + t_k / 2) ].

     The c_base and decode terms of (E1) are excluded here on purpose. They are
     charged once per iteration by the n_c * T_iter term of (E4), not per chunk.

(E4) Composed first-token estimate, the quantity the router scores,

         TTFT_hat = T_adm + n_c * T_iter(B_arrival) + W_p,

     where B_arrival is the batch resident at the request's enqueue instant.
     Two variants share the same compute term n_c * T_iter + W_p:

         oracle      T_adm := realized (E5), isolating the closed-form terms;
         deployable  T_adm := the roll-forward estimator the router runs online,
                     replayed by calibration.admission.analysis.replay.

(E5) Realized quantities. t_arrive is the earliest EngineCore-side arrival
     timestamp, t_sched is t_start of the first step in which the request
     appears, and t_first is t_start of the first step in which computed_r >=
     prompt_len_r (the step at which its prefill has completed and the first
     token exists). Then

         T_adm_realized = t_sched - t_arrive,
         prefill_realized = t_first - t_sched,
         TTFT_realized  = t_first - t_arrive.

     Schema-v2 captures stamp t_engine_arrive in
     EngineCore.preprocess_add_request, before the input queue, and retain the
     later Scheduler.add_request t_enq for decomposition. Schema-v1 captures
     lack the upstream timestamp and therefore fall back to t_enq.

Process restarts
----------------
The gpu-reaper scales the deployment to 0 after an idle period, restarting the
vLLM process. Capture files are opened in append mode so no rows are lost, but a
restart resets the perf_counter epoch, the step counter, and the request-id
namespace. Timestamps and step numbers are therefore only comparable within one
process lifetime, so this driver splits both input files at every reset boundary
and treats each segment independently. Segment count is discovered, not assumed.

Usage
-----
    python -m calibration.admission.ttft_driver \
        --trajectory /mnt/pvc/admission-v2/trajectory.jsonl \
        --events     /mnt/pvc/admission-v2/admission_events.jsonl \
        --meta       /mnt/pvc/admission-v2/meta.json \
        --coeffs     coeffs.json \
        --out-rows   figures/ttft_rows.json \
        --out-dir    calibration/admission

Peak memory is one segment of the trajectory. Pass --segment N to process a
single segment when the whole capture does not fit.
"""
import argparse
import bisect
import gzip
import json
import math
import os
import sys
import types


# ---------------------------------------------------------------------------
# Imports of the frozen analysis code.
#
# calibration.admission.analysis transitively imports calibration.modeb.analysis,
# which imports matplotlib at module scope. The analysis pod runs python:3.11-slim
# with no matplotlib and no root to pip install one. Rather than stub the analysis
# module itself (which would risk diverging from the frozen predict_step), insert a
# minimal matplotlib placeholder so the real frozen module imports unchanged. Only
# modeb.analysis.write_report touches matplotlib, and this driver never calls it.
# ---------------------------------------------------------------------------
def ensure_matplotlib_importable():
    """Insert a placeholder matplotlib into sys.modules if the real one is absent."""
    try:
        import matplotlib  # noqa: F401
        return False
    except ImportError:
        pass
    mpl = types.ModuleType("matplotlib")
    mpl.use = lambda *a, **k: None
    pyplot = types.ModuleType("matplotlib.pyplot")

    def _unavailable(*a, **k):
        raise RuntimeError("matplotlib is not installed; plotting is unavailable "
                           "in this environment")

    pyplot.__getattr__ = lambda name: _unavailable
    mpl.pyplot = pyplot
    sys.modules["matplotlib"] = mpl
    sys.modules["matplotlib.pyplot"] = pyplot
    return True


# ---------------------------------------------------------------------------
# Closed forms (E2) and (E3).
# ---------------------------------------------------------------------------
def n_chunks(prompt_len, kappa):
    """(E2) Number of prefill chunks for a prompt, prefix caching off."""
    if prompt_len <= 0:
        raise ValueError(f"prompt_len must be positive, got {prompt_len!r}")
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa!r}")
    return int(math.ceil(prompt_len / kappa))


def chunk_tokens(prompt_len, kappa):
    """(E3) Per-chunk token counts t_k, k = 0 .. n_c-1. Sums to prompt_len."""
    return [min(kappa, prompt_len - k * kappa)
            for k in range(n_chunks(prompt_len, kappa))]


def prefill_work(prompt_len, kappa, coeffs):
    """(E3) W_p, the request's own prefill compute over its chunk sequence."""
    total = 0.0
    for k, t_k in enumerate(chunk_tokens(prompt_len, kappa)):
        p_k = k * kappa
        total += coeffs["c_pf"] * t_k + coeffs["c_attn"] * t_k * (p_k + t_k / 2.0)
    return total


# ---------------------------------------------------------------------------
# Input reading and process-restart segmentation.
# ---------------------------------------------------------------------------
def _open_text(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def read_jsonl(path):
    """Stream a JSONL file, yielding one parsed object per non-blank line."""
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def split_on_reset(records, time_key, step_key=None, backward_tolerance=1.0):
    """Split append-ordered records into per-process-lifetime segments.

    A restart is detected when the monotonic clock jumps backward by more than
    backward_tolerance seconds, or (when step_key is given) when the counter fails
    to increase. Records must be in the order they were appended, NOT sorted, since
    the reset is only visible in append order.

    Returns a list of lists, always at least one segment (possibly empty).
    """
    segments = [[]]
    prev_t = None
    prev_step = None
    for rec in records:
        t = rec[time_key]
        reset = False
        if prev_t is not None and t < prev_t - backward_tolerance:
            reset = True
        if step_key is not None and prev_step is not None and rec[step_key] <= prev_step:
            reset = True
        if reset:
            segments.append([])
            prev_t = None
            prev_step = None
        segments[-1].append(rec)
        prev_t = t
        prev_step = rec[step_key] if step_key is not None else None
    return segments


def add_step_deltas(steps):
    """Sort a segment ascending by step and attach t_iter / t_end_next.

    Matches calibration.admission.analysis.parse_admission_trajectory exactly,
    including dropping the final step, which has no successor to difference
    against. Operates on an in-memory segment rather than a whole file.
    """
    steps = sorted(steps, key=lambda s: s["step"])
    out = []
    for i in range(len(steps) - 1):
        s = dict(steps[i])
        s["t_iter"] = steps[i + 1]["t_start"] - s["t_start"]
        s["t_end_next"] = steps[i + 1]["t_start"]
        out.append(s)
    return out


def events_by_id(event_records):
    """Collapse enqueue-event records to {req_id: event}, last write wins.

    Same semantics as calibration.admission.analysis.parse_enqueue_events, applied
    to an already-segmented record list.
    """
    out = {}
    for record in event_records:
        event = dict(record)
        req_id = event.pop("req_id")
        out[req_id] = event
    return out


def event_time(event):
    """Earliest comparable engine-side arrival timestamp in an event row."""
    return event.get("t_engine_arrive", event["t_enq"])


# ---------------------------------------------------------------------------
# Enqueue bucketing.
# ---------------------------------------------------------------------------
def bisect_enqueue_bucket(steps, enq_events):
    """Bracket each request's arrival into the step whose [t_start, t_end_next)
    contains it.

    Semantically identical to calibration.admission.analysis.enqueue_bucket, and
    returns the same step objects so replay's identity-keyed index still works.
    That reference implementation scans every step per event, which is O(E*S) and
    does not finish on a capture of ~200k steps and ~20k events. This binary
    search over the (ascending, non-overlapping) t_start values is O(E log S).
    tests/test_ttft_driver.py asserts the two agree.

    Raises ValueError if t_start is not ascending. The frozen linear scan tolerates
    an unordered step list and this binary search cannot, so the precondition is
    checked rather than assumed. add_step_deltas sorts by step number, matching the
    frozen parser, and within one process lifetime perf_counter is monotonic and
    steps run sequentially, so ordering by step does order by t_start. A capture
    that violates that has a problem worth failing on rather than bisecting over.

    Returns (dict[req_id -> step], dropped_count).
    """
    starts = [s["t_start"] for s in steps]
    for i in range(1, len(starts)):
        if starts[i] < starts[i - 1]:
            raise ValueError(
                f"t_start is not ascending at index {i} "
                f"(step {steps[i]['step']}, t_start {starts[i]!r} follows "
                f"step {steps[i - 1]['step']}, t_start {starts[i - 1]!r}). "
                f"Binary-search bucketing needs an ordered segment. Check that "
                f"split_on_reset separated the process lifetimes correctly.")
    bucket = {}
    dropped = 0
    for req_id, enq in enq_events.items():
        arrival = event_time(enq)
        i = bisect.bisect_right(starts, arrival) - 1
        if i < 0 or arrival >= steps[i]["t_end_next"]:
            dropped += 1
        else:
            bucket[req_id] = steps[i]
    return bucket, dropped


# ---------------------------------------------------------------------------
# Realized first-token reconstruction (E5).
# ---------------------------------------------------------------------------
def first_token_times(steps):
    """One pass over a segment, returning {req_id: t_first} per (E5).

    t_first is t_start of the earliest step in which the request's computed count
    has reached its prompt length. A request observed as already decoding on its
    very first captured step is skipped: the capture started mid-flight, so its
    prefill was never observed and its first-token instant is unknown. Those are
    reported as skipped_already_decoding rather than silently dated to capture start.
    """
    first_seen = {}
    first_token = {}
    skipped_already_decoding = set()
    for step in steps:
        t_start = step["t_start"]
        for r in step["reqs"]:
            rid = r["id"]
            done = r["computed"] >= r["prompt_len"]
            if rid not in first_seen:
                first_seen[rid] = True
                if done:
                    skipped_already_decoding.add(rid)
                    continue
            if rid in skipped_already_decoding:
                continue
            if done and rid not in first_token:
                first_token[rid] = t_start
    return first_token, skipped_already_decoding


# ---------------------------------------------------------------------------
# Composition (E4).
# ---------------------------------------------------------------------------
def compose_segment(steps, enq_events, meta, coeffs, estimator_name, seg_index,
                    censored_out=None):
    """Reconstruct realized and predicted first-token time for one segment.

    Returns (rows, diag). Each row carries the fields figures/plot_ttft_full.py
    consumes, plus the intermediate terms so the composition can be audited
    without re-running the driver:

        seg, req_id, nc, prompt_len,
        r_ttft   realized TTFT              (E5)
        r_tadm   realized admission delay   (E5)
        r_prefill realized prefill          (E5)
        compute  n_c * T_iter + W_p         (E4) compute term
        t_adm_deploy  deployable admission estimate
        p_oracle = r_tadm + compute         (E4) oracle variant
        p_deploy = t_adm_deploy + compute   (E4) deployable variant

    A row is emitted only where every term is defined, so both parity series
    cover the same request set. That requires the request to be enqueued,
    bracketed into a step, actually scheduled, and to have reached its first
    token. A request whose later departure is beyond capture end remains
    scoreable: its admission and first-token times are already observed, and
    the deployable replay does not use its eventual output length.
    """
    from calibration.admission import analysis as A

    kappa = meta["max_num_batched_tokens"]
    steps = add_step_deltas(steps)
    if not steps:
        return [], {"segment": seg_index, "n_steps": 0, "note": "empty segment"}

    traces = A.request_traces(steps)
    bucket, dropped_unbracketed = bisect_enqueue_bucket(steps, enq_events)
    first_token, skipped_already_decoding = first_token_times(steps)

    # Deployable admission estimates come from the frozen replay, which owns the
    # censoring rule and the time-ordered RunningMean of realized output lengths.
    # Swap in the bisect bucket for the duration of the call; the reference
    # implementation is O(E*S) and will not finish on a full capture.
    orig_bucket = A.enqueue_bucket
    A.enqueue_bucket = bisect_enqueue_bucket
    try:
        replay_rows, never_scheduled = A.replay(
            steps, enq_events, traces, meta, coeffs,
            estimator_name=estimator_name, use_oracle=False,
            load_of=lambda rid: "all", include_censored=True)
    finally:
        A.enqueue_bucket = orig_bucket
    t_adm_deploy = {r["req_id"]: r["predicted"] for r in replay_rows
                    if not r.get("right_censored")}
    t_first_deploy = {r["req_id"]: r["predicted_first_token"]
                      for r in replay_rows
                      if not r.get("right_censored")
                      and r.get("predicted_first_token") is not None}
    censored_predictions = {r["req_id"]: r["predicted"] for r in replay_rows
                            if r.get("right_censored")}
    censored_first_token = {
        r["req_id"]: r["predicted_first_token"] for r in replay_rows
        if r.get("right_censored")
        and r.get("predicted_first_token") is not None}

    # Requests that never entered a scheduled batch have no realized TTFT.  Do
    # not silently discard them or pretend the capture-end bound is an exact
    # observation: emit one explicitly right-censored row per request.
    capture_end = steps[-1]["t_end_next"]
    censored_rows = []
    for req_id in bucket:
        if req_id in traces:
            continue
        arrival = event_time(enq_events[req_id])
        lower_bound = max(capture_end - arrival, 0.0)
        prompt_len = int(enq_events[req_id]["prompt_len"])
        arrival_batch = [r for r in bucket[req_id]["reqs"]
                         if r["id"] != req_id]
        compute = (n_chunks(prompt_len, kappa)
                   * A.predict_step(arrival_batch, coeffs)
                   + prefill_work(prompt_len, kappa, coeffs))
        t_adm_predicted = censored_predictions.get(req_id)
        p_deploy_composed = (t_adm_predicted + compute
                             if t_adm_predicted is not None else None)
        p_deploy = censored_first_token.get(req_id, p_deploy_composed)
        censored_rows.append({
            "seg": seg_index,
            "req_id": req_id,
            "arrival": arrival,
            "capture_end": capture_end,
            "r_tadm_lower_bound": lower_bound,
            "r_ttft_lower_bound": lower_bound,
            "t_adm_deploy": t_adm_predicted,
            "compute": compute,
            "p_deploy": p_deploy,
            "p_deploy_composed": p_deploy_composed,
            "p_deploy_rollout": censored_first_token.get(req_id),
            "prediction_below_lower_bound": (
                p_deploy < lower_bound if p_deploy is not None else None),
            "known_underprediction_lower_bound": (
                max(lower_bound - p_deploy, 0.0)
                if p_deploy is not None else None),
            "censoring": "right",
            "reason": "enqueued_but_never_scheduled",
        })
    if censored_out is not None:
        censored_out.extend(censored_rows)

    rows = []
    missing_first_token = 0
    self_in_arrival_batch = 0
    for req_id, dep in t_adm_deploy.items():
        t_first = first_token.get(req_id)
        if t_first is None:
            missing_first_token += 1
            continue
        enq = enq_events[req_id]
        prompt_len = enq["prompt_len"]
        arrival = event_time(enq)
        t_sched = traces[req_id]["t_sched"]

        n_c = n_chunks(prompt_len, kappa)
        # T_iter is evaluated on the batch RESIDENT at arrival, which excludes the
        # arriving request. W_p already charges that request's own prefill work, so
        # letting it appear here too would double-count its first chunk. The bracket
        # is closed at its lower edge, so a request whose t_enq coincides with the
        # t_start of the very step that admits it would land in its own batch.
        # Filtering by id enforces the invariant whatever the timestamps do.
        arrival_batch = [r for r in bucket[req_id]["reqs"] if r["id"] != req_id]
        if len(arrival_batch) != len(bucket[req_id]["reqs"]):
            self_in_arrival_batch += 1
        t_iter = A.predict_step(arrival_batch, coeffs)
        compute = n_c * t_iter + prefill_work(prompt_len, kappa, coeffs)
        p_deploy_composed = dep + compute
        p_deploy_rollout = t_first_deploy.get(req_id)

        rows.append({
            "seg": seg_index,
            "req_id": req_id,
            "nc": n_c,
            "prompt_len": prompt_len,
            "r_ttft": t_first - arrival,
            "r_tadm": t_sched - arrival,
            "r_prefill": t_first - t_sched,
            "compute": compute,
            "t_adm_deploy": dep,
            "p_oracle": (t_sched - arrival) + compute,
            "p_deploy": (p_deploy_rollout
                         if p_deploy_rollout is not None
                         else p_deploy_composed),
            "p_deploy_composed": p_deploy_composed,
            "p_deploy_rollout": p_deploy_rollout,
            "t_engine_arrive": enq.get("t_engine_arrive"),
            "t_enq": enq["t_enq"],
            "engine_input_wait": enq.get(
                "engine_input_wait",
                max(enq["t_enq"] - arrival, 0.0)
                if "t_engine_arrive" in enq else None),
        })

    # Requests the deployable replay declined to score, itemized so the row count
    # reconciles against the enqueue-event count. Scheduled requests that are
    # still running at capture end are deliberately replayed above: departure is
    # irrelevant once admission and first token have both been observed.
    censored_excluded = sum(1 for rid in bucket
                            if rid in traces and traces[rid]["censored"]
                            and rid not in t_adm_deploy)
    censored_bounds = sorted(r["r_tadm_lower_bound"] for r in censored_rows)
    predicted_censored = [r for r in censored_rows if r["p_deploy"] is not None]
    violations = sum(r["prediction_below_lower_bound"] for r in predicted_censored)
    known_shortfalls = sorted(
        r["known_underprediction_lower_bound"] for r in predicted_censored)

    rows.sort(key=lambda r: r["r_ttft"])
    diag = {
        "segment": seg_index,
        "n_steps": len(steps),
        "n_enqueue_events": len(enq_events),
        "n_rows": len(rows),
        "kappa": kappa,
        "estimator": estimator_name,
        "dropped_unbracketed": dropped_unbracketed,
        "never_scheduled": never_scheduled,
        "right_censored": {
            "n": len(censored_rows),
            "lower_bound_s_p50": _percentile(censored_bounds, 50),
            "lower_bound_s_p90": _percentile(censored_bounds, 90),
            "prediction_n": len(predicted_censored),
            "prediction_below_lower_bound_n": violations,
            "prediction_below_lower_bound_frac": (
                violations / len(predicted_censored)
                if predicted_censored else None),
            "known_underprediction_s_p50": _percentile(known_shortfalls, 50),
            "known_underprediction_s_p90": _percentile(known_shortfalls, 90),
            "note": ("lower bounds are excluded from MAPE; a prediction below "
                     "its bound is a proven underprediction"),
        },
        "censored_excluded": censored_excluded,
        "skipped_already_decoding": len(skipped_already_decoding),
        "skipped_no_first_token": missing_first_token,
        "self_in_arrival_batch": self_in_arrival_batch,
    }
    return rows, diag


# ---------------------------------------------------------------------------
# Summary views, matching the shape of the committed ttft_seg*.json pins.
# ---------------------------------------------------------------------------
def _percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    i = min(int(q / 100.0 * len(sorted_vals)), len(sorted_vals) - 1)
    return sorted_vals[i]


def view_stats(pairs):
    """Summarize (realized, predicted) pairs.

    median_ratio is realized / predicted, so a value below 1 means the estimate
    runs high. bias_pct is (median_ratio - 1) * 100. mape_pct is the mean of
    |predicted - realized| / realized. Rows with a non-positive realized value
    cannot be scored as a relative error and are excluded from every statistic.
    """
    pairs = [(r, p) for r, p in pairs if r > 0]
    if not pairs:
        return {"n": 0, "median_ratio": None, "bias_pct": None, "mape_pct": None,
                "over_pred_frac": None, "realized_ms_p50": None,
                "realized_ms_p90": None, "pred_ms_p50": None,
                "mae_ms": None, "median_ape_pct": None,
                "p90_ape_pct": None, "wape_pct": None}
    ratios = sorted(r / p for r, p in pairs if p > 0)
    med = _percentile(ratios, 50)
    mape = sum(abs(p - r) / r for r, p in pairs) / len(pairs) * 100.0
    abs_errors = sorted(abs(p - r) for r, p in pairs)
    apes = sorted(abs(p - r) / r * 100.0 for r, p in pairs)
    wape = sum(abs_errors) / sum(r for r, _ in pairs) * 100.0
    over = sum(1 for r, p in pairs if p > r) / len(pairs)
    realized = sorted(r for r, _ in pairs)
    predicted = sorted(p for _, p in pairs)
    return {
        "n": len(pairs),
        "median_ratio": round(med, 4) if med is not None else None,
        "bias_pct": round((med - 1) * 100.0, 1) if med is not None else None,
        "mape_pct": round(mape, 1),
        "over_pred_frac": round(over, 3),
        "realized_ms_p50": round(_percentile(realized, 50) * 1000.0, 3),
        "realized_ms_p90": round(_percentile(realized, 90) * 1000.0, 2),
        "pred_ms_p50": round(_percentile(predicted, 50) * 1000.0, 3),
        "mae_ms": round(sum(abs_errors) / len(abs_errors) * 1000.0, 3),
        "median_ape_pct": round(_percentile(apes, 50), 1),
        "p90_ape_pct": round(_percentile(apes, 90), 1),
        "wape_pct": round(wape, 1),
    }


def summarize(rows, diag, queued_threshold=0.5):
    """Build the per-segment report. Views mirror the committed pins.

    prefill_only scores the compute term against realized prefill, excluding the
    admission delay entirely. The ttft_* views score full composed TTFT (E4).
    The queued split is by realized admission delay, since realized magnitude is
    the axis that separates the free-slot mode from the genuinely queued tail.
    """
    nq = [r for r in rows if r["r_tadm"] <= queued_threshold]
    qd = [r for r in rows if r["r_tadm"] > queued_threshold]
    hist = {}
    for r in rows:
        hist[str(r["nc"])] = hist.get(str(r["nc"]), 0) + 1

    def view(sub, realized_key, predicted_key):
        return view_stats([(r[realized_key], r[predicted_key]) for r in sub])

    return {
        "kappa": diag["kappa"],
        "estimator": diag["estimator"],
        "n_rows": len(rows),
        "n_c_histogram": dict(sorted(hist.items(), key=lambda kv: int(kv[0]))),
        "diagnostics": diag,
        "right_censored": diag.get("right_censored", {
            "n": 0, "lower_bound_s_p50": None, "lower_bound_s_p90": None,
            "note": "lower bounds are excluded from MAPE",
        }),
        "prefill_only": view(rows, "r_prefill", "compute"),
        "prefill_only.not_queued": view(nq, "r_prefill", "compute"),
        "prefill_only.queued_gt500ms": view(qd, "r_prefill", "compute"),
        "ttft_oracle_tadm": view(rows, "r_ttft", "p_oracle"),
        "ttft_deployable": view(rows, "r_ttft", "p_deploy"),
        "ttft_deployable.not_queued": view(nq, "r_ttft", "p_deploy"),
        "ttft_deployable.queued_gt500ms": view(qd, "r_ttft", "p_deploy"),
    }


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trajectory", required=True, help="trajectory.jsonl[.gz]")
    ap.add_argument("--events", required=True, help="admission_events.jsonl[.gz]")
    ap.add_argument("--meta", required=True, help="meta.json from the capture")
    ap.add_argument("--coeffs", required=True, help="frozen coeffs.json")
    ap.add_argument("--estimator", default="rollforward",
                    choices=["token_rollforward", "rollforward", "fluid"],
                    help="deployable estimator to replay (default rollforward)")
    ap.add_argument("--out-rows", required=True,
                    help="pooled per-request rows, consumed by plot_ttft_full.py")
    ap.add_argument("--out-dir", required=True, help="directory for ttft_seg*.json")
    ap.add_argument("--out-censored", default=None,
                    help="pooled right-censored rows (default OUT_DIR/ttft_censored.json)")
    ap.add_argument("--segment", type=int, default=None,
                    help="process only this 1-based segment (lower peak memory)")
    ap.add_argument("--slim-rows", action="store_true",
                    help="emit only the fields the figure needs, for a smaller file")
    a = ap.parse_args(argv)

    if ensure_matplotlib_importable():
        print("note: matplotlib absent, inserted a placeholder so the frozen "
              "analysis modules import; no plots will be written")

    meta = json.load(open(a.meta))
    coeffs = json.load(open(a.coeffs))

    traj_segments = split_on_reset(read_jsonl(a.trajectory), "t_start", step_key="step")
    evt_segments = split_on_reset(read_jsonl(a.events), "t_enq")
    print(f"trajectory segments: {[len(s) for s in traj_segments]}")
    print(f"event segments:      {[len(s) for s in evt_segments]}")
    if len(traj_segments) != len(evt_segments):
        print(f"WARNING: {len(traj_segments)} trajectory segments against "
              f"{len(evt_segments)} event segments. Pairing by index; inspect the "
              f"counts above before trusting the output.")

    os.makedirs(a.out_dir, exist_ok=True)
    all_rows = []
    all_censored = []
    n_seg = min(len(traj_segments), len(evt_segments))
    for i in range(n_seg):
        seg_index = i + 1
        if a.segment is not None and seg_index != a.segment:
            continue
        rows, diag = compose_segment(traj_segments[i], events_by_id(evt_segments[i]),
                                     meta, coeffs, a.estimator, seg_index,
                                     censored_out=all_censored)
        report = summarize(rows, diag)
        path = os.path.join(a.out_dir, f"ttft_seg{seg_index}.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"seg{seg_index}: n={len(rows)} "
              f"oracle MAPE={report['ttft_oracle_tadm']['mape_pct']}% "
              f"deployable MAPE={report['ttft_deployable']['mape_pct']}% -> {path}")
        all_rows.extend(rows)

    if a.slim_rows:
        keep = ("seg", "nc", "r_ttft", "p_oracle", "p_deploy", "r_tadm")
        out_rows = [{k: r[k] for k in keep} for r in all_rows]
    else:
        out_rows = all_rows
    os.makedirs(os.path.dirname(os.path.abspath(a.out_rows)), exist_ok=True)
    with open(a.out_rows, "w") as f:
        json.dump(out_rows, f)
    print(f"wrote {len(out_rows)} pooled rows -> {a.out_rows}")
    censored_path = a.out_censored or os.path.join(a.out_dir, "ttft_censored.json")
    os.makedirs(os.path.dirname(os.path.abspath(censored_path)), exist_ok=True)
    with open(censored_path, "w") as f:
        json.dump(all_censored, f)
    print(f"wrote {len(all_censored)} right-censored rows -> {censored_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
