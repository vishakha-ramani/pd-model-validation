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

__all__ = [
    "load_meta", "parse_admission_trajectory", "parse_enqueue_events",
    "request_traces", "build_context", "block_accounting_diag",
    "predict_step", "load_coeffs",
    "RunningMean", "deployable_rem_steps_est", "enqueue_bucket", "replay",
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
            events[row["req_id"]] = {"t_enq": row["t_enq"], "prompt_len": row["prompt_len"]}
    return events


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


def build_context(step, step_idx, req_id, enq, meta, coeffs, traces, use_oracle, nout_est):
    """Assemble the AdmissionContext dict for req_id evaluated at `step`."""
    block_size = _block_size(meta)
    reqs = step["reqs"]

    # batch_size = TOTAL occupied sequence slots (every scheduled request,
    # prefill or decode, holds a slot) -- not a decode-only count.
    batch_size = len(reqs)
    max_batch_size = meta["max_num_seqs"]

    free_kv_blocks = step.get("free_kv_blocks")
    ctx_extra = {}
    if free_kv_blocks is None:
        free_kv_blocks = meta["num_gpu_blocks"] - sum(
            math.ceil(r["computed"] / block_size) for r in reqs)
        ctx_extra["free_kv_reconstructed"] = True

    req_kv_need = math.ceil(enq["prompt_len"] / block_size)

    waiting_ids = step.get("waiting_ids") or []
    if req_id in waiting_ids:
        queue_depth = waiting_ids.index(req_id)
    else:
        # Truncated snapshot, or a request whose t_enq falls after this
        # step's waiting snapshot was taken: either way its true position
        # is unobserved, so fall back to the step's observed waiting_count
        # as a back-of-queue proxy (replaces the old front-of-queue 0).
        queue_depth = step["waiting_count"]
        ctx_extra["queue_pos_from_count"] = True

    t_iter = predict_step(reqs, coeffs)

    running = []
    for r in reqs:
        tr = traces[r["id"]]
        steps_done = step_idx - tr["first_step_idx"]
        kv_blocks = math.ceil(r["computed"] / block_size)
        true_remaining = (tr["last_step_idx"] - step_idx) if use_oracle else -1
        running.append({"steps_done": steps_done, "kv_blocks": kv_blocks,
                         "true_remaining": true_remaining})

    ctx = {
        "batch_size": batch_size,
        "max_batch_size": max_batch_size,
        "free_kv_blocks": free_kv_blocks,
        "req_kv_need": req_kv_need,
        "t_iter": t_iter,
        "queue_depth": queue_depth,
        "remaining_steps_est": nout_est,
        "running": running,
    }
    ctx.update(ctx_extra)
    return ctx


def block_accounting_diag(step, meta):
    """Compare the captured free-KV-block count against the value
    reconstructed from per-request `computed` tokens, for drift diagnosis."""
    block_size = _block_size(meta)
    captured_free = step.get("free_kv_blocks")
    reconstructed_free = meta["num_gpu_blocks"] - sum(
        math.ceil(r["computed"] / block_size) for r in step["reqs"])
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
        t_enq = enq["t_enq"]
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


def replay(steps, enq_events, traces, meta, coeffs, estimator_name, use_oracle, load_of):
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
    Requests that ARE in `traces` but run past capture end (`censored`)
    are still excluded from `rows` as before, but are NOT counted as
    never-scheduled (they were scheduled; only their departure is
    unobserved).
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
            continue
        if not traces[rid]["censored"]:
            req_ids.append(rid)
    req_ids.sort(key=lambda rid: enq_events[rid]["t_enq"])

    running_mean = RunningMean()
    comp_i = 0
    rows = []
    for req_id in req_ids:
        enq = enq_events[req_id]
        step = bucket[req_id]
        step_idx = idx_of[id(step)]
        t_enq = enq["t_enq"]
        reqs = step["reqs"]

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
            nout_est = deployable_rem_steps_est(step, running_mean.value())

        ctx = build_context(step, step_idx, req_id, enq, meta, coeffs, traces,
                             use_oracle, nout_est)
        predicted = estimator(ctx)
        realized = traces[req_id]["t_sched"] - t_enq
        ratio = (realized / predicted) if predicted > 0 else None
        rows.append({
            "req_id": req_id,
            "predicted": predicted,
            "realized": realized,
            "ratio": ratio,
            "load_bin": load_of(req_id),
            "regime_at_enq": _regime_at_enq(reqs),
        })
    return rows, never_scheduled


def _load_section(load_bin):
    return "overload" if "overload" in str(load_bin) else "sub_capacity"


def admission_report(rows_by_key, diag_rows=None):
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
            mape = (sum(errs) / len(errs) * 100.0) if errs else None
            bias_pct = ((median_ratio - 1) * 100.0) if median_ratio is not None else None
            section = _load_section(load_bin)
            section_dict = report[section]
            section_dict.setdefault(estimator, {}).setdefault(variant, {})[load_bin] = {
                "n": n, "median_ratio": median_ratio, "mape": mape, "bias_pct": bias_pct,
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
