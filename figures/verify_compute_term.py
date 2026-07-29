"""Offline check of the driver's prefill-compute reconstruction on real engine data.

The full TTFT driver needs the admission capture, which carries per-request
enqueue events and lives only on the cluster PVC. The Mode-B capture vendored in
this repo has no enqueue events, so the admission term cannot be reconstructed
from it. Everything downstream of admission can be, and that is what this script
checks, on real captured batches rather than a synthetic fixture.

For every request whose prefill was observed from its first chunk, it compares
the realized prefill duration against two predictions:

  summed law    sum of (E1) T_iter over the request's own prefill steps, using
                the real captured batch composition at each step. Realized
                prefill is exactly the wall-clock length of that same step
                interval, so this isolates the per-iteration law composed over a
                multi-step window.

  closed form   n_c * T_iter(B_resident) + W_p, per (E2), (E3) and the compute
                term of (E4), where B_resident is the batch in the step
                immediately before the request was first scheduled. That is the
                closest available stand-in for the batch resident at arrival, and
                it deliberately excludes the request itself, since W_p accounts
                for the request's own prefill work.

Realized prefill is t_first - t_sched per (E5).

Run from the repository root:

    python figures/verify_compute_term.py
"""
import gzip
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from calibration.admission.ttft_driver import n_chunks, prefill_work  # noqa: E402
from calibration.modeb.analysis import predict_step  # noqa: E402

TRAJ = os.path.join(ROOT, "calibration", "results", "modeb", "trajectory.jsonl.gz")
META = os.path.join(ROOT, "calibration", "results", "modeb", "meta.json")
COEFFS = os.path.join(ROOT, "coeffs.json")


def summarize(name, pairs):
    """pairs are (realized, predicted). Conventions match ttft_driver.view_stats:
    median_ratio is realized/predicted, so below 1 means the estimate runs high."""
    pairs = [(r, p) for r, p in pairs if r > 0 and p > 0]
    ratios = sorted(r / p for r, p in pairs)
    med = statistics.median(ratios)
    mape = sum(abs(p - r) / r for r, p in pairs) / len(pairs) * 100.0
    over = sum(1 for r, p in pairs if p > r) / len(pairs)
    print(f"  {name:<14} n={len(pairs):<6} median_ratio={med:.4f}  "
          f"bias={100 * (med - 1):+.1f}%  MAPE={mape:.1f}%  over_pred={over:.3f}")
    return {"n": len(pairs), "median_ratio": round(med, 4),
            "bias_pct": round(100 * (med - 1), 1), "mape_pct": round(mape, 1),
            "over_pred_frac": round(over, 3)}


def main():
    meta = json.load(open(META))
    coeffs = json.load(open(COEFFS))
    kappa = meta["max_num_batched_tokens"]
    print(f"kappa = {kappa}  (max_num_batched_tokens, prefix caching off)")

    # Streaming single pass. Per request we need: the step index and t_start where
    # it was first scheduled, its prompt length, the resident batch just before it
    # started, the running sum of predicted iteration times across its prefill
    # steps, and the t_start of the step at which its prefill had completed.
    started = {}          # req_id -> {t_sched, prompt_len, t_iter_resident, acc}
    first_token = {}      # req_id -> t_start of first step with prefill complete
    skipped_midflight = 0
    prev_t_iter_pred = None
    n_steps = 0

    with gzip.open(TRAJ, "rt") as f:
        pending = None    # previous step, held back until we know its successor
        for line in f:
            line = line.strip()
            if not line:
                continue
            step = json.loads(line)
            n_steps += 1
            if pending is not None:
                # The authoritative measured iteration time is the inter-step
                # delta of t_start, so a step is only processable once the next
                # one has been read. Predicted time needs no successor.
                _process(pending, started, first_token, coeffs, prev_t_iter_pred)
                prev_t_iter_pred = pending["_pred"]
            pending = step
            pending["_pred"] = predict_step(step["reqs"], coeffs)
        # The final step is dropped, matching add_step_deltas and the frozen parser.

    print(f"steps read = {n_steps}")

    rows = []
    for rid, info in started.items():
        t_first = first_token.get(rid)
        if t_first is None:
            continue
        if info["t_iter_resident"] is None:
            skipped_midflight += 1
            continue
        plen = info["prompt_len"]
        n_c = n_chunks(plen, kappa)
        w_p = prefill_work(plen, kappa, coeffs)
        rows.append({
            "nc": n_c,
            "prompt_len": plen,
            "realized": t_first - info["t_sched"],
            "summed": info["acc"],
            "closed": n_c * info["t_iter_resident"] + w_p,
        })

    if not rows:
        print("no requests with a fully observed prefill; nothing to check")
        return 1

    hist = {}
    for r in rows:
        hist[r["nc"]] = hist.get(r["nc"], 0) + 1
    print(f"requests with fully observed prefill = {len(rows)}")
    print(f"skipped (no resident batch before first schedule) = {skipped_midflight}")
    print(f"n_c histogram = {dict(sorted(hist.items()))}")

    report = {"kappa": kappa, "n_steps": n_steps, "n_requests": len(rows),
              "n_c_histogram": {str(k): v for k, v in sorted(hist.items())}}

    print("\nrealized prefill (t_first - t_sched) against:")
    report["summed_law"] = summarize("summed law", [(r["realized"], r["summed"]) for r in rows])
    report["closed_form"] = summarize("closed form", [(r["realized"], r["closed"]) for r in rows])

    for n_c in sorted(hist):
        sub = [r for r in rows if r["nc"] == n_c]
        if len(sub) < 20:
            continue
        print(f"\n  n_c = {n_c} ({len(sub)} requests)")
        report[f"summed_law.nc{n_c}"] = summarize(
            "summed law", [(r["realized"], r["summed"]) for r in sub])
        report[f"closed_form.nc{n_c}"] = summarize(
            "closed form", [(r["realized"], r["closed"]) for r in sub])

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "compute_term_verification.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")
    return 0


def _process(step, started, first_token, coeffs, prev_t_iter_pred):
    """Fold one step into the per-request accumulators."""
    t_start = step["t_start"]
    t_iter_pred = step["_pred"]
    for r in step["reqs"]:
        rid = r["id"]
        prefilling = r["computed"] < r["prompt_len"]
        if rid not in started and rid not in first_token:
            if not prefilling or r["computed"] != 0:
                # Either already decoding, or first seen mid-prefill: its prefill
                # start was never captured, so its realized prefill is unknown.
                first_token[rid] = None
                continue
            started[rid] = {"t_sched": t_start, "prompt_len": r["prompt_len"],
                            "t_iter_resident": prev_t_iter_pred, "acc": 0.0}
        info = started.get(rid)
        if info is None:
            continue
        if prefilling:
            info["acc"] += t_iter_pred
        elif rid not in first_token:
            first_token[rid] = t_start


if __name__ == "__main__":
    sys.exit(main())
