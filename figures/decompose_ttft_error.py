"""Attribute the composed TTFT error to a term, and bound the probe-placement cost.

The deployable first-token estimate carries a much larger error than the oracle
variant. Both share the compute term of (E4), so the difference is entirely the
admission term, and this script separates the two contributions and quantifies
each.

Every quantity comes from figures/ttft_rows.json, so this runs offline with no
cluster access. The separation is exact rather than fitted. From (E4),

    p_oracle = r_tadm     + n_c * T_iter + W_p
    p_deploy = t_adm_est  + n_c * T_iter + W_p

so subtracting recovers both unknowns without touching the capture,

    compute   = p_oracle - r_tadm            the closed-form compute term
    t_adm_est = p_deploy - compute           the deployable admission estimate

Three things get reported.

  1. A counterfactual table that swaps only the admission term and leaves the
     compute term untouched. It shows how much of the deployable error is the
     admission estimate rather than the model of prefill.

  2. The realized admission delay's distribution on the un-queued subset. This
     is the evidence for the probe-placement problem. t_enq is stamped in a
     Scheduler.add_request patch, and vLLM V1 drains its EngineCore input queue
     and then calls schedule(), so the measured delay begins after the wait for
     the in-flight iteration. A request arriving at a random instant inside a
     ~20 ms iteration would show a delay spread over that whole window. Observing
     nearly all of them under 0.1 ms instead means the clock starts at the step
     boundary, so the wait is real and unmeasured.

  3. A bound on what a correctly placed probe would buy, by adding the missing
     wait back to the measured TTFT and re-scoring the same predictions. The
     correction is a model of the missing quantity rather than a measurement, so
     it brackets the answer instead of settling it. Only a re-run with the probe
     moved upstream measures it.

Run from the repository root:

    python figures/decompose_ttft_error.py
"""
import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS_PATH = os.path.join(HERE, "ttft_rows.json")

# Realized admission delay above which a request counts as genuinely queued.
# Realized magnitude is the axis that separates the free-slot mode from the tail.
QUEUED_S = 0.5


def load_rows(path):
    """Read the row file and make the decomposition explicit on every row.

    The driver emits `compute` and `t_adm_deploy` directly. The slim row file the
    figure consumes omits them, so derive them from the identity above when absent.
    """
    rows = json.load(open(path))
    for r in rows:
        if "compute" not in r:
            r["compute"] = r["p_oracle"] - r["r_tadm"]
        if "t_adm_deploy" not in r:
            r["t_adm_deploy"] = r["p_deploy"] - r["compute"]
    return rows


def mape(pairs):
    pairs = [(r, p) for r, p in pairs if r > 0]
    return 100.0 * sum(abs(p - r) / r for r, p in pairs) / len(pairs)


def median_bias_pct(pairs):
    """Median of predicted/realized, minus one. Positive means the estimate runs high."""
    ratios = sorted(p / r for r, p in pairs if r > 0 and p > 0)
    return 100.0 * (statistics.median(ratios) - 1.0)


def percentile(sorted_vals, q):
    i = min(int(q / 100.0 * len(sorted_vals)), len(sorted_vals) - 1)
    return sorted_vals[i]


def main():
    if not os.path.exists(ROWS_PATH):
        print(f"missing {ROWS_PATH}", file=sys.stderr)
        return 1
    rows = load_rows(ROWS_PATH)
    nq = [r for r in rows if r["r_tadm"] <= QUEUED_S]
    qd = [r for r in rows if r["r_tadm"] > QUEUED_S]
    print(f"rows {len(rows)}   un-queued {len(nq)}   queued>{QUEUED_S * 1000:.0f}ms {len(qd)}")

    report = {"n_rows": len(rows), "n_not_queued": len(nq), "n_queued": len(qd),
              "queued_threshold_s": QUEUED_S}

    # 1. Swap only the admission term.
    print("\n1. Counterfactuals. Only the admission term of (E4) changes.\n")
    variants = [
        ("deployable as shipped", lambda r: r["p_deploy"]),
        ("admission term -> 0", lambda r: r["compute"]),
        ("half the estimate", lambda r: r["compute"] + 0.5 * r["t_adm_deploy"]),
        ("oracle (realized)", lambda r: r["p_oracle"]),
    ]
    print(f"  {'variant':<24}{'all':>9}{'un-queued':>12}{'queued':>9}"
          f"{'nq bias':>10}")
    report["counterfactuals"] = {}
    for name, f in variants:
        a = mape([(r["r_ttft"], f(r)) for r in rows])
        n = mape([(r["r_ttft"], f(r)) for r in nq])
        q = mape([(r["r_ttft"], f(r)) for r in qd])
        b = median_bias_pct([(r["r_ttft"], f(r)) for r in nq])
        print(f"  {name:<24}{a:>8.1f}%{n:>11.1f}%{q:>8.1f}%{b:>9.1f}%")
        report["counterfactuals"][name] = {
            "mape_all_pct": round(a, 1), "mape_not_queued_pct": round(n, 1),
            "mape_queued_pct": round(q, 1), "median_bias_not_queued_pct": round(b, 1)}

    oracle_nq = report["counterfactuals"]["oracle (realized)"]["mape_not_queued_pct"]
    zeroed_nq = report["counterfactuals"]["admission term -> 0"]["mape_not_queued_pct"]
    print(f"\n  Zeroing the admission term reaches {zeroed_nq}% on the un-queued "
          f"subset against\n  the oracle's {oracle_nq}%. The compute term already "
          f"carries oracle fidelity there,\n  so the deployable gap on that subset "
          f"is the admission estimate alone.")
    report["compute_term_matches_oracle_on_bulk"] = (
        abs(zeroed_nq - oracle_nq) < 0.15)

    # 2. Evidence for the probe-placement problem.
    print("\n2. Realized admission delay on the un-queued subset.\n")
    vals = sorted(r["r_tadm"] for r in nq)
    dist = {}
    for q in (1, 25, 50, 75, 90, 99, 99.9):
        dist[f"p{q}_ms"] = round(percentile(vals, q) * 1000.0, 4)
        print(f"  p{q:<5} {percentile(vals, q) * 1000:9.4f} ms")
    under = sum(1 for v in vals if v < 0.0001) / len(vals)
    t_iter_proxy = statistics.median(r["t_adm_deploy"] for r in nq)
    print(f"\n  fraction under 0.1 ms      {under:.4f}")
    print(f"  one-iteration floor (p50)  {t_iter_proxy * 1000:.2f} ms")
    print(f"\n  A uniform arrival inside a {t_iter_proxy * 1000:.0f} ms iteration would "
          f"sit near {t_iter_proxy * 500:.0f} ms.\n  The observed median is "
          f"{percentile(vals, 50) * 1000:.4f} ms, so the clock starts at the step boundary\n"
          f"  and the wait for the in-flight iteration is real and unmeasured.")
    report["realized_tadm_not_queued"] = dist
    report["realized_tadm_not_queued"]["frac_under_0.1ms"] = round(under, 4)
    report["one_iteration_floor_p50_ms"] = round(t_iter_proxy * 1000.0, 2)

    # 3. Bound the cost of the probe placement.
    print("\n3. Adding the missing wait back, then re-scoring the same predictions.\n")
    rng = random.Random(0)
    corrections = [
        ("none (as measured)", lambda r: 0.0),
        ("+ T_iter/2 (mean phase)", lambda r: 0.5 * r["t_adm_deploy"]),
        ("+ U[0,T_iter] (seed 0)", lambda r: rng.random() * r["t_adm_deploy"]),
        ("+ T_iter (upper bound)", lambda r: r["t_adm_deploy"]),
    ]
    print(f"  {'correction':<26}{'un-queued MAPE':>16}{'bias':>9}")
    report["probe_correction_bound"] = {}
    for name, corr in corrections:
        pairs = [(r["r_ttft"] + corr(r), r["p_deploy"]) for r in nq]
        m, b = mape(pairs), median_bias_pct(pairs)
        print(f"  {name:<26}{m:>15.1f}%{b:>8.1f}%")
        report["probe_correction_bound"][name] = {
            "mape_not_queued_pct": round(m, 1), "median_bias_pct": round(b, 1)}

    print("\n  The queued tail is unchanged by any of this. It stays near "
          f"{report['counterfactuals']['deployable as shipped']['mape_queued_pct']}% "
          "and zeroing\n  the admission term makes it slightly worse, so that error is "
          "the estimator's\n  blindness to work already committed ahead of the request "
          "rather than a\n  measurement artifact.")

    out = os.path.join(HERE, "ttft_error_decomposition.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
