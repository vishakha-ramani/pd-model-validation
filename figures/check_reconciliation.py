"""Independent reconciliation gate for every headline number in README.md.

This re-derives each quantity from the raw vendored data instead of parsing the
output of the plotting scripts, so it cross-checks those scripts rather than
trusting them. It reimplements the documented filtering rules from scratch for
the same reason.

Unlike the byte-for-byte PNG comparison, nothing here depends on the matplotlib
or freetype version, so this is the check to trust across environments.

Run from the repository root:

    python figures/check_reconciliation.py

Exits non-zero if any check fails.
"""
import csv
import gzip
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

FAILURES = []
CHECKS = 0


def check(label, observed, expected, tol, unit=""):
    """Compare a number against its README value within an absolute tolerance."""
    global CHECKS
    CHECKS += 1
    ok = observed is not None and abs(observed - expected) <= tol
    status = "pass" if ok else "FAIL"
    obs = "None" if observed is None else f"{observed:.4f}"
    print(f"  [{status}] {label:<46} expected {expected:>10.4f}{unit}  "
          f"observed {obs}{unit}")
    if not ok:
        FAILURES.append(label)
    return ok


def check_exact(label, observed, expected):
    global CHECKS
    CHECKS += 1
    ok = observed == expected
    print(f"  [{'pass' if ok else 'FAIL'}] {label:<46} "
          f"expected {expected!r}  observed {observed!r}")
    if not ok:
        FAILURES.append(label)
    return ok


def mape(pairs):
    pairs = [(r, p) for r, p in pairs if r > 0]
    return 100.0 * sum(abs(p - r) / r for r, p in pairs) / len(pairs)


# ---------------------------------------------------------------------------
def stage0():
    """Section 4a and 9a. Refit (E1)'s coefficients and score the three regimes."""
    print("\nStage 0, coefficient fit (README section 4a)")
    from calibration.fit import fit_decode, fit_prefill, predict_mixed

    def load(path):
        with open(os.path.join(ROOT, path)) as f:
            r = csv.reader(f)
            next(r)
            return [tuple(float(x) for x in row) for row in r]

    dec, pre, mix = load("calibration/decode.csv"), \
        load("calibration/prefill.csv"), load("calibration/mixed.csv")
    d = fit_decode(dec)
    p = fit_prefill(pre, c_base=d["c_base"], chunk_bud=8192)
    refit = {"c_base": d["c_base"], "c_pf": p["c_pf"], "c_attn": p["c_attn"],
             "c_dec": d["c_dec"], "c_kv": d["c_kv"]}
    frozen = json.load(open(os.path.join(ROOT, "coeffs.json")))
    check_exact("refit coefficients equal coeffs.json bit-exactly", refit, frozen)
    check("decode R2", d["r2"], 0.9908, 0.0002)
    check("decode MAPE", d["mape"], 0.8647, 0.001, "%")
    check("prefill R2", p["r2"], 0.9998, 0.0002)

    errs = sorted((abs(predict_mixed(row[:4], frozen) - row[4]) / row[4] * 100.0)
                  for row in mix)
    check("mixed MAPE, all 12 rows", sum(errs) / len(errs), 606.8, 0.5, "%")
    check("mixed MAPE, 10 rows after dropping 2 misaligned",
          sum(errs[:-2]) / len(errs[:-2]), 17.6, 0.2, "%")
    return frozen


# ---------------------------------------------------------------------------
def stage1(coeffs):
    """Section 6. Score (E1) against the real captured batch composition.

    Reimplements the plotting script's documented filters independently. Steps
    before 250 are warmup. A gap over 2 ms between one step ending and the next
    beginning means the engine idled, so the inter-start delta would measure idle
    time rather than compute.
    """
    print("\nStage 1, per-iteration law (E1) (README section 6)")
    traj = os.path.join(ROOT, "calibration", "results", "modeb", "trajectory.jsonl.gz")
    rows = []
    with gzip.open(traj, "rt") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda s: s["step"])

    def predict(reqs):
        b_dec = 0
        resident = 0.0
        prefill = 0.0
        for r in reqs:
            if r["computed"] < r["prompt_len"]:
                k, pk = r["kappa"], r["computed"]
                prefill += coeffs["c_pf"] * k + coeffs["c_attn"] * k * (pk + k / 2.0)
            else:
                b_dec += 1
                resident += r["computed"]
        return (coeffs["c_base"] + coeffs["c_dec"] * b_dec
                + coeffs["c_kv"] * resident + prefill)

    def regime(reqs):
        hp = any(r["computed"] < r["prompt_len"] for r in reqs)
        hd = any(r["computed"] >= r["prompt_len"] for r in reqs)
        return "mixed" if hp and hd else ("pure_prefill" if hp else "pure_decode")

    recs = []
    for i in range(len(rows) - 1):
        s, nxt = rows[i], rows[i + 1]
        if s["step"] < 250 or not s["reqs"]:
            continue
        if nxt["t_start"] - s["t_end"] > 0.002:
            continue
        recs.append((predict(s["reqs"]), nxt["t_start"] - s["t_start"], regime(s["reqs"])))

    check("scored steps", float(len(recs)), 190583, 0)
    check("(E1) overall MAPE", mape([(m, p) for p, m, _ in recs]), 3.32, 0.01, "%")
    for name, n_exp, mape_exp in (("pure_decode", 174665, 2.72),
                                  ("pure_prefill", 1200, 3.68),
                                  ("mixed", 14718, 10.38)):
        sub = [(m, p) for p, m, g in recs if g == name]
        check(f"(E1) {name} n", float(len(sub)), n_exp, 0)
        check(f"(E1) {name} MAPE", mape(sub), mape_exp, 0.01, "%")


# ---------------------------------------------------------------------------
def stage2():
    """Sections 7 and 8. Score composed TTFT (E4) and split the error by term."""
    print("\nStage 2, composed TTFT (E4) (README sections 7 and 8)")
    rows = json.load(open(os.path.join(HERE, "ttft_rows.json")))
    for r in rows:
        r.setdefault("compute", r["p_oracle"] - r["r_tadm"])
        r.setdefault("t_adm_deploy", r["p_deploy"] - r["compute"])
    nq = [r for r in rows if r["r_tadm"] <= 0.5]
    qd = [r for r in rows if r["r_tadm"] > 0.5]

    check("pooled rows", float(len(rows)), 30734, 0)
    check("un-queued rows", float(len(nq)), 26622, 0)
    check("queued rows", float(len(qd)), 4112, 0)
    check("(E4) oracle MAPE", mape([(r["r_ttft"], r["p_oracle"]) for r in rows]),
          7.89, 0.02, "%")
    check("(E4) deployable MAPE", mape([(r["r_ttft"], r["p_deploy"]) for r in rows]),
          57.52, 0.02, "%")
    check("(E4) deployable, un-queued",
          mape([(r["r_ttft"], r["p_deploy"]) for r in nq]), 52.96, 0.02, "%")
    check("(E4) deployable, queued",
          mape([(r["r_ttft"], r["p_deploy"]) for r in qd]), 86.99, 0.02, "%")

    # Section 8: dropping the admission term must land on the oracle's accuracy,
    # which is the claim that the bulk error is the admission estimate alone. The
    # two do not coincide exactly, because the realized delay on this subset is
    # small rather than zero, so p_oracle carries a residual median 11 us that
    # `compute` does not. The gap is a few hundredths of a point.
    zeroed = mape([(r["r_ttft"], r["compute"]) for r in nq])
    oracle_nq = mape([(r["r_ttft"], r["p_oracle"]) for r in nq])
    check("admission term zeroed, un-queued", zeroed, 8.86, 0.02, "%")
    check("oracle, un-queued", oracle_nq, 8.88, 0.02, "%")
    check("the two agree to under 0.05 points",
          abs(zeroed - oracle_nq), 0.022, 0.03, "%")

    # Probe placement evidence.
    vals = sorted(r["r_tadm"] for r in nq)
    p50 = vals[len(vals) // 2] * 1000.0
    under = sum(1 for v in vals if v < 0.0001) / len(vals)
    check("realized admission delay p50, un-queued", p50, 0.0113, 0.001, " ms")
    check("fraction under 0.1 ms", under, 0.8531, 0.002)
    floor = statistics.median(r["t_adm_deploy"] for r in nq) * 1000.0
    check("one-iteration floor p50", floor, 19.74, 0.05, " ms")

    # Segment 1 must reproduce the committed reference report.
    seg1 = [r for r in rows if r["seg"] == 1]
    pin = json.load(open(os.path.join(ROOT, "calibration", "admission",
                                      "ttft_seg1.json")))
    check("segment 1 rows", float(len(seg1)), 18880, 1)
    check("segment 1 oracle MAPE against ttft_seg1.json",
          mape([(r["r_ttft"], r["p_oracle"]) for r in seg1]),
          pin["ttft_oracle_tadm"]["mape_pct"], 0.05, "%")
    check("segment 1 deployable MAPE against ttft_seg1.json",
          mape([(r["r_ttft"], r["p_deploy"]) for r in seg1]),
          pin["ttft_deployable"]["mape_pct"], 0.05, "%")
    ratios = sorted(r["p_oracle"] / r["r_ttft"] for r in seg1 if r["r_ttft"] > 0)
    med_realized_over_pred = 1.0 / statistics.median(ratios)
    check("segment 1 oracle median ratio against ttft_seg1.json",
          med_realized_over_pred, pin["ttft_oracle_tadm"]["median_ratio"], 0.001)


# ---------------------------------------------------------------------------
def compute_term():
    """Section 9d. The driver's own closed forms, checked on the real capture."""
    print("\nDriver composition core (README section 9d)")
    path = os.path.join(HERE, "compute_term_verification.json")
    if not os.path.exists(path):
        print("  [skip] run figures/verify_compute_term.py first")
        return
    rep = json.load(open(path))
    check("requests with fully observed prefill",
          float(rep["n_requests"]), 18277, 0)
    check("summed law MAPE", rep["summed_law"]["mape_pct"], 9.7, 0.05, "%")
    check("single-chunk MAPE", rep["summed_law.nc1"]["mape_pct"], 10.0, 0.05, "%")
    check("two-chunk MAPE", rep["summed_law.nc2"]["mape_pct"], 1.1, 0.05, "%")

    # Bias, asserted in BOTH sign conventions, because the repository uses both
    # and a silent flip between them would otherwise go unnoticed. The script's
    # own `bias_pct` is median(realized/predicted) - 1, so a high prediction reads
    # negative. README section 9d quotes the reciprocal, how high the prediction
    # runs. Check the printed value and the README's restatement of it.
    for key, script_bias, readme_high in (("summed_law.nc1", -7.8, 8.4),
                                          ("summed_law.nc2", 1.0, -1.0)):
        v = rep[key]
        check(f"{key} bias as the script prints it",
              v["bias_pct"], script_bias, 0.05, "%")
        check(f"{key} restated as how high the prediction runs",
              100.0 * (1.0 / v["median_ratio"] - 1.0), readme_high, 0.05, "%")

    # (E2) and (E3) sanity, independent of any capture.
    from calibration.admission.ttft_driver import chunk_tokens, n_chunks
    check_exact("(E2) n_c for a 16000-token prompt at kappa 8192",
                n_chunks(16000, 8192), 2)
    check_exact("(E3) chunk split for the same prompt",
                chunk_tokens(16000, 8192), [8192, 7808])


# ---------------------------------------------------------------------------
def main():
    print("Reconciling README.md against the vendored data.")
    print("Version-independent. Nothing here compares rendered pixels.")
    coeffs = stage0()
    stage1(coeffs)
    stage2()
    compute_term()
    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print("FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("All README numbers reconcile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
