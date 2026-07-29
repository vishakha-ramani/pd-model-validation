"""Mode-B teacher-forced analysis: feed the REAL captured per-step batch
composition into the calibrated per-iteration latency law and score
predicted-vs-measured T_iter per regime.

Handles two capture realities the raw inter-step delta must be cleaned of:
  1. probe warmup rows (step < PROBE_ROWS) are excluded.
  2. inter-arrival IDLE gaps: at low sweep rates the engine sits idle between
     requests, so t_start[i+1]-t_start[i] includes wait time the compute-law
     never models. We isolate genuine back-to-back (compute-bound) iterations
     via gap_after = t_start[i+1]-t_end[i]; a small gap => the next step began
     as this one ended => the inter-step delta is a real iteration time.

Run: python3 analyze_modeb.py           (reads ./trajectory.jsonl.gz)
"""
import gzip, json, os, sys, math

HERE = os.path.dirname(os.path.abspath(__file__))
COEFFS = json.load(open(os.path.join(HERE, "..", "..", "..", "coeffs.json")))
TRAJ = os.path.join(HERE, "trajectory.jsonl.gz")
PROBE_ROWS = 250   # rows 0..249 = warmup sanity probe, excluded

# archetype row ranges (from the sequential-driver wc -l checkpoints)
ARCHETYPES = [
    ("decode-corner", 250, 78400),
    ("balanced",      78400, 114050),
    ("prefill-lean",  114050, 142500),
    ("prefill-bound", 142500, 151600),
    ("conversation",  151600, 191800),
]


def is_prefill(r):
    return r["computed"] < r["prompt_len"]


def predict_step(reqs, c):
    B_dec = 0; resident = 0.0; pf = 0.0
    for r in reqs:
        if is_prefill(r):
            k = r["kappa"]; pk = r["computed"]
            pf += c["c_pf"] * k + c["c_attn"] * k * (pk + k / 2.0)
        else:
            B_dec += 1; resident += r["computed"]
    return c["c_base"] + c["c_dec"] * B_dec + c["c_kv"] * resident + pf


def regime_of(reqs):
    hp = any(is_prefill(r) for r in reqs)
    hd = any(not is_prefill(r) for r in reqs)
    if hp and hd: return "mixed"
    if hp: return "pure_prefill"
    return "pure_decode"


def load():
    rows = []
    with gzip.open(TRAJ, "rt") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda s: s["step"])
    return rows


def build(rows):
    """Attach t_iter (inter-step delta), compute (in-step), gap_after, regime."""
    out = []
    for i in range(len(rows) - 1):
        s = rows[i]; nxt = rows[i + 1]
        rec = {
            "step": s["step"],
            "reqs": s["reqs"],
            "t_iter": nxt["t_start"] - s["t_start"],
            "compute": s["t_end"] - s["t_start"],
            "gap_after": nxt["t_start"] - s["t_end"],
            "regime": regime_of(s["reqs"]),
            "n_reqs": len(s["reqs"]),
        }
        out.append(rec)
    return out


def pctl(xs, p):
    if not xs: return None
    xs = sorted(xs); k = (len(xs) - 1) * p
    lo = int(math.floor(k)); hi = int(math.ceil(k))
    if lo == hi: return xs[lo]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def stats(recs, measured_key="t_iter"):
    if not recs:
        return {"n": 0}
    resid_pct = []
    me = []; pr = []
    for r in recs:
        p = predict_step(r["reqs"], COEFFS)
        m = r[measured_key]
        pr.append(p); me.append(m)
        resid_pct.append((p - m) / m * 100.0)
    mean_me = sum(me) / len(me)
    ss_res = sum((me[j] - pr[j]) ** 2 for j in range(len(me)))
    ss_tot = sum((v - mean_me) ** 2 for v in me)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        "n": len(recs),
        "mape": sum(abs(v) for v in resid_pct) / len(resid_pct),
        "bias_pct": sum(resid_pct) / len(resid_pct),
        "r2": r2,
        "median_meas_ms": pctl(me, 0.5) * 1000,
    }


def main():
    rows = load()
    print(f"loaded {len(rows)} rows")
    recs = build(rows)
    # exclude probe + empty steps
    recs = [r for r in recs if r["step"] >= PROBE_ROWS and r["n_reqs"] > 0]
    print(f"post probe+empty filter: {len(recs)}")

    gaps = [r["gap_after"] for r in recs]
    print("\n=== gap_after (idle between steps) distribution, ms ===")
    for p in (0.5, 0.9, 0.95, 0.99):
        print(f"  p{int(p*100)}: {pctl(gaps, p)*1000:.2f}")
    print(f"  max: {max(gaps)*1000:.1f}")
    for thr in (0.001, 0.002, 0.005, 0.01):
        frac = sum(1 for g in gaps if g <= thr) / len(gaps)
        print(f"  frac gap<= {thr*1000:.0f}ms: {frac*100:.1f}%")

    GAP = 0.002  # 2ms: next step began essentially as this one ended -> back-to-back
    bb = [r for r in recs if r["gap_after"] <= GAP]
    print(f"\n=== back-to-back subset (gap_after <= {GAP*1000:.0f}ms): {len(bb)} steps ===")

    print("\n--- PER REGIME (measured = inter-step t_iter, back-to-back only) ---")
    for name in ("pure_decode", "pure_prefill", "mixed", "all"):
        sub = bb if name == "all" else [r for r in bb if r["regime"] == name]
        st = stats(sub)
        if st["n"]:
            print(f"  {name:13s} n={st['n']:7d}  MAPE={st['mape']:6.2f}%  "
                  f"bias={st['bias_pct']:+6.2f}%  R2={st['r2']:.4f}  "
                  f"med={st['median_meas_ms']:.2f}ms")
        else:
            print(f"  {name:13s} n=0")

    print("\n--- cross-check: measured = in-step compute (t_end-t_start), back-to-back ---")
    for name in ("pure_decode", "pure_prefill", "mixed", "all"):
        sub = bb if name == "all" else [r for r in bb if r["regime"] == name]
        st = stats(sub, measured_key="compute")
        if st["n"]:
            print(f"  {name:13s} n={st['n']:7d}  MAPE={st['mape']:6.2f}%  "
                  f"bias={st['bias_pct']:+6.2f}%  R2={st['r2']:.4f}")

    print("\n--- MIXED residual vs max prefill P_k (the mislabeling test) ---")
    mixed = [r for r in bb if r["regime"] == "mixed"]
    def maxpk(r): return max([q["computed"] for q in r["reqs"] if is_prefill(q)] or [0])
    bins = [(0, 1), (1, 2048), (2048, 4096), (4096, 8192), (8192, 16000)]
    for lo, hi in bins:
        sub = [r for r in mixed if lo <= maxpk(r) < hi]
        st = stats(sub)
        if st["n"]:
            print(f"  P_k[{lo:5d},{hi:5d}) n={st['n']:6d}  MAPE={st['mape']:6.2f}%  bias={st['bias_pct']:+6.2f}%")

    print("\n--- PURE_PREFILL residual vs P_k (multi-chunk) ---")
    pp = [r for r in bb if r["regime"] == "pure_prefill"]
    for lo, hi in bins:
        sub = [r for r in pp if maxpk(r) >= lo and maxpk(r) < hi]
        st = stats(sub)
        if st["n"]:
            print(f"  P_k[{lo:5d},{hi:5d}) n={st['n']:6d}  MAPE={st['mape']:6.2f}%  bias={st['bias_pct']:+6.2f}%")

    print("\n--- PER ARCHETYPE (back-to-back, all regimes) ---")
    per_arch = {}
    for name, lo, hi in ARCHETYPES:
        sub = [r for r in bb if lo <= r["step"] < hi]
        st = stats(sub)
        per_arch[name] = st
        if st["n"]:
            print(f"  {name:14s} n={st['n']:7d}  MAPE={st['mape']:6.2f}%  bias={st['bias_pct']:+6.2f}%  R2={st['r2']:.4f}")

    # persist summary
    summary = {
        "coeffs": COEFFS,
        "n_rows_total": len(rows),
        "n_after_probe_empty": len(recs),
        "gap_threshold_ms": GAP * 1000,
        "n_back_to_back": len(bb),
        "per_regime": {name: stats(bb if name == "all" else [r for r in bb if r["regime"] == name])
                       for name in ("pure_decode", "pure_prefill", "mixed", "all")},
        "per_archetype": per_arch,
    }
    json.dump(summary, open(os.path.join(HERE, "modeb_report.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(HERE, 'modeb_report.json')}")

    # scatter plot
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        colors = {"pure_decode": "tab:blue", "pure_prefill": "tab:green", "mixed": "tab:red"}
        plt.figure(figsize=(6, 6))
        for name in ("pure_decode", "pure_prefill", "mixed"):
            sub = [r for r in bb if r["regime"] == name]
            if not sub: continue
            xs = [r["t_iter"] * 1000 for r in sub]
            ys = [predict_step(r["reqs"], COEFFS) * 1000 for r in sub]
            plt.scatter(xs, ys, s=4, alpha=0.3, label=f"{name} (n={len(sub)})", color=colors[name])
        allm = [r["t_iter"] * 1000 for r in bb]
        lo, hi = min(allm), max(allm)
        plt.plot([lo, hi], [lo, hi], "k--", lw=1)
        plt.xlabel("measured T_iter (ms)"); plt.ylabel("predicted T_iter (ms)")
        plt.legend(); plt.title("Mode-B teacher-forced: predicted vs measured")
        plt.savefig(os.path.join(HERE, "modeb_pred_vs_meas.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"wrote {os.path.join(HERE, 'modeb_pred_vs_meas.png')}")
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
