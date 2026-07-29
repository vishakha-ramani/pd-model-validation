"""Offline Mode-B analysis: derive the law's variables from raw captured fields,
classify each step's regime, and score predicted-vs-measured per regime."""
import json


def load_coeffs(path):
    with open(path) as f:
        return json.load(f)


def _is_prefill(r):
    return r["computed"] < r["prompt_len"]


def predict_step(reqs, coeffs, resident_convention="computed"):
    B_dec = 0
    resident = 0.0
    prefill_terms = 0.0
    for r in reqs:
        if _is_prefill(r):
            k = r["kappa"]
            pk = r["computed"]
            prefill_terms += coeffs["c_pf"] * k + coeffs["c_attn"] * k * (pk + k / 2.0)
        else:
            B_dec += 1
            ctx = r["computed"] + (1 if resident_convention == "computed_plus_1" else 0)
            resident += ctx
    return coeffs["c_base"] + coeffs["c_dec"] * B_dec + coeffs["c_kv"] * resident + prefill_terms


def classify_regime(reqs):
    has_prefill = any(_is_prefill(r) for r in reqs)
    has_decode = any(not _is_prefill(r) for r in reqs)
    if has_prefill and has_decode:
        return "mixed"
    if has_prefill:
        return "pure_prefill"
    return "pure_decode"


def parse_trajectory(jsonl_path):
    with open(jsonl_path) as f:
        steps = [json.loads(line) for line in f if line.strip()]
    steps.sort(key=lambda s: s["step"])
    out = []
    for i in range(len(steps) - 1):          # drop final step (no successor for the delta)
        s = steps[i]
        s = dict(s)
        s["t_iter"] = steps[i + 1]["t_start"] - s["t_start"]
        s["regime"] = classify_regime(s["reqs"])
        out.append(s)
    return out


import os
import json as _json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _n_prefill_chunks(reqs):
    return sum(1 for r in reqs if _is_prefill(r))


def _max_prefill_pk(reqs):
    pks = [r["computed"] for r in reqs if _is_prefill(r)]
    return max(pks) if pks else 0


def _b_dec(reqs):
    return sum(1 for r in reqs if not _is_prefill(r))


def per_regime_report(steps, coeffs, resident_convention="computed"):
    predicted, measured, regime = [], [], []
    for s in steps:
        predicted.append(predict_step(s["reqs"], coeffs, resident_convention))
        measured.append(s["t_iter"])
        regime.append(s["regime"])

    def _stats(idx):
        if not idx:
            return {"n": 0, "mape": None, "r2": None, "bias_pct": None,
                    "resid_pct": [], "p_k": [], "b_dec": [], "n_prefill_chunks": []}
        pr = [predicted[i] for i in idx]
        me = [measured[i] for i in idx]
        resid_pct = [(pr[j] - me[j]) / me[j] * 100.0 for j in range(len(idx))]
        mean_me = sum(me) / len(me)
        ss_res = sum((me[j] - pr[j]) ** 2 for j in range(len(idx)))
        ss_tot = sum((v - mean_me) ** 2 for v in me)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        mape = sum(abs(v) for v in resid_pct) / len(resid_pct)
        bias = sum(resid_pct) / len(resid_pct)
        return {"n": len(idx), "mape": mape, "r2": r2, "bias_pct": bias,
                "resid_pct": resid_pct,
                "p_k": [_max_prefill_pk(steps[i]["reqs"]) for i in idx],
                "b_dec": [_b_dec(steps[i]["reqs"]) for i in idx],
                "n_prefill_chunks": [_n_prefill_chunks(steps[i]["reqs"]) for i in idx]}

    by_regime = {}
    for name in ("pure_decode", "pure_prefill", "mixed"):
        by_regime[name] = _stats([i for i, r in enumerate(regime) if r == name])
    by_regime["all"] = _stats(list(range(len(steps))))
    return {"by_regime": by_regime, "predicted": predicted,
            "measured": measured, "regime": regime}


def write_report(steps, coeffs, out_dir, resident_convention="computed"):
    rep = per_regime_report(steps, coeffs, resident_convention)
    os.makedirs(out_dir, exist_ok=True)
    # strip long arrays out of the persisted summary; keep resid arrays for plots only
    summary = {"by_regime": {k: {kk: vv for kk, vv in v.items()
                                 if kk not in ("resid_pct", "p_k", "b_dec", "n_prefill_chunks")}
                             for k, v in rep["by_regime"].items()}}
    _json.dump(summary, open(os.path.join(out_dir, "modeb_report.json"), "w"), indent=2)

    colors = {"pure_decode": "tab:blue", "pure_prefill": "tab:green", "mixed": "tab:red"}
    plt.figure()
    for name in ("pure_decode", "pure_prefill", "mixed"):
        xs = [rep["measured"][i] for i, r in enumerate(rep["regime"]) if r == name]
        ys = [rep["predicted"][i] for i, r in enumerate(rep["regime"]) if r == name]
        if xs:
            plt.scatter(xs, ys, s=6, alpha=0.5, label=name, color=colors[name])
    allm = rep["measured"]
    if allm:
        lo, hi = min(allm), max(allm)
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel("measured T_iter (s)")
    plt.ylabel("predicted T_iter (s)")
    plt.legend()
    plt.title("Mode-B: predicted vs measured per regime")
    plt.savefig(os.path.join(out_dir, "modeb_pred_vs_meas.png"), dpi=150, bbox_inches="tight")
    plt.close()
    return rep


import csv as _csv


def guidellm_curves(benchmarks_csv):
    with open(benchmarks_csv) as f:
        reader = _csv.DictReader(f)
        rows = []
        for row in reader:
            rate = row.get("rate", "")
            rows.append({
                "strat": row["strat"],
                "rate": float(rate) if rate not in ("", None) else None,
                "ttft_mean_ms": float(row["ttft_mean"]),
                "itl_mean_ms": float(row["itl_mean"]),
                "concurrency": float(row["conc"]),
            })
        return rows
