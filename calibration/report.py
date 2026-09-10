import argparse
import csv
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calibration.fit import fit_decode, fit_prefill, predict_mixed, mape


def load_csv(path):
    with open(path) as f:
        r = csv.reader(f)
        next(r)  # header
        return [tuple(float(x) for x in row) for row in r]


def load_csv_inputs(paths):
    """Load one CSV path or concatenate several compatible CSV shards."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    return [row for path in paths for row in load_csv(path)]


def run_report(decode_csv, prefill_csv, mixed_csv, chunk_bud, out_dir):
    dec = load_csv_inputs(decode_csv)
    pre = load_csv_inputs(prefill_csv)
    mix = load_csv_inputs(mixed_csv)

    d = fit_decode(dec)
    p = fit_prefill(pre, c_base=d["c_base"], chunk_bud=chunk_bud)
    coeffs = {"c_base": d["c_base"], "c_pf": p["c_pf"], "c_attn": p["c_attn"],
              "c_dec": d["c_dec"], "c_kv": d["c_kv"]}

    obs = [row[4] for row in mix]
    pred = [predict_mixed(row[:4], coeffs) for row in mix]
    mixed_mape = mape(pred, obs)

    summary = {"coeffs": coeffs, "decode_mape": d["mape"], "decode_r2": d["r2"],
               "prefill_mape": p["mape"], "prefill_r2": p["r2"], "mixed_mape": mixed_mape}

    os.makedirs(out_dir, exist_ok=True)
    json.dump(coeffs, open(os.path.join(out_dir, "coeffs.json"), "w"), indent=2)
    residuals = {"mixed_abs_pct": [abs(a - b) / abs(b) * 100 for a, b in zip(pred, obs)],
                 "mixed_mape": mixed_mape,
                 "decode_mape": d["mape"], "prefill_mape": p["mape"]}
    json.dump(residuals, open(os.path.join(out_dir, "residuals.json"), "w"), indent=2)

    plt.figure()
    plt.scatter(obs, pred, s=8)
    lo, hi = min(obs), max(obs)
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    plt.xlabel("realized step time (s)")
    plt.ylabel("predicted step time (s)")
    plt.title(f"mixed-batch: MAPE {mixed_mape:.2f}%")
    plt.savefig(os.path.join(out_dir, "predicted_vs_realized.png"), dpi=150, bbox_inches="tight")
    plt.close()
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode", action="append", required=True)
    ap.add_argument("--prefill", action="append", required=True)
    ap.add_argument("--mixed", action="append", required=True)
    ap.add_argument("--chunk-bud", type=int, default=8192)
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    s = run_report(a.decode, a.prefill, a.mixed, a.chunk_bud, a.out_dir)
    print(json.dumps(s, indent=2))
