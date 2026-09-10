import argparse
import csv
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TRANSFER_RE = re.compile(
    r"Avg xfer time \(ms\)=([0-9.]+).*"
    r"Avg MB per transfer=([0-9.]+).*"
    r"Throughput \(MB/s\)=([0-9.]+)"
)


def parse_transfer_metrics(lines):
    rows = []
    for line in lines:
        match = TRANSFER_RE.search(line)
        if not match:
            continue
        xfer_ms, size_mib, throughput_mib_s = map(float, match.groups())
        rows.append({
            "xfer_ms": xfer_ms,
            "size_mib": size_mib,
            "throughput_mib_s": throughput_mib_s,
        })
    return rows


def fit_size_aware_transfer(rows):
    if len(rows) < 2:
        raise ValueError("at least two transfer points are required")
    size_bytes = np.asarray([row["size_mib"] * 2**20 for row in rows], dtype=float)
    observed_s = np.asarray([row["xfer_ms"] / 1000.0 for row in rows], dtype=float)
    design = np.column_stack([np.ones_like(size_bytes), size_bytes])
    beta, _, _, _ = np.linalg.lstsq(design, observed_s, rcond=None)
    predicted_s = design @ beta
    base_s, seconds_per_byte = map(float, beta)
    if seconds_per_byte <= 0:
        raise ValueError("fitted transfer slope must be positive")
    errors = np.abs(predicted_s - observed_s) / observed_s * 100.0
    residual = float(np.sum((observed_s - predicted_s) ** 2))
    total = float(np.sum((observed_s - np.mean(observed_s)) ** 2))
    return {
        "xfer_base_us": base_s * 1e6,
        "xfer_bandwidth_decimal_gbps": 1.0 / seconds_per_byte / 1e9,
        "xfer_bandwidth_gibps": 1.0 / seconds_per_byte / 2**30,
        "r2": 1.0 - residual / total if total > 0 else 1.0,
        "mape_pct": float(np.mean(errors)),
        "p95_abs_pct_error": float(np.percentile(errors, 95)),
        "max_abs_pct_error": float(np.max(errors)),
        "predicted_ms": list(predicted_s * 1000.0),
    }


def run_report(log_path, skip_first, out_dir):
    with open(log_path, errors="replace") as stream:
        all_rows = parse_transfer_metrics(stream)
    if skip_first < 0 or skip_first >= len(all_rows):
        raise ValueError("skip_first must leave at least one transfer point")
    included = all_rows[skip_first:]
    fit = fit_size_aware_transfer(included)
    summary = {
        key: value for key, value in fit.items() if key != "predicted_ms"
    }
    summary.update({
        "all_points": len(all_rows),
        "excluded_leading_points": skip_first,
        "fit_points": len(included),
        "size_unit_in_vllm_log": "MiB (despite MB label)",
        "bandwidth_unit_for_policy": "decimal GB/s",
    })

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "transfer-points.csv"), "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "index", "included", "size_mib", "xfer_ms", "reported_mib_s",
            "predicted_ms",
        ])
        prediction_by_index = {
            index + skip_first: predicted
            for index, predicted in enumerate(fit["predicted_ms"])
        }
        for index, row in enumerate(all_rows):
            writer.writerow([
                index,
                index >= skip_first,
                row["size_mib"],
                row["xfer_ms"],
                row["throughput_mib_s"],
                prediction_by_index.get(index, ""),
            ])
    with open(os.path.join(out_dir, "transfer-fit.json"), "w") as stream:
        json.dump(summary, stream, indent=2)

    sizes = np.asarray([row["size_mib"] for row in included])
    order = np.argsort(sizes)
    plt.figure()
    if skip_first:
        plt.scatter(
            [row["size_mib"] for row in all_rows[:skip_first]],
            [row["xfer_ms"] for row in all_rows[:skip_first]],
            marker="x",
            label="excluded cold/warmup",
        )
    plt.scatter(sizes, [row["xfer_ms"] for row in included], s=18, label="steady observed")
    predicted = np.asarray(fit["predicted_ms"])
    plt.plot(sizes[order], predicted[order], label="size-aware fit")
    plt.xlabel("KV transfer size (MiB)")
    plt.ylabel("NIXL transfer time (ms)")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "transfer-fit.png"), dpi=150, bbox_inches="tight")
    plt.close()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--skip-first", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_report(args.log, args.skip_first, args.out_dir), indent=2))
