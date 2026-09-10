"""Cross-replicate validation for real-engine calibration measurements."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from calibration.fit import fit_decode, fit_prefill, predict_mixed


def load_rows(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        return [tuple(float(value) for value in row) for row in reader]


def error_summary(predicted, observed):
    pred = np.asarray(predicted, dtype=float)
    obs = np.asarray(observed, dtype=float)
    if len(obs) == 0:
        raise ValueError("cannot evaluate an empty observation set")
    ape = np.abs(pred - obs) / np.abs(obs) * 100.0
    residual = obs - pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
    return {
        "n": int(len(obs)),
        "mape_pct": float(np.mean(ape)),
        "median_ape_pct": float(np.median(ape)),
        "p95_ape_pct": float(np.percentile(ape, 95)),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
    }


def aggregate_decode(rows):
    cells = defaultdict(list)
    for batch, context, step, observed in rows:
        cells[(batch, context)].append((step, observed))
    return [
        (
            batch,
            context,
            float(np.median([point[0] for point in points])),
            float(np.median([point[1] for point in points])),
        )
        for (batch, context), points in sorted(cells.items())
    ]


def aggregate_prefill(rows):
    cells = defaultdict(list)
    for prompt_tokens, observed in rows:
        cells[prompt_tokens].append(observed)
    return [
        (prompt_tokens, float(np.median(observations)))
        for prompt_tokens, observations in sorted(cells.items())
    ]


def fit_coefficients(directories, chunk_bud):
    decode = []
    prefill = []
    for directory in directories:
        decode.extend(load_rows(directory / "decode.csv"))
        prefill.extend(load_rows(directory / "prefill.csv"))
    decode_fit = fit_decode(decode)
    prefill_fit = fit_prefill(
        prefill, c_base=decode_fit["c_base"], chunk_bud=chunk_bud
    )
    return {
        "c_base": decode_fit["c_base"],
        "c_dec": decode_fit["c_dec"],
        "c_kv": decode_fit["c_kv"],
        "c_pf": prefill_fit["c_pf"],
        "c_attn": prefill_fit["c_attn"],
    }


def evaluate(directory, coeffs, chunk_bud):
    decode = aggregate_decode(load_rows(directory / "decode.csv"))
    decode_pred = [
        coeffs["c_base"]
        + coeffs["c_dec"] * batch
        + coeffs["c_kv"] * batch * (context + step)
        for batch, context, step, _ in decode
    ]
    decode_obs = [row[3] for row in decode]

    prefill = aggregate_prefill(load_rows(directory / "prefill.csv"))
    prefill_pred = [
        math.ceil(prompt_tokens / chunk_bud) * coeffs["c_base"]
        + coeffs["c_pf"] * prompt_tokens
        + coeffs["c_attn"] * prompt_tokens**2 / 2.0
        for prompt_tokens, _ in prefill
    ]
    prefill_obs = [row[1] for row in prefill]

    mixed = load_rows(directory / "mixed.csv")
    mixed_pred = [predict_mixed(row[:4], coeffs) for row in mixed]
    mixed_obs = [row[4] for row in mixed]
    mixed_by_batch = {}
    for batch in sorted({row[0] for row in mixed}):
        indices = [index for index, row in enumerate(mixed) if row[0] == batch]
        mixed_by_batch[str(int(batch))] = error_summary(
            [mixed_pred[index] for index in indices],
            [mixed_obs[index] for index in indices],
        )

    return {
        "decode_cells": error_summary(decode_pred, decode_obs),
        "prefill_lengths": error_summary(prefill_pred, prefill_obs),
        "mixed_rows": error_summary(mixed_pred, mixed_obs),
        "mixed_by_batch": mixed_by_batch,
    }


def coverage(directory):
    metadata = json.loads((directory / "metadata.json").read_text())
    decode = load_rows(directory / "decode.csv")
    prefill = load_rows(directory / "prefill.csv")
    mixed = load_rows(directory / "mixed.csv")

    decode_cells = defaultdict(int)
    for batch, context, _, _ in decode:
        decode_cells[(batch, context)] += 1
    prefill_cells = defaultdict(int)
    for prompt_tokens, _ in prefill:
        prefill_cells[prompt_tokens] += 1
    mixed_cells = defaultdict(int)
    for batch, _, _, _, _ in mixed:
        mixed_cells[batch] += 1

    expected_decode_cells = len(metadata["decode_batches"]) * len(
        metadata["decode_contexts"]
    )
    minimum_decode_rows_per_cell = max(
        1,
        min(
            32,
            metadata["decode_max_tokens"] - metadata["decode_warmup_tokens"],
        ),
    )
    expected_prefill_rows = len(metadata["prefill_lengths"]) * metadata[
        "prefill_repeats"
    ]
    expected_mixed_rows = sum(
        math.ceil(metadata["mixed_inject"] / metadata["chunk_bud"])
        for _ in metadata["mixed_batches"]
        for _ in range(metadata["mixed_repeats"])
    )
    issued_requests = (
        sum(metadata["decode_batches"]) * len(metadata["decode_contexts"])
        + len(metadata["prefill_lengths"]) * metadata["prefill_repeats"]
        + (sum(metadata["mixed_batches"]) + len(metadata["mixed_batches"]))
        * metadata["mixed_repeats"]
    )

    checks = {
        "decode_cells_complete": len(decode_cells) == expected_decode_cells,
        "decode_steady_state_barrier": bool(
            metadata.get("decode_steady_state_barrier", False)
        ),
        "decode_rows_per_cell_sufficient": (
            min(decode_cells.values(), default=0) >= minimum_decode_rows_per_cell
        ),
        "prefill_rows_complete": len(prefill) == expected_prefill_rows,
        "prefill_repeats_at_least_5": min(prefill_cells.values(), default=0) >= 5,
        "mixed_rows_complete": len(mixed) == expected_mixed_rows,
        "mixed_rows_per_batch_at_least_15": min(mixed_cells.values(), default=0) >= 15,
    }
    return {
        "directory": str(directory),
        "issued_requests": issued_requests,
        "decode_rows": len(decode),
        "decode_cells": len(decode_cells),
        "min_decode_rows_per_cell": min(decode_cells.values(), default=0),
        "required_decode_rows_per_cell": minimum_decode_rows_per_cell,
        "prefill_rows": len(prefill),
        "min_prefill_rows_per_length": min(prefill_cells.values(), default=0),
        "mixed_rows": len(mixed),
        "min_mixed_rows_per_batch": min(mixed_cells.values(), default=0),
        "checks": checks,
        "complete": all(checks.values()),
    }


def run_report(replicate_dirs, chunk_bud):
    directories = [Path(directory) for directory in replicate_dirs]
    if len(directories) < 3:
        raise ValueError("at least three independent replicate directories are required")

    coverage_rows = [coverage(directory) for directory in directories]
    folds = []
    for held_out, directory in enumerate(directories):
        training = [candidate for i, candidate in enumerate(directories) if i != held_out]
        coeffs = fit_coefficients(training, chunk_bud)
        folds.append({
            "held_out": str(directory),
            "training": [str(path) for path in training],
            "coeffs": coeffs,
            "validation": evaluate(directory, coeffs, chunk_bud),
        })

    stability = {}
    for name in ("c_base", "c_dec", "c_kv", "c_pf", "c_attn"):
        values = [fold["coeffs"][name] for fold in folds]
        mean = statistics.fmean(values)
        std = statistics.stdev(values)
        stability[name] = {
            "mean": mean,
            "std": std,
            "cv_pct": abs(std / mean) * 100.0 if mean else math.inf,
            "min": min(values),
            "max": max(values),
        }

    final_coeffs = fit_coefficients(directories, chunk_bud)
    return {
        "replicates": len(directories),
        "chunk_bud": chunk_bud,
        "total_issued_requests": sum(row["issued_requests"] for row in coverage_rows),
        "coverage": coverage_rows,
        "sample_sufficiency_pass": all(row["complete"] for row in coverage_rows),
        "leave_one_replicate_out": folds,
        "coefficient_stability": stability,
        "final_coeffs": final_coeffs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicate-dir", action="append", required=True)
    parser.add_argument("--chunk-bud", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = run_report(args.replicate_dir, args.chunk_bud)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
