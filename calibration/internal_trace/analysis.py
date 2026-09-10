"""Fit and validate router coefficients from exact vLLM 0.26 iteration rows."""
import argparse
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


REQUEST_RE = re.compile(
    r"internal-v1:s(?P<seed>\d+):(?P<workload>decode|prefill|mixed):"
)
DECODE_RE = re.compile(r":decode:B(?P<B>\d+):N(?P<N>\d+):")
PREFILL_RE = re.compile(r":prefill:N(?P<N>\d+):r(?P<repeat>\d+)")
MIXED_RE = re.compile(r":mixed:B(?P<B>\d+):r(?P<repeat>\d+):")


def load_steps(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _request_metadata(step):
    matches = [REQUEST_RE.search(request["id"]) for request in step["requests"]]
    if not matches or any(match is None for match in matches):
        return None
    seeds = {int(match.group("seed")) for match in matches}
    workloads = {match.group("workload") for match in matches}
    if len(seeds) != 1 or len(workloads) != 1:
        return None
    return next(iter(seeds)), next(iter(workloads))


def annotate_steps(steps):
    annotated = []
    for raw in steps:
        step = dict(raw)
        metadata = _request_metadata(step)
        if metadata is None:
            step.update(seed=None, workload="unlabelled")
        else:
            step.update(seed=metadata[0], workload=metadata[1])
        annotated.append(step)
    return annotated


def _ols(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _median_groups(rows, key_fields, value_fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in key_fields)].append(row)
    result = []
    for key, members in groups.items():
        record = dict(zip(key_fields, key))
        for field in value_fields:
            record[field] = float(np.median([member[field] for member in members]))
        record["samples"] = len(members)
        result.append(record)
    return result


def _decode_cell(step):
    matches = [DECODE_RE.search(request["id"]) for request in step["requests"]]
    if not matches or any(match is None for match in matches):
        return None
    cells = {(int(match.group("B")), int(match.group("N"))) for match in matches}
    return next(iter(cells)) if len(cells) == 1 else None


def decode_cells(steps, seeds=None):
    rows = []
    for step in steps:
        if step["workload"] != "decode" or step["regime"] != "pure_decode":
            continue
        if seeds is not None and step["seed"] not in seeds:
            continue
        cell = _decode_cell(step)
        if cell is None:
            continue
        B, N = cell
        rows.append(
            {
                "seed": step["seed"],
                "B": B,
                "N": N,
                "K_decode": step["K_decode"],
                "engine_step_s": step["engine_step_s"],
            }
        )
    return _median_groups(
        rows,
        ("seed", "B", "N"),
        ("K_decode", "engine_step_s"),
    )


def prefill_cells(steps, seeds=None):
    rows = []
    for step in steps:
        if step["workload"] != "prefill" or step["regime"] != "pure_prefill":
            continue
        if seeds is not None and step["seed"] not in seeds:
            continue
        request = step["requests"][0]
        match = PREFILL_RE.search(request["id"])
        if match is None:
            continue
        rows.append(
            {
                "seed": step["seed"],
                "N": int(match.group("N")),
                "computed": request["computed_tokens"],
                "S_prefill": step["S_prefill"],
                "U_prefill": step["U_prefill"],
                "engine_step_s": step["engine_step_s"],
            }
        )
    return _median_groups(
        rows,
        ("seed", "N", "computed", "S_prefill"),
        ("U_prefill", "engine_step_s"),
    )


def fit_coefficients(steps, seeds=None):
    decode = decode_cells(steps, seeds)
    prefill = prefill_cells(steps, seeds)
    if len(decode) < 3 or len(prefill) < 3:
        raise ValueError("not enough labelled pure-regime cells to fit coefficients")

    decode_beta = _ols(
        [[1.0, row["B"], row["K_decode"]] for row in decode],
        [row["engine_step_s"] for row in decode],
    )
    prefill_beta = _ols(
        [[1.0, row["S_prefill"], row["U_prefill"]] for row in prefill],
        [row["engine_step_s"] for row in prefill],
    )
    return {
        "alphaD": float(decode_beta[0]),
        "alphaP": float(prefill_beta[0]),
        "c0": float(decode_beta[1]),
        "c1": float(decode_beta[2]),
        "cPf": float(prefill_beta[1]),
        "cAttn": float(prefill_beta[2]),
    }


def predict(step, coefficients, scheduled_batch=False):
    alpha = (
        coefficients["alphaP"]
        if step["regime"] == "pure_prefill"
        else coefficients["alphaD"]
    )
    batch = step["B_scheduled"] if scheduled_batch else step["B_decode"]
    return (
        alpha
        + coefficients["c0"] * batch
        + coefficients["c1"] * step["K_decode"]
        + coefficients["cPf"] * step["S_prefill"]
        + coefficients["cAttn"] * step["U_prefill"]
    )


def error_summary(steps, coefficients, scheduled_batch=False):
    if not steps:
        return {"n": 0, "mape_pct": None, "p95_ape_pct": None, "r2": None}
    observed = np.asarray([step["engine_step_s"] for step in steps], dtype=float)
    predicted = np.asarray(
        [predict(step, coefficients, scheduled_batch) for step in steps], dtype=float
    )
    ape = np.abs(predicted - observed) / observed * 100.0
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    return {
        "n": len(steps),
        "mape_pct": float(np.mean(ape)),
        "median_ape_pct": float(np.median(ape)),
        "p95_ape_pct": float(np.percentile(ape, 95)),
        "bias_pct": float(np.mean((predicted - observed) / observed * 100.0)),
        "rmse_us": float(math.sqrt(np.mean((predicted - observed) ** 2)) * 1e6),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
    }


def _evaluation_rows(steps, seed, workload, regime):
    return [
        step
        for step in steps
        if step["seed"] == seed
        and step["workload"] == workload
        and step["regime"] == regime
    ]


def _coefficient_cv(folds, name):
    values = np.asarray([fold["coefficients"][name] for fold in folds], dtype=float)
    return float(np.std(values, ddof=1) / abs(np.mean(values)) * 100.0)


def build_report(raw_steps):
    steps = annotate_steps(raw_steps)
    seeds = sorted({step["seed"] for step in steps if step["seed"] is not None})
    if len(seeds) < 2:
        raise ValueError(f"need at least two labelled seeds, found {seeds}")

    folds = []
    for held_out in seeds:
        training = set(seeds) - {held_out}
        coefficients = fit_coefficients(steps, training)
        folds.append(
            {
                "held_out_seed": held_out,
                "coefficients": coefficients,
                "decode": error_summary(
                    _evaluation_rows(steps, held_out, "decode", "pure_decode"),
                    coefficients,
                ),
                "prefill": error_summary(
                    _evaluation_rows(steps, held_out, "prefill", "pure_prefill"),
                    coefficients,
                ),
                "mixed_exact_B_decode": error_summary(
                    _evaluation_rows(steps, held_out, "mixed", "mixed"),
                    coefficients,
                ),
                "mixed_router_B_scheduled": error_summary(
                    _evaluation_rows(steps, held_out, "mixed", "mixed"),
                    coefficients,
                    scheduled_batch=True,
                ),
            }
        )

    final = fit_coefficients(steps, set(seeds))
    coefficient_names = ("alphaD", "alphaP", "c0", "c1", "cPf", "cAttn")
    counts = {
        "total_steps": len(steps),
        "labelled_steps": sum(step["seed"] is not None for step in steps),
        "decode_cells": len(decode_cells(steps)),
        "prefill_cells": len(prefill_cells(steps)),
        "mixed_steps": sum(
            step["workload"] == "mixed" and step["regime"] == "mixed"
            for step in steps
        ),
        "unexpected_decode_grants": sum(
            step["decode_scheduled_tokens"] != step["B_decode"]
            for step in steps
            if step["B_decode"]
        ),
        "over_budget_steps": sum(
            step["total_scheduled_tokens"] > 8192 for step in steps
        ),
        "vllm_aggregate_mismatches": sum(
            bool(step.get("vllm_iteration_details"))
            and (
                step["B_decode"]
                != step["vllm_iteration_details"]["generation_requests"]
                or step["S_prefill"]
                != step["vllm_iteration_details"]["context_tokens"]
            )
            for step in steps
        ),
    }
    coefficient_cv = {
        name: _coefficient_cv(folds, name) for name in coefficient_names
    }
    final_validation = {
        "decode": error_summary(
            [s for s in steps if s["workload"] == "decode" and s["regime"] == "pure_decode"],
            final,
        ),
        "prefill": error_summary(
            [s for s in steps if s["workload"] == "prefill" and s["regime"] == "pure_prefill"],
            final,
        ),
        "mixed_exact_B_decode": error_summary(
            [s for s in steps if s["workload"] == "mixed" and s["regime"] == "mixed"],
            final,
        ),
        "mixed_router_B_scheduled": error_summary(
            [s for s in steps if s["workload"] == "mixed" and s["regime"] == "mixed"],
            final,
            scheduled_batch=True,
        ),
    }
    gates = {
        "three_independent_seeds": len(seeds) == 3,
        "all_105_decode_cells": counts["decode_cells"] == 105,
        "all_45_prefill_chunk_cells": counts["prefill_cells"] == 45,
        "at_least_360_exact_mixed_steps": counts["mixed_steps"] >= 360,
        "one_decode_token_per_scheduled_decode_request": counts[
            "unexpected_decode_grants"
        ]
        == 0,
        "scheduler_budget_respected": counts["over_budget_steps"] == 0,
        "derived_phase_counts_match_vllm": counts["vllm_aggregate_mismatches"]
        == 0,
        "all_coefficients_positive": all(final[name] > 0 for name in coefficient_names),
        "all_coefficient_cv_below_5pct": all(
            coefficient_cv[name] < 5.0 for name in coefficient_names
        ),
        "held_out_mixed_median_ape_below_5pct": all(
            fold["mixed_exact_B_decode"]["median_ape_pct"] < 5.0 for fold in folds
        ),
        "held_out_mixed_p95_ape_below_15pct": all(
            fold["mixed_exact_B_decode"]["p95_ape_pct"] < 15.0 for fold in folds
        ),
    }
    return {
        "schema_version": 1,
        "timing_source": "wall time around vLLM EngineCore.step",
        "state_source": "the matching vLLM SchedulerOutput",
        "seeds": seeds,
        "coverage": counts,
        "final_coefficients_seconds": final,
        "epp_coefficients_microseconds": {
            "alphaD": final["alphaD"] * 1e6,
            "alphaP": final["alphaP"] * 1e6,
            "c0": final["c0"] * 1e6,
            "c1": final["c1"] * 1e6,
            "cPf": final["cPf"] * 1e6,
            "cAttn": final["cAttn"] * 1e6,
        },
        "leave_one_seed_out": folds,
        "coefficient_cv_pct": coefficient_cv,
        "final_validation": final_validation,
        "reliability_gates": gates,
        "all_reliability_gates_pass": all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = build_report(load_steps(args.trajectory))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
