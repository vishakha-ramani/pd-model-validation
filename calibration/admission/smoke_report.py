"""Summarize TTFT smoke-validation rows by enqueue-order workload slice."""

import argparse
import json
import math


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = min(max(int(math.ceil(q * len(ordered))) - 1, 0), len(ordered) - 1)
    return ordered[index]


def error_metrics(rows, predicted, realized):
    pairs = [(float(row[predicted]), float(row[realized])) for row in rows
             if row.get(predicted) is not None and row.get(realized) is not None
             and float(row[realized]) > 0]
    if not pairs:
        return {"n": 0}
    absolute = [abs(pred - real) for pred, real in pairs]
    ape = [err / real for err, (_, real) in zip(absolute, pairs)]
    real_sum = sum(real for _, real in pairs)
    return {
        "n": len(pairs),
        "realized_s_p50": percentile([real for _, real in pairs], 0.50),
        "realized_s_p90": percentile([real for _, real in pairs], 0.90),
        "predicted_s_p50": percentile([pred for pred, _ in pairs], 0.50),
        "mae_s": sum(absolute) / len(absolute),
        "mape_pct": 100.0 * sum(ape) / len(ape),
        "median_ape_pct": 100.0 * percentile(ape, 0.50),
        "p90_ape_pct": 100.0 * percentile(ape, 0.90),
        "wape_pct": 100.0 * sum(absolute) / real_sum,
        "bias_pct": 100.0 * sum(pred - real for pred, real in pairs) / real_sum,
    }


def censored_metrics(rows):
    predicted = [row for row in rows if row.get("p_deploy") is not None]
    below = [row for row in predicted if row["p_deploy"] < row["r_ttft_lower_bound"]]
    known = [max(row["r_ttft_lower_bound"] - row["p_deploy"], 0.0)
             for row in predicted]
    return {
        "n": len(rows),
        "prediction_n": len(predicted),
        "prediction_below_lower_bound_n": len(below),
        "prediction_below_lower_bound_frac": len(below) / len(predicted)
        if predicted else None,
        "lower_bound_s_p50": percentile(
            [row["r_ttft_lower_bound"] for row in rows], 0.50),
        "known_underprediction_s_p50": percentile(known, 0.50),
    }


def parse_slice(value):
    name, bounds = value.split("=", 1)
    start, end = bounds.split(":", 1)
    return name, int(start), int(end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--censored", required=True)
    parser.add_argument("--slice", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.events) as f:
        event_ids = [json.loads(line)["req_id"] for line in f if line.strip()]
    with open(args.rows) as f:
        rows = json.load(f)
    with open(args.censored) as f:
        censored = json.load(f)

    report = {}
    for value in args.slice:
        name, start, end = parse_slice(value)
        ids = set(event_ids[start:end])
        selected = [row for row in rows if row["req_id"] in ids]
        selected_censored = [row for row in censored if row["req_id"] in ids]
        report[name] = {
            "event_range": [start, end],
            "enqueued": len(ids),
            "completed_with_ttft": len(selected),
            "rollout_ttft": error_metrics(selected, "p_deploy_rollout", "r_ttft"),
            "legacy_composed_ttft": error_metrics(
                selected, "p_deploy_composed", "r_ttft"),
            "admission": error_metrics(selected, "t_adm_deploy", "r_tadm"),
            "censored": censored_metrics(selected_censored),
        }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
