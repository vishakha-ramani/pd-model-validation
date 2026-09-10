import csv
import json
import math

from calibration.replicate_report import run_report


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def make_replicate(path, seed):
    c_base, c_dec, c_kv = 0.02, 0.0011, 2.5e-7
    c_pf, c_attn = 3e-5, 1e-8
    chunk = 8
    metadata = {
        "seed": seed,
        "chunk_bud": chunk,
        "decode_batches": [1, 2],
        "decode_contexts": [8, 16],
        "decode_max_tokens": 8,
        "decode_warmup_tokens": 4,
        "decode_steady_state_barrier": True,
        "prefill_lengths": [4, 8],
        "prefill_repeats": 5,
        "mixed_batches": [1, 2],
        "mixed_inject": 8,
        "mixed_repeats": 15,
    }
    path.mkdir()
    (path / "metadata.json").write_text(json.dumps(metadata))

    decode = []
    for batch in metadata["decode_batches"]:
        for context in metadata["decode_contexts"]:
            for step in range(4, 8):
                observed = c_base + c_dec * batch + c_kv * batch * (context + step)
                decode.append((batch, context, step, observed))
    write_csv(path / "decode.csv", ["B", "n", "k", "t_iter"], decode)

    prefill = []
    for prompt in metadata["prefill_lengths"]:
        observed = (
            math.ceil(prompt / chunk) * c_base
            + c_pf * prompt
            + c_attn * prompt**2 / 2
        )
        prefill.extend((prompt, observed) for _ in range(5))
    write_csv(path / "prefill.csv", ["n", "ttft"], prefill)

    mixed = []
    for batch in metadata["mixed_batches"]:
        for _ in range(15):
            context_sum = batch * 16
            observed = (
                c_base
                + c_dec * batch
                + c_kv * context_sum
                + c_pf * chunk
                + c_attn * chunk * (chunk / 2)
            )
            mixed.append((batch, context_sum, chunk, 0, observed))
    write_csv(
        path / "mixed.csv",
        ["B", "resident_context_sum", "kappa", "P_k", "t_iter_observed"],
        mixed,
    )


def test_cross_replicate_report_counts_requests_and_holds_out_each_seed(tmp_path):
    directories = []
    for seed in range(3):
        directory = tmp_path / f"replicate-{seed}"
        make_replicate(directory, seed)
        directories.append(directory)

    report = run_report(directories, chunk_bud=8)

    assert report["replicates"] == 3
    assert report["sample_sufficiency_pass"]
    assert report["total_issued_requests"] == 3 * (6 + 10 + 75)
    assert len(report["leave_one_replicate_out"]) == 3
    for fold in report["leave_one_replicate_out"]:
        assert fold["validation"]["decode_cells"]["mape_pct"] < 1e-8
        assert fold["validation"]["prefill_lengths"]["mape_pct"] < 1e-8
        assert fold["validation"]["mixed_rows"]["mape_pct"] < 1e-8
