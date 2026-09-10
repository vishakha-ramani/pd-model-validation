import csv, math, json, os
from calibration.report import load_csv_inputs, run_report

def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

def test_run_report_end_to_end(tmp_path):
    c_base, c_dec, c_kv = 0.020, 0.0011, 2.5e-7
    c_pf, c_attn = 3.0e-5, 1.0e-8
    chunk_bud = 8192
    d = tmp_path
    dec = [(B, n, k, c_base + B*c_dec + c_kv*B*(n+k))
           for B in [1,2,4,8,16,32,64] for n in [64,256,1024,4096] for k in range(0,256,8)]
    _write(d/"decode.csv", ["B","n","k","t_iter"], dec)
    pre = [(n, math.ceil(n/chunk_bud)*c_base + c_pf*n + c_attn*(n**2/2))
           for n in [64,128,256,512,1024,2048,4096,8192,16384]]
    _write(d/"prefill.csv", ["n","ttft"], pre)
    mix = []
    for B in [8,16,32]:
        for k in range(0, 50):
            rcs = B*(1024+k); kappa=8192; P_k=0
            t = (c_base + B*c_dec + c_kv*rcs + c_pf*kappa + c_attn*kappa*(P_k+kappa/2))
            mix.append((B, rcs, kappa, P_k, t))
    _write(d/"mixed.csv", ["B","resident_context_sum","kappa","P_k","t_iter_observed"], mix)

    summary = run_report(str(d/"decode.csv"), str(d/"prefill.csv"), str(d/"mixed.csv"),
                         chunk_bud=chunk_bud, out_dir=str(d))
    assert summary["mixed_mape"] < 0.01
    assert os.path.exists(d/"coeffs.json")
    assert os.path.exists(d/"residuals.json")
    assert os.path.exists(d/"predicted_vs_realized.png")
    coeffs = json.load(open(d/"coeffs.json"))
    assert abs(coeffs["c_attn"] - c_attn) < 1e-11


def test_load_csv_inputs_accepts_multiple_shards(tmp_path):
    header = ["B", "n", "k", "t_iter"]
    _write(tmp_path / "first.csv", header, [(1, 64, 4, 0.1)])
    _write(tmp_path / "second.csv", header, [(2, 256, 8, 0.2)])

    assert load_csv_inputs([
        tmp_path / "first.csv",
        tmp_path / "second.csv",
    ]) == [(1.0, 64.0, 4.0, 0.1), (2.0, 256.0, 8.0, 0.2)]
