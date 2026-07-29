# calibration/modeb/tests/test_analysis.py
import json
from calibration.modeb.analysis import (load_coeffs, predict_step, classify_regime,
                                         parse_trajectory)
from calibration.fit import predict_mixed

COEFFS = {"c_base": 0.015977, "c_dec": 4.4947e-05, "c_kv": 2.6932e-08,
          "c_pf": 6.3524e-05, "c_attn": 1.2893e-09}


def test_predict_step_matches_predict_mixed_on_single_prefill_row():
    # B=3 decodes with total resident context 900, one prefill chunk kappa=2048 at P_k=4096
    reqs = ([{"id": f"d{i}", "kappa": 1, "computed": 300, "prompt_len": 100} for i in range(3)]
            + [{"id": "p", "kappa": 2048, "computed": 4096, "prompt_len": 16000}])
    resident_ctx_sum = 3 * 300
    row = (3, resident_ctx_sum, 2048, 4096)
    assert abs(predict_step(reqs, COEFFS) - predict_mixed(row, COEFFS)) < 1e-12


def test_predict_step_sums_multiple_prefill_chunks():
    reqs = [{"id": "p1", "kappa": 1000, "computed": 0, "prompt_len": 5000},
            {"id": "p2", "kappa": 1000, "computed": 2000, "prompt_len": 5000}]
    expected = (COEFFS["c_base"]
                + COEFFS["c_pf"] * 2000
                + COEFFS["c_attn"] * (1000 * (0 + 500) + 1000 * (2000 + 500)))
    assert abs(predict_step(reqs, COEFFS) - expected) < 1e-12


def test_classify_regime():
    dec = [{"id": "a", "kappa": 1, "computed": 500, "prompt_len": 100}]
    pre = [{"id": "b", "kappa": 2048, "computed": 0, "prompt_len": 4096}]
    assert classify_regime(dec) == "pure_decode"
    assert classify_regime(pre) == "pure_prefill"
    assert classify_regime(dec + pre) == "mixed"


def test_parse_trajectory_computes_delta_and_drops_last(tmp_path):
    p = tmp_path / "trajectory.jsonl"
    steps = [
        {"step": 0, "t_start": 1.00, "reqs": [{"id": "a", "kappa": 1, "computed": 500, "prompt_len": 100}]},
        {"step": 1, "t_start": 1.02, "reqs": [{"id": "a", "kappa": 1, "computed": 501, "prompt_len": 100}]},
        {"step": 2, "t_start": 1.05, "reqs": [{"id": "a", "kappa": 1, "computed": 502, "prompt_len": 100}]},
    ]
    p.write_text("\n".join(json.dumps(s) for s in steps) + "\n")
    out = parse_trajectory(str(p))
    assert len(out) == 2                      # last step dropped
    assert abs(out[0]["t_iter"] - 0.02) < 1e-9
    assert abs(out[1]["t_iter"] - 0.03) < 1e-9
    assert out[0]["regime"] == "pure_decode"


def test_load_coeffs_roundtrip(tmp_path):
    p = tmp_path / "coeffs.json"
    p.write_text(json.dumps(COEFFS))
    assert load_coeffs(str(p)) == COEFFS


def test_per_regime_report_recovers_zero_error_on_synthetic():
    from calibration.modeb.analysis import per_regime_report
    # build steps whose measured t_iter EQUALS the law prediction -> ~0 MAPE
    reqs_dec = [{"id": "a", "kappa": 1, "computed": 500, "prompt_len": 100}]
    reqs_mix = [{"id": "a", "kappa": 1, "computed": 500, "prompt_len": 100},
                {"id": "p", "kappa": 2048, "computed": 0, "prompt_len": 8192}]
    steps = []
    for reqs in (reqs_dec, reqs_mix, reqs_dec):
        t = predict_step(reqs, COEFFS)
        steps.append({"reqs": reqs, "t_iter": t, "regime": classify_regime(reqs)})
    rep = per_regime_report(steps, COEFFS)
    assert rep["by_regime"]["pure_decode"]["mape"] < 1e-6
    assert rep["by_regime"]["mixed"]["mape"] < 1e-6
    assert rep["by_regime"]["mixed"]["n"] == 1
    assert rep["by_regime"]["all"]["n"] == 3


def test_write_report_emits_json_and_png(tmp_path):
    from calibration.modeb.analysis import write_report
    reqs = [{"id": "a", "kappa": 1, "computed": 500, "prompt_len": 100}]
    steps = [{"reqs": reqs, "t_iter": predict_step(reqs, COEFFS), "regime": "pure_decode"}]
    rep = write_report(steps, COEFFS, str(tmp_path))
    assert (tmp_path / "modeb_report.json").exists()
    assert (tmp_path / "modeb_pred_vs_meas.png").exists()
    assert rep["by_regime"]["pure_decode"]["n"] == 1


def test_guidellm_curves_parses_by_header(tmp_path):
    from calibration.modeb.analysis import guidellm_curves
    p = tmp_path / "curves.csv"
    p.write_text(
        "strat,rate,reqs,conc,ttft_med,ttft_mean,ttft_p95,itl_med,itl_mean,itl_p95,outtps,n\n"
        "synchronous,,0.23,0.99,81.4,82.9,84.3,16.28,16.31,16.46,60.4,29\n"
        "constant,1.04,1.0,4.70,91.9,92.4,99.9,17.76,17.77,17.89,259.5,122\n"
    )
    rows = guidellm_curves(str(p))
    assert rows[0]["strat"] == "synchronous"
    assert abs(rows[0]["itl_mean_ms"] - 16.31) < 1e-6
    assert abs(rows[1]["ttft_mean_ms"] - 92.4) < 1e-6
    assert abs(rows[1]["concurrency"] - 4.70) < 1e-6
    assert rows[1]["rate"] == 1.04
