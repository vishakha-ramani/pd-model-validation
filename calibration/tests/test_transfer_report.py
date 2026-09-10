from calibration.transfer_report import fit_size_aware_transfer, parse_transfer_metrics


def test_parse_transfer_metrics():
    lines = [
        "noise",
        "KV Transfer metrics: Num successful transfers=1, "
        "Avg xfer time (ms)=10.5, P90 xfer time (ms)=10.5, "
        "Avg MB per transfer=100.0, Throughput (MB/s)=9523.81",
    ]
    assert parse_transfer_metrics(lines) == [{
        "xfer_ms": 10.5,
        "size_mib": 100.0,
        "throughput_mib_s": 9523.81,
    }]


def test_fit_size_aware_transfer_recovers_line():
    base_s = 0.0005
    bandwidth = 12.0e9
    rows = []
    for size_mib in [100.0, 500.0, 2000.0]:
        seconds = base_s + size_mib * 2**20 / bandwidth
        rows.append({
            "xfer_ms": seconds * 1000.0,
            "size_mib": size_mib,
            "throughput_mib_s": size_mib / seconds,
        })
    fit = fit_size_aware_transfer(rows)
    assert abs(fit["xfer_base_us"] - 500.0) < 1e-6
    assert abs(fit["xfer_bandwidth_decimal_gbps"] - 12.0) < 1e-9
    assert fit["r2"] > 1 - 1e-12
