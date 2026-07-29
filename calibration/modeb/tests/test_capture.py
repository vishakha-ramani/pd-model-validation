# calibration/modeb/tests/test_capture.py
from calibration.modeb.modeb_capture import extract_reqs, build_step_record


def test_extract_new_prefill_and_decode_rows():
    cache = {}
    # new req A: prompt 1000, computed 0 (fresh prefill); new req B: prompt 512, computed 512 (just finished prefill, now decode)
    new = [("A", 1000, 0), ("B", 512, 512)]
    reqs = extract_reqs({"A": 256, "B": 1}, new, [], [], cache)
    by_id = {r["id"]: r for r in reqs}
    assert by_id["A"] == {"id": "A", "kappa": 256, "computed": 0, "prompt_len": 1000}
    assert by_id["B"] == {"id": "B", "kappa": 1, "computed": 512, "prompt_len": 512}
    assert cache == {"A": 1000, "B": 512}  # prompt lengths cached from new reqs


def test_extract_cached_uses_struct_of_arrays_and_cached_prompt_len():
    cache = {"A": 1000}  # A was new on a prior step
    # A continues (cached): computed advanced to 256; struct-of-arrays parallel lists
    reqs = extract_reqs({"A": 256}, [], ["A"], [256], cache)
    assert reqs == [{"id": "A", "kappa": 256, "computed": 256, "prompt_len": 1000}]


def test_extract_mixed_step_new_prefill_plus_cached_decode():
    cache = {"D": 300}  # decode req seen earlier
    new = [("P", 4096, 0)]
    reqs = extract_reqs({"D": 1, "P": 2048}, new, ["D"], [300], cache)
    by_id = {r["id"]: r for r in reqs}
    assert by_id["D"] == {"id": "D", "kappa": 1, "computed": 300, "prompt_len": 300}
    assert by_id["P"] == {"id": "P", "kappa": 2048, "computed": 0, "prompt_len": 4096}


def test_build_step_record_shape():
    rec = build_step_record(7, 1.5, 1.53, 2049, 1,
                            [{"id": "P", "kappa": 2048, "computed": 0, "prompt_len": 4096}])
    assert rec == {"step": 7, "t_start": 1.5, "t_end": 1.53,
                   "total_scheduled": 2049, "num_running": 1,
                   "reqs": [{"id": "P", "kappa": 2048, "computed": 0, "prompt_len": 4096}]}
