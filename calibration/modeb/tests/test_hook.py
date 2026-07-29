# calibration/modeb/tests/test_hook.py
import json
import types
from calibration.modeb import sitecustomize as hook


def test_import_without_vllm_is_noop():
    # vllm is not installed in the test env; install() must return False, not raise.
    assert hook.install() is False


def test_jsonl_writer_buffers_and_flushes(tmp_path):
    p = tmp_path / "trace.jsonl"
    w = hook.JsonlWriter(str(p), flush_every=2)
    w.write({"step": 0})
    assert not p.exists() or p.read_text() == ""      # not flushed yet
    w.write({"step": 1})                                # hits flush_every -> flush
    lines = p.read_text().splitlines()
    assert [json.loads(l)["step"] for l in lines] == [0, 1]
    w.write({"step": 2})
    w.flush()
    lines = p.read_text().splitlines()
    assert [json.loads(l)["step"] for l in lines] == [0, 1, 2]


def test_capture_meta_maps_config_fields():
    cfg = types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(
            max_num_batched_tokens=8192, chunked_prefill_enabled=True,
            async_scheduling=False, max_num_partial_prefills=1,
            long_prefill_token_threshold=0, max_model_len=131072),
        cache_config=types.SimpleNamespace(num_gpu_blocks=29332),
        model_config=types.SimpleNamespace(model="meta-llama/Llama-3.3-70B-Instruct"),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=4),
    )
    meta = hook.capture_meta(cfg)
    assert meta["max_num_batched_tokens"] == 8192
    assert meta["chunked_prefill_enabled"] is True
    assert meta["async_scheduling"] is False
    assert meta["num_gpu_blocks"] == 29332
    assert meta["model"] == "meta-llama/Llama-3.3-70B-Instruct"
    assert meta["tensor_parallel_size"] == 4
    assert meta["max_num_partial_prefills"] == 1
    assert meta["long_prefill_token_threshold"] == 0
