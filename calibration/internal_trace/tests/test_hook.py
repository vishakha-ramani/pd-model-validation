import importlib
import json
import sys
import types

from calibration.internal_trace import sitecustomize as unpatched_hook


def test_v026_hook_pairs_scheduler_output_with_same_engine_step(monkeypatch, tmp_path):
    class Scheduler:
        def __init__(self):
            self.running = [object(), object()]
            self.waiting = []
            self.skipped_waiting = []
            self.calls = 0

        def schedule(self, throttle_prefills=False):
            assert throttle_prefills is True
            self.calls += 1
            if self.calls == 1:
                return types.SimpleNamespace(
                    total_num_scheduled_tokens=1024,
                    num_scheduled_tokens={"decode": 1024},
                    scheduled_new_reqs=[
                        types.SimpleNamespace(
                            req_id="decode",
                            prompt_token_ids=list(range(1024)),
                            prompt_embeds=None,
                            num_computed_tokens=0,
                        )
                    ],
                    scheduled_cached_reqs=types.SimpleNamespace(
                        req_ids=[], num_computed_tokens=[], num_output_tokens=[]
                    ),
                )
            return types.SimpleNamespace(
                total_num_scheduled_tokens=2049,
                num_scheduled_tokens={"decode": 1, "prefill": 2048},
                scheduled_new_reqs=[
                    types.SimpleNamespace(
                        req_id="prefill",
                        prompt_token_ids=list(range(4096)),
                        prompt_embeds=None,
                        num_computed_tokens=0,
                    )
                ],
                scheduled_cached_reqs=types.SimpleNamespace(
                    req_ids=["decode"],
                    num_computed_tokens=[1024],
                    num_output_tokens=[8],
                ),
            )

    scheduler_config = types.SimpleNamespace(
        max_num_batched_tokens=8192,
        max_num_seqs=128,
        max_model_len=32768,
        enable_chunked_prefill=True,
        async_scheduling=False,
    )
    cache_config = types.SimpleNamespace(
        block_size=16, num_gpu_blocks=1000, enable_prefix_caching=False
    )
    config = types.SimpleNamespace(
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        model_config=types.SimpleNamespace(model="test-model", revision="abc"),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=4),
    )

    class EngineCore:
        def __init__(self):
            self.scheduler = Scheduler()
            self.vllm_config = config

        def step(self):
            self.scheduler.schedule(True)
            return ({}, True)

    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.core": types.ModuleType("vllm.v1.core"),
        "vllm.v1.core.sched": types.ModuleType("vllm.v1.core.sched"),
        "vllm.v1.core.sched.scheduler": types.ModuleType(
            "vllm.v1.core.sched.scheduler"
        ),
        "vllm.v1.engine": types.ModuleType("vllm.v1.engine"),
        "vllm.v1.engine.core": types.ModuleType("vllm.v1.engine.core"),
        "vllm.v1.utils": types.ModuleType("vllm.v1.utils"),
    }
    modules["vllm"].__version__ = "0.26.0"
    modules["vllm.v1.core.sched.scheduler"].Scheduler = Scheduler
    modules["vllm.v1.engine.core"].EngineCore = EngineCore
    modules["vllm.v1.utils"].compute_iteration_details = lambda output: types.SimpleNamespace(
        num_ctx_requests=1,
        num_ctx_tokens=2048,
        num_generation_requests=1,
        num_generation_tokens=1,
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("ITERTRACE_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("ITERTRACE_FLUSH_EVERY", "1")

    hook = importlib.reload(unpatched_hook)
    assert hook._PATCHED is True

    engine = EngineCore()
    engine.step()  # caches the decode request's immutable prompt length
    result = engine.step()
    assert result == ({}, True)
    rows = [
        json.loads(line)
        for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()
    ]
    row = rows[1]
    assert row["regime"] == "mixed"
    assert row["B_decode"] == 1
    assert row["K_decode"] == 1024
    assert row["S_prefill"] == 2048
    assert row["U_prefill"] == 2048 * 1024
    assert row["engine_step_s"] >= 0
    assert row["vllm_iteration_details"]["context_tokens"] == 2048

    metadata = json.loads((tmp_path / "meta.json").read_text())
    assert metadata["vllm_version"] == "0.26.0"
    assert metadata["tensor_parallel_size"] == 4
