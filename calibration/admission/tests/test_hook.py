import importlib


def test_import_is_noop_without_vllm():
    # vllm is absent in the test env; importing the hook must not raise.
    mod = importlib.import_module("calibration.admission.sitecustomize")
    assert mod.install() is False  # vllm absent -> no-op


def test_max_waiting_ids_env_parsed(monkeypatch):
    monkeypatch.setenv("ADMISSION_MAX_WAITING_IDS", "128")
    mod = importlib.reload(importlib.import_module("calibration.admission.sitecustomize"))
    assert mod.MAX_WAITING_IDS == 128


def test_max_waiting_ids_env_default(monkeypatch):
    monkeypatch.delenv("ADMISSION_MAX_WAITING_IDS", raising=False)
    mod = importlib.reload(importlib.import_module("calibration.admission.sitecustomize"))
    assert mod.MAX_WAITING_IDS == 512


def test_max_waiting_ids_env_garbage_falls_back(monkeypatch):
    monkeypatch.setenv("ADMISSION_MAX_WAITING_IDS", "not-an-int")
    mod = importlib.reload(importlib.import_module("calibration.admission.sitecustomize"))
    assert mod.MAX_WAITING_IDS == 512
