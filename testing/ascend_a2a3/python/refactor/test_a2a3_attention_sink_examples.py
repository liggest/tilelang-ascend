"""Original attention-sink assets: import, runtime lifetime, CLI and source contracts."""

import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = json.loads((ROOT / "maint/ascend_a2a3/ecosystem_attention_sink_ports.json").read_text())
ENTRIES = MANIFEST["entries"]
TEST = ROOT / "examples/ascend_a2a3/attention_sink/example_gqa_sink_fwd_varlen/test_gqa_sink_fwd_varlen.py"


def _load(path=TEST):
    spec = importlib.util.spec_from_file_location("_attention_sink_component_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e["source_path"])
def test_import_is_inert(entry, monkeypatch):
    import tilelang
    import torch

    def forbidden(*args, **kwargs):
        raise AssertionError("import performed runtime operation")

    prior = sys.modules.get("torch_npu")
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", forbidden)
    for obj, names in [
        (tilelang, ["compile", "lower", "disable_cache"]),
        (torch, ["manual_seed", "device", "set_default_device", "empty", "zeros", "full", "randn", "tensor"]),
    ]:
        for name in names:
            monkeypatch.setattr(obj, name, forbidden)
    _load(ROOT / entry["destination_path"])
    assert sys.modules.get("torch_npu") is prior


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("npu_default", [False, True])
def test_runtime_scope_restores_cache_and_lexical_default(enabled, fail, npu_default, monkeypatch):
    import tilelang
    from tilelang.env import CacheState

    module = _load()
    events = []
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))

    @contextmanager
    def device(name):
        events.append(("enter", name))
        try:
            yield
        finally:
            events.append(("exit", name))

    monkeypatch.setattr(module.torch, "device", device)
    original_error = ValueError("original host failure")
    result = object()

    @module._runtime_scope(npu_default=npu_default)
    def original_host():
        assert events == ([("enter", "npu")] if npu_default else [])
        tilelang.disable_cache()
        if fail:
            raise original_error
        return result

    prior = CacheState.is_enabled()
    try:
        CacheState.enable() if enabled else CacheState.disable()
        if fail:
            with pytest.raises(ValueError) as error:
                original_host()
            assert error.value is original_error
        else:
            assert original_host() is result
        assert CacheState.is_enabled() == enabled
        assert events == ([("enter", "npu"), ("exit", "npu")] if npu_default else [])
    finally:
        CacheState.enable() if prior else CacheState.disable()


@pytest.mark.parametrize("level", ["l0", "l1", "l2", "boundary", "all", "bench"])
@pytest.mark.parametrize("positive", [False, True])
def test_original_cli_blocking_and_nonblocking_exit(level, positive, monkeypatch):
    module = _load()
    events = []
    monkeypatch.setattr(sys, "argv", [str(TEST), "--level", level])
    monkeypatch.setattr(module, "_setup", lambda: events.append("setup"))
    for name in ["l0", "l1", "l2", "boundary"]:
        monkeypatch.setattr(module, "test_gqa_sink_fwd_" + name, lambda n=name: events.append(n) or positive)
    monkeypatch.setattr(module, "run_bench", lambda: events.append("bench") or positive)
    with pytest.raises(SystemExit) as error:
        module.main.__wrapped__()
    assert error.value.code == (0 if positive or level in ["l2", "boundary"] else 1)
    assert events == ["setup"] + (["l0", "l1", "l2", "boundary"] if level == "all" else [level])


@pytest.mark.parametrize("case", MANIFEST["factory_requests"], ids=lambda r: r["id"])
def test_complete_original_source_request(case):
    import tilelang
    from tilelang import tvm
    from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs

    if tvm.ffi.get_global_func("target.build.tilelang_ascend_a2a3", allow_missing=True) is None:
        pytest.skip("A2A3 native source codegen is not built")
    entry = next(e for e in ENTRIES if e["source_path"] == case["source_path"])
    module = _load(ROOT / entry["destination_path"])
    factory = getattr(module, case["factory"])
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    if case.get("expected_frontend_error"):
        assert case["expected_frontend_error"] == "AssertionError"
        with target, pytest.raises(AssertionError):
            factory.get_tir(*case["args"], **case["kwargs"])
        return
    with target:
        function = factory.get_tir(*case["args"], **case["kwargs"])
    configs = normalize_a2a3_pass_configs(target, factory.pass_configs)
    with tvm.transform.PassContext(opt_level=3, config=configs):
        artifact = tilelang.lower(function, target=target)
    assert 'extern "C" void call(' in artifact.kernel_source


@pytest.mark.parametrize("level", ["l0", "l1", "l2", "boundary", "all"])
@pytest.mark.parametrize("positive", [False, True])
def test_original_backward_cli_exit_and_nonblocking_levels(level, positive, monkeypatch):
    path = TEST.parent.parent / "example_gqa_sink_bwd_bhsd/test_gqa_sink_bwd_bhsd.py"
    module = _load(path)
    calls = []
    monkeypatch.setattr(sys, "argv", [str(path), "--level", level])
    monkeypatch.setattr(module.tilelang, "disable_cache", lambda: None)
    monkeypatch.setattr(module.torch, "manual_seed", lambda seed: calls.append(seed))
    for name in ["l0", "l1", "l2", "boundary"]:
        monkeypatch.setattr(module, "test_gqa_sink_bwd_bhsd_" + name, lambda n=name: (calls.append(n) or positive, []))
    with pytest.raises(SystemExit) as error:
        module.main.__wrapped__()
    assert error.value.code == (0 if positive or level in ["l2", "boundary"] else 1)
    assert calls == [42] + (["l0", "l1", "l2", "boundary"] if level == "all" else [level])
