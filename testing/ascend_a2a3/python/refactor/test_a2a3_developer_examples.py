"""Original developer examples: import/runtime isolation and full source requests."""

import argparse
import builtins
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = json.loads((ROOT / "maint/ascend_a2a3/ecosystem_developer_ports.json").read_text())
ENTRIES = MANIFEST["entries"]
REQUESTS = MANIFEST["factory_requests"]
BY_SOURCE = {entry["source_path"]: entry for entry in ENTRIES}


def _load(entry):
    path = ROOT / entry["destination_path"]
    spec = importlib.util.spec_from_file_location("_developer_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e["source_path"])
def test_import_does_not_execute_original_runtime(entry, monkeypatch):
    import tilelang
    import torch
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

    def forbidden(*args, **kwargs):
        raise AssertionError("import executed original runtime")

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", forbidden)
    monkeypatch.setattr(A2A3CythonKernelCache, "clear_cache", forbidden)
    for obj, names in [
        (tilelang, ["disable_cache", "compile", "lower"]),
        (torch, ["manual_seed", "set_default_device", "randn", "empty", "zeros"]),
    ]:
        for name in names:
            monkeypatch.setattr(obj, name, forbidden)
    previous = sys.modules.get("torch_npu")
    module = _load(entry)
    assert callable(module.main)
    assert sys.modules.get("torch_npu") is previous


@pytest.mark.parametrize("entry", [e for e in ENTRIES if e["source_runtime"]["clear_cache"]], ids=lambda e: e["source_path"])
def test_clear_requires_explicit_owned_cache(entry, monkeypatch):
    module = _load(entry)
    monkeypatch.delenv("TILELANG_CACHE_DIR", raising=False)
    with pytest.raises(RuntimeError, match="dedicated TILELANG_CACHE_DIR"):
        module.main()


@pytest.mark.parametrize("entry", [e for e in ENTRIES if e["source_runtime"]["clear_cache"]], ids=lambda e: e["source_path"])
def test_resolved_cache_override_conflict_cannot_clear_other_cache(entry, tmp_path, monkeypatch):
    from tilelang import env
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

    module = _load(entry)
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "owned"))
    monkeypatch.setattr(vars(type(env))["TILELANG_CACHE_DIR"], "_forced_value", str(tmp_path / "other"))

    def forbidden(*args):
        raise AssertionError("other caller cache was cleared")

    monkeypatch.setattr(A2A3CythonKernelCache, "clear_cache", forbidden)
    with pytest.raises(RuntimeError, match="resolved cache differs"):
        module.main()


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e["source_path"])
@pytest.mark.parametrize("enabled", [False, True])
def test_original_cache_mode_and_registration_during_runtime_restore_caller_on_failure(entry, enabled, tmp_path, monkeypatch):
    import tilelang
    import torch
    from tilelang.env import CacheState
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

    module = _load(entry)
    monkeypatch.setattr(sys, "argv", [entry["destination_path"]])
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "owned"))
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(torch, "manual_seed", lambda *args: None)
    if entry["source_runtime"]["set_default_device"]:
        # Exercise the public CPU context stack without selecting an NPU.
        class HostTorch:
            def device(self, name):
                assert name == "npu"
                return torch.device("cpu")

            def __getattr__(self, name):
                return getattr(torch, name)

        monkeypatch.setattr(module, "torch", HostTorch())
    events = []
    original_import = builtins.__import__
    original_disable = tilelang.disable_cache

    def observe_import(name, *args, **kwargs):
        if name == "torch_npu":
            events.append("registration")
        return original_import(name, *args, **kwargs)

    def disable():
        events.append("disable")
        original_disable()

    monkeypatch.setattr(builtins, "__import__", observe_import)
    monkeypatch.setattr(tilelang, "disable_cache", disable)
    monkeypatch.setattr(A2A3CythonKernelCache, "clear_cache", lambda *args: events.append("clear"))

    class StopFactory(Exception):
        pass

    def stop(*args, **kwargs):
        events.append("inputs" if entry["source_runtime"].get("inputs_before_factory") else "factory")
        assert CacheState.is_enabled() == (False if entry["source_runtime"]["disable_cache"] else enabled)
        raise StopFactory

    request = next(r for r in REQUESTS if r["source_path"] == entry["source_path"])
    monkeypatch.setattr(module, request["factory"], stop)
    if entry["source_runtime"].get("inputs_before_factory"):
        # Preserve original PTO allocations before factory construction; fail
        # at the first input boundary without allocating any device tensor.
        monkeypatch.setattr(torch, "randn", stop)
    before = CacheState.is_enabled()
    modes_before = torch.overrides._get_current_function_mode_stack()
    try:
        CacheState.enable() if enabled else CacheState.disable()
        with pytest.raises(StopFactory):
            module.main()
        assert CacheState.is_enabled() == enabled
        modes_after = torch.overrides._get_current_function_mode_stack()
        assert len(modes_before) == len(modes_after)
        assert all(a is b for a, b in zip(modes_before, modes_after))
        assert events == [
            "registration",
            "disable" if entry["source_runtime"]["disable_cache"] else "clear",
            "inputs" if entry["source_runtime"].get("inputs_before_factory") else "factory",
        ]
    finally:
        CacheState.enable() if before else CacheState.disable()


@pytest.mark.parametrize("case", REQUESTS, ids=lambda r: r["id"])
def test_full_original_factory_lowers_without_device(case, monkeypatch):
    import tilelang
    from tilelang import tvm
    from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs

    for name in ("tl.transform.A2A3HostLegalize", "target.build.tilelang_ascend_a2a3", "target.build.tilelang_pto_a2a3"):
        if tvm.ffi.get_global_func(name, allow_missing=True) is None:
            pytest.skip("requires USE_ASCEND=ON with the complete A2A3 pipeline")
    monkeypatch.setenv("TL_PLATFORM", "A3")
    factory = getattr(_load(BY_SOURCE[case["source_path"]]), case["factory"])
    dialect = "pto" if factory.target == "pto" else "ascendc"
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": [dialect, "ascend"]})
    with target:
        prim = factory.get_tir(*case["args"], **case["kwargs"])
    config = normalize_a2a3_pass_configs(target, factory.pass_configs)
    with tvm.transform.PassContext(opt_level=3, config=config):
        artifact = tilelang.lower(prim, target=target)
    assert 'extern "C" void call(' in artifact.kernel_source
