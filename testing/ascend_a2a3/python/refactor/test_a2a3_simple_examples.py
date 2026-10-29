"""Original simple examples: import safety and complete source-only requests."""

import argparse
import builtins
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[4]
MANIFEST = json.loads((REPO / "maint/ascend_a2a3/ecosystem_simple_ports.json").read_text())
ENTRIES = MANIFEST["entries"]
REQUESTS = MANIFEST["factory_requests"]
BY_SOURCE = {entry["source_path"]: entry for entry in ENTRIES}


def _load(entry):
    path = REPO / entry["destination_path"]
    spec = importlib.util.spec_from_file_location("_simple_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e["source_path"])
def test_import_does_not_run_original_runtime(entry, monkeypatch):
    import tilelang
    import torch
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

    def forbidden(*args, **kwargs):
        raise AssertionError("import executed the example runtime")

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", forbidden)
    monkeypatch.setattr(A2A3CythonKernelCache, "clear_cache", forbidden)
    for obj, names in [
        (tilelang, ["disable_cache", "compile", "lower"]),
        (torch, ["manual_seed", "set_default_device", "randn", "arange", "empty", "zeros"]),
    ]:
        for name in names:
            monkeypatch.setattr(obj, name, forbidden)
    previous_npu_module = sys.modules.get("torch_npu")
    module = _load(entry)
    assert callable(module.main)
    assert sys.modules.get("torch_npu") is previous_npu_module


@pytest.mark.parametrize("entry", [e for e in ENTRIES if e["source_runtime"]["clear_cache"]], ids=lambda e: e["source_path"])
def test_cache_clearing_cli_requires_an_explicit_directory(entry, monkeypatch):
    module = _load(entry)
    monkeypatch.delenv("TILELANG_CACHE_DIR", raising=False)
    with pytest.raises(RuntimeError, match="dedicated TILELANG_CACHE_DIR"):
        module.main()


@pytest.mark.parametrize("entry", [e for e in ENTRIES if e["source_runtime"]["clear_cache"]], ids=lambda e: e["source_path"])
def test_conflicting_caller_cache_override_cannot_be_cleared(entry, tmp_path, monkeypatch):
    from tilelang import env
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

    module = _load(entry)
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "owned"))
    descriptor = vars(type(env))["TILELANG_CACHE_DIR"]
    monkeypatch.setattr(descriptor, "_forced_value", str(tmp_path / "caller-other"))

    def forbidden_clear(*args):
        raise AssertionError("conflicting caller cache was cleared")

    monkeypatch.setattr(A2A3CythonKernelCache, "clear_cache", forbidden_clear)
    with pytest.raises(RuntimeError, match="resolved cache differs"):
        module.main()


@pytest.mark.parametrize(
    "entry",
    [e for e in ENTRIES if e["source_runtime"]["disable_cache"] or e["source_runtime"]["set_default_device"]],
    ids=lambda e: e["source_path"],
)
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("caller_context", [False, True])
def test_runtime_failure_restores_cache_and_default_device(entry, enabled, caller_context, monkeypatch):
    import torch
    from tilelang.env import CacheState
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

    module = _load(entry)
    monkeypatch.setattr(sys, "argv", [entry["destination_path"]])
    monkeypatch.setenv("TILELANG_CACHE_DIR", "/tmp/simple-fixture-unused-cache")
    monkeypatch.setattr(A2A3CythonKernelCache, "clear_cache", lambda *args: None)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    requested_devices = []
    npu_imports = []
    original_import = builtins.__import__

    def observe_import(name, *args, **kwargs):
        if name == "torch_npu":
            npu_imports.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", observe_import)

    class HostTorch:
        # Exercise the real public CPU context stack without requiring or
        # allocating on an NPU. Only the host isolation boundary is under test.
        def device(self, name):
            requested_devices.append(name)
            assert name == "npu"
            return torch.device("cpu")

        def manual_seed(self, *args):
            return None

        def randn(self, *args, **kwargs):
            return object()

        def set_default_device(self, *args):
            raise AssertionError("must scope the default without replacing global state")

        def __getattr__(self, name):
            return getattr(torch, name)

    monkeypatch.setattr(module, "torch", HostTorch())

    class StopFactory(Exception):
        pass

    def stop(*args, **kwargs):
        modes = torch.overrides._get_current_function_mode_stack()
        assert len(modes) == len(before_modes) + 1
        assert str(modes[-1].device) == "cpu"
        raise StopFactory

    request = next(r for r in REQUESTS if r["source_path"] == entry["source_path"])
    monkeypatch.setattr(module, request["factory"], stop)
    before = CacheState.is_enabled()
    try:
        CacheState.enable() if enabled else CacheState.disable()
        with torch.device("cpu") if caller_context else nullcontext():
            before_modes = torch.overrides._get_current_function_mode_stack()
            with pytest.raises(StopFactory):
                module.main()
            assert CacheState.is_enabled() == enabled
            after_modes = torch.overrides._get_current_function_mode_stack()
            assert len(after_modes) == len(before_modes)
            assert all(a is b for a, b in zip(before_modes, after_modes))
        assert requested_devices == ["npu"]
        assert npu_imports == ["torch_npu"]
    finally:
        CacheState.enable() if before else CacheState.disable()


@pytest.mark.parametrize("case", REQUESTS, ids=lambda r: r["id"])
def test_complete_original_factory_request_lowers_without_device(case, monkeypatch):
    import tilelang
    from tilelang import tvm
    from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs

    for name in ("tl.transform.A2A3HostLegalize", "target.build.tilelang_ascend_a2a3", "target.build.tilelang_pto_a2a3"):
        if tvm.ffi.get_global_func(name, allow_missing=True) is None:
            pytest.skip("requires USE_ASCEND=ON with the native A2/A3 pipeline")
    monkeypatch.setenv("TL_PLATFORM", "A3")
    module = _load(BY_SOURCE[case["source_path"]])
    factory = getattr(module, case["factory"])
    dialect = "pto" if factory.target == "pto" else "ascendc"
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": [dialect, "ascend"]})
    with target:
        prim = factory.get_tir(*case["args"], **case["kwargs"])
    configs = normalize_a2a3_pass_configs(target, factory.pass_configs)
    with tvm.transform.PassContext(opt_level=3, config=configs):
        artifact = tilelang.lower(prim, target=target)
    assert artifact.kernel_source
    assert 'extern "C" void call(' in artifact.kernel_source
