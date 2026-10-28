"""Original experiment examples: import/runtime isolation and full source requests."""

import argparse
import builtins
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / "maint/ascend_a2a3/ecosystem_experiment_rest_ports.json").read_text())
ENTRIES = MANIFEST["entries"]
REQUESTS = MANIFEST["factory_requests"]
BY_SOURCE = {entry["source_path"]: entry for entry in ENTRIES}


def _load(entry):
    path = ROOT / entry["destination_path"]
    spec = importlib.util.spec_from_file_location("_experiment_" + path.stem, path)
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
    if (
        entry["source_runtime"]["set_default_device"]
        or entry["source_runtime"].get("host_availability_gate")
        or entry["source_runtime"].get("constructs_device_object_before_factory")
    ):
        # Exercise the public CPU context stack without selecting an NPU.
        class HostTorch:
            def device(self, name):
                assert name == "npu"
                return torch.device("cpu")

            def __getattr__(self, name):
                if name == "npu" and entry["source_runtime"].get("host_availability_gate"):
                    return SimpleNamespace(is_available=lambda: True)
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

    class StopFactory(BaseException):
        pass

    def stop(*args, **kwargs):
        events.append("inputs" if entry["source_runtime"].get("inputs_before_factory") else "factory")
        assert CacheState.is_enabled() == (False if entry["source_runtime"]["disable_cache"] else enabled)
        raise StopFactory

    request = next(r for r in REQUESTS if r["source_path"] == entry.get("source_request_owner", entry["source_path"]))
    monkeypatch.setattr(module, request["factory"], stop)
    if entry["source_runtime"].get("inputs_before_factory"):
        # Preserve original input allocation before factory construction; fail
        # at the first input boundary without allocating any device tensor.
        monkeypatch.setattr(torch, entry["source_runtime"].get("input_constructor", "randn"), stop)
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
        expected = ["registration"]
        if entry["source_runtime"]["disable_cache"]:
            expected.append("disable")
        elif entry["source_runtime"]["clear_cache"]:
            expected.extend(["clear"] * entry["source_runtime"]["clear_count"])
        expected.append("inputs" if entry["source_runtime"].get("inputs_before_factory") else "factory")
        assert events == expected
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
    if case.get("jit_target_override"):
        factory.target = case["jit_target_override"]
    dialect = "pto" if factory.target == "pto" else "ascendc"
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": [dialect, "ascend"]})
    with target:
        args = [tuple(value) if i in case.get("tuple_arg_indices", []) else value for i, value in enumerate(case["args"])]
        if case.get("expected_frontend_error"):
            assert case["expected_frontend_error"] == "AssertionError"
            with pytest.raises(AssertionError):
                factory.get_tir(*args, **case["kwargs"])
            return
        prim = factory.get_tir(*args, **case["kwargs"])
    config = normalize_a2a3_pass_configs(target, factory.pass_configs)
    with tvm.transform.PassContext(opt_level=3, config=config):
        artifact = tilelang.lower(prim, target=target)
    assert 'extern "C" void call(' in artifact.kernel_source


@pytest.mark.parametrize(
    "argv,expected",
    [
        ([], (1024, 128, 42, 256)),
        (["--m", "129", "--block_size", "64", "--seed", "17"], (129, 64, 17, 128)),
    ],
)
def test_random_original_parser_reconfigures_factory_globals(argv, expected, tmp_path, monkeypatch):
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

    entry = next(e for e in ENTRIES if "/random_1d/" in e["source_path"])
    module = _load(entry)
    assert (module.M, module.BLOCK_SIZE, module.SEED, module.TOTAL_BLOCK_SIZE) == (1024, 128, 42, 256)
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "owned"))
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(A2A3CythonKernelCache, "clear_cache", lambda *args: None)
    monkeypatch.setattr(sys, "argv", [entry["destination_path"], *argv])

    class StopFactory(BaseException):
        pass

    def original_factory_boundary(M, block_size, seed, a, c):
        assert (M, block_size, seed, module.TOTAL_BLOCK_SIZE) == expected
        assert (a, c) == (1103515245, 12345)
        raise StopFactory

    monkeypatch.setattr(module, "random_1d", original_factory_boundary)
    with pytest.raises(StopFactory):
        module.main()


def test_mla_decode_layered_uses_exact_private_sibling():
    entry = next(e for e in ENTRIES if e["batch"] == "E4a" and "/test_" in e["source_path"])
    module = _load(entry)
    assert Path(module._mla_module.__file__).resolve() == (ROOT / BY_SOURCE[entry["source_request_owner"]]["destination_path"]).resolve()
    assert module._col_indices_cpu is None


def test_mla_decode_lazy_indices_real_CPU_values(tmp_path):
    import subprocess

    entry = next(e for e in ENTRIES if e["batch"] == "E4a" and "/test_" in e["source_path"])
    script = ROOT / entry["destination_path"]
    # Explicit CPU-only metadata probe in a separate process; parent tests keep
    # all real tensor factories forbidden and do not execute kernel/golden.
    code = (
        "import os,ctypes\nif os.environ.get('SKIP_LOADING_TILELANG_SO')=='1':\n import tvm\n ctypes.CDLL(os.path.join(os.environ['TVM_LIBRARY_PATH'],'libtilelang.so'),mode=ctypes.RTLD_GLOBAL)\nimport importlib.util,torch; s=importlib.util.spec_from_file_location('mla_cpu_metadata',"
        + repr(str(script))
        + "); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert m._col_indices_cpu is None; before=torch.random.get_rng_state().clone(); x=m._get_col_indices_cpu(); assert x.device.type=='cpu' and x.dtype==torch.float32 and x.tolist()==list(range(m.BLOCK_N)); assert m._get_col_indices_cpu() is x and torch.equal(before,torch.random.get_rng_state()); print('CPU indices exact, cached, RNG unchanged')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=dict(__import__("os").environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CPU indices exact, cached, RNG unchanged" in result.stdout
