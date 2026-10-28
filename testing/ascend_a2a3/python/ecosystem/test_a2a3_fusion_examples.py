"""Complete original fusion and MoE factories and import/cache boundaries, without NPU execution."""

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = json.loads((ROOT / "maint/ascend_a2a3/ecosystem_fusion_ports.json").read_text())
ENTRIES = MANIFEST["entries"]
_CACHE_CLEAR_ENTRIES = [entry for entry in ENTRIES if entry["source_runtime"]["clear_cache"]]
REQUESTS = MANIFEST["factory_requests"]
CONTRACTS = {row["id"]: row for row in MANIFEST["source_contracts"]}


def _load(entry):
    path = ROOT / entry["destination_path"]
    spec = importlib.util.spec_from_file_location("_fusion_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda row: row["source_path"])
def test_import_has_no_cli_device_or_cache_side_effects(entry, monkeypatch):
    import tilelang
    import torch
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

    def forbidden(*args, **kwargs):
        raise AssertionError("import executed the example runtime")

    for owner, names in [
        (argparse.ArgumentParser, ["parse_args"]),
        (A2A3CythonKernelCache, ["clear_cache"]),
        (tilelang, ["compile", "lower", "disable_cache"]),
        (torch, ["manual_seed", "set_default_device", "randn", "randint", "empty", "zeros"]),
    ]:
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)
    prior_npu_module = sys.modules.get("torch_npu")
    assert callable(_load(entry).main)
    assert sys.modules.get("torch_npu") is prior_npu_module


@pytest.mark.parametrize("entry", _CACHE_CLEAR_ENTRIES, ids=lambda row: row["source_path"])
def test_cli_requires_an_explicit_cache_namespace(entry, monkeypatch):
    import builtins

    module = _load(entry)
    monkeypatch.delenv("TILELANG_CACHE_DIR", raising=False)
    original_import = builtins.__import__

    def no_device_import(name, *args, **kwargs):
        if name == "torch_npu":
            raise AssertionError("cache guard reached device registration")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_device_import)
    with pytest.raises(RuntimeError, match="dedicated TILELANG_CACHE_DIR"):
        module.main()


@pytest.mark.parametrize("entry", _CACHE_CLEAR_ENTRIES, ids=lambda row: row["source_path"])
def test_cli_cannot_clear_a_conflicting_caller_cache(entry, monkeypatch, tmp_path):
    import builtins

    from tilelang import env
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

    module = _load(entry)
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "owned"))
    monkeypatch.setattr(vars(type(env))["TILELANG_CACHE_DIR"], "_forced_value", str(tmp_path / "caller"))

    def forbidden(*args):
        raise AssertionError("caller cache was cleared")

    monkeypatch.setattr(A2A3CythonKernelCache, "clear_cache", forbidden)
    original_import = builtins.__import__

    def no_device_import(name, *args, **kwargs):
        if name == "torch_npu":
            raise AssertionError("cache guard reached device registration")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_device_import)
    with pytest.raises(RuntimeError, match="resolved cache differs"):
        module.main()


@pytest.mark.parametrize("case", REQUESTS, ids=lambda row: row["id"])
@pytest.mark.a2a3_native
@pytest.mark.usefixtures("isolated_a2a3_cache")
def test_complete_original_factory_source_contract(case):
    import tilelang
    from tilelang import tvm
    from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs

    entry = next(row for row in ENTRIES if row["source_path"] == case["source_path"])
    module = _load(entry)
    factory = getattr(module, case["factory"])
    contract = CONTRACTS[case["id"]]
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    with target:
        function = factory.get_tir(*case["args"], **case["kwargs"])
    params = [
        {
            "name": var.name,
            "dtype": str(function.buffer_map[var].dtype) if var in function.buffer_map else str(var.dtype),
            "shape": [str(dim) for dim in function.buffer_map[var].shape] if var in function.buffer_map else [],
        }
        for var in function.params
    ]
    assert params == contract["frontend_params"]
    assert factory.compile_flags == contract["original_compile_flags"]
    configs = normalize_a2a3_pass_configs(target, getattr(module, case.get("pass_configs_attribute", "pass_configs"), {}))
    with tvm.transform.PassContext(opt_level=3, config=configs):
        artifact = tilelang.lower(function, target=target)
    calls = Counter()

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and isinstance(node.op, tvm.ir.Op):
            calls[node.op.name.replace("tirx.", "tir.")] += 1

    tvm.tirx.stmt_functor.post_order_visit(next(iter(artifact.device_mod.functions.values())).body, visit)
    assert dict(calls) == contract["final_op_calls"]
    allocations = re.findall(r"auto\s+(\w+)\s*=\s*(ascend_\w+)\.GetWithOffset<([^>]+)>\((\d+),\s*(\d+)\);", artifact.kernel_source)
    assert sorted([list(item) for item in allocations]) == contract["allocation_views"]
    assert re.search(r'extern "C" void call\(([^\n]+)\)', artifact.kernel_source)[1] == contract["host_prototype"]


_MHC_POST = next(entry for entry in ENTRIES if entry["source_path"] == "examples/mhc_post/example_mhc_post.py")


@pytest.mark.parametrize("case", _MHC_POST["original_CLI_cases"])
def test_original_mhc_post_block_selection(case):
    module = _load(_MHC_POST)
    assert module._select_h_blk(case["h"], case["hc"]) == case["h_blk"]


@pytest.mark.parametrize("enabled", [False, True])
def test_mhc_cli_failure_restores_exact_caller_cache_state(enabled, monkeypatch):
    from types import ModuleType

    from tilelang import env
    from tilelang.env import CacheState

    module = _load(_MHC_POST)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(CacheState, "_enabled", enabled)
    descriptor = vars(type(env))["TILELANG_CACHE_DIR"]
    original_descriptor_override = descriptor._forced_value

    class StopBeforeDevice(RuntimeError):
        pass

    def original_test_boundary():
        assert CacheState._enabled is False
        raise StopBeforeDevice

    monkeypatch.setattr(module, "test", original_test_boundary)
    with pytest.raises(StopBeforeDevice):
        module.main()
    assert CacheState._enabled is enabled
    assert descriptor._forced_value is original_descriptor_override


_MOE_AUX = next(entry for entry in ENTRIES if entry["source_path"].endswith("/moe_aux_fi.py"))
_MOE_TOPK = next(entry for entry in ENTRIES if entry["source_path"].endswith("/moe_topk_gate.py"))


@pytest.mark.parametrize("full", [False, True])
def test_aux_complete_original_parameter_domains(full, monkeypatch):
    if full:
        monkeypatch.setenv("TK_FULL_TEST", "1")
    else:
        monkeypatch.delenv("TK_FULL_TEST", raising=False)
    module = _load(_MOE_AUX)
    expected = _MOE_AUX["original_pytest_matrix"]["full" if full else "default"]
    actual = module.generate_test_params()
    assert actual == expected["ordered_params"]
    assert [module.make_param_id(params) for params in actual] == expected["ids"]
    assert len(actual) == (158 if full else 24)
    if full:
        # The original num_aux_topk loop deliberately yields the zero case twice.
        assert actual[0] == actual[1] and actual[0]["num_send_tokens"] == 0


def test_topk_complete_original_forward_backward_parameter_domains():
    module = _load(_MOE_TOPK)
    expected = _MOE_TOPK["original_pytest_matrix"]
    actual = module.generate_test_params()
    assert actual == expected["ordered_params"]
    assert [module.make_param_id(params) for params in actual] == expected["ids"]
    assert len(actual) == 20
    for name in expected["test_functions"]:
        function = getattr(module, name)
        mark = next(mark for mark in function.pytestmark if mark.name == "parametrize")
        assert mark.args[0] == "params" and mark.args[1] == actual


@pytest.mark.parametrize("entry", [_MOE_AUX, _MOE_TOPK], ids=lambda row: Path(row["source_path"]).stem)
@pytest.mark.parametrize("exit_code", [0, 1, 2])
def test_moe_original_pytest_CLI_exit_and_success_marker(entry, exit_code, monkeypatch, tmp_path, capsys):
    from types import ModuleType

    from tilelang import env
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

    module = _load(entry)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    owned = tmp_path / "dedicated-cache"
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(owned))
    monkeypatch.setattr(vars(type(env))["TILELANG_CACHE_DIR"], "_forced_value", str(owned))
    clear_calls = []
    monkeypatch.setattr(A2A3CythonKernelCache, "clear_cache", lambda self: clear_calls.append(True))
    pytest_calls = []

    def original_pytest_boundary(argv):
        pytest_calls.append(argv)
        return exit_code

    monkeypatch.setattr(module.pytest, "main", original_pytest_boundary)
    with pytest.raises(SystemExit) as result:
        module.main()
    assert result.value.code == exit_code
    assert pytest_calls == [[module.__file__, "-v"]]
    assert len(clear_calls) == int(entry["source_runtime"]["clear_cache"])
    assert ("Kernel Output Match!" in capsys.readouterr().out) is (exit_code == 0)
