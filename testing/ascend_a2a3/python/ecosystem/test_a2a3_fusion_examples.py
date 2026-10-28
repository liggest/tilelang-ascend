"""Full original F1 factories and import/cache boundaries, without NPU execution."""

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


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda row: row["source_path"])
def test_cli_requires_an_explicit_cache_namespace(entry, monkeypatch):
    module = _load(entry)
    monkeypatch.delenv("TILELANG_CACHE_DIR", raising=False)
    with pytest.raises(RuntimeError, match="dedicated TILELANG_CACHE_DIR"):
        module.main()


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda row: row["source_path"])
def test_cli_cannot_clear_a_conflicting_caller_cache(entry, monkeypatch, tmp_path):
    from tilelang import env
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

    module = _load(entry)
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "owned"))
    monkeypatch.setattr(vars(type(env))["TILELANG_CACHE_DIR"], "_forced_value", str(tmp_path / "caller"))

    def forbidden(*args):
        raise AssertionError("caller cache was cleared")

    monkeypatch.setattr(A2A3CythonKernelCache, "clear_cache", forbidden)
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
    configs = normalize_a2a3_pass_configs(target, getattr(module, "pass_configs", {}))
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
