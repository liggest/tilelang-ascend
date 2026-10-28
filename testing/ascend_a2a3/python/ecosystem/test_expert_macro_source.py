"""Complete original expert factories use the supported ordinary eager macros."""

import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import pytest
import tilelang
from tilelang import tvm


_ROOT = Path(__file__).resolve().parents[4]
_NAMESPACE = _ROOT / "examples/ascend_a2a3"
_CONTRACTS = json.loads((_NAMESPACE / "flash_attention/fa_opt/expert_macro_source_contracts.json").read_text())["rows"]


def _load(row):
    path = _NAMESPACE / row["source_path"].removeprefix("examples/")
    spec = importlib.util.spec_from_file_location("_a2a3_expert_macro_" + row["case"], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("row", _CONTRACTS, ids=lambda row: row["case"])
@pytest.mark.a2a3_native
@pytest.mark.usefixtures("isolated_a2a3_cache")
def test_expert_macro_complete_source_contract(row):
    from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs

    module = _load(row)
    factory = getattr(module, row["factory"])
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    with target:
        func = factory.get_tir(**row["kwargs"]) if hasattr(factory, "get_tir") else factory(**row["kwargs"])
    assert [
        {
            "name": var.name,
            "dtype": str(func.buffer_map[var].dtype) if var in func.buffer_map else str(var.dtype),
            "shape": [str(dim) for dim in func.buffer_map[var].shape] if var in func.buffer_map else [],
        }
        for var in func.params
    ] == row["frontend_params"]
    configs = normalize_a2a3_pass_configs(target, getattr(module, "pass_configs", getattr(module, "PASS_CONFIGS", {})))
    with tvm.transform.PassContext(opt_level=3, config=configs):
        artifact = tilelang.lower(func, target=target)
    calls = Counter()

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and isinstance(node.op, tvm.ir.Op):
            calls[node.op.name.replace("tirx.", "tir.")] += 1

    tvm.tirx.stmt_functor.post_order_visit(next(iter(artifact.device_mod.functions.values())).body, visit)
    assert dict(calls) == row["final_op_calls"]
    code = artifact.kernel_source
    allocations = re.findall(r"auto\s+(\w+)\s*=\s*(ascend_\w+)\.GetWithOffset<([^>]+)>\((\d+),\s*(\d+)\);", code)
    assert sorted([list(item) for item in allocations]) == row["allocation_views"]
    assert re.search(r'extern "C" void call\(([^\n]+)\)', code)[1] == row["host_prototype"]


@pytest.mark.parametrize("row", [_CONTRACTS[0], _CONTRACTS[1], _CONTRACTS[3]], ids=lambda row: row["case"])
def test_expert_macro_import_has_no_cli_or_cache_side_effect(row, monkeypatch):
    import argparse
    import torch

    def forbidden(*args, **kwargs):
        raise AssertionError("import attempted CLI, cache state, compilation or default device mutation")

    for owner, name in [
        (argparse.ArgumentParser, "parse_args"),
        (tilelang, "disable_cache"),
        (tilelang, "compile"),
        (torch, "set_default_device"),
    ]:
        monkeypatch.setattr(owner, name, forbidden)
    _load(row)
