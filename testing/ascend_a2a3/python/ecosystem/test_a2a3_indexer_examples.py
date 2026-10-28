"""Indexer host/import contracts. No NPU allocation, properties, context or launch."""

import ast
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = json.loads((ROOT / "maint/ascend_a2a3/ecosystem_indexer_ports.json").read_text())
ENTRIES = MANIFEST["entries"]


def _functions(entry, names, namespace):
    path = ROOT / entry["destination_path"]
    tree = ast.parse(path.read_bytes())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def _load(entry):
    path = ROOT / entry["destination_path"]
    spec = importlib.util.spec_from_file_location("_indexer_contract_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: Path(e["source_path"]).stem)
def test_complete_original_definitions_and_committed_bytes(entry):
    path = ROOT / entry["destination_path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["destination_sha256"]
    functions = {n.name: n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef)}
    for name, expected in entry["original_function_AST_sha256"].items():
        assert hashlib.sha256(ast.dump(functions[name], include_attributes=False).encode()).hexdigest() == expected


@pytest.mark.parametrize("entry", ENTRIES)
def test_import_defers_cache_seed_compiler_and_device_operations(entry, monkeypatch):
    import builtins
    import tilelang
    import torch

    def forbidden(*args, **kwargs):
        raise AssertionError("module import reached a compiler, cache, RNG or device operation")

    for owner, names in [(tilelang, ["compile", "lower", "disable_cache"]), (torch, ["manual_seed", "randn", "empty", "zeros"])]:
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(get_device_properties=forbidden, set_device=forbidden, synchronize=forbidden), raising=False
    )
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch_npu":
            forbidden()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    module = _load(entry)
    assert callable(module.main) and callable(module.test_indexer)


@pytest.mark.parametrize("entry", ENTRIES)
def test_original_CPU_golden_relu_weights_and_topk_layout(entry):
    import torch

    ns = _functions(entry, ["index_golden"], {"torch": torch, "TOP_K": 2})
    q = torch.tensor([[[[[1.0, -2.0]]]]])
    k = torch.tensor([[[[1.0, 0.0]], [[2.0, 0.0]], [[-1.0, -2.0]], [[-1.0, 2.0]]]])
    weights = torch.tensor([[[[[3.0]]]]])
    score, indices = ns["index_golden"](q, k, weights)
    assert torch.equal(score, torch.tensor([[[[[1.0, 2.0, 3.0, 0.0]]]]]))
    assert indices.dtype == torch.int32 and torch.equal(indices, torch.tensor([[[[2, 1]]]], dtype=torch.int32))


@pytest.mark.parametrize("entry", ENTRIES)
def test_original_topk_comparison_is_multiset_not_order_or_set(entry):
    import torch

    compare = _functions(entry, ["count_mismatches_last_dim"], {"Counter": Counter})["count_mismatches_last_dim"]
    assert compare(torch.tensor([[1, 1, 2]]), torch.tensor([[2, 1, 1]])) == 0
    assert compare(torch.tensor([[1, 1, 2]]), torch.tensor([[1, 2, 2]])) == 2
    with pytest.raises(AssertionError, match="last dimension"):
        compare(torch.tensor([[1, 2]]), torch.tensor([[1]]))


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize(
    "values, expected", [([[1, 2]], "completely the same"), ([[2, 1]], "2 different elements"), ([[1]], "different shapes")]
)
def test_original_compare_diagnostics(entry, values, expected, capsys):
    import torch

    compare = _functions(entry, ["compare_tensors"], {"torch": torch})["compare_tensors"]
    assert compare(torch.tensor([[1, 2]]), torch.tensor(values)) is None
    assert expected in capsys.readouterr().out


class _ShapeOnly:
    """Route original host operations without allocating data or invoking devices."""

    def __init__(self, shape):
        self.shape = tuple(shape)

    def view(self, *shape):
        count = math.prod(self.shape)
        self.shape = tuple(count // -math.prod(shape) if x == -1 else x for x in shape)
        return self

    def half(self):
        return self

    float = half
    npu = half
    cpu = half

    def to(self, dtype):
        return self


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("successful", [True, False])
def test_original_CLI_request_shapes_print_and_failure_exit0(entry, successful, capsys):
    params = entry["original_cli_local_constants"]
    B, S1, N2, G, D, S2, K = [params[k] for k in ["B", "S1", "N2", "G", "D", "S2", "TOP_K"]]
    shapes, requests, calls, syncs = [], [], [], []
    output = _ShapeOnly((B, N2, S1, K))

    def randn(*shape):
        shapes.append(shape)
        return _ShapeOnly(shape)

    def factory(*args):
        requests.append(args)

        def kernel(*tensors):
            calls.append([x.shape for x in tensors])
            return output

        return kernel

    denominator = B * S1 * N2 * K
    mismatches = math.floor(denominator * 0.01) if successful else math.ceil(denominator * 0.01)
    ns = dict(
        params,
        indexer=factory,
        torch=SimpleNamespace(randn=randn, int32="int32", npu=SimpleNamespace(synchronize=lambda: syncs.append(True))),
        index_golden=lambda *args: (None, output),
        count_mismatches_last_dim=lambda *args: mismatches,
    )
    assert _functions(entry, ["test_indexer"], ns)["test_indexer"]() is None
    assert requests == [tuple(entry["original_factory_args"])]
    assert shapes == [(B, S1, N2, G, D), (B, S2, N2, D), (B, S1, N2, G, 1)]
    assert calls == [[(B, S1, N2, G * D), (B, S2, N2, D), (B, S1, N2, G, 1)]]
    assert syncs == [True, True]
    text = capsys.readouterr().out
    assert ("Test passed!" if successful else "Test failed! The precision is not correct!") in text


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("failure", [True, False])
def test_main_owns_cache_and_restores_exact_raw_enabled_bit(entry, enabled, failure, monkeypatch, tmp_path):
    class State:
        _enabled = enabled

        @classmethod
        def is_enabled(cls):
            return cls._enabled

    calls = []
    module = ModuleType("tilelang")
    module.env = SimpleNamespace(TILELANG_CACHE_DIR=str(tmp_path))
    module.disable_cache = lambda: setattr(State, "_enabled", False)
    env = ModuleType("tilelang.env")
    env.CacheState = State
    monkeypatch.setitem(sys.modules, "tilelang", module)
    monkeypatch.setitem(sys.modules, "tilelang.env", env)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path))

    def original_CLI():
        calls.append(("CLI", State._enabled))
        if failure:
            raise RuntimeError("original CLI failure")

    ns = _functions(
        entry,
        ["main"],
        dict(
            os=os,
            Path=Path,
            tilelang=module,
            torch=SimpleNamespace(manual_seed=lambda s: calls.append(("seed", s))),
            test_indexer=original_CLI,
        ),
    )
    if failure:
        with pytest.raises(RuntimeError, match="original CLI failure"):
            ns["main"]()
    else:
        assert ns["main"]() is None
    assert calls == [("seed", 2), ("CLI", False)] and State._enabled is enabled


@pytest.mark.a2a3_native
@pytest.mark.parametrize("entry", ENTRIES)
def test_original_full_frontend_parameter_and_dynamic_symbol_order(entry):
    from tilelang import tvm

    func = _load(entry).indexer.get_tir(*entry["original_factory_args"])
    restored = tvm.ir.load_json(tvm.ir.save_json(func))
    tvm.ir.assert_structural_equal(func, restored)
    assert [func.buffer_map[v].name for v in func.params] == ["Query", "KEY", "QK_RES", "WEIGHTS", "OUT"]
    assert [str(func.buffer_map[v].dtype) for v in func.params] == ["float16", "float16", "float32", "float32", "int32"]
    if entry["original_raw_target"] == "ascendc":
        buffers = [func.buffer_map[v] for v in func.params]
        assert [str(x) for x in buffers[0].shape] == ["B", "S1", "1", "2048"]
        assert [str(x) for x in buffers[1].shape] == ["B", "S2", "1", "64"]
        assert [str(x) for x in buffers[2].shape] == ["B", "1", "S1", "32", "S2"]
