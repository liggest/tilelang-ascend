"""SFA1 complete host/frontend contracts; no actual device or large allocation."""

import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = json.loads((ROOT / "maint/ascend_a2a3/ecosystem_sparse_regular_ports.json").read_text())
ENTRIES = MANIFEST["entries"]


def _tree(entry):
    return ast.parse((ROOT / entry["destination_path"]).read_bytes())


def _selected(entry, names, namespace):
    nodes = [n for n in _tree(entry).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), entry["destination_path"], "exec"), namespace)
    return namespace


def _load(entry):
    spec = importlib.util.spec_from_file_location("_SFA1_host_" + Path(entry["source_path"]).stem, ROOT / entry["destination_path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


@pytest.mark.parametrize("entry", ENTRIES)
def test_complete_original_factory_reference_and_CLI_operations(entry):
    functions = {n.name: n for n in _tree(entry).body if isinstance(n, ast.FunctionDef)}
    for name, expected in entry["original_definition_AST_sha256"].items():
        assert _digest(functions[name]) == expected
    assert _digest(ast.Module(body=functions["_run_original_cli"].body, type_ignores=[])) == entry["original_CLI_operations_AST_sha256"]
    assert hashlib.sha256((ROOT / entry["destination_path"]).read_bytes()).hexdigest() == entry["destination_sha256"]


@pytest.mark.parametrize("entry", ENTRIES)
def test_import_defers_original_top_level_compile_data_RNG_cache_and_device(entry, monkeypatch):
    import builtins
    import tilelang
    import torch

    def forbidden(*args, **kwargs):
        raise AssertionError("SFA1 import executed original eager compiler/data/cache/device state")

    for owner, names in [
        (tilelang, ["compile", "lower", "disable_cache"]),
        (torch, ["manual_seed", "set_default_device", "randn", "full", "randperm"]),
    ]:
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch_npu":
            forbidden()
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    module = _load(entry)
    assert callable(module.main)
    if entry["original_core_num"] is not None:
        assert module.core_num == 24  # Literal source constant, not a device-property guess.


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("q_start", [1, None])
def test_original_CPU_reference_sentinel_and_compressed_causal_mask(entry, q_start):
    import torch

    reference = _selected(entry, ["ref_sparse_attention_fwd_interface"], {"torch": torch})["ref_sparse_attention_fwd_interface"]
    q = torch.zeros((1, 2, 2, 576))
    kv = torch.zeros((1, 4, 1, 576))
    kv[0, :, 0, 0] = torch.tensor([0.0, 2.0, 4.0, 8.0])
    indices = torch.tensor([[[[0, 2, 4]], [[1, 2, 4]]]], dtype=torch.int32)
    actual = reference(q, kv, indices, q_start, kv_stride=1)
    dtype = getattr(torch, entry["original_dtype"])
    expected = torch.zeros((1, 2, 2, 512), dtype=dtype)
    expected[0, 0, :, 0] = 0 if q_start == 1 else 2
    expected[0, 1, :, 0] = 3
    assert actual.dtype == dtype and torch.equal(actual, expected)


class _ShapeOnly:
    def __init__(self, shape, assignments=None):
        self.shape = tuple(shape)
        self.assignments = assignments

    def __setitem__(self, key, value):
        self.assignments.append((key[:3], key[3].stop, len(value)))


class _Permutation:
    def __init__(self, count):
        self.count = count

    def __getitem__(self, item):
        return _Permutation(min(self.count, item.stop))

    def __len__(self):
        return self.count


@pytest.mark.parametrize("entry", ENTRIES)
def test_original_full_CLI_data_shapes_rng_schedule_kernel_reference_and_assert_dispatch(entry, capsys):
    p = entry["original_runtime_shapes"]
    B, S, SKV, H, HKV, DQK, DV, K = [p[k] for k in ["B", "S", "SKV", "H", "HKV", "DQK", "DV", "topk"]]
    calls, allocations, assignments, permutes, assertions = [], [], [], [], []
    output = _ShapeOnly((B, S, H, DV))

    def randn(shape, **kwargs):
        allocations.append((shape, kwargs))
        return _ShapeOnly(shape)

    def full(shape, fill, **kwargs):
        allocations.append((shape, kwargs, fill))
        return _ShapeOnly(shape, assignments)

    def randperm(count):
        permutes.append(count)
        return _Permutation(count)

    def factory(**kwargs):
        assert kwargs == entry["original_factory_kwargs"]
        calls.append("compile")

        def kernel(*args):
            calls.append(("runtime", [a.shape for a in args]))
            return output

        return kernel

    def reference(*args):
        calls.append(("reference", [a.shape for a in args[:3]], args[3:]))
        return output

    def assert_close(expected, actual, **kwargs):
        assert expected.shape == actual.shape == (B, S, H, DV)
        assertions.append(kwargs)

    torch = SimpleNamespace(
        manual_seed=lambda seed: calls.append(("seed", seed)),
        float16="float16",
        bfloat16="bfloat16",
        int32="int32",
        randn=randn,
        full=full,
        randperm=randperm,
        npu=SimpleNamespace(synchronize=lambda: calls.append("sync")),
        testing=SimpleNamespace(assert_close=assert_close),
    )
    ns = dict(
        torch=torch,
        tilelang=SimpleNamespace(disable_cache=lambda: calls.append("disable")),
        sparse_attention_fwd=factory,
        ref_sparse_attention_fwd_interface=reference,
    )
    assert _selected(entry, ["_run_original_cli"], ns)["_run_original_cli"]() is None
    assert calls == [
        ("seed", 0),
        "disable",
        "compile",
        "sync",
        ("runtime", [(B, S, H, DQK), (B, SKV, HKV, DQK), (B, S, HKV, K)]),
        "sync",
        ("reference", [(B, S, H, DQK), (B, SKV, HKV, DQK), (B, S, HKV, K)], (28672, 1)),
        "sync",
    ]
    assert allocations == [
        ((B, S, H, DQK), {"dtype": entry["original_dtype"]}),
        ((B, SKV, HKV, DQK), {"dtype": entry["original_dtype"]}),
        ((B, S, HKV, K), {"dtype": "int32"}, SKV),
    ]
    assert permutes == [28672 + t for b in range(B) for t in range(S) for h in range(HKV)]
    assert assignments == [((b, t, h), 2048, 2048) for b in range(B) for t in range(S) for h in range(HKV)]
    assert assertions == [entry["original_golden"]]
    assert capsys.readouterr().out.endswith("Test Passed!\n")


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("failure", [True, False])
def test_main_scopes_device_and_restores_cache_after_original_assert_or_success(entry, enabled, failure, monkeypatch, tmp_path):
    class State:
        _enabled = enabled

        @classmethod
        def is_enabled(cls):
            return cls._enabled

    calls = []
    module = ModuleType("tilelang")
    module.env = SimpleNamespace(TILELANG_CACHE_DIR=str(tmp_path))
    env = ModuleType("tilelang.env")
    env.CacheState = State
    for name, value in {"tilelang": module, "tilelang.env": env, "torch_npu": ModuleType("torch_npu")}.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path))

    @contextmanager
    def device(name):
        calls.append(("enter", name))
        try:
            yield
        finally:
            calls.append(("leave", name))

    def original():
        State._enabled = False
        calls.append("original")
        if failure:
            raise AssertionError("original golden")

    ns = _selected(
        entry, ["main"], dict(os=os, Path=Path, tilelang=module, torch=SimpleNamespace(device=device), _run_original_cli=original)
    )
    if failure:
        with pytest.raises(AssertionError, match="original golden"):
            ns["main"]()
    else:
        assert ns["main"]() is None
    assert State._enabled is enabled and calls == [("enter", "npu"), "original", ("leave", "npu")]


@pytest.mark.a2a3_native
@pytest.mark.parametrize("entry", ENTRIES)
def test_complete_original_frontend_buffers_workspace_dtype_and_dynamic_symbols(entry):
    from tilelang import tvm

    module = _load(entry)
    assert module.sparse_attention_fwd.out_idx == [3]
    assert module.sparse_attention_fwd.workspace_idx == [4, 5, 6, 7, 8]
    func = module.sparse_attention_fwd.get_tir(**entry["original_factory_kwargs"])
    tvm.ir.assert_structural_equal(func, tvm.ir.load_json(tvm.ir.save_json(func)))
    assert len(func.params) == 9
    assert [str(func.buffer_map[v].dtype) for v in func.params[:2]] == [entry["original_dtype"]] * 2
    if entry["original_core_num"] is not None:
        shape = [str(x) for x in func.buffer_map[func.params[0]].shape]
        assert shape == ["batch", "seq_len", "128", "576"]
