"""Complete HISA source/host contracts; no NPU allocations, properties or launches."""

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
MANIFEST = json.loads((ROOT / "maint/ascend_a2a3/ecosystem_hisa_ports.json").read_text())
ENTRIES = MANIFEST["entries"]


def _tree(entry):
    return ast.parse((ROOT / entry["destination_path"]).read_bytes())


def _selected(entry, names, namespace):
    nodes = [n for n in _tree(entry).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), entry["destination_path"], "exec"), namespace)
    return namespace


def _load(entry):
    spec = importlib.util.spec_from_file_location("_HISA_host_" + Path(entry["source_path"]).stem, ROOT / entry["destination_path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


@pytest.mark.parametrize("entry", ENTRIES)
def test_complete_original_definitions_parser_and_CLI_body(entry):
    functions = {n.name: n for n in _tree(entry).body if isinstance(n, ast.FunctionDef)}
    for name, expected in entry["original_definition_AST_sha256"].items():
        assert _digest(functions[name]) == expected
    assert _digest(ast.Module(body=functions["_parse_args"].body[:-1], type_ignores=[])) == entry["original_parser_AST_sha256"]
    assert (
        _digest(ast.Module(body=functions["_run_original_cli"].body, type_ignores=[]))
        == entry["original_CLI_after_default_device_AST_sha256"]
    )
    assert hashlib.sha256((ROOT / entry["destination_path"]).read_bytes()).hexdigest() == entry["destination_sha256"]


@pytest.mark.parametrize("entry", ENTRIES)
def test_original_no_argument_parser_and_help_before_device_setup(entry, monkeypatch, capsys):
    import argparse

    namespace = _selected(entry, ["_parse_args"], {"argparse": argparse})
    monkeypatch.setattr(sys, "argv", ["original.py"])
    args = namespace["_parse_args"]()
    assert vars(args) == {key.removeprefix("--"): value for key, value in entry["original_CLI_defaults"].items()}
    monkeypatch.setattr(sys, "argv", ["original.py", "--help"])
    with pytest.raises(SystemExit) as result:
        namespace["_parse_args"]()
    assert result.value.code == 0 and "--topk" in capsys.readouterr().out


@pytest.mark.parametrize("entry", ENTRIES)
def test_original_CLI_dispatch_all_parameters_and_success_marker(entry, capsys):
    calls = []
    params = {key.removeprefix("--"): value for key, value in entry["original_CLI_defaults"].items()}
    ns = dict(
        torch=SimpleNamespace(manual_seed=lambda seed: calls.append(("seed", seed))),
        tilelang=SimpleNamespace(disable_cache=lambda: calls.append("disable")),
        get_npu_core_num=lambda: 20,
    )
    test_name = "test_block_sparse_mqa_attn" if entry is ENTRIES[0] else "test_paged_block_sparse_mqa_attn"
    ns[test_name] = lambda **kwargs: calls.append(("dispatch", kwargs))
    _selected(entry, ["_run_original_cli"], ns)["_run_original_cli"](SimpleNamespace(**params))
    expected = dict(params)
    if entry is ENTRIES[0]:
        expected["grid_size"] = 20
    else:
        expected.pop("dtype")  # The original paged CLI never forwards this option.
    assert calls == [("seed", 42), "disable", ("dispatch", expected)]
    assert capsys.readouterr().out.endswith("Kernel Output Match!\n")


@pytest.mark.parametrize("cube", [20, 24, "32"])
def test_original_core_query_requires_actual_positive_property_without_fallback(cube, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    calls = []

    def query(index):
        calls.append(index)
        return SimpleNamespace(cube_core_num=cube)

    f = _selected(ENTRIES[0], ["get_npu_core_num"], {"torch": SimpleNamespace(npu=SimpleNamespace(get_device_properties=query))})[
        "get_npu_core_num"
    ]
    assert f() == int(cube) and calls == [0]


@pytest.mark.parametrize("properties", [SimpleNamespace(), SimpleNamespace(cube_core_num=0), SimpleNamespace(cube_core_num=-1)])
def test_original_core_query_diagnoses_absent_nonpositive_property(properties, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    f = _selected(
        ENTRIES[0], ["get_npu_core_num"], {"torch": SimpleNamespace(npu=SimpleNamespace(get_device_properties=lambda index: properties))}
    )["get_npu_core_num"]
    with pytest.raises(RuntimeError, match="could not determine NPU cube_core_num"):
        f()


def test_original_core_query_failure_is_not_replaced_by_a_guessed_platform(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))

    def unavailable(index):
        raise ValueError("original runtime property query unavailable")

    f = _selected(ENTRIES[0], ["get_npu_core_num"], {"torch": SimpleNamespace(npu=SimpleNamespace(get_device_properties=unavailable))})[
        "get_npu_core_num"
    ]
    with pytest.raises(ValueError, match="original runtime"):
        f()


def test_original_CPU_block_reference_relu_weights_and_inclusive_exclusive_mask():
    import torch

    ns = _selected(ENTRIES[0], ["ref_block_sparse_mqa_attn"], {"torch": torch})
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
    k = torch.tensor([[1.0, -1.0], [2.0, 3.0], [-1.0, 4.0], [2.0, -2.0]])
    actual = ns["ref_block_sparse_mqa_attn"](
        q, k, torch.tensor([[0, 1], [1, 0]]), 2, torch.tensor([[1.0, 2.0], [1.0, 1.0]]), torch.tensor([0, 1]), torch.tensor([3, 4])
    )
    assert torch.equal(actual, torch.tensor([[[1.0, 8.0], [8.0, -float("inf")]], [[4.0, 2.0], [-float("inf"), 5.0]]]))


def test_original_CPU_paged_reference_uses_physical_table_and_context_tail():
    import torch

    ns = _selected(ENTRIES[1], ["ref_paged_block_sparse_mqa_attn"], {"torch": torch})
    q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    kv = torch.tensor([[[[1.0, -1.0]], [[2.0, 3.0]]], [[[4.0, 0.0]], [[-1.0, 4.0]]]])
    actual = ns["ref_paged_block_sparse_mqa_attn"](
        q, kv, torch.tensor([[[0, 1]]]), 2, torch.tensor([[[1.0, 2.0]]]), torch.tensor([3]), torch.tensor([[1, 0]])
    )
    assert torch.equal(actual, torch.tensor([[[4.0, 8.0, 1.0, -float("inf")]]]))


@pytest.mark.parametrize("entry", ENTRIES)
def test_import_does_not_run_original_module_cache_disable_or_device_setup(entry, monkeypatch):
    import builtins
    import tilelang
    import torch

    def forbidden(*args, **kwargs):
        raise AssertionError("HISA import executed compiler/cache/CLI/device state")

    for owner, names in [
        (tilelang, ["compile", "lower", "disable_cache"]),
        (torch, ["manual_seed", "set_default_device", "rand", "empty", "zeros"]),
    ]:
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch_npu":
            forbidden()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    module = _load(entry)
    assert callable(module.main)


@pytest.mark.a2a3_native
@pytest.mark.parametrize("request_spec", MANIFEST["factory_requests"], ids=lambda row: row["id"])
def test_complete_original_PTO_frontend_and_JSON(request_spec):
    from tilelang import tvm

    entry = next(e for e in ENTRIES if e["source_path"] == request_spec["source_path"])
    module = _load(entry)
    factory = getattr(module, request_spec["factory"])
    assert factory.target == "pto" and factory.workspace_idx == [-1] and factory.out_idx == [3]
    func = factory.get_tir(*request_spec["args"], **request_spec["kwargs"])
    tvm.ir.assert_structural_equal(func, tvm.ir.load_json(tvm.ir.save_json(func)))
    assert len(func.params) == 8


def _main_state(entry, enabled, monkeypatch, tmp_path):
    class State:
        _enabled = enabled

        @classmethod
        def is_enabled(cls):
            return cls._enabled

    calls = []
    module = ModuleType("tilelang")
    module.env = SimpleNamespace(TILELANG_CACHE_DIR=str(tmp_path))
    module.disable_cache = lambda: calls.append(("disable", setattr(State, "_enabled", False)))
    env = ModuleType("tilelang.env")
    env.CacheState = State
    monkeypatch.setitem(sys.modules, "tilelang", module)
    monkeypatch.setitem(sys.modules, "tilelang.env", env)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path))

    @contextmanager
    def device(name):
        calls.append(("enter", name))
        try:
            yield
        finally:
            calls.append(("leave", name))

    ns = dict(os=os, Path=Path, tilelang=module, torch=SimpleNamespace(device=device), _parse_args=lambda: "original args")
    return _selected(entry, ["main"], ns), State, module.env, calls


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("failure", [True, False])
def test_main_restores_cache_and_scoped_default_device_on_CLI_exit(entry, enabled, failure, monkeypatch, tmp_path):
    ns, state, env, calls = _main_state(entry, enabled, monkeypatch, tmp_path)

    def original(args):
        assert args == "original args" and state._enabled is False
        calls.append("original CLI")
        if failure:
            raise RuntimeError("original host failure")

    ns["_run_original_cli"] = original
    if failure:
        with pytest.raises(RuntimeError, match="original host failure"):
            ns["main"]()
    else:
        assert ns["main"]() is None
    assert calls == [("disable", None), ("enter", "npu"), "original CLI", ("leave", "npu")]
    assert state._enabled is enabled and str(tmp_path) == env.TILELANG_CACHE_DIR


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("conflict", [True, False])
def test_main_cache_guard_precedes_registration_disable_or_default_device(entry, conflict, monkeypatch, tmp_path):
    ns, state, env, calls = _main_state(entry, True, monkeypatch, tmp_path)
    if conflict:
        env.TILELANG_CACHE_DIR = str(tmp_path / "caller")
    else:
        monkeypatch.delenv("TILELANG_CACHE_DIR")
    with pytest.raises(RuntimeError, match="resolved cache differs" if conflict else "dedicated TILELANG_CACHE_DIR"):
        ns["main"]()
    assert state._enabled is True and not calls
