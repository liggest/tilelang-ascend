"""Complete DeepSeek assets: AST/import/lifetime/source, never numeric fixtures."""

import argparse
import ast
from contextlib import contextmanager
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = json.loads((ROOT / "maint/ascend_a2a3/ecosystem_deepseek_v4_ports.json").read_text())
ENTRY = MANIFEST["entries"][0]
SOURCE = ROOT / ENTRY["destination_path"]


def _load():
    spec = importlib.util.spec_from_file_location("_deepseek_component", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ast_sha(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def test_complete_original_functions_CLI_and_global_assignments():
    module = ast.parse(SOURCE.read_bytes())
    functions = {n.name: n for n in module.body if isinstance(n, ast.FunctionDef)}
    macro = functions["_merge_cached_sort_sources"]
    assert [ast.unparse(d) for d in macro.decorator_list] == ["T.macro"]
    assert [a.arg for a in macro.args.args] == ["merged_ub", "cache_buf", "s1_local", "_num_cached"]
    assert not any(
        isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For, ast.While, ast.With, ast.Return)) for n in ast.walk(macro)
    )
    inlines = []

    class Inline(ast.NodeTransformer):
        def visit_Expr(self, node):
            call = node.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == macro.name:
                assert [ast.unparse(a) for a in call.args] == ["merged_ub", "cache_buf", "s1_local", "_num_cached"]
                assert not call.keywords
                inlines.append(node)
                return copy.deepcopy(macro.body[0])
            return self.generic_visit(node)

    for name, expected in ENTRY["original_complete_function_AST_sha256"].items():
        function = copy.deepcopy(functions[name])
        if name in ["make_lightning_indexer_kernel", "lightning_indexer"]:
            assert ast.unparse(function.decorator_list.pop(0)) == "_runtime_scope(npu_default=False)"
        if name == "make_lightning_indexer_kernel":
            function = Inline().visit(function)
        assert _ast_sha(function) == expected, name
    assert len(inlines) == 2
    assert _ast_sha(ast.Module(body=functions["main"].body, type_ignores=[])) == ENTRY["original_normalized_CLI_body_AST_sha256"]
    assert [_ast_sha(n) for n in module.body if isinstance(n, ast.Assign)] == ENTRY["original_global_assignments_AST_sha256"]


def test_import_has_no_runtime_or_device_side_effect(monkeypatch):
    import tilelang
    import torch

    def forbidden(*args, **kwargs):
        raise AssertionError("module import performed a runtime operation")

    prior = sys.modules.get("torch_npu")
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", forbidden)
    for obj, names in [
        (tilelang, ["compile", "lower", "disable_cache", "enable_cache"]),
        (torch, ["manual_seed", "set_default_device", "device", "empty", "zeros", "full", "randn", "tensor"]),
    ]:
        for name in names:
            monkeypatch.setattr(obj, name, forbidden)
    _load()
    assert sys.modules.get("torch_npu") is prior


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("npu_default", [False, True])
def test_runtime_scope_preserves_caller_cache_return_exception_and_default(enabled, fail, npu_default, monkeypatch):
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
    result = object()
    error = ValueError("original host exception")

    @module._runtime_scope(npu_default=npu_default)
    def original_host():
        CacheState.disable() if enabled else CacheState.enable()
        if fail:
            raise error
        return result

    prior = CacheState.is_enabled()
    try:
        CacheState.enable() if enabled else CacheState.disable()
        if fail:
            with pytest.raises(ValueError) as caught:
                original_host()
            assert caught.value is error
        else:
            assert original_host() is result
        assert CacheState.is_enabled() == enabled
        assert events == ([("enter", "npu:0"), ("exit", "npu:0")] if npu_default else [])
    finally:
        CacheState.enable() if prior else CacheState.disable()


@pytest.mark.parametrize("case", MANIFEST["factory_requests"], ids=lambda c: c["id"])
def test_complete_original_source_request_with_real_nested_JIT(case, monkeypatch):
    import tilelang
    from tilelang import tvm
    from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs

    if tvm.ffi.get_global_func("target.build.tilelang_ascend_a2a3", allow_missing=True) is None:
        pytest.skip("A2A3 native source codegen is not built")
    module = _load()
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    actual_jit = module.jit
    decorators = []
    captured = []

    def explicit_source_target(*args, **kwargs):
        decorators.append(dict(kwargs))
        kwargs["target"] = target
        return actual_jit(*args, **kwargs)

    def only_source(function, **kwargs):
        captured.append(kwargs)
        return function

    # This is a source-construction fixture returning actual PrimFunc, never
    # a fake Kernel/output. Real JITKernel cold/restore is tested separately.
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(module, "jit", explicit_source_target)
    monkeypatch.setattr(importlib.import_module("tilelang.jit"), "compile", only_source)
    with target:
        function = getattr(module, case["factory"])(*case["args"], **case["kwargs"])
    assert isinstance(function, tvm.tirx.PrimFunc)
    assert len(captured) == len(decorators) == 1
    assert decorators[0]["out_idx"] == [6, 7] and decorators[0]["workspace_idx"] == [3, 8]
    configs = normalize_a2a3_pass_configs(target, decorators[0]["pass_configs"])
    with tvm.transform.PassContext(opt_level=3, config=configs):
        artifact = tilelang.lower(function, target=target)
    launch = int(re.search(r"main_kernel<<<(\d+),", artifact.kernel_source)[1])
    assert launch == MANIFEST["actual_source_domains"][case["id"]]["actual_logical_core_launch"]
    assert "KERNEL_TYPE_MIX_AIC_1_2" in artifact.kernel_source
    assert 'extern "C" void call(' in artifact.kernel_source
