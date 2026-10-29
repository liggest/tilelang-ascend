"""Complete sparse MLA assets: host contracts and source, never fake numerical outputs."""

import argparse
import ast
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
MANIFEST = json.loads((ROOT / "maint/ascend_a2a3/ecosystem_deepseek_v4_sparse_mla_ports.json").read_text())
DIRECTORY = ROOT / "examples/ascend_a2a3/deepseek_v4/sparse_flash_mla"


def _load(monkeypatch):
    # A unique package exercises the real relative imports without mutating an
    # unrelated direct-script module in sys.modules.
    package = ModuleType("_a2a3_sparse_mla_component")
    package.__path__ = [str(DIRECTORY)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    modules = {}
    for name in ["sparse_flash_mla_kernel", "sparse_flash_mla_golden", "sparse_flash_mla_api"]:
        qualified = package.__name__ + "." + name
        spec = importlib.util.spec_from_file_location(qualified, DIRECTORY / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, qualified, module)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def _ast_sha(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


@pytest.mark.parametrize("entry", MANIFEST["entries"], ids=lambda e: Path(e["source_path"]).name)
def test_complete_original_functions_globals_and_CLI(entry):
    path = ROOT / entry["destination_path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["destination_sha256"]
    module = ast.parse(path.read_bytes())
    functions = {n.name: n for n in module.body if isinstance(n, ast.FunctionDef)}
    for name, expected in entry["original_complete_function_AST_sha256"].items():
        assert _ast_sha(functions[name]) == expected, name
    assert [_ast_sha(n) for n in module.body if isinstance(n, (ast.Assign, ast.AnnAssign))] == entry[
        "original_global_assignments_AST_sha256"
    ]
    expected_CLI = entry["original_normalized_CLI_body_AST_sha256"]
    if expected_CLI is not None:
        main = functions["main"]
        assert [ast.unparse(n) for n in main.decorator_list] == ["_compiler_cache_scope"]
        body = copy.deepcopy(main.body)
        if path.name == "sparse_flash_mla_api.py":
            assert ast.unparse(body.pop(0)) == "import torch_npu"
            imports = body.pop(0)
            assert isinstance(imports, ast.If) and ast.unparse(imports.test) == "__package__"
            assert isinstance(imports.body[0], ast.ImportFrom) and imports.body[0].level == 1
            relative = copy.deepcopy(imports.body[0])
            relative.level = 0
            assert _ast_sha(relative) == _ast_sha(imports.orelse[0])
            body = imports.orelse + body
        assert _ast_sha(ast.Module(body=body, type_ignores=[])) == expected_CLI


def test_import_preserves_cache_and_performs_no_runtime_operation(monkeypatch):
    import tilelang
    import torch
    from tilelang.env import CacheState

    def forbidden(*args, **kwargs):
        raise AssertionError("import performed a runtime operation")

    raw_cache = CacheState._enabled
    prior_npu = sys.modules.get("torch_npu")
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", forbidden)
    for obj, names in [
        (tilelang, ["compile", "lower", "disable_cache", "enable_cache"]),
        (torch, ["manual_seed", "set_default_device", "empty", "zeros", "ones", "randn", "randint", "tensor"]),
    ]:
        for name in names:
            monkeypatch.setattr(obj, name, forbidden)
    _load(monkeypatch)
    assert CacheState._enabled is raw_cache
    assert sys.modules.get("torch_npu") is prior_npu


@pytest.mark.parametrize("raw_cache", [None, False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_explicit_CLI_cache_scope_restores_raw_state_and_return_or_exception(raw_cache, fail, monkeypatch):
    from tilelang.env import CacheState

    module = _load(monkeypatch)["sparse_flash_mla_kernel"]
    monkeypatch.setattr(CacheState, "_enabled", raw_cache)
    result = object()
    error = ValueError("original failure")

    @module._compiler_cache_scope
    def original_CLI():
        assert CacheState._enabled is False
        if fail:
            raise error
        return result

    if fail:
        with pytest.raises(ValueError) as caught:
            original_CLI()
        assert caught.value is error
    else:
        assert original_CLI() is result
    assert CacheState._enabled is raw_cache


def test_original_dtype_rejection_and_constructor_boundaries(monkeypatch):
    import torch

    modules = _load(monkeypatch)
    api = modules["sparse_flash_mla_api"]
    assert api._torch_to_tilelang_dtype(torch.bfloat16) == "bfloat16"
    assert api._torch_to_tilelang_dtype(torch.float16) == "float16"
    with pytest.raises(ValueError, match="unsupported torch dtype"):
        api._torch_to_tilelang_dtype(torch.float32)
    kernel = modules["sparse_flash_mla_kernel"]
    base = MANIFEST["factory_requests"][0]["kwargs"]
    for override, exception, message in [
        ({"scenario": 4}, NotImplementedError, "only SWA"),
        ({"n_heads": 32}, ValueError, "kernel assumes"),
        ({"n_kv_heads": 2}, ValueError, "kernel assumes"),
        ({"head_dim": 256}, ValueError, "kernel assumes"),
        ({"ori_win_left": 128}, ValueError, "exceeds single-tile"),
    ]:
        with pytest.raises(exception, match=message):
            kernel.build_sparse_flash_mla(**(base | override))


@pytest.mark.parametrize("case", MANIFEST["factory_requests"], ids=lambda c: c["id"])
def test_complete_original_request_source_and_workspace_ABI(case, monkeypatch):
    import tilelang
    from tilelang import tvm
    from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs

    if tvm.ffi.get_global_func("target.build.tilelang_ascend_a2a3", allow_missing=True) is None:
        pytest.skip("A2A3 native source codegen is not built")
    module = _load(monkeypatch)["sparse_flash_mla_kernel"]
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    actual_jit = tilelang.jit
    decorators = []
    captured = []

    def source_target(*args, **kwargs):
        decorators.append(dict(kwargs))
        kwargs["target"] = target
        return actual_jit(*args, **kwargs)

    def only_source(function, **kwargs):
        captured.append(kwargs)
        return function

    # Source instrumentation retains the complete original factory and actual
    # PrimFunc. Genuine public cold/restore is separately archived in the report.
    monkeypatch.setattr(tilelang, "jit", source_target)
    monkeypatch.setattr(importlib.import_module("tilelang.jit"), "compile", only_source)
    with target:
        function = module.build_sparse_flash_mla(**case["kwargs"])
    assert isinstance(function, tvm.tirx.PrimFunc)
    assert len(captured) == len(decorators) == 1
    decorator = decorators[0]
    assert decorator["target"] == "ascendc" and decorator["out_idx"] == [10, 11]
    expected_count = {1: 15, 2: 17, 3: 18}[case["kwargs"]["scenario"]]
    assert decorator["workspace_idx"] == list(range(12, expected_count))
    assert len(function.params) == expected_count
    configs = normalize_a2a3_pass_configs(target, decorator.get("pass_configs", {}))
    with tvm.transform.PassContext(opt_level=3, config=configs):
        artifact = tilelang.lower(function, target=target)
    assert int(re.search(r"sparse_flash_mla_(?:swa|hca|csa)_kernel<<<(\d+),", artifact.kernel_source)[1]) == case["kwargs"]["core_num"]
    assert "KERNEL_TYPE_MIX_AIC_1_2" in artifact.kernel_source
    assert 'extern "C" void call(' in artifact.kernel_source
