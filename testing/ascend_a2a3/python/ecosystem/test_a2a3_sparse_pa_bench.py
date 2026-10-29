"""Original complete PA benchmark flow, safe CLI parsing and exact local leaves."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = json.loads((ROOT / "maint/ascend_a2a3/ecosystem_sparse_pa_bench_ports.json").read_text())
BENCH = next(entry for entry in MANIFEST["entries"] if entry["source_path"].endswith(".py"))
PATH = ROOT / BENCH["destination_path"]


def _load():
    spec = importlib.util.spec_from_file_location("_owned_sparse_pa_benchmark", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def test_both_original_functions_and_whole_executable_flow_are_complete():
    tree = ast.parse(PATH.read_bytes())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name, expected in BENCH["original_definition_AST_sha256"].items():
        assert _digest(functions[name]) == expected

    class OriginalImports(ast.NodeTransformer):
        def visit_Assign(self, node):
            value = node.value
            if isinstance(value, ast.Attribute) and value.attr == "sparse_attn_tilelang" and isinstance(value.value, ast.Call):
                call = value.value
                if isinstance(call.func, ast.Name) and call.func.id == "_load_leaf":
                    return ast.ImportFrom(module=call.args[0].value, names=[ast.alias(name="sparse_attn_tilelang")], level=0)
            return node

    flow = ast.Module(
        body=[
            node for node in functions["_run_original_cli"].body if not (isinstance(node, ast.Import) and node.names[0].name == "torch_npu")
        ],
        type_ignores=[],
    )
    assert _digest(OriginalImports().visit(flow)) == BENCH["original_executable_flow_AST_sha256"]
    assert hashlib.sha256(PATH.read_bytes()).hexdigest() == BENCH["destination_sha256"]


def test_original_empty_CPP_has_no_synthetic_implementation():
    entry = next(row for row in MANIFEST["entries"] if row["source_path"].endswith(".cpp"))
    raw = (ROOT / entry["destination_path"]).read_bytes()
    assert raw == b"" and hashlib.sha256(raw).hexdigest() == entry["source_sha256"] == entry["destination_sha256"]


def test_import_and_help_never_register_NPU_initialize_or_touch_cache(monkeypatch, capsys):
    import tilelang
    import torch
    from tilelang.env import CacheState
    from tilelang.jit import JITImpl

    def forbidden(*args, **kwargs):
        raise AssertionError("benchmark import/help must not initialize, compile or allocate")

    before_path = sys.path.copy()
    before_cache = CacheState._enabled
    with monkeypatch.context() as scoped:
        for name in ("set_default_device", "manual_seed", "empty", "zeros", "rand", "randn", "arange"):
            scoped.setattr(torch, name, forbidden)
        scoped.setattr(tilelang, "disable_cache", forbidden)
        scoped.setattr(JITImpl, "get_tir", forbidden)
        module = _load()
        scoped.setattr(module, "init_test", forbidden)
        scoped.setitem(sys.modules, "torch_npu", None)
        scoped.setattr(sys, "argv", [str(PATH), "--help"])
        with pytest.raises(SystemExit) as error:
            module.main()
        assert error.value.code == 0
    assert "--file" in capsys.readouterr().out
    assert CacheState._enabled is before_cache and sys.path == before_path


@pytest.mark.parametrize(
    "name",
    ["sparse_flash_attn_pa_baseline", "sparse_flash_attn_pa_developer", "sparse_flash_attn_pa", "sparse_flash_attn_pa_no_cv_pipeline"],
)
def test_selected_leaf_and_unconditional_pipeline_are_exact_local_files(name, monkeypatch):
    sentinel = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, sentinel)
    before = sys.path.copy()
    module = _load()
    leaf = module._load_leaf(name)
    assert Path(leaf.__file__).resolve() == PATH.parent / (name + ".py")
    assert leaf.T.__name__ == "tilelang.ascend.a2a3.language"
    assert module.init_test is module._pipeline.init_test
    assert module.sparse_attention_fwd is module._pipeline.sparse_attention_fwd
    assert Path(module._pipeline.__file__).resolve() == PATH.parent / "sparse_flash_attn_pa.py"
    assert sys.modules[name] is sentinel and sys.path == before


def test_original_unknown_file_rejection_precedes_initializer_or_NPU_import(monkeypatch):
    module = _load()

    def forbidden():
        raise AssertionError("invalid file must be rejected before original initializer")

    monkeypatch.setattr(module, "init_test", forbidden)
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    monkeypatch.setattr(sys, "argv", [str(PATH), "--file", "unknown"])
    with pytest.raises(RuntimeError, match="not supported file, only the following files are supported"):
        module.main()
    with pytest.raises(ValueError, match="unknown local sparse PA leaf"):
        module._load_leaf("../foreign")


@pytest.mark.parametrize("enabled", [True, False, None])
@pytest.mark.parametrize("raises", [False, True])
def test_new_main_boundary_restores_raw_cache_flag_on_exit_or_exception(enabled, raises, monkeypatch):
    import tilelang
    from tilelang.env import CacheState

    module = _load()
    original = CacheState._enabled

    def stop_before_tensors_or_device():
        tilelang.disable_cache()
        if raises:
            raise RuntimeError("stop at CLI boundary before NPU")

    monkeypatch.setattr(module, "_run_original_cli", stop_before_tensors_or_device)
    try:
        CacheState._enabled = enabled
        if raises:
            with pytest.raises(RuntimeError, match="stop at CLI boundary"):
                module.main()
        else:
            module.main()
        assert CacheState._enabled is enabled
    finally:
        CacheState._enabled = original
