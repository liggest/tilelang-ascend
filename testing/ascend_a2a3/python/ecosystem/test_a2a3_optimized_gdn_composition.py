"""Optimized GDN composition: complete original CLI and exact local leaf imports."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = json.loads((ROOT / "maint/ascend_a2a3/ecosystem_gdn_ports.json").read_text())
ENTRY = next(e for e in MANIFEST["entries"] if e.get("component") == "G6")
REQUESTS = [r for r in MANIFEST["factory_requests"] if r.get("component") == "G6"]


def _load():
    path = ROOT / ENTRY["destination_path"]
    spec = importlib.util.spec_from_file_location("_GDN_composition", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def test_complete_original_composition_and_original_CLI_domain():
    path = ROOT / ENTRY["destination_path"]
    functions = {n.name: n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef)}
    for name, expected in ENTRY["original_definition_AST_sha256"].items():
        assert _digest(functions[name]) == expected
    body = functions["_run_original_cli"].body
    assert isinstance(body[0], ast.ImportFrom) and body[0].module == "tilelang.ascend.a2a3.kernel_cache"
    assert ast.unparse(body[1]) == "A2A3CythonKernelCache().clear_cache()"
    original_clear = ast.parse("tilelang.cache.clear_cache()").body[0]
    assert _digest(ast.Module(body=[original_clear, *body[2:]], type_ignores=[])) == ENTRY["original_CLI_AST_sha256"]
    configs = next(
        n.value for n in body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "test_configs" for t in n.targets)
    )
    assert ast.literal_eval(configs) == [(1, 2, 1024, 128, 128, 128, 128, 128)]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ENTRY["destination_sha256"]


def test_exact_local_leaves_ignore_foreign_gdn_and_keep_caller_paths_and_modules(monkeypatch):
    foreign = ModuleType("opt_gdn")
    sentinels = {"opt_gdn": foreign}
    for name in ENTRY["original_leaf_dependencies"]:
        sentinels[name] = ModuleType(name)
    for name, value in sentinels.items():
        monkeypatch.setitem(sys.modules, name, value)
    original_path = sys.path.copy()
    module = _load()
    assert sys.path == original_path and all(sys.modules[name] is value for name, value in sentinels.items())
    assert len(module._opt_gdn_leaves) == 6
    for name, leaf in module._opt_gdn_leaves.items():
        assert Path(leaf.__file__).resolve() == (ROOT / ENTRY["destination_path"]).with_name("opt_gdn") / (name + ".py")
        assert leaf.__name__ not in sys.modules
        assert leaf.T.__name__ == "tilelang.ascend.a2a3.language"


def test_original_sequential_reference_CPU_algebra(monkeypatch):
    import torch

    # Scoped CPU transfer identity: reference algebra, not an NPU/default CLI run.
    def identity(tensor, *args, **kwargs):
        assert tensor.device.type == "cpu"
        return tensor

    monkeypatch.setattr(torch.Tensor, "npu", identity, raising=False)
    module = _load()
    q = torch.tensor([[[[1, 0], [0, 1], [1, 0], [0, 1]]]], dtype=torch.float16)
    v = torch.tensor([[[[1, 2], [3, 4], [5, 6], [7, 8]]]], dtype=torch.float16)
    g, beta = torch.zeros((1, 1, 4)), torch.full((1, 1, 4), 0.5, dtype=torch.float16)
    expected = torch.tensor([[[[0.5, 1], [1.5, 2], [2.75, 3.5], [4.25, 5]]]], dtype=torch.float16)
    assert torch.equal(module.ref_seq_gdn(q, q, v, g, beta), expected)


@pytest.mark.a2a3_native
@pytest.mark.parametrize("request_spec", REQUESTS, ids=lambda row: row["id"])
def test_complete_composed_default_leaf_frontends_and_JSON(request_spec):
    from tilelang import tvm

    module = _load()
    leaf = module._opt_gdn_leaves[Path(request_spec["source_path"]).stem]
    factory = getattr(leaf, request_spec["factory"])
    func = factory.get_tir(*request_spec["args"], **request_spec["kwargs"])
    tvm.ir.assert_structural_equal(func, tvm.ir.load_json(tvm.ir.save_json(func)))
    assert any(int(dim) == 1024 for param in func.params for dim in func.buffer_map[param].shape)
