"""Remaining optimized GDN leaves: complete provenance and CPU/frontend checks."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = json.loads((ROOT / "maint/ascend_a2a3/ecosystem_gdn_ports.json").read_text())
ENTRIES = [e for e in MANIFEST["entries"] if e.get("component") == "G5"]
REQUESTS = [r for r in MANIFEST["factory_requests"] if r.get("component") == "G5"]


def _load(entry):
    path = ROOT / entry["destination_path"]
    spec = importlib.util.spec_from_file_location("_GDN_remaining_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


@pytest.mark.parametrize("entry", ENTRIES)
def test_all_original_definitions_complete_CLI_and_single_clear(entry):
    path = ROOT / entry["destination_path"]
    functions = {n.name: n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef)}
    for name, original in entry["original_definition_AST_sha256"].items():
        expected = entry.get("adapted_definition_AST_sha256", {}).get(name, original)
        assert _digest(functions[name]) == expected
    body = functions["_run_original_cli"].body
    assert isinstance(body[0], ast.ImportFrom) and body[0].module == "tilelang.ascend.a2a3.kernel_cache"
    assert ast.unparse(body[1]) == "A2A3CythonKernelCache().clear_cache()"
    original_clear = ast.parse("tilelang.cache.clear_cache()").body[0]
    assert _digest(ast.Module(body=[original_clear, *body[2:]], type_ignores=[])) == entry["original_CLI_AST_sha256"]
    configs = next(
        n.value for n in body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "test_configs" for t in n.targets)
    )
    assert ast.literal_eval(configs) == [tuple(row) for row in entry["original_test_configs"]]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["destination_sha256"]


@pytest.fixture
def cpu_reference_transfer(monkeypatch):
    import torch

    # CPU algebra checks only; these do not execute an NPU or the default CLI.
    def identity(tensor, *args, **kwargs):
        assert tensor.device.type == "cpu"
        return tensor

    monkeypatch.setattr(torch.Tensor, "npu", identity, raising=False)
    return torch


def test_original_hidden_reference_recurrence(cpu_reference_transfer):
    torch = cpu_reference_transfer
    module = _load(next(e for e in ENTRIES if e["original_factory"] == "chunk_h_ker"))
    k = torch.tensor([[[[1, 0], [0, 1], [1, 0], [0, 1]]]], dtype=torch.float16)
    w = torch.zeros_like(k)
    u = torch.tensor([[[[1, 2], [3, 4], [5, 6], [7, 8]]]], dtype=torch.float16)
    s, new_v, final_s = module.ref_chunk_h(k, w, u, torch.zeros((1, 1, 4)), 2)
    expected_s = torch.tensor([[[[[0, 0], [0, 0]], [[1, 2], [3, 4]]]]], dtype=torch.float16)
    assert torch.equal(s, expected_s) and torch.equal(new_v, u)
    assert torch.equal(final_s, torch.tensor([[[[6, 8], [10, 12]]]], dtype=torch.float16))


def test_original_output_reference_combines_history_and_causal_tile(cpu_reference_transfer):
    torch = cpu_reference_transfer
    module = _load(next(e for e in ENTRIES if e["original_factory"] == "chunk_o_ker"))
    k = torch.tensor([[[[1, 0], [0, 1], [1, 0], [0, 1]]]], dtype=torch.float16)
    v = torch.tensor([[[[1, 2], [3, 4], [5, 6], [7, 8]]]], dtype=torch.float16)
    s = torch.tensor([[[[[0, 0], [0, 0]], [[1, 2], [3, 4]]]]], dtype=torch.float16)
    actual = module.ref_chunk_o(k, k, v, s, torch.zeros((1, 1, 4)), 2)
    assert torch.equal(actual, torch.tensor([[[[1, 2], [3, 4], [6, 8], [10, 12]]]], dtype=torch.float16))


def test_original_triangular_reference_preserves_dtype_and_mutation_contract(cpu_reference_transfer):
    torch = cpu_reference_transfer
    module = _load(next(e for e in ENTRIES if e["original_factory"] == "solve_tril_128_ker"))
    a = torch.tensor([[[[0, 0, 0], [2, 0, 0], [5, 3, 0]]]], dtype=torch.float16)
    before = a.clone()
    expected = torch.tensor([[[[1, 0, 0], [-2, 1, 0], [1, -3, 1]]]], dtype=torch.float16)
    assert torch.equal(module.ref_solve_tril(a), expected)
    assert torch.equal(a, before)
    float_input = a.float()
    assert torch.equal(module.solve_triangular(float_input), expected.float())
    assert torch.equal(float_input[0, 0, 2], torch.tensor([-1.0, 3.0, 0.0]))


@pytest.mark.a2a3_native
@pytest.mark.parametrize("request_spec", REQUESTS, ids=lambda row: row["id"])
def test_complete_original_frontend_JSON_and_explicit_workspace_ABI(request_spec):
    from tilelang import tvm

    entry = next(e for e in ENTRIES if e["source_path"] == request_spec["source_path"])
    module = _load(entry)
    factory = getattr(module, request_spec["factory"])
    assert factory.out_idx == entry["out_idx"] and factory.workspace_idx == entry["workspace_idx"]
    func = factory.get_tir(*request_spec["args"], **request_spec["kwargs"])
    tvm.ir.assert_structural_equal(func, tvm.ir.load_json(tvm.ir.save_json(func)))
    actual = [str(func.buffer_map[param].name) for param in func.params]
    expected = {
        "chunk_h_ker": ["K", "W", "U", "G", "workspace_1", "workspace_2", "workspace_3", "workspace_4", "S", "V", "FS"],
        "chunk_o_ker": ["Q", "K", "V", "S", "G", "Msk", "workspace_1", "workspace_2", "workspace_3", "O"],
        "solve_tril_ker": ["A", "I", "O"],
    }
    key = "solve_tril_ker" if request_spec["factory"].startswith("solve_tril") else request_spec["factory"]
    assert actual == expected[key]


@pytest.mark.parametrize("entry", ENTRIES)
def test_original_manual_sync_and_explicit_planning_configs(entry):
    module = _load(entry)
    actual = {getattr(key, "value", key): value for key, value in module.pass_configs.items()}
    expected = {"tl.ascend_auto_sync": False}
    if entry["original_factory"] in ("chunk_h_ker", "chunk_o_ker"):
        expected["tl.ascend_memory_planning"] = True
    assert actual == expected
