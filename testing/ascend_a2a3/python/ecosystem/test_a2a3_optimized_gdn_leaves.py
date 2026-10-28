"""Complete optimized GDN leaves: provenance, CPU reference and native frontend."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = json.loads((ROOT / "maint/ascend_a2a3/ecosystem_gdn_ports.json").read_text())
ENTRIES = [e for e in MANIFEST["entries"] if e.get("component") == "G4" and e.get("original_factory")]
REQUESTS = [r for r in MANIFEST["factory_requests"] if r.get("component") == "G4"]


def _load(entry):
    path = ROOT / entry["destination_path"]
    spec = importlib.util.spec_from_file_location("_optimized_GDN_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


@pytest.mark.parametrize("entry", ENTRIES)
def test_all_original_factory_reference_helpers_and_complete_CLI(entry):
    path = ROOT / entry["destination_path"]
    functions = {n.name: n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef)}
    for name, expected in entry["original_definition_AST_sha256"].items():
        assert _digest(functions[name]) == expected
    body = functions["_run_original_cli"].body
    assert isinstance(body[0], ast.ImportFrom) and body[0].module == "tilelang.ascend.a2a3.kernel_cache"
    assert ast.unparse(body[1]) == "A2A3CythonKernelCache().clear_cache()"
    old_clear = ast.parse("tilelang.cache.clear_cache()").body[0]
    assert _digest(ast.Module(body=[old_clear, *body[2:]], type_ignores=[])) == entry["original_CLI_AST_sha256"]
    configs = next(
        n.value for n in body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "test_configs" for t in n.targets)
    )
    assert ast.literal_eval(configs) == [tuple(row) for row in entry["original_test_configs"]]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["destination_sha256"]


def test_original_optimized_empty_package_stays_empty():
    entry = next(e for e in MANIFEST["entries"] if e.get("component") == "G4" and e["source_path"].endswith("__init__.py"))
    assert (ROOT / entry["destination_path"]).read_bytes() == b""


@pytest.mark.parametrize("entry", ENTRIES)
def test_explicit_manual_sync_and_memory_planning_original_configs(entry):
    module = _load(entry)
    actual = {getattr(key, "value", key): value for key, value in module.pass_configs.items()}
    expected = {"tl.ascend_auto_sync": False}
    if entry["original_factory"] == "kkt_ker":
        expected["tl.ascend_memory_planning"] = True
    assert actual == expected
    factory = getattr(module, entry["original_factory"])
    assert factory.out_idx == entry["out_idx"] and factory.workspace_idx == entry["workspace_idx"]


@pytest.mark.parametrize("entry", ENTRIES)
def test_original_reference_algebra_CPU_only(entry, monkeypatch):
    import torch

    # Explicitly scoped CPU transfer stand-in: no original device CLI is run.
    def identity(tensor, *args, **kwargs):
        assert tensor.device.type == "cpu"
        return tensor

    monkeypatch.setattr(torch.Tensor, "npu", identity, raising=False)
    module = _load(entry)
    if entry["original_factory"] == "cumsum_ker":
        source = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        assert torch.equal(module.ref_chunk_cumsum(source, 2), torch.tensor([[[1.0, 3.0, 3.0, 7.0]]]))
        with pytest.raises(RuntimeError):
            module.ref_chunk_cumsum(torch.ones((1, 1, 5)), 2)
        return
    k = torch.tensor([[[[1.0, 2.0], [3.0, 4.0], [2.0, 0.0], [-1.0, 1.0]]]], dtype=torch.float16)
    beta, g = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]), torch.zeros((1, 1, 4))
    if entry["original_factory"] == "kkt_ker":
        expected = torch.tensor([[[[0.0, 0.0], [22.0, 0.0], [0.0, 0.0], [-8.0, 0.0]]]], dtype=torch.float16)
        assert torch.equal(module.ref_kkt(k, beta, g, 2), expected)
    else:
        v = torch.tensor([[[[1.0], [2.0], [3.0], [4.0]]]], dtype=torch.float16)
        a = torch.tensor([[[[1.0, 0.0], [1.0, 1.0], [2.0, 0.0], [0.0, 1.0]]]])
        w, u = module.ref_wy_fast(k, v, beta, g, a, 2)
        assert torch.equal(w, torch.tensor([[[[1.0, 2.0], [7.0, 10.0], [12.0, 0.0], [-4.0, 4.0]]]], dtype=torch.float16))
        assert torch.equal(u, torch.tensor([[[[1.0], [5.0], [18.0], [16.0]]]], dtype=torch.float16))


@pytest.mark.a2a3_native
@pytest.mark.parametrize("request_spec", REQUESTS, ids=lambda row: row["id"])
def test_complete_large_frontend_JSON_manual_event_schema_and_optional_tilings(request_spec):
    from tilelang import tvm

    entry = next(e for e in ENTRIES if e["source_path"] == request_spec["source_path"])
    module = _load(entry)
    factory = getattr(module, request_spec["factory"])
    func = factory.get_tir(*request_spec["args"], **request_spec["kwargs"])
    tvm.ir.assert_structural_equal(func, tvm.ir.load_json(tvm.ir.save_json(func)))
    explicit = {"CC": 8} if request_spec["factory"] == "cumsum_ker" else {"BK": 128}
    if request_spec["factory"] == "wy_fast_ker":
        explicit["BV"] = 128
    tvm.ir.assert_structural_equal(func, factory.get_tir(*request_spec["args"], **explicit))
    events = {"tl.ascend_set_flag": [], "tl.ascend_wait_flag": []}

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and getattr(node.op, "name", "") in events:
            assert len(node.args) == 2
            events[node.op.name].append((node.args[0].value, int(node.args[1])))

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    assert events["tl.ascend_set_flag"] and events["tl.ascend_set_flag"] == events["tl.ascend_wait_flag"]
