"""First standard GDN leaves: CPU algebra/import/state and native frontend only."""

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
G1_SOURCES = {
    "examples/linear_attention_and_rnn/gdn/" + name for name in ("gdn_chunk_cumsum.py", "gdn_chunk_scaled_dot_kkt.py", "gdn_wy_fast.py")
}
ENTRIES = [e for e in MANIFEST["entries"] if e["source_path"] in G1_SOURCES]
SOURCE_REQUESTS = [r for r in MANIFEST["factory_requests"] if r["source_path"] in G1_SOURCES and r.get("component", "G1") == "G1"]
ALL_LEAF_AND_COMPOSITION_ENTRIES = [e for e in MANIFEST["entries"] if e.get("original_factory") or e.get("component") in ("G3", "G6")]


def _tree(entry):
    return ast.parse((ROOT / entry["destination_path"]).read_bytes())


def _load(entry):
    path = ROOT / entry["destination_path"]
    spec = importlib.util.spec_from_file_location("_GDN_leaf_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


@pytest.mark.parametrize("entry", ENTRIES)
def test_complete_original_definitions_CLI_matrix_and_single_clear(entry):
    functions = {n.name: n for n in _tree(entry).body if isinstance(n, ast.FunctionDef)}
    for name, expected in entry["original_definition_AST_sha256"].items():
        assert _digest(functions[name]) == expected
    body = functions["_run_original_cli"].body
    assert isinstance(body[0], ast.ImportFrom) and body[0].module == "tilelang.ascend.a2a3.kernel_cache"
    original = ast.parse("tilelang.cache.clear_cache()").body[0]
    assert ast.unparse(body[1]) == "A2A3CythonKernelCache().clear_cache()"
    assert _digest(ast.Module(body=[original, *body[2:]], type_ignores=[])) == entry["original_CLI_AST_sha256"]
    configs = next(
        n.value for n in body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "test_configs" for t in n.targets)
    )
    assert ast.literal_eval(configs) == [tuple(row) for row in entry["original_test_configs"]]
    assert hashlib.sha256((ROOT / entry["destination_path"]).read_bytes()).hexdigest() == entry["destination_sha256"]


def test_original_empty_package_is_exactly_empty():
    entry = next(e for e in MANIFEST["entries"] if e["source_path"].endswith("__init__.py"))
    assert (ROOT / entry["destination_path"]).read_bytes() == b""


@pytest.mark.parametrize("entry", ALL_LEAF_AND_COMPOSITION_ENTRIES)
def test_import_does_not_clear_seed_print_compile_or_allocate(entry, monkeypatch):
    import builtins
    import tilelang
    import torch
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

    def forbidden(*args, **kwargs):
        raise AssertionError("GDN leaf import reached original CLI side effects")

    for owner, names in [
        (tilelang, ["compile", "lower", "disable_cache"]),
        (A2A3CythonKernelCache, ["clear_cache"]),
        (torch, ["manual_seed", "set_printoptions", "randn", "empty", "zeros"]),
    ]:
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch_npu":
            forbidden()
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert callable(_load(entry).main)


def test_original_CPU_cumsum_chunk_reset_and_nondivisible_reference_error():
    import torch

    module = _load(ENTRIES[0])
    data = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
    assert torch.equal(module.ref_chunk_cumsum(data, 2), torch.tensor([[[1.0, 3.0, 3.0, 7.0], [5.0, 11.0, 7.0, 15.0]]]))
    with pytest.raises(RuntimeError):
        module.ref_chunk_cumsum(torch.ones((1, 1, 5)), 2)


@pytest.mark.parametrize("entry", ENTRIES[1:])
def test_original_reference_algebra_with_explicit_CPU_only_npu_standin(entry, monkeypatch):
    import torch

    # These are CPU helper checks, not device or original-CLI numerical results.
    transfers = []

    def cpu_only(tensor, *args, **kwargs):
        assert tensor.device.type == "cpu"
        transfers.append(tuple(tensor.shape))
        return tensor

    monkeypatch.setattr(torch.Tensor, "npu", cpu_only, raising=False)
    module = _load(entry)
    k = torch.tensor([[[[1.0, 2.0], [3.0, 4.0], [2.0, 0.0], [-1.0, 1.0]]]])
    beta = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    g = torch.zeros((1, 1, 4))
    if "kkt" in entry["original_factory"]:
        actual = module.ref_kkt(k, beta, g, 2)
        assert torch.equal(actual, torch.tensor([[[[0.0, 0.0], [22.0, 0.0], [0.0, 0.0], [-8.0, 0.0]]]], dtype=torch.float16))
        assert transfers == [(1, 1, 4, 2)]
    else:
        v = torch.tensor([[[[1.0], [2.0], [3.0], [4.0]]]])
        a = torch.tensor([[[[1.0, 0.0], [1.0, 1.0], [2.0, 0.0], [0.0, 1.0]]]])
        w, u = module.ref_wy_fast(k, v, beta, g, a, 2)
        assert torch.equal(w, torch.tensor([[[[1.0, 2.0], [7.0, 10.0], [12.0, 0.0], [-4.0, 4.0]]]], dtype=torch.float16))
        assert torch.equal(u, torch.tensor([[[[1.0], [5.0], [18.0], [16.0]]]], dtype=torch.float16))
        assert transfers == [(1, 1, 4, 2), (1, 1, 4, 1)]


@pytest.mark.usefixtures("isolated_a2a3_cache")
@pytest.mark.parametrize("entry", ALL_LEAF_AND_COMPOSITION_ENTRIES)
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("failure", [True, False])
def test_actual_cache_descriptor_memory_staging_and_printopts_restore(entry, enabled, failure, isolated_a2a3_cache, monkeypatch, tmp_path):
    import tilelang
    import torch
    from torch._tensor_str import PRINT_OPTS
    from tilelang import env
    from tilelang.env import CacheState
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache
    from tilelang.cache.kernel_cache import KernelCache

    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(isolated_a2a3_cache))
    monkeypatch.setattr(CacheState, "_enabled", enabled)
    cache = A2A3CythonKernelCache()
    caller_memory = {"caller_sentinel": object()}
    monkeypatch.setattr(cache, "_memory_cache", caller_memory)
    descriptor = vars(type(env))["TILELANG_CACHE_DIR"]
    forced = descriptor._forced_value
    print_object, print_values = PRINT_OPTS, vars(PRINT_OPTS).copy()
    staging = KernelCache._last_cleaned_staging_root
    outside = tmp_path / "caller_file"
    outside.write_text("keep")
    module = _load(entry)
    calls = []

    def prefix_only():
        assert cache._memory_cache == {} and cache._memory_cache is not caller_memory
        cache.clear_cache()  # Real existing implementation; task-owned paths only.
        calls.append("clear")
        torch.set_printoptions(threshold=float("inf"), sci_mode=True)
        CacheState._enabled = not enabled
        if failure:
            raise AssertionError("original golden failed")

    monkeypatch.setattr(module, "_run_original_cli", prefix_only)
    if failure:
        with pytest.raises(AssertionError, match="original golden failed"):
            module.main()
    else:
        assert module.main() is None
    assert calls == ["clear"] and CacheState._enabled is enabled
    assert cache._memory_cache is caller_memory and list(caller_memory) == ["caller_sentinel"]
    assert descriptor._forced_value is forced and KernelCache._last_cleaned_staging_root is staging
    assert PRINT_OPTS is print_object and vars(PRINT_OPTS) == print_values and outside.read_text() == "keep"
    assert tilelang.env is env


@pytest.mark.a2a3_native
@pytest.mark.parametrize("spec", SOURCE_REQUESTS, ids=lambda row: row["id"])
def test_complete_original_frontend_and_JSON_workspace_metadata(spec):
    from tilelang import tvm

    entry = next(e for e in ENTRIES if e["source_path"] == spec["source_path"])
    module = _load(entry)
    factory = getattr(module, spec["factory"])
    assert factory.out_idx == entry["out_idx"]
    assert factory.workspace_idx == entry["workspace_idx"]
    func = factory.get_tir(*spec["args"], **spec["kwargs"])
    tvm.ir.assert_structural_equal(func, tvm.ir.load_json(tvm.ir.save_json(func)))
