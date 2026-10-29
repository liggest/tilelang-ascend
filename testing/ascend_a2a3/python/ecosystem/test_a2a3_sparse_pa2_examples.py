"""Complete pipeline and developer PA leaf APIs, import safety, original defaults and native metadata."""

import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = json.loads((ROOT / "maint/ascend_a2a3/ecosystem_sparse_pa2_ports.json").read_text())
ENTRIES = MANIFEST["entries"]


def _load(entry):
    spec = importlib.util.spec_from_file_location("_sparse_pa_owned", ROOT / entry["destination_path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda row: Path(row["source_path"]).stem)
def test_six_complete_original_definitions_and_hashes(entry):
    path = ROOT / entry["destination_path"]
    tree = ast.parse(path.read_bytes())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert len(functions) == 3
    actual = {name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() for name, node in functions.items()}
    assert actual == entry["original_definition_AST_sha256"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["destination_sha256"]
    assert not any(isinstance(node, ast.If) and "__main__" in ast.unparse(node.test) for node in tree.body)


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda row: Path(row["source_path"]).stem)
def test_import_has_no_cli_initializer_cache_path_or_tensor_side_effects(entry, monkeypatch):
    import tilelang
    import torch
    from tilelang.env import CacheState
    from tilelang.jit import JITImpl

    def forbidden(*args, **kwargs):
        raise AssertionError("leaf import must not execute initializer, parse CLI or construct IR/tensors")

    before_path = sys.path.copy()
    before_cache = CacheState._enabled
    with monkeypatch.context() as scoped:
        for name in ("manual_seed", "set_default_device", "empty", "zeros", "rand", "randn", "arange"):
            scoped.setattr(torch, name, forbidden)
        scoped.setattr(tilelang, "disable_cache", forbidden)
        scoped.setattr(JITImpl, "get_tir", forbidden)
        module = _load(entry)
    assert sys.path == before_path and CacheState._enabled is before_cache
    assert module.T.__name__ == "tilelang.ascend.a2a3.language"
    assert not hasattr(module, "main")


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda row: Path(row["source_path"]).stem)
def test_public_wrapper_signature_and_explicit_initializer_boundary(entry):
    module = _load(entry)
    assert not inspect.signature(module.init_test).parameters
    parameters = inspect.signature(module.sparse_attn_tilelang).parameters
    assert list(parameters)[:8] == [
        "query",
        "key",
        "value",
        "sparse_indices",
        "scale_value",
        "sparse_block_size",
        "actual_seq_lengths_query",
        "actual_seq_lengths_kv",
    ]
    assert parameters["layout_query"].default == "BSND"
    assert parameters["layout_kv"].default == "BSND"
    assert parameters["sparse_mode"].default == 3
    assert module.sparse_attention_fwd.out_idx == [3]
    assert module.sparse_attention_fwd.workspace_idx == [7, 8, 9, 10, 11]


@pytest.mark.a2a3_native
@pytest.mark.parametrize("factory_request", MANIFEST["factory_requests"], ids=lambda row: row["id"])
def test_complete_default_frontend_dynamic_carriers_and_five_explicit_workspaces(factory_request):
    from tilelang import tvm

    request = factory_request
    entry = next(row for row in ENTRIES if row["source_path"] == request["source_path"])
    module = _load(entry)
    function = module.sparse_attention_fwd.get_tir(**request["kwargs"])
    # Layout iterator variables are free ObjectRefs in annotation values;
    # compare their alpha mapping after JSON rebuild as well.
    tvm.ir.assert_structural_equal(function, tvm.ir.load_json(tvm.ir.save_json(function)), map_free_vars=True)
    assert len(function.params) == 12
    buffers = [function.buffer_map[parameter] for parameter in function.params]
    assert str(buffers[0].shape[0]) == "batch" and str(buffers[0].shape[1]) == "seq_len"
    if "block_num" in request["kwargs"]:
        assert int(buffers[1].shape[0]) == request["kwargs"]["block_num"]
    else:
        assert str(buffers[1].shape[0]) == "block_num"
    assert str(buffers[6].shape[1]) == "block_table_len"
    assert int(buffers[2].shape[-1]) == request["kwargs"]["topk"]
    assert [int(buffer.shape[0]) for buffer in buffers[7:]] == [24] * 5
    assert [str(buffer.dtype) for buffer in buffers[7:]] == ["bfloat16", "bfloat16", "float32", "bfloat16", "float32"]


@pytest.mark.a2a3_native
@pytest.mark.parametrize("topk", [0, 64, 192])
def test_original_pipeline_rejects_zero_or_incomplete_two_stage_group(topk):
    entry = next(row for row in ENTRIES if row["source_path"].endswith("/sparse_flash_attn_pa.py"))
    module = _load(entry)
    request = next(row for row in MANIFEST["factory_requests"] if row["source_path"] == entry["source_path"])
    kwargs = dict(request["kwargs"], topk=topk)
    message = "topk must be positive" if topk == 0 else r"must be a multiple of 128"
    with pytest.raises(AssertionError, match=message):
        module.sparse_attention_fwd.get_tir(**kwargs)
