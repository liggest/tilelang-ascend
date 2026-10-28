"""Complete sparse GQA imports, input-device reference, state and dynamic ABI."""

import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = json.loads((ROOT / "maint/ascend_a2a3/ecosystem_sparse_gqa_ports.json").read_text())
ENTRIES = MANIFEST["entries"]


def _load(entry):
    path = ROOT / entry["destination_path"]
    spec = importlib.util.spec_from_file_location("_sparse_gqa_owned", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda row: Path(row["source_path"]).stem)
def test_complete_original_definitions_and_executable_CLI(entry):
    path = ROOT / entry["destination_path"]
    functions = {n.name: n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef)}
    for name, expected in entry["original_definition_AST_sha256"].items():
        assert _digest(functions[name]) == expected
    setter = ast.parse('torch.set_default_device("npu")').body[0]
    assert (
        _digest(ast.Module(body=[setter, *functions["_run_original_cli"].body], type_ignores=[]))
        == entry["original_executable_CLI_AST_sha256"]
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["destination_sha256"]


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda row: Path(row["source_path"]).stem)
def test_import_does_not_compile_seed_toggle_cache_or_allocate(entry, monkeypatch):
    import tilelang
    import torch
    from tilelang.env import CacheState
    from tilelang.jit import JITImpl

    def forbidden(*args, **kwargs):
        raise AssertionError("import must not execute CLI or construct an IR/kernel/tensor")

    before = CacheState._enabled
    with monkeypatch.context() as scoped:
        for name in ("manual_seed", "set_default_device", "empty", "zeros", "rand", "randn", "arange"):
            scoped.setattr(torch, name, forbidden)
        scoped.setattr(tilelang, "disable_cache", forbidden)
        scoped.setattr(tilelang, "lower", forbidden)
        scoped.setattr(JITImpl, "get_tir", forbidden)
        module = _load(entry)
    assert CacheState._enabled is before
    assert module.T.__name__ == "tilelang.ascend.a2a3.language"
    reference = module.ref_sparse_attention_fwd_interface_gqa
    assert inspect.signature(reference) == inspect.signature(reference.__wrapped__)


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda row: Path(row["source_path"]).stem)
def test_reference_uses_input_device_and_keeps_causal_mask(entry):
    import torch

    module = _load(entry)
    q = torch.zeros((1, 2, 4, 128), dtype=torch.float16)
    kv = torch.stack((torch.full((2, 128), 2.0), torch.full((2, 128), 6.0))).unsqueeze(0).half()
    indices = torch.tensor([[[[0, 1], [0, 1]], [[0, 1], [0, 1]]]], dtype=torch.int32)
    # Call from a different default-device context: the original bare arange tensors
    # must follow q, and the caller's context must remain active afterward.
    with torch.device("meta"):
        actual = module.ref_sparse_attention_fwd_interface_gqa(q, kv, indices, q_start_index_s=0, kv_stride=1)
        assert torch.empty(0).device.type == "meta"
    expected = torch.full_like(q, 4.0)
    expected[:, 0] = 2.0
    assert actual.device.type == "cpu" and torch.equal(actual, expected)


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda row: Path(row["source_path"]).stem)
@pytest.mark.parametrize("enabled", [True, False, None])
@pytest.mark.parametrize("raises", [False, True])
def test_cli_restores_raw_cache_state_and_device_context(entry, enabled, raises, monkeypatch):
    import tilelang
    import torch
    from tilelang.env import CacheState

    module = _load(entry)
    original = CacheState._enabled
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    # CPU stand-in for the CLI device boundary, not a replacement kernel or
    # output. Exercise the real cache transition and success/exception cleanup.
    module.torch = SimpleNamespace(device=lambda name: torch.device("cpu"))

    def cli_prefix():
        assert torch.empty(0).device.type == "cpu"
        tilelang.disable_cache()
        if raises:
            raise RuntimeError("stop before original tensors/compile/launch")

    module._run_original_cli = cli_prefix
    try:
        CacheState._enabled = enabled
        with torch.device("meta"):
            if raises:
                with pytest.raises(RuntimeError, match="stop before original tensors"):
                    module.main()
            else:
                module.main()
            assert torch.empty(0).device.type == "meta"
        assert CacheState._enabled is enabled
    finally:
        CacheState._enabled = original


@pytest.mark.a2a3_native
@pytest.mark.parametrize("entry", ENTRIES, ids=lambda row: Path(row["source_path"]).stem)
def test_full_default_frontend_and_independent_workspace_symbol(entry):
    from tilelang import tvm

    module = _load(entry)
    request = next(r for r in MANIFEST["factory_requests"] if r["source_path"] == entry["source_path"])
    function = module.sparse_attention_fwd.get_tir(**request["kwargs"])
    tvm.ir.assert_structural_equal(function, tvm.ir.load_json(tvm.ir.save_json(function)))
    shapes = {buffer.name: list(buffer.shape) for buffer in function.buffer_map.values()}
    ws = shapes["workspace_1"]
    if "developer" in entry["source_path"]:
        assert isinstance(ws[0], tvm.tirx.Var) and ws[0].name == "workspace_block_num"
        q = shapes["Q"]
        assert not ws[0].same_as(q[0]) and not ws[0].same_as(q[1])
        assert module.sparse_attention_fwd.workspace_idx is None
    elif request["dialect"] == "pto":
        assert int(ws[0]) == 2184
    assert module.sparse_attention_fwd.out_idx == [3]
