"""Complete masked sparse attention imports, input-device reference, state and dynamic ABI."""

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
MANIFEST = json.loads((ROOT / "maint/ascend_a2a3/ecosystem_sparse_mask_ports.json").read_text())
ENTRIES = [e for e in MANIFEST["entries"] if e["source_path"].endswith(("mask.py", "mask_pa.py"))]


def _load(entry):
    path = ROOT / entry["destination_path"]
    spec = importlib.util.spec_from_file_location("_sparse_mask_owned", path)
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
    for statement in entry["retained_original_module_constants_AST"]:
        assert any(ast.unparse(n) == statement for n in ast.parse(path.read_bytes()).body)
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
    reference = module.ref_sparse_attention_fwd_interface
    assert inspect.signature(reference) == inspect.signature(reference.__wrapped__)


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda row: Path(row["source_path"]).stem)
def test_reference_uses_input_device_and_keeps_causal_mask(entry):
    import torch

    module = _load(entry)
    q = torch.zeros((1, 2, 4, 576), dtype=torch.bfloat16)
    kv = torch.stack((torch.full((1, 576), 2.0), torch.full((1, 576), 6.0))).unsqueeze(0).bfloat16()
    indices = torch.tensor([[[[0, 1]], [[0, 1]]]], dtype=torch.int32)
    with torch.device("meta"):
        actual = module.ref_sparse_attention_fwd_interface(q, kv, indices)
        assert torch.empty(0).device.type == "meta"
    expected = torch.full((1, 2, 4, 512), 4.0, dtype=torch.bfloat16)
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
def test_full_default_frontend_and_five_core24_workspaces(entry):
    from tilelang import tvm

    module = _load(entry)
    request = next(r for r in MANIFEST["factory_requests"] if r["source_path"] == entry["source_path"])
    function = module.sparse_attention_fwd.get_tir(**request["kwargs"])
    tvm.ir.assert_structural_equal(function, tvm.ir.load_json(tvm.ir.save_json(function)))
    assert len(function.params) == (12 if "_pa" in entry["source_path"] else 9)
    workspace_start = 7 if "_pa" in entry["source_path"] else 4
    for parameter in function.params[workspace_start:]:
        assert int(function.buffer_map[parameter].shape[0]) == 24
    assert module.sparse_attention_fwd.workspace_idx == entry["original_workspace_idx"]
    assert module.sparse_attention_fwd.out_idx == [3]


def _golden():
    path = ROOT / "examples/ascend_a2a3/sparse_flash_attention/sfa_golden.py"
    spec = importlib.util.spec_from_file_location("_sfa_cpu_golden_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_eight_original_golden_definitions_and_PA_import_are_local(monkeypatch):
    module = _golden()
    entry = next(e for e in MANIFEST["entries"] if e["source_path"].endswith("sfa_golden.py"))
    functions = {n.name: n for n in ast.parse(Path(module.__file__).read_bytes()).body if isinstance(n, ast.FunctionDef)}
    assert len(functions) == 8
    assert {name: _digest(node) for name, node in functions.items()} == entry["original_definition_AST_sha256"]
    foreign = ModuleType("sfa_golden")
    monkeypatch.setitem(sys.modules, "sfa_golden", foreign)
    before = sys.path.copy()
    pa = _load(next(e for e in ENTRIES if "_pa" in e["source_path"]))
    assert sys.path == before and sys.modules["sfa_golden"] is foreign
    assert Path(pa.ref.__file__).resolve() == Path(module.__file__).resolve()
    assert pa.ref.__name__ not in sys.modules


def test_original_page_table_gather_preserves_repeated_pages_and_unused_capacity():
    import torch

    golden = _golden()
    pages = torch.arange(8).view(4, 2, 1, 1).float()
    actual = golden.pa_to_bsnd(pages, torch.tensor([[3, 0], [1, 2]]), [2, 4])
    expected = torch.tensor([[[[6]], [[7]], [[0]], [[0]]], [[[2]], [[3]], [[4]], [[5]]]], dtype=torch.float32)
    assert torch.equal(actual, expected)
    repeated = golden.pa_to_bsnd(pages, torch.zeros((1, 2), dtype=torch.int32), [4])
    assert torch.equal(repeated[0, :4, 0, 0], torch.tensor([0.0, 1, 0, 1]))


def test_original_TND_conversion_and_cumulative_sequence_error_contract(capsys):
    import numpy as np
    import torch

    golden = _golden()
    raw = np.arange(12).reshape(3, 2, 2)
    bsnd = golden.trans_tnd_to_bsnd(raw, raw.shape, [1, 0, 2])
    actual = golden.trans_bnsd_to_tnd(torch.from_numpy(bsnd).transpose(1, 2), None, [1, 0, 2])
    assert np.array_equal(actual.numpy(), raw)
    assert golden.trans_tnd_actseq([1, 1, 3]) == [1, 0, 2]
    assert golden.trans_tnd_actseq([3, 2]) == [3]
    assert "Wrong input actseq" in capsys.readouterr().out
    with pytest.raises(IndexError):
        golden.trans_tnd_actseq([])


def test_original_sparse_tail_and_causal_mask_and_stable_softmax():
    import numpy as np

    golden = _golden()
    source = np.arange(5).reshape(1, 1, 5, 1)
    key, value = golden.gather_kv(source, source * 2, [0, 2, -1, 1], 2, 4, 0, 0, 0, 5)
    assert key[:, 0].tolist() == [0, 1, 4] and value[:, 0].tolist() == [0, 2, 8]
    masked = golden.mask(np.zeros((1, 3)), 2, 5, [0, 2, -1], 0, 2)
    assert np.array_equal(masked, np.array([[0.0, 0.0, -1e12]]))
    probability = golden.softmax(np.array([[1000, 1000]], dtype=np.float64))
    assert probability.dtype == np.float32 and np.array_equal(probability, np.array([[0.5, 0.5]], dtype=np.float32))


def test_complete_original_numpy_golden_CPU_causal_selected_keys():
    import numpy as np
    import torch

    golden = _golden()
    q = torch.zeros((1, 2, 2, 2))
    key = torch.tensor([[[[2.0, 2]], [[6.0, 6]], [[10.0, 10]]]])
    indices = torch.tensor([[[[0, 1]], [[1, 2]]]], dtype=torch.int32)
    # Original helper intentionally derives V from key.copy(), retaining its
    # original value-argument behavior rather than introducing a new reference.
    actual = golden.cpu_sparse_flash_attention(
        q,
        key,
        torch.zeros_like(key),
        indices,
        1.0,
        1,
        torch.tensor([2]),
        torch.tensor([3]),
        query_rope=torch.zeros((1, 2, 2, 1)),
        key_rope=torch.zeros((1, 3, 1, 1)),
    )
    expected = np.array([[[[4, 4], [4, 4]], [[8, 8], [8, 8]]]], dtype=np.float32)
    assert np.array_equal(actual, expected)
