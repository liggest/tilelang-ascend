"""Complete original recommendation helpers, catalog, imports and scoped cache."""

import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import random
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = json.loads((ROOT / "maint/ascend_a2a3/ecosystem_generative_ports.json").read_text())
ENTRIES = MANIFEST["entries"]
MAIN = ENTRIES[0]


def _digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def _load(entry=MAIN):
    spec = importlib.util.spec_from_file_location("_owned_generative_example", ROOT / entry["destination_path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda row: Path(row["source_path"]).stem)
def test_all_original_definitions_helpers_and_CLI_are_complete(entry):
    path = ROOT / entry["destination_path"]
    tree = ast.parse(path.read_bytes())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name, expected in entry["original_definition_AST_sha256"].items():
        node = functions[name]
        if name == "test":
            node.decorator_list = []

            class OriginalClear(ast.NodeTransformer):
                def visit_Call(self, call):
                    self.generic_visit(call)
                    if isinstance(call.func, ast.Name) and call.func.id == "_clear_owned_cache":
                        call.func = ast.parse("tilelang.cache.clear_cache()", mode="eval").body.func
                    return call

            node = OriginalClear().visit(node)
        assert _digest(node) == expected
    if "whole_original_module_AST_sha256" in entry:
        assert _digest(tree) == entry["whole_original_module_AST_sha256"]
    else:
        assert _digest(ast.Module(body=functions["_run_original_cli"].body, type_ignores=[])) == entry["original_CLI_AST_sha256"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["destination_sha256"]


def test_main_import_is_lazy_without_seed_cache_tensor_or_path_side_effects(monkeypatch):
    import tilelang
    import torch
    from tilelang.env import CacheState
    from tilelang.jit import JITImpl

    def forbidden(*args, **kwargs):
        raise AssertionError("import must not build IR, allocate, seed, clear or launch")

    before = sys.path.copy()
    enabled = CacheState._enabled
    with monkeypatch.context() as scoped:
        for name in ("manual_seed", "set_default_device", "empty", "zeros", "randn", "arange"):
            scoped.setattr(torch, name, forbidden)
        scoped.setattr(tilelang, "disable_cache", forbidden)
        scoped.setattr(JITImpl, "get_tir", forbidden)
        module = _load()
    assert module._HELPERS == {} and sys.path == before and CacheState._enabled is enabled
    assert module.T.__name__ == "tilelang.ascend.a2a3.language"


@pytest.mark.parametrize("name", ["golden", "testcase"])
def test_lazy_helpers_are_exact_local_files_and_preserve_public_signatures(name, monkeypatch):
    sentinel = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, sentinel)
    before = sys.path.copy()
    module = _load()
    helper = module._load_local_helper(name)
    assert Path(helper.__file__).resolve() == (ROOT / MAIN["destination_path"]).with_name(name + ".py")
    public_name = "prepare_data" if name == "testcase" else "golden_attention_simulated_kernel"
    assert inspect.signature(getattr(module, public_name)) == inspect.signature(getattr(helper, public_name))
    assert module._load_local_helper(name) is helper
    assert sys.modules[name] is sentinel and sys.path == before
    with pytest.raises(ValueError, match="unknown local generative"):
        module._load_local_helper("../foreign")


def test_all_462_original_generated_configs_and_flags_without_RNG_mutation():
    import torch

    original = random.getstate()
    rng = torch.get_rng_state().clone()
    helper = _load()._load_local_helper("testcase")
    actual = json.dumps(helper.test_configs, sort_keys=True, separators=(",", ":")).encode()
    assert len(helper.test_configs) == 462
    assert hashlib.sha256(actual).hexdigest() == MANIFEST["validation"]["complete_catalog_canonical_SHA256"]
    assert random.getstate() == original and torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("function", ["golden_attention_float64", "golden_attention_simulated_kernel"])
def test_CPU_reference_causal_prefix_and_GQA_uniform_means(function):
    import torch

    with torch.random.fork_rng(devices=[]):
        module = _load()
        data = module.prepare_data(dict(H=4, D=4, seg_lengths=[[128, 1, 1]], rules=[0, 1, 2], matched_prefix_arr=[128], kv_group=2))
        data["query_snd"].zero_()
        data["key_snd"].zero_()
        data["key_cache"].zero_()
        data["value_snd"].fill_(3)
        data["value_cache"].fill_(1)
        keys = [
            "query_snd",
            "key_snd",
            "value_snd",
            "segment_offsets_i32",
            "segment_rules_i32",
            "q_seq_starts_i32",
            "matched_prefix_lens_i32",
            "key_cache",
            "value_cache",
            "block_table_tensor",
            "block_size",
            "sm_scale",
        ]
        result = getattr(module._load_local_helper("golden"), function)(*(data[key] for key in keys))
        expected = torch.full((2, 4, 4), 131 / 129, dtype=torch.float64)
        expected[1] = 134 / 130
        tolerance = 1e-2 if "simulated" in function else 1e-12
        torch.testing.assert_close(result.double(), expected, rtol=tolerance, atol=tolerance)


def test_original_unaligned_cached_prefix_is_rejected_before_RNG_or_tensor_creation():
    import torch

    module = _load()
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="matched_prefix_arr\\[0\\] = 1 is not a multiple of block_N=128"):
        module.prepare_data(dict(H=4, D=4, seg_lengths=[[128, 1, 1]], rules=[0, 1, 2], matched_prefix_arr=[1], kv_group=2))
    assert torch.equal(torch.get_rng_state(), rng)


def test_missing_explicit_cache_blocks_test_and_CLI_before_NPU_import(monkeypatch):
    module = _load()
    monkeypatch.delenv("TILELANG_CACHE_DIR", raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    with pytest.raises(RuntimeError, match="dedicated TILELANG_CACHE_DIR"):
        module.test(MAIN["original_default_configs"][0])
    with pytest.raises(RuntimeError, match="dedicated TILELANG_CACHE_DIR"):
        module.main()


@pytest.mark.parametrize("enabled", [None, False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_actual_owned_clear_and_raw_caller_memory_restore(enabled, raises, isolated_a2a3_cache, monkeypatch):
    import tilelang
    from tilelang import env
    from tilelang.env import CacheState
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache
    from tilelang.cache.kernel_cache import KernelCache

    module = _load()
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(isolated_a2a3_cache))
    isolated_a2a3_cache.mkdir(parents=True, exist_ok=True)
    sibling = isolated_a2a3_cache.parent / "caller.txt"
    sibling.write_text("keep")
    cache = A2A3CythonKernelCache()
    owned_root = Path(cache._get_cache_root())
    assert owned_root.is_relative_to(isolated_a2a3_cache)
    owned_root.mkdir(parents=True, exist_ok=True)
    owned = owned_root / "owned.txt"
    owned.write_text("task-only")
    memory = cache._memory_cache
    contents = memory.copy()
    original = CacheState._enabled
    staging = KernelCache._last_cleaned_staging_root
    descriptor = vars(type(env))["TILELANG_CACHE_DIR"]
    forced = descriptor._forced_value

    @module._owned_test_state
    def stop_before_NPU():
        tilelang.disable_cache()
        module._clear_owned_cache()
        assert cache._memory_cache is not memory and CacheState._enabled is False
        if raises:
            raise RuntimeError("CPU stop before original tensors or NPU")

    try:
        CacheState._enabled = enabled
        if raises:
            with pytest.raises(RuntimeError, match="CPU stop before"):
                stop_before_NPU()
        else:
            stop_before_NPU()
        assert cache._memory_cache is memory and memory == contents
        assert CacheState._enabled is enabled and KernelCache._last_cleaned_staging_root is staging
        assert descriptor._forced_value is forced and sibling.read_text() == "keep" and not owned.exists()
    finally:
        CacheState._enabled = original


@pytest.mark.a2a3_native
def test_full_default_frontend_three_explicit_workspaces_and_int32_bin_iters():
    from tilelang import tvm

    module = _load()
    function = module.mtgr_ragged_segment_attention_kernel.get_tir(**MANIFEST["factory_requests"][0]["kwargs"])
    tvm.ir.assert_structural_equal(function, tvm.ir.load_json(tvm.ir.save_json(function)), map_free_vars=True)
    assert len(function.params) == 15 and len(function.buffer_map) == 14
    assert function.params[-1].name == "bin_iters" and str(function.params[-1].dtype) == "int32"
    buffers = [function.buffer_map[value] for value in function.params[:14]]
    assert [list(map(int, buffer.shape)) for buffer in buffers[7:10]] == [[24, 14, 128, 128]] * 3
    assert module.mtgr_ragged_segment_attention_kernel.workspace_idx is None
    assert module.mtgr_ragged_segment_attention_kernel.out_idx is None
