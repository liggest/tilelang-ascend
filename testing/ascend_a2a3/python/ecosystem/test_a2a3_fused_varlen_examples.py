"""VL1 host contracts; no NPU registration, allocation or kernel execution."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = json.loads((ROOT / "maint/ascend_a2a3/ecosystem_fused_varlen_ports.json").read_text())
ENTRY = MANIFEST["entries"][0]
SOURCE = ROOT / ENTRY["destination_path"]


def _load():
    spec = importlib.util.spec_from_file_location("_vl1_host_contract", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_original_definitions_and_bytes():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == ENTRY["destination_sha256"]
    definitions = {node.name: node for node in ast.parse(SOURCE.read_bytes()).body if isinstance(node, ast.FunctionDef)}
    for name, expected in ENTRY["original_definition_AST_sha256"].items():
        assert hashlib.sha256(ast.dump(definitions[name], include_attributes=False).encode()).hexdigest() == expected


def test_import_and_help_defer_cache_rng_compiler_and_npu(monkeypatch, capsys):
    import builtins
    import tilelang
    import torch
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache
    from tilelang.jit import JITImpl

    def forbidden(*args, **kwargs):
        raise AssertionError("import/help reached a cache, RNG, compiler or NPU operation")

    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch_npu":
            forbidden()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    for name in ("manual_seed", "empty", "zeros", "randn", "set_default_device"):
        monkeypatch.setattr(torch, name, forbidden)
    monkeypatch.setattr(A2A3CythonKernelCache, "clear_cache", forbidden)
    monkeypatch.setattr(JITImpl, "get_tir", forbidden)
    monkeypatch.setattr(tilelang, "disable_cache", forbidden)
    monkeypatch.setattr(sys, "argv", [str(SOURCE), "--help"])
    module = _load()
    with pytest.raises(SystemExit) as stopped:
        module.cli_main()
    assert stopped.value.code == 0 and "--threads" in capsys.readouterr().out
    assert module.T.__name__ == "tilelang.ascend.a2a3.language"


def test_cache_guard_rejects_missing_or_foreign_descriptor(monkeypatch, tmp_path):
    from tilelang import env

    module = _load()
    monkeypatch.delenv("TILELANG_CACHE_DIR", raising=False)
    with pytest.raises(RuntimeError, match="dedicated TILELANG_CACHE_DIR"):
        module._require_owned_cache()
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "explicit"))
    descriptor = vars(type(env))["TILELANG_CACHE_DIR"]
    monkeypatch.setattr(descriptor, "_forced_value", str(tmp_path / "foreign"))
    with pytest.raises(RuntimeError, match="resolved cache differs"):
        module._require_owned_cache()


@pytest.mark.parametrize("enabled", [None, True, False])
@pytest.mark.parametrize("failure", [False, True])
def test_owned_clear_and_exact_caller_state_restoration(enabled, failure, monkeypatch, tmp_path):
    from tilelang import env
    from tilelang.env import CacheState
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache
    from tilelang.cache.kernel_cache import KernelCache

    module = _load()
    descriptor = vars(type(env))["TILELANG_CACHE_DIR"]
    monkeypatch.setattr(descriptor, "_forced_value", None)
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "owned"))
    cache = A2A3CythonKernelCache()
    original_memory = {"caller": object()}
    monkeypatch.setattr(cache, "_memory_cache", original_memory)
    monkeypatch.setattr(CacheState, "_enabled", enabled)
    staging = object()
    monkeypatch.setattr(KernelCache, "_last_cleaned_staging_root", staging)
    root = Path(cache._get_cache_root())
    root.mkdir(parents=True, exist_ok=True)
    old_entry = root / "old-task-entry"
    old_entry.write_text("remove only this task namespace")
    outside = tmp_path / "caller-sentinel"
    outside.write_text("preserve")

    def run():
        with module._owned_cli_cache():
            assert not old_entry.exists()
            assert cache._memory_cache is not original_memory and cache._memory_cache == {}
            CacheState._enabled = not enabled
            if failure:
                raise ValueError("original case failure")

    if failure:
        with pytest.raises(ValueError, match="original case failure"):
            run()
    else:
        run()
    assert cache._memory_cache is original_memory and list(original_memory) == ["caller"]
    assert CacheState._enabled is enabled and KernelCache._last_cleaned_staging_root is staging
    assert descriptor._forced_value is None and outside.read_text() == "preserve"


@pytest.mark.parametrize("threads", [1, 2, 8])
def test_original_cli_schedule_and_thread_failure_contract(threads, monkeypatch, tmp_path, capsys):
    """Exercise real Python scheduling, not a numerical substitute for the kernel."""
    import threading
    from tilelang import env
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

    module = _load()
    monkeypatch.setattr(vars(type(env))["TILELANG_CACHE_DIR"], "_forced_value", None)
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "task"))
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(sys, "argv", [str(SOURCE), "--threads", str(threads)])
    observed, failures, clears = [], [], []
    clear = A2A3CythonKernelCache.clear_cache

    def clear_once(self):
        clears.append(True)
        return clear(self)

    def case_observer(**kwargs):
        observed.append(kwargs)
        if threads > 1 and kwargs["seed"] == 41:
            raise ValueError("original thread failure stays in the thread")

    monkeypatch.setattr(A2A3CythonKernelCache, "clear_cache", clear_once)
    monkeypatch.setattr(module, "main", case_observer)
    monkeypatch.setattr(threading, "excepthook", lambda args: failures.append(args.exc_value))
    module.cli_main()
    assert sorted(observed, key=lambda row: row["seed"]) == sorted(ENTRY["original_CLI_cases"], key=lambda row: row["seed"])
    assert clears == [True] and len(failures) == int(threads > 1)
    assert "All tests passed!" in capsys.readouterr().out


@pytest.mark.parametrize("initial_index", [-1, 0])
@pytest.mark.parametrize("softplus_beta", [1.0, 2.0])
def test_original_cpu_golden_empty_sequence_and_negative_cache_index(initial_index, softplus_beta):
    import torch

    module = _load()
    query = torch.ones((1, 1, 1, 1))
    output, state = module.golden(
        torch.zeros(1),
        torch.zeros((1, 1)),
        torch.zeros(1),
        query,
        query,
        torch.full_like(query, 4),
        torch.zeros((1, 1)),
        torch.tensor([[[[2.0]]], [[[5.0]]]]),
        torch.tensor([initial_index, 1]),
        torch.tensor([0, 1, 1]),
        scale=0.25,
        use_qk_l2norm=False,
        softplus_beta=softplus_beta,
    )
    expected = 2.0 if initial_index < 0 else 2.0 + 2 ** (-1 / softplus_beta)
    torch.testing.assert_close(output, torch.full_like(output, expected / 4))
    torch.testing.assert_close(state[:, 0, 0, 0], torch.tensor([expected, 5.0]))


@pytest.mark.a2a3_native
@pytest.mark.parametrize("changes, message", [({"nv": 3}, "nv must be divisible"), ({"block_v": 3}, "block_v must be divisible")])
def test_original_factory_rejections_before_ir(changes, message):
    request = MANIFEST["factory_requests"][0]["kwargs"].copy()
    request.update(changes)
    with pytest.raises(ValueError, match=message):
        _load().kernel.get_tir(**request)
