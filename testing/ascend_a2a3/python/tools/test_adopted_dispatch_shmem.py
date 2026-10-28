"""Preserved distributed asset: host/conditional source only, never a worker."""

import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch
import tilelang
from tilelang.env import CacheState
from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache
from tilelang.cache.kernel_cache import KernelCache

ASSET = Path(__file__).parents[4] / "examples/ascend_a2a3/dispatch_combine/dispatch_combine_shmem.py"
EXPECTED = json.loads((Path(__file__).parent / "data/shmem_dispatch_cpu_expected.json").read_text())


def _module():
    spec = importlib.util.spec_from_file_location("adopted_dispatch_shmem", ASSET)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_help_and_invalid_args_do_not_load_runtime_clear_or_spawn(monkeypatch, capsys):
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"shmem", "torch_npu"}:
            pytest.fail("import/help loaded distributed runtime")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    monkeypatch.setattr(A2A3CythonKernelCache, "clear_cache", lambda self: pytest.fail("import/help cleared cache"))
    module = _module()
    monkeypatch.setattr(module.mp, "Process", lambda *args, **kwargs: pytest.fail("import/help created worker"))
    with pytest.raises(SystemExit) as caught:
        module.main(["--help"])
    assert caught.value.code == 0 and "16-rank" in capsys.readouterr().out
    with pytest.raises(SystemExit) as caught:
        module.main(["--shape", "1"])
    assert caught.value.code == 2


@pytest.mark.parametrize("conflict", [False, True])
def test_cache_guard_precedes_runtime_and_clear(conflict, tmp_path, monkeypatch):
    module = _module()
    if conflict:
        monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "requested"))
        monkeypatch.setattr(vars(type(tilelang.env))["TILELANG_CACHE_DIR"], "_forced_value", str(tmp_path / "caller"))
        message = "conflicts with the active cache-directory override"
    else:
        monkeypatch.delenv("TILELANG_CACHE_DIR", raising=False)
        message = "Set a dedicated TILELANG_CACHE_DIR"
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: pytest.fail("refused CLI queried runtime dependency"))
    with pytest.raises(RuntimeError, match=message):
        module.main([])


def test_actual_missing_shmem_diagnosed_before_registration_or_clear(tmp_path, monkeypatch):
    module = _module()
    if importlib.util.find_spec("shmem") is not None:
        pytest.skip("requires the actual missing Python extension")
    path = str(tmp_path / "owned")
    monkeypatch.setenv("TILELANG_CACHE_DIR", path)
    monkeypatch.setattr(vars(type(tilelang.env))["TILELANG_CACHE_DIR"], "_forced_value", path)
    monkeypatch.setattr(A2A3CythonKernelCache, "clear_cache", lambda self: pytest.fail("missing runtime cleared cache"))
    with pytest.raises(ModuleNotFoundError, match="pinned shmem Python extension"):
        module.main([])


def test_all_original_CPU_rank_inputs_match_fixed_source_and_conditional_counts_without_fake_dispatch():
    module = _module()
    before = random.getstate()
    cpu = torch.random.get_rng_state().clone()
    counts = [0] * 16
    try:
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            for expected in EXPECTED["CPU_input_hashes"]:
                x, ids = module.init_input(expected["rank"], 64, 7168, 4, 16, 3)
                assert x.device.type == ids.device.type == "cpu"
                for tensor, key in [(x, "x_bytes_sha256"), (ids, "expert_ids_bytes_sha256")]:
                    assert hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() == expected[key]
                for dest in range(16):
                    counts[dest] += int(((ids // 3) == dest).sum().item())
    finally:
        random.setstate(before)
    assert counts == EXPECTED["conditional_expected_counts"] and sum(counts) == 4096
    assert torch.equal(torch.random.get_rng_state(), cpu) and random.getstate() == before


@pytest.mark.parametrize("enabled", [False, True])
def test_scope_failure_before_real_workers_restores_exact_globals_rng_cache_and_sentinels(enabled, tmp_path, monkeypatch):
    module = _module()
    path = str(tmp_path / "owned")
    monkeypatch.setenv("TILELANG_CACHE_DIR", path)
    monkeypatch.setattr(vars(type(tilelang.env))["TILELANG_CACHE_DIR"], "_forced_value", path)
    cache = A2A3CythonKernelCache()
    caller = {"caller": object()}
    monkeypatch.setattr(cache, "_memory_cache", caller)
    monkeypatch.setattr(CacheState, "_enabled", enabled)
    staging = KernelCache._last_cleaned_staging_root
    original_random = random.getstate()
    original_rng = torch.random.get_rng_state().clone()
    caller_H = object()
    module.H = caller_H
    names = ("Bs", "H", "K", "ep_world_size", "local_expert_num", "aiv_num", "num_processes")
    old_names = {name: module.__dict__[name] for name in names if name in module.__dict__}
    cache_root = Path(cache._get_cache_root())
    cache_root.mkdir(parents=True, exist_ok=True)
    sentinel = cache_root / "owned-clear-sentinel"
    sentinel.write_text("owned")
    outside = tmp_path / "caller-sentinel"
    outside.write_text("keep")
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: object())
    original_import = builtins.__import__

    def import_boundary(name, *args, **kwargs):
        if name in {"torch_npu", "shmem"}:
            return SimpleNamespace()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_boundary)

    def stop_before_real_worker():
        assert cache._memory_cache is not caller and not sentinel.exists()
        module.H = 7168
        module.Bs = 64
        random.seed(1)
        torch.manual_seed(1)
        tilelang.disable_cache()
        raise RuntimeError("CPU sentinel before original process or tensor program")

    monkeypatch.setattr(module, "_run_original_cli", stop_before_real_worker)
    with pytest.raises(RuntimeError, match="CPU sentinel before original process"):
        module.main([])
    assert {name: module.__dict__[name] for name in names if name in module.__dict__} == old_names
    assert cache._memory_cache is caller and CacheState._enabled is enabled
    assert KernelCache._last_cleaned_staging_root == staging and outside.read_text() == "keep"
    assert random.getstate() == original_random and torch.equal(torch.random.get_rng_state(), original_rng)


@pytest.mark.a2a3_native
@pytest.mark.parametrize("rank", range(16))
def test_original_default_dispatch_same_source_zero_dimension_rejection(rank):
    from tilelang import tvm

    module = _module()
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    func = module.moe_dispatch_kernel._get_tir_with_target(target, 64, 7168, 4, 16, 3, rank, 7424, 48)
    with tvm.transform.PassContext(config=module.pass_configs), pytest.raises(tvm.error.InternalError, match="Divide by zero"):
        tilelang.lower(func, target=target)


@pytest.mark.a2a3_native
@pytest.mark.parametrize("rank", range(16))
def test_conditional_expected_count_combine_source_is_not_an_actual_dispatch_result(rank, tmp_path, request):
    from tilelang import tvm

    module = _module()
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    func = module.moe_combine_kernel._get_tir_with_target(
        target, 64, EXPECTED["conditional_expected_counts"][rank], 7168, 4, 16, 3, rank, 48
    )
    with tvm.transform.PassContext(config=module.pass_configs):
        artifact = tilelang.lower(func, target=target)
    assert len(artifact.params) == 7
    assert [int(x) for x in artifact.params[0].shape] == [EXPECTED["conditional_expected_counts"][rank], 7168]
    assert "tl::ascend::shmem_ub_put_nbi<bfloat16_t>" in artifact.kernel_source
    path = tmp_path / ("conditional_CPU_expected-combine-rank" + str(rank) + ".cc")
    path.write_text(artifact.kernel_source)
    request.node.user_properties.append(("shmem_current_source", str(path)))
