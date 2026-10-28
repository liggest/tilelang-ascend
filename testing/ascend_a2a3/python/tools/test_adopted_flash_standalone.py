"""Complete standalone Flash assets: host isolation and real source, no device."""

import builtins
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import tilelang
from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache
from tilelang.cache.kernel_cache import KernelCache
from tilelang.env import CacheState

ROOT = Path(__file__).parents[4]
NAMES = ["flash_attn_bhsd_cc_sync", "paged_flash_attn_bhsd"]


def _module(name):
    path = ROOT / "examples/ascend_a2a3/flash_attention" / (name + ".py")
    spec = importlib.util.spec_from_file_location("standalone_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", NAMES)
def test_import_help_preserve_rng_cache_and_skip_registration(name, monkeypatch, capsys):
    original = builtins.__import__

    def guard(module, *args, **kwargs):
        if module == "torch_npu":
            pytest.fail("import/help registered NPU")
        return original(module, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    monkeypatch.setattr(A2A3CythonKernelCache, "clear_cache", lambda self: pytest.fail("import/help cleared cache"))
    raw = CacheState._enabled
    rng = torch.random.get_rng_state().clone()
    module = _module(name)
    with pytest.raises(SystemExit) as caught:
        module.main(["--help"])
    assert caught.value.code == 0 and "usage:" in capsys.readouterr().out
    assert CacheState._enabled is raw and torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("conflict", [False, True])
def test_owned_cache_validation_precedes_registration_clear_or_allocation(name, conflict, tmp_path, monkeypatch):
    module = _module(name)
    if conflict:
        monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "requested"))
        monkeypatch.setattr(vars(type(tilelang.env))["TILELANG_CACHE_DIR"], "_forced_value", str(tmp_path / "caller"))
        message = "conflicts with the active cache-directory override"
    else:
        monkeypatch.delenv("TILELANG_CACHE_DIR", raising=False)
        message = "Set a dedicated TILELANG_CACHE_DIR"
    monkeypatch.setattr(module, "_load_execution_module", lambda: pytest.fail("refused CLI loaded execution factory"))
    with pytest.raises(RuntimeError, match=message):
        module.main([])


@pytest.mark.parametrize("name", NAMES)
def test_fresh_namespace_uses_actual_JIT_constructor_cache_policy(name):
    module = _module(name)
    execution = module._load_execution_module()
    factory_name = "flash_attention_fwd" if name == NAMES[0] else "paged_flash_attention_fwd"
    original = getattr(module, factory_name)
    fresh = getattr(execution, factory_name)
    assert type(fresh) is type(original) and fresh is not original
    assert type(fresh._call_form_cache) is type(original._call_form_cache)
    assert fresh._call_form_cache is not original._call_form_cache
    assert fresh.func.__code__.co_filename == original.func.__code__.co_filename


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("enabled", [False, True])
def test_actual_cache_scope_restores_caller_memory_raw_staging_and_rng(name, enabled, tmp_path, monkeypatch):
    module = _module(name)
    owned = str(tmp_path / "owned")
    monkeypatch.setenv("TILELANG_CACHE_DIR", owned)
    monkeypatch.setattr(vars(type(tilelang.env))["TILELANG_CACHE_DIR"], "_forced_value", owned)
    cache = A2A3CythonKernelCache()
    caller = {"caller": object()}
    monkeypatch.setattr(cache, "_memory_cache", caller)
    monkeypatch.setattr(CacheState, "_enabled", enabled)
    staging = KernelCache._last_cleaned_staging_root
    cpu = torch.random.get_rng_state().clone()
    sentinel = Path(cache._get_cache_root()) / "owned-sentinel"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("owned")
    outside = tmp_path / "caller-sentinel"
    outside.write_text("keep")
    execution = module._load_execution_module()
    monkeypatch.setattr(module, "_load_execution_module", lambda: execution)
    original_import = builtins.__import__

    def registration_boundary(value, *args, **kwargs):
        return SimpleNamespace() if value == "torch_npu" else original_import(value, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", registration_boundary)
    real_device = torch.device
    monkeypatch.setattr(module, "torch", SimpleNamespace(random=torch.random, device=lambda value: real_device("cpu")))

    def stop_before_tensor_program(*args):
        if name == NAMES[0]:
            assert not CacheState.is_enabled() and sentinel.exists() and cache._memory_cache is caller
        else:
            assert not sentinel.exists() and cache._memory_cache is not caller
        raise RuntimeError("CPU sentinel before any NPU tensor or kernel")

    if name == NAMES[0]:
        monkeypatch.setattr(execution, "_run_original", stop_before_tensor_program)
    else:
        # Preserve the real original clear position; stop at the next seed call,
        # before check_case allocates its original tensors.
        monkeypatch.setattr(torch, "manual_seed", stop_before_tensor_program)
    with pytest.raises(RuntimeError, match="CPU sentinel before any NPU tensor or kernel"):
        module.main([])
    assert cache._memory_cache is caller and CacheState._enabled is enabled
    assert KernelCache._last_cleaned_staging_root == staging
    assert outside.read_text() == "keep" and torch.equal(cpu, torch.random.get_rng_state())


def test_paged_original_h_is_heads_and_unknown_args_reported(capsys):
    module = _module(NAMES[1])
    args = module.parse_args(["-h", "3", "--batch", "2", "--seq_len", "256", "--hidden_dim", "512", "--original-unknown"])
    assert (args.batch, args.heads, args.seq_len, args.hidden_dim) == (2, 3, 256, 512)
    assert "Unknown args:" in capsys.readouterr().out


@pytest.mark.a2a3_native
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("dialect", ["ascendc", "pto"])
def test_original_default_full_source(name, dialect, tmp_path, request):
    from tilelang import tvm

    module = _module(name)
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": [dialect, "ascend"]})
    if name == NAMES[0]:
        factory = module.flash_attention_fwd
        args = (1, 512)
        configs = module.pass_configs
        shape = [1, 1, 128, 512]
    else:
        factory = module.paged_flash_attention_fwd
        args = (1, 16, 128, 128, 1, 1)
        configs = factory.pass_configs
        shape = [1, 16, 128, 128]
    func = factory._get_tir_with_target(target, *args)
    with tvm.transform.PassContext(config=configs):
        artifact = tilelang.lower(func, target=target)
    assert [int(value) for value in artifact.params[0].shape] == shape
    assert "void call(" in artifact.kernel_source
    path = tmp_path / (name + "_" + dialect + ".cc")
    path.write_text(artifact.kernel_source)
    request.node.user_properties.append(("flash_standalone_current_source", str(path)))


def test_paged_original_default_CPU_cache_table_generation_is_not_device_execution():
    module = _module(NAMES[1])
    data = torch.zeros((1, 16, 128, 128), dtype=torch.float16, device="cpu")
    table = torch.full((1, 1), -1, dtype=torch.int32, device="cpu")
    cache = module.gen_cache(data, table, 128)
    assert cache.shape == (1, 128, 16, 128) and cache.device.type == "cpu"
    assert table.tolist() == [[0]] and torch.equal(cache[0], data[0].permute(1, 0, 2))
