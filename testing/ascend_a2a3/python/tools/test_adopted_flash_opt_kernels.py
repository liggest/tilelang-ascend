"""Full fa_opt kernels and original CLI/bench shapes; never execute NPU code."""

import builtins
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import tilelang
from tilelang.env import CacheState

ROOT = Path(__file__).parents[4]
ASSETS = ["auto_pipeline/h16_d128.py", "auto_pipeline/h32_d512.py", "expert_v1/kernel.py"]
CASES = [
    (ASSETS[0], (4, 4096, 16, 16, 128)),
    (ASSETS[1], (4, 4096, 32, 32, 512)),
    (ASSETS[2], (4, 4096, 16, 16, 128)),
    (ASSETS[0], (2, 131072, 12, 1, 128)),
    (ASSETS[0], (2, 65536, 12, 1, 128)),
    (ASSETS[0], (1, 32768, 12, 1, 128)),
]


def _module(relative):
    path = ROOT / "examples/ascend_a2a3/flash_attention/fa_opt" / relative
    spec = importlib.util.spec_from_file_location("fa_opt_" + relative.replace("/", "_").replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("relative", ASSETS)
def test_import_help_have_no_registration_rng_or_cache_side_effect(relative, monkeypatch, capsys):
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "torch_npu":
            pytest.fail("import/help registered NPU")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    raw = CacheState._enabled
    cpu = torch.random.get_rng_state().clone()
    module = _module(relative)
    with pytest.raises(SystemExit) as caught:
        module.main(["--help"])
    assert caught.value.code == 0 and "usage:" in capsys.readouterr().out
    assert CacheState._enabled is raw and torch.equal(cpu, torch.random.get_rng_state())


@pytest.mark.parametrize("relative", ASSETS)
@pytest.mark.parametrize("conflict", [False, True])
def test_owned_cache_guard_precedes_registration_or_factory_loading(relative, conflict, tmp_path, monkeypatch):
    module = _module(relative)
    if conflict:
        monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "requested"))
        monkeypatch.setattr(vars(type(tilelang.env))["TILELANG_CACHE_DIR"], "_forced_value", str(tmp_path / "caller"))
        message = "conflicts with the active cache-directory override"
    else:
        monkeypatch.delenv("TILELANG_CACHE_DIR", raising=False)
        message = "Set a dedicated TILELANG_CACHE_DIR"
    monkeypatch.setattr(module, "_load_execution_module", lambda: pytest.fail("refused CLI loaded factory"))
    with pytest.raises(RuntimeError, match=message):
        module.main([])


@pytest.mark.parametrize("relative", ASSETS)
def test_fresh_module_preserves_real_factory_memo_constructor_types(relative):
    module = _module(relative)
    fresh = module._load_execution_module()
    a, b = module.flash_attention_fwd, fresh.flash_attention_fwd
    assert type(a) is type(b) and a is not b
    assert type(a._call_form_cache) is type(b._call_form_cache) and a._call_form_cache is not b._call_form_cache
    assert a.func.__code__.co_filename == b.func.__code__.co_filename


@pytest.mark.parametrize("relative", ASSETS)
@pytest.mark.parametrize("enabled", [False, True])
def test_original_disable_cache_and_RNG_scope_restore_on_failure(relative, enabled, tmp_path, monkeypatch):
    module = _module(relative)
    owned = str(tmp_path / "owned")
    monkeypatch.setenv("TILELANG_CACHE_DIR", owned)
    monkeypatch.setattr(vars(type(tilelang.env))["TILELANG_CACHE_DIR"], "_forced_value", owned)
    monkeypatch.setattr(CacheState, "_enabled", enabled)
    cpu = torch.random.get_rng_state().clone()
    original = builtins.__import__
    monkeypatch.setattr(
        builtins, "__import__", lambda name, *args, **kwargs: SimpleNamespace() if name == "torch_npu" else original(name, *args, **kwargs)
    )
    execution = module._load_execution_module()
    monkeypatch.setattr(module, "_load_execution_module", lambda: execution)
    real_device = torch.device
    monkeypatch.setattr(module, "torch", SimpleNamespace(random=torch.random, device=lambda value: real_device("cpu")))

    def before_NPU_tensor_or_kernel(args):
        assert not CacheState.is_enabled()
        torch.manual_seed(0)
        raise RuntimeError("CPU sentinel before original numerical program")

    monkeypatch.setattr(execution, "_run_original", before_NPU_tensor_or_kernel)
    with pytest.raises(RuntimeError, match="CPU sentinel before original numerical program"):
        module.main([])
    assert CacheState._enabled is enabled and torch.equal(cpu, torch.random.get_rng_state())


@pytest.mark.parametrize("relative,heads,dim,cross", [(ASSETS[0], 16, 128, 1), (ASSETS[1], 32, 512, None), (ASSETS[2], 16, 128, 2)])
def test_complete_original_CLI_defaults_and_explicit_head_flags(relative, heads, dim, cross):
    module = _module(relative)
    args = module.parse_args([])
    assert (args.B, args.S, args.H, args.D, args.q_heads, args.kv_heads, args.no_check) == (4, 4096, heads, dim, None, None, False)
    assert getattr(args, "cross_interval", None) == cross
    args = module.parse_args(["--B", "2", "--S", "131072", "--q-heads", "12", "--kv-heads", "1", "--no-check"])
    assert (args.B, args.S, args.q_heads, args.kv_heads, args.no_check) == (2, 131072, 12, 1, True)


def test_original_h16_KNOWN_BROKEN_warning_remains_source_not_runtime_acceptance():
    module = _module(ASSETS[0])
    assert "KNOWN BROKEN (UB out of bounds at runtime)" in module.__doc__
    assert "256KB" in module.__doc__ and "Kept as reference" in module.__doc__


@pytest.mark.a2a3_native
@pytest.mark.parametrize("relative,args", CASES)
def test_full_original_default_and_bench_zip_sources(relative, args, tmp_path, request):
    from tilelang import tvm

    module = _module(relative)
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    func = module.flash_attention_fwd._get_tir_with_target(target, *args)
    with tvm.transform.PassContext(config=module.pass_configs):
        artifact = tilelang.lower(func, target=target)
    assert [int(value) for value in artifact.params[0].shape] == [args[0], args[2], args[1], args[4]]
    assert "void call(" in artifact.kernel_source
    name = relative.replace("/", "_").removesuffix(".py") + "_" + "_".join(map(str, args)) + ".cc"
    path = tmp_path / name
    path.write_text(artifact.kernel_source)
    request.node.user_properties.append(("flash_opt_current_source", str(path)))


@pytest.mark.a2a3_native
@pytest.mark.parametrize(
    "relative,args,kwargs,message",
    [
        (ASSETS[0], (4, 4096, 3, 2, 128), {}, "heads_q must be a multiple of heads_kv"),
        (ASSETS[1], (4, 4096, 3, 2, 512), {}, "heads_q must be a multiple of heads_kv"),
        (ASSETS[2], (4, 4096, 3, 2, 128), {}, "heads_q must be a multiple of heads_kv"),
        (ASSETS[0], (4, 4096, 16, 16, 256), {}, "dim must be 128"),
        (ASSETS[2], (4, 4096, 16, 16, 256), {}, "dim must be 128"),
        (ASSETS[0], (4, 129, 16, 16, 128), {}, "seq_len (129) must be divisible by block_N (128)"),
        (ASSETS[2], (4, 129, 16, 16, 128), {}, "seq_len (129) must be divisible by block_N (128)"),
        (ASSETS[0], (4, 4096, 16, 16, 128), {"num_stages": 3}, "num_stages must be even for double buffering"),
        (ASSETS[2], (4, 4096, 16, 16, 128), {"num_stages": 3}, "num_stages must be even for double buffering"),
        (ASSETS[0], (4, 4096, 16, 16, 128), {"num_stages": 6}, "num_iters (32) must be divisible by num_stages (6)"),
    ],
)
def test_original_factory_validation_inputs_rejected_before_kernel(relative, args, kwargs, message):
    from tilelang import tvm

    module = _module(relative)
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    with pytest.raises(AssertionError) as error:
        module.flash_attention_fwd._get_tir_with_target(target, *args, **kwargs)
    assert str(error.value) == message
