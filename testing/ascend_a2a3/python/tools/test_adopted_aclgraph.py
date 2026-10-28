"""Original graph asset import/caller contracts; graph execution is separate."""

import builtins
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import tilelang
from tilelang.env import CacheState

ASSET = Path(__file__).parents[4] / "examples/ascend_a2a3/aclgraph/rms_rope_aclgraph.py"


def _module():
    spec = importlib.util.spec_from_file_location("adopted_graph", ASSET)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_help_no_seed_cache_compile_or_backend_registration(monkeypatch, capsys):
    original = builtins.__import__
    attempted = []

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] == "torch_npu":
            attempted.append(name)
            raise AssertionError("import/help cannot register an NPU backend")
        return original(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("import/help cannot query devices, seed, compile or alter cache")

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(torch, "device", forbidden)
    monkeypatch.setattr(torch, "manual_seed", forbidden)
    monkeypatch.setattr(tilelang, "compile", forbidden)
    monkeypatch.setattr(tilelang, "lower", forbidden)
    monkeypatch.setattr(tilelang, "disable_cache", forbidden)
    module = _module()
    assert module.device == "npu"
    with pytest.raises(SystemExit) as caught:
        module.main(["--help"])
    assert caught.value.code == 0 and "--shape" in capsys.readouterr().out
    assert attempted == []


@pytest.mark.parametrize(
    "args,shape,eps", [([], [16, 64, 512, 256], 1e-6), (["--shape", "1", "32", "128", "64", "--eps", "0.25"], [1, 32, 128, 64], 0.25)]
)
def test_original_argument_parser(args, shape, eps):
    result = _module().parse_args(args)
    assert result.shape == shape and result.eps == eps


def test_malformed_shape_stops_before_backend_or_cache(capsys):
    with pytest.raises(SystemExit) as caught:
        _module().main(["--shape", "1", "2", "3"])
    assert caught.value.code == 2 and "expected 4 arguments" in capsys.readouterr().err


@pytest.mark.parametrize("conflicts", [False, True])
def test_explicit_owned_cache_is_required_before_backend_or_execution(conflicts, tmp_path, monkeypatch):
    module = _module()
    original = builtins.__import__

    def forbidden_import(name, *args, **kwargs):
        if name.split(".")[0] == "torch_npu":
            pytest.fail("cache refusal must precede NPU registration")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbidden_import)
    monkeypatch.setattr(tilelang, "disable_cache", lambda: pytest.fail("refused CLI altered caller cache"))
    monkeypatch.setattr(torch, "manual_seed", lambda *args: pytest.fail("refused CLI seeded caller RNG"))
    if conflicts:
        monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "explicit"))
        monkeypatch.setattr(vars(type(tilelang.env))["TILELANG_CACHE_DIR"], "_forced_value", str(tmp_path / "caller"))
        message = "conflicts with the active cache-directory override"
    else:
        monkeypatch.delenv("TILELANG_CACHE_DIR", raising=False)
        message = "Set a dedicated TILELANG_CACHE_DIR"
    with pytest.raises(RuntimeError, match=message):
        module.main([])


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_exception_restores_exact_caller_memos_device_cache_and_CPU_rng(enabled, tmp_path, monkeypatch):
    module = _module()
    token = object()
    module.device = token
    prior = []
    for factory in (module.rms_norm_kernel, module.rope_kernel_in_place):
        for name in ("_kernel_cache", "_call_form_cache"):
            value = {"caller": token}
            setattr(factory, name, value)
            prior.append((factory, name, value, value.copy()))
    original_import = builtins.__import__
    backend_imports = []

    def fake_backend_import(name, *args, **kwargs):
        if name == "torch_npu":
            backend_imports.append(name)
            return SimpleNamespace()
        return original_import(name, *args, **kwargs)

    fake_device = object()
    monkeypatch.setattr(builtins, "__import__", fake_backend_import)
    monkeypatch.setattr(module, "torch", SimpleNamespace(device=lambda name: fake_device, random=torch.random))
    monkeypatch.setattr(CacheState, "_enabled", enabled)
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "owned"))
    monkeypatch.setattr(vars(type(tilelang.env))["TILELANG_CACHE_DIR"], "_forced_value", str(tmp_path / "owned"))
    cpu_rng = torch.random.get_rng_state().clone()

    def before_real_graph_execution(args):
        # This is a host-only exception-boundary test, not a fake kernel or a
        # successful graph. Never enter the original tensor/graph body.
        assert module.device is fake_device
        for factory, name, old, _ in prior:
            assert getattr(factory, name) is not old and getattr(factory, name) == {}
        tilelang.disable_cache()
        torch.manual_seed(42)
        raise RuntimeError("CPU-only boundary before real graph execution")

    monkeypatch.setattr(module, "_run_original", before_real_graph_execution)
    with pytest.raises(RuntimeError, match="CPU-only boundary before real graph execution"):
        module.main([])
    assert backend_imports == ["torch_npu"]
    assert CacheState._enabled is enabled and module.device is token
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    assert all(getattr(factory, name) is old and old == snapshot for factory, name, old, snapshot in prior)


def test_original_CPU_references_preserve_rotation_and_normalization():
    module = _module()
    q = torch.ones(2, 2, 4)
    assert torch.equal(module.rms_norm_reference(q, 1e-6), torch.ones_like(q, dtype=torch.float16))
    zeros = torch.zeros(2, 4)
    ones = torch.ones(2, 4)
    assert torch.equal(module.torch_rope_ref(q, zeros, ones), q.half())
    expected = torch.tensor([-1.0, 1.0, -1.0, 1.0]).expand(2, 2, 4).half()
    assert torch.equal(module.torch_rope_ref(q, ones, zeros), expected)


SOURCE_REQUESTS = json.loads(Path(__file__).with_name("data").joinpath("aclgraph_default_requests.json").read_text())["requests"]


@pytest.mark.a2a3_native
@pytest.mark.parametrize("source_request", SOURCE_REQUESTS, ids=lambda row: row["factory"])
def test_original_default_factory_source_and_void_host_ABI(source_request):
    from tilelang import tvm
    from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs

    module = _module()
    factory = getattr(module, source_request["factory"])
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    func = factory._get_tir_with_target(target, *source_request["args"], **source_request["kwargs"])
    with tvm.transform.PassContext(config=normalize_a2a3_pass_configs(target, factory.pass_configs)):
        artifact = tilelang.lower(func, target=target)
    actual = [{"dtype": str(p.dtype).removeprefix("torch."), "shape": [str(d) for d in p.shape]} for p in artifact.params]
    assert actual == source_request["TA_expected_params"]
    assert (
        next(line.strip() for line in artifact.kernel_source.splitlines() if "void call(" in line)
        == source_request["TA_expected_host_prototype"]
    )
