"""Original paged expert import/geometry protocols and explicit source controls."""

import builtins
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import tilelang
from tilelang.env import CacheState

PATH = Path(__file__).parents[4] / "examples/ascend_a2a3/xattention/expert_v1/paged.py"


def _module():
    spec = importlib.util.spec_from_file_location("expert_paged_host_scope", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _protocol_geometry(module, cores):
    """Explicit CPU protocol metadata; never an actual NPU property sample."""
    calls = []
    original = module.torch

    def get_properties(device):
        calls.append(device)
        return SimpleNamespace(cube_core_num=cores)

    module.torch = SimpleNamespace(npu=SimpleNamespace(get_device_properties=get_properties))
    try:
        module._ensure_geometry()
        module._ensure_geometry()
    finally:
        module.torch = original
    assert calls == ["npu"]


def test_import_help_avoids_registration_geometry_and_cache_rng_effects(monkeypatch, capsys):
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "torch_npu":
            pytest.fail("import/help registered NPU")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    raw = CacheState._enabled
    rng = torch.random.get_rng_state().clone()
    module = _module()
    assert not module._GEOMETRY_READY and not hasattr(module, "NUM_CORES")
    with pytest.raises(SystemExit) as error:
        module.main(["--help"])
    assert error.value.code == 0 and "usage:" in capsys.readouterr().out
    assert not module._GEOMETRY_READY and not hasattr(module, "NUM_CORES")
    assert CacheState._enabled is raw and torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("conflict", [False, True])
def test_owned_cache_guard_precedes_registration_and_geometry(conflict, tmp_path, monkeypatch):
    module = _module()
    if conflict:
        monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "requested"))
        monkeypatch.setattr(vars(type(tilelang.env))["TILELANG_CACHE_DIR"], "_forced_value", str(tmp_path / "caller"))
        message = "conflicts with the active cache-directory override"
    else:
        monkeypatch.delenv("TILELANG_CACHE_DIR", raising=False)
        message = "Set a dedicated TILELANG_CACHE_DIR"
    with pytest.raises(RuntimeError, match=message):
        module.main([])
    assert not module._GEOMETRY_READY


@pytest.mark.parametrize("cores", [20, 24])
def test_original_nine_geometry_assignments_once_per_fresh_module_CPU_protocol(cores):
    module = _module()
    _protocol_geometry(module, cores)
    assert cores == module.NUM_CORES and cores - 4 == module.SHARED_CORES
    assert cores * 2 == module.MERGE_WORKERS and module.MERGE_TILES_PER_WORKER == 2
    assert module.SHARED_RESIDENT_GROUPS_PER_CORE == 1 and module.SHARED_ACTIVE_RESIDENT_CORES == 8
    assert module.SHARED_PAIRS_PER_CORE == 2 and module.SHARED_TASKS_PER_CORE == 4 and module.SHARED_PIPE_STEPS == 6
    assert module._GEOMETRY_READY


@pytest.mark.parametrize("config,expected", [("", False), ("arbitrary_OPPROF_path", True)])
def test_original_simulator_environment_predicate(config, expected, monkeypatch):
    module = _module()
    monkeypatch.setenv("CAMODEL_CONFIG_PATH", config)
    assert module._is_simulator() is expected and not module._GEOMETRY_READY


@pytest.mark.a2a3_native
@pytest.mark.parametrize("cores", [20, 24])
def test_complete_original_factory_source_API_explicit_geometry_control(cores, tmp_path, request):
    from tilelang import tvm

    module = _module()
    _protocol_geometry(module, cores)
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    factory = module.xattention_paged
    func = factory._get_tir_with_target(target)
    with tvm.transform.PassContext(config=factory.pass_configs):
        artifact = tilelang.lower(func, target=target)
    assert "void call(" in artifact.kernel_source and len(artifact.params) == 27
    assert list(artifact.params[4].shape) == [3, cores, 128, 512]
    path = tmp_path / f"paged_core{cores}.cc"
    path.write_text(artifact.kernel_source)
    request.node.user_properties.append(("xattention_current_source", str(path)))
