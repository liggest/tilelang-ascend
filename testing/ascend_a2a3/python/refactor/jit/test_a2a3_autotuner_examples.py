"""Import-safe original autotune assets and their static-shape input contract."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _example(name):
    path = Path(__file__).parents[5] / "examples" / "ascend_a2a3" / "autotune" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_original_default_search_space_builds_every_kernel():
    module = _example("example_gemm_autotune.py")
    for config in module.get_config():
        program = module.matmul.jit_impl.get_tir(1024, 1024, 1024, **config)
        shapes = [[int(dim) for dim in program.buffer_map[param].shape] for param in program.params]
        assert shapes == [[1024, 1024], [1024, 1024], [1024, 1024]]


def test_input_supplier_uses_logical_shapes_and_preserves_precision(monkeypatch):
    module = _example("example_gemm_autotune.py")
    monkeypatch.setattr(torch.Tensor, "npu", lambda self: self, raising=False)
    a, b = module.supply_prog([SimpleNamespace(shape=(32, 64)), SimpleNamespace(shape=(64, 48))])
    assert a.shape == (32, 64) and b.shape == (64, 48)
    assert a.dtype == torch.float16 and b.dtype == torch.float16
    a_again, b_again = module.supply_prog([SimpleNamespace(shape=(32, 64)), SimpleNamespace(shape=(64, 48))])
    assert torch.equal(a, a_again) and torch.equal(b, b_again)


def test_carver_asset_reports_the_deferred_architecture_without_static_replacement():
    module = _example("example_gemm_carver.py")
    with pytest.raises(NotImplementedError, match="Ascend.*Carver.*deferred"):
        module.get_config(1024, 1024, 1024)


def test_legacy_carver_module_presence_does_not_enable_the_deferred_asset(monkeypatch):
    import sys
    from types import ModuleType

    module = _example("example_gemm_carver.py")
    legacy = ModuleType("tilelang.carver.arch.ascend")
    legacy.Ascend = object
    monkeypatch.setitem(sys.modules, legacy.__name__, legacy)
    monkeypatch.setattr(module, "_original_recommend_configs", lambda *args: pytest.fail("deferred asset ran legacy Carver"))
    with pytest.raises(NotImplementedError, match="deferred"):
        module.make_matmul(1024, 1024, 1024)
