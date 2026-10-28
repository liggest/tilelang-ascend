"""Capture original R1 numerical requests before their NPU allocation/call."""

import importlib
import inspect

import pytest
import torch

import tilelang
from tilelang import tvm
from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs
from testing.ascend_a2a3.python.tools.test_adopted_dtype_sources import _parameterizations

pytestmark = pytest.mark.a2a3_native
_MODULE = "testing.ascend_a2a3.python.language."


def _cases():
    cases = []
    for suffix in ("alloc_var", "tile_arith", "tile_atomic_add"):
        module = importlib.import_module(_MODULE + "test_tilelang_ascend_language_" + suffix)
        for name, function in vars(module).items():
            if not name.startswith("test_") or not inspect.isfunction(function):
                continue
            if name in {
                "test_tile_arith_size_mismatch_src0",
                "test_tile_arith_size_mismatch_region",
                "test_tile_arith_div_int_compile_error",
            }:
                continue  # original native error assertions run directly
            inherited = [mark for mark in getattr(function, "pytestmark", []) if mark.name in {"ci_skip", "low_priority"}]
            for index, (values, marks) in enumerate(_parameterizations(function)):
                cases.append(pytest.param(function, values, marks=inherited + marks, id=f"{suffix}-{name}-{index}"))
    return cases


class _SourceCaptured(Exception):
    pass


@pytest.mark.parametrize("original_test,values", _cases())
def test_original_r1_request_full_source(original_test, values, monkeypatch):
    captured = []

    def capture(func, *args, **kwargs):
        assert not args
        target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": [kwargs["target"], "ascend"]})
        effective = normalize_a2a3_pass_configs(target, kwargs.get("pass_configs"))
        before = tvm.ir.save_json(func)
        with tvm.transform.PassContext(config=effective):
            artifact = tilelang.lower(func, target=target)
        assert tvm.ir.save_json(func) == before
        assert "void call(" in artifact.kernel_source
        for param, lowered in zip(func.params, artifact.params):
            declared = func.buffer_map[param]
            assert str(lowered.dtype) == str(declared.dtype)
            assert [str(dim) for dim in lowered.shape] == [str(dim) for dim in declared.shape]
        captured.append(artifact)
        raise _SourceCaptured

    def no_device(*args, **kwargs):
        pytest.fail("source capture must stop before NPU input allocation or kernel invocation")

    monkeypatch.setattr(tilelang, "compile", capture)
    monkeypatch.setattr(torch.Tensor, "npu", no_device, raising=False)
    if hasattr(torch, "npu"):
        monkeypatch.setattr(torch.npu, "synchronize", no_device)
    for name in ("randn", "rand", "randint", "empty", "zeros", "ones", "full", "arange"):
        real = getattr(torch, name)

        def guarded(*args, _real=real, **kwargs):
            if str(kwargs.get("device", "")).startswith("npu"):
                no_device()
            return _real(*args, **kwargs)

        monkeypatch.setattr(torch, name, guarded)
    with pytest.raises(_SourceCaptured):
        original_test(**values)
    assert len(captured) == 1


@pytest.mark.parametrize("auto_sync", [True, False])
def test_original_negative_sync_request_full_source(auto_sync, monkeypatch):
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache
    from testing.ascend_a2a3.python.language import test_ascend_compile_flags as original

    captured = []

    def capture(self, func=None, out_idx=None, *args, **kwargs):
        assert not args and out_idx == [1]
        from tilelang.backend.module import create_backend_context

        context = kwargs.get("backend_context") or create_backend_context(kwargs["target"], execution_backend="cython")
        configs = normalize_a2a3_pass_configs(context.target, kwargs.get("pass_configs"))
        assert bool(configs["tl.ascend_auto_sync"]) is auto_sync
        assert kwargs["compile_flags"] == ["--cce-auto-sync=off", "-O3"]
        before = tvm.ir.save_json(func)
        with tvm.transform.PassContext(config=configs):
            artifact = tilelang.lower(func, target=context.target)
        assert tvm.ir.save_json(func) == before
        assert "void call(" in artifact.kernel_source
        captured.append(artifact)
        raise _SourceCaptured

    monkeypatch.setattr(A2A3CythonKernelCache, "cached", capture)
    with pytest.raises(_SourceCaptured):
        original._build_sync_dependency_kernel(auto_sync)
    assert len(captured) == 1
