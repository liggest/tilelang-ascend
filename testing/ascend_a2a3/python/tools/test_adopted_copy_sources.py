"""Build the complete original copy-test requests before any NPU inputs."""

import importlib
import inspect

import pytest
import torch

import tilelang
from tilelang import tvm
from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs
from tilelang.jit import JITImpl
from testing.ascend_a2a3.python.tools.test_adopted_dtype_sources import _parameterizations

pytestmark = pytest.mark.a2a3_native


def _module(suffix):
    return importlib.import_module(f"testing.ascend_a2a3.python.language.test_tilelang_ascend_language_{suffix}")


def _cases():
    cases = []
    cast = _module("cast_on_copy").TestTileLangKernels()
    for name in vars(type(cast)):
        if name.startswith("test_"):
            cases.append(pytest.param(getattr(cast, name), {}, id=f"cast-{name}"))
    for suffix in ("copy_row_slice_stride", "copy_runtime_extent", "copy_ub_to_gm_alignment"):
        for name, function in vars(_module(suffix)).items():
            if not name.startswith("test_") or not inspect.isfunction(function):
                continue
            if "_codegen" in name:
                # These original source assertions already run in their module.
                continue
            inherited = [mark for mark in getattr(function, "pytestmark", []) if mark.name in ("ci_skip", "low_priority")]
            for index, (values, marks) in enumerate(_parameterizations(function)):
                cases.append(pytest.param(function, values, marks=inherited + marks, id=f"{suffix}-{name}-{index}"))
    return cases


class _SourceCaptured(Exception):
    pass


@pytest.mark.parametrize("original_test,values", _cases())
def test_original_copy_request_full_source(original_test, values, monkeypatch):
    captured = []

    def target_for(requested):
        assert requested in (None, "auto", "ascendc", "pto")
        dialect = "pto" if requested == "pto" else "ascendc"
        return tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": [dialect, "ascend"]})

    def build(func, target, pass_configs):
        before = tvm.ir.save_json(func)
        effective = normalize_a2a3_pass_configs(target, pass_configs)
        with tvm.transform.PassContext(config=effective):
            artifact = tilelang.lower(func, target=target)
        assert tvm.ir.save_json(func) == before
        assert "void call(" in artifact.kernel_source
        assert len(artifact.params) >= len(func.params)
        for param, lowered in zip(func.params, artifact.params):
            declared = func.buffer_map[param]
            assert str(lowered.dtype) == str(declared.dtype)
            assert [str(dimension) for dimension in lowered.shape] == [str(dimension) for dimension in declared.shape]
        captured.append(artifact)
        raise _SourceCaptured

    def compile_request(func, *args, **kwargs):
        assert not args
        return build(func, target_for(kwargs["target"]), kwargs.get("pass_configs"))

    def jit_request(factory, *args, **kwargs):
        requested_target = factory.target
        target = target_for(requested_target)
        func = factory._get_tir_with_target(target, *args, **kwargs)
        assert factory.target is requested_target
        return build(func, target, factory.pass_configs)

    def no_device(*args, **kwargs):
        pytest.fail("source capture must stop before NPU input allocation, invocation or hardware traps")

    monkeypatch.setattr(tilelang, "compile", compile_request)
    monkeypatch.setattr(JITImpl, "__call__", jit_request)
    monkeypatch.setattr(torch.Tensor, "npu", no_device, raising=False)
    if hasattr(torch, "npu"):
        monkeypatch.setattr(torch.npu, "synchronize", no_device)
    for name in ("randn", "rand", "randint", "empty", "zeros", "ones", "full", "arange"):
        original = getattr(torch, name)

        def guarded(*args, _original=original, **kwargs):
            if str(kwargs.get("device", "")).startswith("npu"):
                no_device()
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, guarded)
    arguments = {name: values.get(name) for name in inspect.signature(original_test).parameters}
    with pytest.raises(_SourceCaptured):
        original_test(**arguments)
    assert len(captured) == 1
