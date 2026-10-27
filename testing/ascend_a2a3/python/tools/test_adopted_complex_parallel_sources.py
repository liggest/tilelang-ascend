"""Elaborate the exact original Parallel test requests, then stop before inputs."""

import importlib
import inspect
import itertools

import pytest
import torch

import tilelang
from tilelang import tvm
from tilelang.jit import JITImpl

pytestmark = pytest.mark.a2a3_native


def _module(suffix):
    tail = f"_{suffix}" if suffix else ""
    return importlib.import_module(f"testing.ascend_a2a3.python.language.parallel.test_tilelang_ascend_language_parallel{tail}")


def _parameterizations(function):
    options = [{}]
    for mark in getattr(function, "pytestmark", []):
        if mark.name != "parametrize":
            continue
        names = [name.strip() for name in mark.args[0].split(",")]
        rows = [dict(zip(names, (value,) if len(names) == 1 else value)) for value in mark.args[1]]
        options = [left | right for left, right in itertools.product(options, rows)]
    return options


def _cases():
    cases = []
    original = _module("").TestTileLangKernels()
    for name in vars(type(original)):
        if name.startswith("test_"):
            cases.append(pytest.param(getattr(original, name), {}, id=f"original-{name}"))
    for suffix in ("complex", "auto_copy", "discrete"):
        for name, function in vars(_module(suffix)).items():
            if not name.startswith("test_") or not inspect.isfunction(function):
                continue
            for index, values in enumerate(_parameterizations(function)):
                cases.append(pytest.param(function, values, id=f"{suffix}-{name}-{index}"))
    return cases


class _SourceCaptured(Exception):
    pass


@pytest.mark.parametrize("original_test,values", _cases())
def test_original_parallel_request_full_source(original_test, values, monkeypatch):
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    captured = []

    def capture(factory, *args, **kwargs):
        requested_target = factory.target
        func = factory._get_tir_with_target(target, *args, **kwargs)
        assert factory.target is requested_target
        with tvm.transform.PassContext(config=factory.pass_configs or {}):
            artifact = tilelang.lower(func, target=target)
        assert "void call(" in artifact.kernel_source
        assert len(artifact.params) >= len(func.params)
        for param, result in zip(func.params, artifact.params):
            declared = func.buffer_map[param]
            assert str(result.dtype) == str(declared.dtype)
            assert [str(dim) for dim in result.shape] == [str(dim) for dim in declared.shape]
        captured.append(artifact)
        # All these original tests elaborate before allocating their NPU input.
        # Stop here: this asserts source construction, not numerical execution.
        raise _SourceCaptured

    def no_device(*args, **kwargs):
        pytest.fail("source request capture must stop before numerical NPU execution")

    monkeypatch.setattr(JITImpl, "__call__", capture)
    monkeypatch.setattr(torch.Tensor, "npu", no_device, raising=False)
    if hasattr(torch, "npu"):
        monkeypatch.setattr(torch.npu, "synchronize", no_device)
    for name in ("randn", "rand", "randint", "empty", "zeros", "ones", "full"):
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
