"""Elaborate every original elementwise request without executing an NPU call."""

import importlib
import inspect

import pytest
import torch

import tilelang
from tilelang import tvm
from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs
from testing.ascend_a2a3.python.tools.test_adopted_dtype_sources import _parameterizations

pytestmark = pytest.mark.a2a3_native


def _cases():
    module = importlib.import_module("testing.ascend_a2a3.python.language.test_tilelang_ascend_language_elementwise")
    cases = []
    for name, function in vars(module).items():
        if not name.startswith("test_") or not inspect.isfunction(function):
            continue
        if not any(mark.name == "a2a3_device" for mark in getattr(function, "pytestmark", [])):
            continue
        inherited = [mark for mark in getattr(function, "pytestmark", []) if mark.name in {"ci_skip", "low_priority"}]
        for index, (values, marks) in enumerate(_parameterizations(function)):
            cases.append(pytest.param(function, values, marks=inherited + marks, id=f"{name}-{index}"))
    return cases


class _SourceCaptured(Exception):
    pass


@pytest.mark.parametrize("original,values", _cases())
def test_original_elementwise_request_full_source(original, values, monkeypatch, isolated_a2a3_cache):
    requests = []

    def capture(func, *args, **kwargs):
        assert not args
        target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": [kwargs["target"], "ascend"]})
        effective = normalize_a2a3_pass_configs(target, kwargs.get("pass_configs"))
        before = tvm.ir.save_json(func)
        with tvm.transform.PassContext(config=effective):
            artifact = tilelang.lower(func, target=target)
        assert tvm.ir.save_json(func) == before
        assert "void call(" in artifact.kernel_source
        assert len(artifact.params) >= len(func.params)
        for parameter, lowered in zip(func.params, artifact.params):
            declared = func.buffer_map[parameter]
            assert str(lowered.dtype) == str(declared.dtype)
            assert [str(dim) for dim in lowered.shape] == [str(dim) for dim in declared.shape]
        requests.append(artifact)
        raise _SourceCaptured

    def blocked(*args, **kwargs):
        pytest.fail("source-only request must stop before NPU allocation or invocation")

    monkeypatch.setattr(tilelang, "compile", capture)
    monkeypatch.setattr(torch.Tensor, "npu", blocked, raising=False)
    if hasattr(torch, "npu"):
        monkeypatch.setattr(torch.npu, "synchronize", blocked)
    for name in ("randn", "rand", "randint", "empty", "zeros", "ones", "full", "arange", "tensor"):
        real = getattr(torch, name)

        def guarded(*args, _real=real, **kwargs):
            if str(kwargs.get("device", "")).startswith("npu"):
                blocked()
            return _real(*args, **kwargs)

        monkeypatch.setattr(torch, name, guarded)
    with pytest.raises(_SourceCaptured):
        if original.__name__ == "test_bilinear_interpolation":
            # The original helper transfers inputs before elaborating this
            # factory. Only their exact .shape is read by the factory; build
            # the same real tensors on CPU, without replacing .npu or outputs.
            module = inspect.getmodule(original)
            src0 = torch.arange(1, 513, dtype=torch.float16).reshape(1, -1)
            src0offset_int = torch.arange(0, 1024, 32, dtype=torch.int64).reshape(1, -1)
            src0offset = src0offset_int.to(dtype=torch.uint32)
            src1 = torch.arange(2, 18, dtype=torch.float16).reshape(1, -1)
            func = module.bilinear_interpolation(128, 2, False, 1, 128, 2, src0, src0offset_int, src0offset, src1)
            tilelang.compile(func, out_idx=[-1], pass_configs=module.pass_configs, target=values["target"])
        else:
            original(**values)
    assert len(requests) == 1
