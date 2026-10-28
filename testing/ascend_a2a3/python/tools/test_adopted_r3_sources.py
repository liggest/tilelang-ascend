"""Run complete original R3 compile/source assertions before any NPU transfer."""

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
    cases = []
    for suffix in ("reduce2d_v2", "reduce_dtype_coverage", "select", "transpose"):
        module = importlib.import_module("testing.ascend_a2a3.python.language.test_tilelang_ascend_language_" + suffix)
        for name, function in vars(module).items():
            if not name.startswith("test_") or not inspect.isfunction(function):
                continue
            if not any(mark.name == "a2a3_device" for mark in getattr(function, "pytestmark", [])):
                continue
            inherited = [mark for mark in getattr(function, "pytestmark", []) if mark.name in {"ci_skip", "low_priority"}]
            for index, (values, marks) in enumerate(_parameterizations(function)):
                cases.append(pytest.param(function, values, marks=inherited + marks, id=f"{suffix}-{name}-{index}"))
    return cases


class _BeforeDevice(Exception):
    pass


@pytest.mark.parametrize("original_test,values", _cases())
def test_original_r3_request_full_source_and_prefix_assertions(original_test, values, monkeypatch, isolated_a2a3_cache):
    compiled = []
    stopped = []

    def stop(*args, **kwargs):
        stopped.append(True)
        raise _BeforeDevice

    class SourceKernel:
        def __init__(self, artifact):
            self.artifact = artifact

        def get_kernel_source(self):
            return self.artifact.kernel_source

        __call__ = stop

    def capture(func, *args, **kwargs):
        assert not args
        requested = kwargs["target"]
        target = (
            requested
            if isinstance(requested, tvm.target.Target)
            else tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": [requested, "ascend"]})
        )
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
        compiled.append(artifact)
        # Let the unchanged original GetWithOffset/Reduce2D source assertions
        # execute. This facade never executes or allocates kernel outputs.
        return SourceKernel(artifact)

    monkeypatch.setattr(tilelang, "compile", capture)
    monkeypatch.setattr(torch.Tensor, "npu", stop, raising=False)
    real_to = torch.Tensor.to

    def to(self, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if isinstance(device, (str, torch.device)) and str(device).startswith("npu"):
            stop()
        return real_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    if hasattr(torch, "npu"):
        monkeypatch.setattr(torch.npu, "synchronize", stop)
    for name in ("randn", "rand", "randint", "empty", "zeros", "ones", "full", "arange", "tensor"):
        real = getattr(torch, name)

        def guarded(*args, _real=real, **kwargs):
            if str(kwargs.get("device", "")).startswith("npu"):
                stop()
            return _real(*args, **kwargs)

        monkeypatch.setattr(torch, name, guarded)
    if original_test.__name__ == "test_transpose_fallback_int64" and values["target"] == "pto":
        # Both fixed TA and current codegen reject B64 in PrintType. The
        # unchanged original numerical node is ci_skip and remains unsupported.
        with pytest.raises(tvm.error.InternalError, match="Unsupported datatype"):
            original_test(**values)
        assert compiled == stopped == []
    else:
        with pytest.raises(_BeforeDevice):
            original_test(**values)
        assert len(compiled) == len(stopped) == 1
