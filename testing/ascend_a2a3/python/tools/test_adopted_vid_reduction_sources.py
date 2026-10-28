"""Construct every original vid-reduction request before NPU inputs."""

import inspect

import pytest
import torch

import tilelang
from tilelang import tvm
from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs
from testing.ascend_a2a3.python.language.cvseparate import test_tilelang_ascend_language_vid_reduction as original
from testing.ascend_a2a3.python.tools.test_adopted_dtype_sources import _parameterizations

pytestmark = pytest.mark.a2a3_native


def _cases():
    cases = []
    for name, function in vars(original).items():
        if not name.startswith("test_") or not inspect.isfunction(function):
            continue
        for index, (values, marks) in enumerate(_parameterizations(function)):
            cases.append(pytest.param(function, values, marks=marks, id=f"{name}-{index}"))
    return cases


class _SourceCaptured(Exception):
    pass


@pytest.mark.parametrize("original_test,values", _cases())
def test_original_vid_reduction_request_full_source(original_test, values, monkeypatch):
    captured = []

    def capture(func, *args, **kwargs):
        assert not args
        dialect = kwargs["target"]
        target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": [dialect, "ascend"]})
        before = tvm.ir.save_json(func)
        effective = normalize_a2a3_pass_configs(target, kwargs.get("pass_configs"))
        with tvm.transform.PassContext(config=effective):
            artifact = tilelang.lower(func, target=target)
        assert tvm.ir.save_json(func) == before
        assert "void call(" in artifact.kernel_source
        assert len(artifact.params) >= len(func.params)
        for param, lowered in zip(func.params, artifact.params):
            declared = func.buffer_map[param]
            assert str(lowered.dtype) == str(declared.dtype)
            assert [str(d) for d in lowered.shape] == [str(d) for d in declared.shape]
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
    # The original seed fixture is only relevant to later numerical inputs;
    # its parameter is not read by the original test body/compile prefix.
    with pytest.raises(_SourceCaptured):
        original_test(setup_random_seed=None, **values)
    assert len(captured) == 1
