"""Capture original R2 numerical requests before their NPU allocation/call."""

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
    for suffix in ("gemm_v0", "gemm_v0_k_tail", "gemm_v0_n_actual", "gemm_v0_n_tiling"):
        module = importlib.import_module(_MODULE + "test_tilelang_ascend_language_" + suffix)
        for name, function in vars(module).items():
            if not name.startswith("test_") or not inspect.isfunction(function):
                continue
            if "invalid_" in name or "dtype_mismatch" in name:
                continue  # original native validation assertions run directly
            inherited = [mark for mark in getattr(function, "pytestmark", []) if mark.name in {"ci_skip", "low_priority"}]
            for index, (values, marks) in enumerate(_parameterizations(function)):
                if values.get("expected") != "reject":
                    cases.append(pytest.param(function, values, marks=inherited + marks, id=f"{suffix}-{name}-{index}"))
    return cases


class _SourceCaptured(Exception):
    pass


@pytest.mark.parametrize("original_test,values", _cases())
def test_original_r2_request_full_source(original_test, values, monkeypatch):
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


def _rejected_cases():
    module = importlib.import_module(_MODULE + "test_tilelang_ascend_language_gemm_v0_k_tail")
    rows = []
    for name in ("test_gemm_v0_ktail_int8", "test_gemm_v0_ktail_int8_kl0size_c0"):
        function = getattr(module, name)
        for values, marks in _parameterizations(function):
            if values["expected"] == "reject":
                rows.append(pytest.param(function, values, marks=marks, id=f"{name}-{values}"))
    return rows


@pytest.mark.parametrize("original_test,values", _rejected_cases())
def test_original_r2_rejected_request(original_test, values, monkeypatch, isolated_a2a3_cache):
    def no_device(*args, **kwargs):
        pytest.fail("rejected GEMM requests must stop before any NPU use")

    monkeypatch.setattr(torch.Tensor, "npu", no_device, raising=False)
    if hasattr(torch, "npu"):
        monkeypatch.setattr(torch.npu, "synchronize", no_device)
    for name in ("randn", "rand", "randint", "empty", "zeros", "ones", "full"):
        real = getattr(torch, name)

        def guarded(*args, _real=real, **kwargs):
            if str(kwargs.get("device", "")).startswith("npu"):
                no_device()
            return _real(*args, **kwargs)

        monkeypatch.setattr(torch, name, guarded)
    # The unchanged original branch contains its precise Compilation Failed
    # assertion and returns before inputs. Use the actual compiler here.
    original_test(**values)
