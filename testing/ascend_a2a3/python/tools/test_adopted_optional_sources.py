"""Build the original arithmetic and optional-loop requests without NPU input."""

import pytest
import torch

import tilelang
from tilelang import tvm
from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs
from testing.ascend_a2a3.python.language import test_ascend_sync_insert_zero_trip as optional
from testing.ascend_a2a3.python.language import test_tilelang_ascend_language_arith_progression_dtype_coverage as arithmetic
from testing.ascend_a2a3.python.tools.test_adopted_dtype_sources import _parameterizations

pytestmark = pytest.mark.a2a3_native


def _cases():
    cases = []
    for name, function in vars(arithmetic).items():
        if not name.startswith("test_") or "_raises" in name:
            continue
        inherited = [mark for mark in getattr(function, "pytestmark", []) if mark.name in ("ci_skip", "low_priority")]
        for index, (values, marks) in enumerate(_parameterizations(function)):
            cases.append(pytest.param(function, values, marks=inherited + marks, id=f"{name}-{index}"))
    cases.append(pytest.param(optional.optional_loop_kernel.__wrapped__, {}, id="optional-loop-dynamic-count"))
    return cases


class _SourceCaptured(Exception):
    pass


@pytest.mark.parametrize("original_request,values", _cases())
def test_original_arithmetic_and_optional_loop_source(original_request, values, monkeypatch):
    captured = []

    def capture(func, *args, **kwargs):
        assert not args
        requested = kwargs["target"]
        target = requested
        if isinstance(requested, str):
            target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": [requested, "ascend"]})
        effective = normalize_a2a3_pass_configs(target, kwargs["pass_configs"])
        with tvm.transform.PassContext(config=effective):
            artifact = tilelang.lower(func, target=target)
        assert "void call(" in artifact.kernel_source
        for param, lowered in zip(func.params, artifact.params):
            declared = func.buffer_map[param]
            assert str(lowered.dtype) == str(declared.dtype)
            assert [str(dimension) for dimension in lowered.shape] == [str(dimension) for dimension in declared.shape]
        if original_request.__name__ == "optional_loop_kernel":
            assert kwargs["compile_flags"] == ["--cce-auto-sync=off"]
            assert kwargs["out_idx"] == []
            config = kwargs["pass_configs"]
            assert config[tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC] is True
            for key in (
                tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING,
                tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE,
                tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC,
            ):
                assert config[key] is False
            assert str(target.attrs["model"]) == "A3"
            assert "int64_t n" in artifact.kernel_source
        captured.append(artifact)
        raise _SourceCaptured

    def no_device(*args, **kwargs):
        pytest.fail("source capture must stop before allocating or executing NPU tensors")

    monkeypatch.setattr(tilelang, "compile", capture)
    monkeypatch.setattr(torch.Tensor, "npu", no_device, raising=False)
    if hasattr(torch, "npu"):
        monkeypatch.setattr(torch.npu, "synchronize", no_device)
    with pytest.raises(_SourceCaptured):
        if original_request.__name__ == "optional_loop_kernel":
            original_request(monkeypatch=monkeypatch)
        else:
            original_request(**values)
    assert len(captured) == 1
