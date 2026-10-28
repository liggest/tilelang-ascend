"""Lower complete original copy-path requests without allocating NPU inputs."""

import importlib
import inspect

import pytest
import torch

import tilelang
from tilelang import tvm
from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs
from testing.ascend_a2a3.python.tools.test_adopted_dtype_sources import _parameterizations

pytestmark = pytest.mark.a2a3_native


def _module(suffix):
    return importlib.import_module(f"testing.ascend_a2a3.python.language.test_tilelang_ascend_language_{suffix}")


def _cases():
    result = []
    for suffix in ("gm_to_ub", "gm_to_l1", "l1_to_l0", "copy_cv"):
        for name, function in vars(_module(suffix)).items():
            if not name.startswith("test_") or not inspect.isfunction(function):
                continue
            markers = getattr(function, "pytestmark", [])
            if any(mark.name == "a2a3_native" for mark in markers):
                continue  # The original source assertions run in their own module.
            inherited = [mark for mark in markers if mark.name in ("ci_skip", "low_priority", "skip")]
            for index, (values, marks) in enumerate(_parameterizations(function)):
                marks = [mark for mark in marks if mark.name != "a2a3_original_skip_a5"]
                result.append(pytest.param(suffix, function, values, marks=inherited + marks, id=f"{suffix}-{name}-{index}"))
    return result


class _SourceCaptured(Exception):
    pass


@pytest.mark.parametrize("suffix,original_test,values", _cases())
def test_original_copy_path_request_full_source(suffix, original_test, values, monkeypatch):
    captured = []

    def capture(func, *args, **kwargs):
        assert not args
        dialect = kwargs["target"]
        assert dialect in ("ascendc", "pto")
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
        if suffix != "copy_cv":
            raise _SourceCaptured

    def no_device(*args, **kwargs):
        pytest.fail("source capture must not allocate NPU inputs or invoke a kernel")

    monkeypatch.setattr(tilelang, "compile", capture)
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

    if suffix == "copy_cv":
        module = _module(suffix)

        # The two special-shapes tests issue two requests. Keep their original
        # branches running, while stopping each case helper at its compile prefix.
        def case_prefix(
            kernel_func,
            M=128,
            N=128,
            K=128,
            dtype="float16",
            accum_dtype="float",
            target="pto",
            expert=False,
            dtype_cast=False,
            manual_workspace=False,
        ):
            program = kernel_func(M=M, N=N, K=K, dtype=dtype, accum_dtype=accum_dtype)
            module._compile(program, target=target, expert=expert)

        def combined_prefix(kernel_func, M=128, N=128, K=128, target="pto"):
            program = kernel_func(M=M, N=N, K=K, dtype="float16", accum_dtype="float")
            module._compile(program, target=target)

        monkeypatch.setattr(module, "_ub_to_l1_case", case_prefix)
        monkeypatch.setattr(module, "_l0c_to_ub_case", case_prefix)
        monkeypatch.setattr(module, "_combined_case", combined_prefix)
        arguments = {name: "A3" if name == "platform" else values[name] for name in inspect.signature(original_test).parameters}
        original_test(**arguments)
        expected = 2 if "special_shapes" in original_test.__name__ and values["target"] == "pto" else 1
    else:
        with pytest.raises(_SourceCaptured):
            original_test(**values)
        expected = 1
    assert len(captured) == expected
