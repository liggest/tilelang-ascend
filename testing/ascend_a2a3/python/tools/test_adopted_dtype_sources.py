"""Build original dtype-coverage requests without allocating NPU inputs."""

import importlib
import inspect
import itertools

import pytest
import torch

import tilelang
from tilelang import tvm
from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs

pytestmark = pytest.mark.a2a3_native


def _module(suffix):
    return importlib.import_module(f"testing.ascend_a2a3.python.language.test_tilelang_ascend_language_tile_{suffix}_dtype_coverage")


def _parameterizations(function):
    options = [({}, [])]
    for mark in getattr(function, "pytestmark", []):
        if mark.name != "parametrize":
            continue
        names = mark.args[0]
        names = [name.strip() for name in names.split(",")] if isinstance(names, str) else list(names)
        rows = []
        for value in mark.args[1]:
            if hasattr(value, "values") and hasattr(value, "marks"):
                values, marks = value.values, list(value.marks)
            else:
                values = (value,) if len(names) == 1 else value
                marks = []
            rows.append((dict(zip(names, values)), marks))
        options = [
            (left | right, left_marks + right_marks) for (left, left_marks), (right, right_marks) in itertools.product(options, rows)
        ]
    return options


def _cases():
    cases = []
    for suffix in ("atomic_add", "broadcast", "compare", "createvecindex", "gather"):
        for name, function in vars(_module(suffix)).items():
            if not name.startswith("test_") or not inspect.isfunction(function) or "_raises" in name:
                continue
            inherited = [mark for mark in getattr(function, "pytestmark", []) if mark.name in ("ci_skip", "low_priority")]
            for index, (values, marks) in enumerate(_parameterizations(function)):
                cases.append(pytest.param(function, values, marks=inherited + marks, id=f"{suffix}-{name}-{index}"))
    return cases


class _SourceCaptured(Exception):
    pass


@pytest.mark.parametrize("original_test,values", _cases())
def test_original_dtype_request_full_source(original_test, values, monkeypatch):
    captured = []

    def capture(func, *args, **kwargs):
        assert not args
        dialect = kwargs["target"]
        assert dialect in ("ascendc", "pto")
        target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": [dialect, "ascend"]})
        effective = normalize_a2a3_pass_configs(target, kwargs.get("pass_configs"))
        with tvm.transform.PassContext(config=effective):
            artifact = tilelang.lower(func, target=target)
        assert "void call(" in artifact.kernel_source
        assert len(artifact.params) >= len(func.params)
        for param, result in zip(func.params, artifact.params):
            declared = func.buffer_map[param]
            assert str(result.dtype) == str(declared.dtype)
            assert [str(dim) for dim in result.shape] == [str(dim) for dim in declared.shape]
        captured.append(artifact)
        raise _SourceCaptured

    def no_device(*args, **kwargs):
        pytest.fail("source capture must not allocate or execute NPU tensors")

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
    with pytest.raises(_SourceCaptured):
        if original_test.__name__ == "test_gather_base_addr":
            # This original test allocates inputs before elaborating. Build its
            # unchanged factory with that test's exact literal arguments instead
            # of substituting fake tensors or running the numerical body.
            module = _module("gather")
            func = module.make_gather_baseaddr_kernel(1024, 512, "float32", 64)
            tilelang.compile(func, out_idx=[-1], pass_configs=module.PASS_CONFIGS, target="ascendc")
        else:
            original_test(**values)
    assert len(captured) == 1
