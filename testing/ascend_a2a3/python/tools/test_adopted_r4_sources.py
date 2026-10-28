"""Full original R4 source requests, separating multi-helper and trap cases."""

import importlib
import inspect

import pytest
import torch

import tilelang
from tilelang import tvm
from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs
from testing.ascend_a2a3.python.tools.test_adopted_dtype_sources import _parameterizations

pytestmark = pytest.mark.a2a3_native

# These are all original helper calls of the three non-parameterized outer
# merge tests. Source tests invoke one real helper per request; they do not
# replace kernel outputs or continue an original numerical test with fake data.
_MERGE_CALLS = {
    "test_merge_sort_basic_ascendc": [("run_test_merge", {"block_len": 64, "num_ways": ways, "target": "ascendc"}) for ways in (2, 3, 4)],
    "test_merge_sort_sizes_ascendc": [
        ("run_test_merge", {"block_len": 16, "num_ways": 2, "target": "ascendc"}),
        ("run_test_merge", {"block_len": 64, "num_ways": 2, "target": "ascendc"}),
        ("run_test_merge_unequal", {"sizes": [16, 8], "target": "ascendc"}),
        ("run_test_merge_unequal", {"sizes": [48, 16], "target": "ascendc"}),
    ],
    "test_merge_sort_edges_ascendc": [
        ("run_test_merge", {"block_len": 1, "num_ways": 2, "target": "ascendc"}),
        ("run_test_merge_region", {"M": 2, "per_row": 128, "target": "ascendc"}),
    ],
}


def _cases():
    rows = []
    for suffix in ("merge_sort", "sort", "topk", "tail_block"):
        module = importlib.import_module("testing.ascend_a2a3.python.language.test_tilelang_ascend_language_" + suffix)
        for name, function in vars(module).items():
            if not name.startswith("test_") or not inspect.isfunction(function):
                continue
            inherited = [mark for mark in getattr(function, "pytestmark", []) if mark.name in {"ci_skip", "low_priority"}]
            if suffix == "merge_sort" and name in _MERGE_CALLS:
                for index, (helper, values) in enumerate(_MERGE_CALLS[name]):
                    rows.append(
                        pytest.param(getattr(module, helper), values, "helper", marks=inherited, id=f"{suffix}-{name}-helper-{index}")
                    )
            else:
                for index, (values, marks) in enumerate(_parameterizations(function)):
                    kind = "trap-construction-only" if name == "test_sort_actual_num_zero_crashes" else "original-single-request"
                    rows.append(pytest.param(function, values, kind, marks=inherited + marks, id=f"{suffix}-{name}-{index}"))
    return rows


class _SourceCaptured(Exception):
    pass


@pytest.mark.parametrize("original,values,kind", _cases())
def test_original_r4_request_full_source(original, values, kind, monkeypatch, isolated_a2a3_cache):
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
    if kind == "trap-construction-only":
        module = inspect.getmodule(original)
        actual_prim_func = module.T.prim_func
        constructed = []

        def observe(*args, **kwargs):
            result = actual_prim_func(*args, **kwargs)
            constructed.append(result)
            return result

        monkeypatch.setattr(module.T, "prim_func", observe)

        # The original trap test constructs its real PrimFunc before the first
        # input transfer. Observe that object, never inject the hardware trap.
        def before_device(*args, **kwargs):
            raise _SourceCaptured

        monkeypatch.setattr(torch.Tensor, "npu", before_device)
        with pytest.raises(_SourceCaptured):
            original(**values)
        assert len(constructed) == 1 and requests == []
        with pytest.raises(tvm.error.InternalError, match="Expected a non-empty workspace view for Sort<"):
            capture(constructed[0], out_idx=[-1], pass_configs=module.PASS_CONFIGS, target="ascendc")
        assert requests == []
        return
    else:
        with pytest.raises(_SourceCaptured):
            original(**values)
    assert len(requests) == 1
