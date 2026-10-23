"""Run the A2A3 automatic-memory/void-launch ABI with CPU tensors and a C library."""

import ctypes
import subprocess
from types import SimpleNamespace

import pytest
import torch
from tilelang import tvm
from tilelang.engine.param import KernelParam, dump_kernel_params, load_kernel_params
from tilelang.ascend.a2a3.runtime import configure_void_call, normalize_memory_indices, resolve_allocation_shape
from tilelang_cython_wrapper import CythonKernelWrapper


tir = tvm.tirx


def _wrapper(params, result_idx, workspace_idx=(), auto_gm_idx=(), symbolic=None, *, a2a3=True, library=None):
    if library is None:
        library = SimpleNamespace(call=lambda *args: None if a2a3 else 0)
    wrapper = CythonKernelWrapper(result_idx, params, library, lambda: torch.device("cpu"), lambda device: 0)
    wrapper.set_dynamic_symbolic_map(symbolic or {})
    wrapper.set_dynamic_symbolic_sources({})
    wrapper.set_buffer_device_map({})
    wrapper.set_buffer_dtype_map({})
    wrapper.set_static_shape_map({})
    wrapper.set_static_strides_map({})
    wrapper.set_static_contiguous_list([])
    wrapper.set_ptr_map({})
    if a2a3:
        wrapper.set_a2a3_runtime(workspace_idx, auto_gm_idx)
    return wrapper


def _params(*shapes, dtype="float32"):
    return [KernelParam(tvm.DataType(dtype), list(shape)) for shape in shapes]


@pytest.mark.parametrize("value,expected", [(None, []), (0, [0]), (-1, [3]), (-4, [0]), ([0, -1, -3], [0, 3, 1]), ([1, 1], [1, 1])])
def test_source_index_forms_and_negative_indices(value, expected):
    before = value.copy() if isinstance(value, list) else value
    assert normalize_memory_indices(value, 4, "workspace_idx") == expected
    assert value == before


@pytest.mark.parametrize("value", [4, -5, [4], [-5], (0,), "0", ["0"]])
def test_source_index_rejections(value):
    with pytest.raises(ValueError, match="workspace_idx"):
        normalize_memory_indices(value, 4, "workspace_idx")


def test_dynamic_workspace_expressions_and_parameter_reflection():
    n = tir.Var("N", "int32")
    params = load_kernel_params(dump_kernel_params(_params((n * 2 + 1,), (n,), (tir.ceildiv(n, 4),))))
    # Cache params and the frontend own separate Var handles, so resolve by name.
    assert resolve_allocation_shape(params[0].shape, {n: (0, 1, 0, 1)}, {}, [None, torch.zeros(9)]) == [19]
    assert resolve_allocation_shape(params[2].shape, {n: (0, 1, 0, 1)}, {}, [None, torch.zeros(9)]) == [3]


def test_shape_can_use_a_later_available_carrier():
    n = tir.Var("N", "int32")
    assert resolve_allocation_shape([n * 2], {n: (0, 0, 0, 1)}, {"N": [(0, 0, 0, 1), (0, 1, 0, 1)]}, [None, torch.zeros(7)]) == [14]


def test_missing_symbol_keeps_source_error_boundary():
    with pytest.raises(KeyError, match="Unfounded symbolic var"):
        resolve_allocation_shape([tir.Var("missing", "int32") * 2], {}, {}, [])


def test_compiler_gm_dynamic_shapes_remain_explicitly_unsupported():
    n = tir.Var("N", "int32")
    with pytest.raises(NotImplementedError, match="dynamic auto_gm_idx"):
        resolve_allocation_shape([n * 2], {n: (0, 0, 0, 1)}, {}, [torch.zeros(7)], True)


def test_interleaved_auto_allocations_and_implicit_native_shape_order(monkeypatch):
    n, m = tir.Var("N", "int32"), tir.Var("M", "int32")
    source, other = torch.zeros(7), torch.zeros(3)
    calls, allocations = [], []
    empty = torch.empty

    def allocate(*shape, **kwargs):
        allocations.append(tuple(shape))
        return empty(*shape, **kwargs)

    monkeypatch.setattr(torch, "empty", allocate)
    params = _params((n * 2,), (n,), (n + 1,), (m,), (16,))
    wrapper = _wrapper(
        params, [2], [0], [4], {n: (0, 1, 0, 1), m: (0, 3, 0, 1)}, library=SimpleNamespace(call=lambda *args: calls.append(args))
    )
    output = wrapper.forward([source, other], stream=42)
    assert tuple(output.shape) == (8,)
    assert allocations == [(14,), (8,), (16,)]
    assert len(calls[0]) == 8
    assert calls[0][1].value == source.data_ptr()
    assert calls[0][3].value == other.data_ptr()
    assert [value.value for value in calls[0][-3:]] == [7, 3, 42]


def test_output_workspace_overlap_is_allocated_once(monkeypatch):
    allocations = []
    empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *shape, **kwargs: (allocations.append(shape), empty(*shape, **kwargs))[1])
    wrapper = _wrapper(_params((4,), (4,)), [1], [1], [1])
    output = wrapper.forward([torch.zeros(4)], stream=0)
    assert output.shape == (4,)
    assert len(allocations) == 1


def test_workspace_without_output_returns_source_empty_list():
    wrapper = _wrapper(_params((4,), (4,)), [], [1])
    assert wrapper.forward([torch.zeros(4)], stream=0) == []


def test_raw_source_extra_python_shape_arguments_are_ignored():
    wrapper = _wrapper(_params((4,), (4,)), [1])
    assert wrapper.forward([torch.zeros(4), 4], stream=0).shape == (4,)


def test_shared_default_abi_still_checks_arity_and_static_shapes():
    wrapper = _wrapper(_params((4,), (4,)), [1], a2a3=False)
    with pytest.raises(ValueError, match="Expected"):
        wrapper.forward([torch.zeros(4), 4], stream=0)
    wrapper.set_static_shape_map({"source": (0, [(0, 4)])})
    with pytest.raises(ValueError, match="Static shape mismatch"):
        wrapper.forward([torch.zeros(3)], stream=0)
    source_wrapper = _wrapper(_params((4,), (4,)), [1])
    source_wrapper.set_static_shape_map({"source": (0, [(0, 4)])})
    assert source_wrapper.forward([torch.zeros(3)], stream=0).shape == (4,)


def test_a2a3_still_validates_dtype_device_and_contiguity():
    wrapper = _wrapper(_params((4,), (4,)), [1])
    wrapper.set_buffer_dtype_map({"source": (0, torch.float32)})
    with pytest.raises(ValueError, match="dtype mismatch"):
        wrapper.forward([torch.zeros(4, dtype=torch.float64)], stream=0)
    wrapper.set_buffer_dtype_map({})
    wrapper.set_buffer_device_map({"source": (0, torch.device("cuda"))})
    with pytest.raises(ValueError, match="device mismatch"):
        wrapper.forward([torch.zeros(4)], stream=0)
    wrapper.set_buffer_device_map({})
    wrapper.set_static_contiguous_list([(0, "source")])
    with pytest.raises(ValueError, match="contiguous"):
        wrapper.forward([torch.zeros(8)[::2]], stream=0)


def test_native_void_call_with_dynamic_workspace(tmp_path):
    path = tmp_path / "void_launcher.so"
    source = """
    #include <stdint.h>
    extern "C" void call(float *workspace, float *input, float *output, int64_t n, void *stream) {
      for (int64_t i = 0; i < n; ++i) {
        workspace[2*i] = input[i] * 2;
        output[i] = workspace[2*i] + 1;
      }
    }
    """
    subprocess.run(["c++", "-shared", "-fPIC", "-x", "c++", "-", "-o", str(path)], input=source, text=True, capture_output=True, check=True)
    library = ctypes.CDLL(str(path))
    configure_void_call(library)
    assert library.call.restype is None
    assert not hasattr(library, "init")
    assert not hasattr(library, "get_last_error")
    n = tir.Var("N", "int32")
    wrapper = _wrapper(_params((n * 2,), (n,), (n,)), [2], [0], symbolic={n: (0, 1, 0, 1)}, library=library)
    source = torch.arange(13, dtype=torch.float32)
    output = wrapper.forward([source], stream=0)
    torch.testing.assert_close(output, source * 2 + 1)


def test_shared_default_status_return_errors_remain_visible():
    library = SimpleNamespace(call=lambda *args: 7, get_last_error=lambda: b"original status failure")
    wrapper = _wrapper(_params((4,), (4,)), [1], a2a3=False, library=library)
    with pytest.raises(RuntimeError, match="original status failure"):
        wrapper.forward([torch.zeros(4)], stream=0)
