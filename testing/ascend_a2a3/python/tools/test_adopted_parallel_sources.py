"""Build each original distinct parallel regression factory without NPU."""

import importlib

import pytest

import tilelang
from tilelang import tvm

pytestmark = pytest.mark.a2a3_native


def _module(suffix):
    return importlib.import_module(f"testing.ascend_a2a3.python.language.parallel.test_tilelang_ascend_language_parallel_{suffix}")


def _params(test):
    [mark] = [mark for mark in test.pytestmark if mark.name == "parametrize"]
    return mark.args[1]


def _cases():
    one = _module("1d")
    copy = _module("copy")
    mixed = _module("mixed")
    stride = _module("ub_stride")
    cases = []
    for name, op, _, _ in _params(one.test_parallel_1d_unary):
        cases.append(pytest.param(one.unary_1d_kernel, (1024, 128, op), id=f"unary-{name}"))
    for name, op, _ in _params(one.test_parallel_1d_binary_float):
        cases.append(pytest.param(one.binary_1d_kernel, (1024, 128, op), id=f"binary-{name}"))
    for name, op, _ in _params(one.test_parallel_1d_binary_int):
        cases.append(pytest.param(one.binary_1d_kernel, (1024, 128, op, "int16"), id=f"integer-{name}"))
    for name, op, _ in _params(one.test_parallel_1d_scalar_immediate):
        cases.append(pytest.param(one.scalar_1d_kernel, (1024, 128, op), id=f"scalar-{name}"))
    cases.append(pytest.param(one.offset_1d_kernel, (1024, 128), id="offset"))
    for dtype in _params(copy.test_parallel_copy_gm_ub_gm):
        cases.append(pytest.param(copy.gm_ub_gm_kernel, (1024, 1024, 128, 128, dtype), id=f"copy-{dtype}"))
    cases.append(pytest.param(copy.gm_ub_gm_1d_kernel, (1024, 128), id="copy-1d"))
    cases.append(pytest.param(copy.gm_l1_l0c_gm_kernel, (1024, 1024, 1024, 128, 256, 64), id="copy-cube"))
    cases += [
        pytest.param(mixed.parallel_fill_kernel, (1024, 1024, 128, 128), id="mixed-fill"),
        pytest.param(mixed.parallel_cast_kernel, (1024, 1024, 128, 128), id="mixed-cast"),
        pytest.param(mixed.parallel_reduce_kernel, (1024, 64, 64, 64), id="mixed-reduce"),
        pytest.param(mixed.parallel_serial_nest_kernel, (1024, 1024, 128, 128), id="mixed-serial"),
    ]
    for src_cols, dst_cols, dtype in _params(stride.test_parallel_compact_to_aligned):
        cases.append(
            pytest.param(stride.compact_to_aligned_kernel, (256, src_cols, dst_cols, dtype), id=f"stride-{src_cols}-{dst_cols}-{dtype}")
        )
    cases.append(pytest.param(stride.relu_if_else_1d_kernel, (1024, 128), id="serial-if-else"))
    return cases


@pytest.mark.parametrize("factory,args", _cases())
def test_original_parallel_factory_full_source(factory, args):
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    # Use the already-selected target entry shared with autotuning without
    # changing the original JIT target or probing an auto device on the host.
    requested_target = factory.target
    func = factory._get_tir_with_target(target, *args)
    assert factory.target is requested_target
    with tvm.transform.PassContext(config=factory.pass_configs):
        artifact = tilelang.lower(func, target=target)
    assert "void call(" in artifact.kernel_source
    assert len(artifact.params) >= len(func.params)
    for param, lowered in zip(func.params, artifact.params):
        declared = func.buffer_map[param]
        assert str(lowered.dtype) == str(declared.dtype)
        assert [str(dim) for dim in lowered.shape] == [str(dim) for dim in declared.shape]
    parallel = []
    for lowered in artifact.device_mod.functions.values():
        tvm.tirx.stmt_functor.post_order_visit(
            lowered.body,
            lambda node: parallel.append(node) if isinstance(node, tvm.tirx.For) and node.kind == tvm.tirx.ForKind.PARALLEL else None,
        )
    assert not parallel
