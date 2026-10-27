"""Host contracts for the original adopted test factories and dtype collector."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from testing.ascend_a2a3.python import base
from testing.ascend_a2a3.python.base import binary_op, unary_op


@pytest.mark.parametrize("dtype", ["float16", "float32", "int16", "int32", "uint8"])
def test_cpu_data_and_original_precision_contract(dtype):
    data = base.make_test_data((2, 4), dtype, device="cpu", low=0, high=7)
    assert data.dtype == base.DTYPE_MAP[dtype] and data.shape == (2, 4)
    base.assert_close_npu(data, data.clone(), dtype)
    bad = data.clone()
    bad[0, 0] += 1
    with pytest.raises(AssertionError):
        base.assert_close_npu(data, bad, dtype)


def test_original_spec_registration_preserves_all_generated_classes():
    # The original API registers in the actual caller namespace and returns
    # private bases so consumer tests can preserve their custom subclasses.
    binary = base.BinaryOpSpec("adopted_binary", lambda *args: None, torch.maximum, ["float32"])
    unary = base.UnaryOpSpec("adopted_unary", lambda *args: None, torch.abs, ["float32"])
    created = []
    try:
        for register, spec in ((base.register_binary_op_tests, binary), (base.register_unary_op_tests, unary)):
            classes = register(spec)
            for suffix, private in zip(("Compile", "E2E", "Boundary"), classes):
                name = f"TestTile{spec.name.capitalize()}{suffix}"
                created.append(name)
                public = globals()[name]
                assert public.op is spec and private.op is spec
                assert public.__bases__ == private.__bases__
                assert private.__name__ == f"_{name}"
    finally:
        for name in created:
            globals().pop(name, None)


def test_original_dtype_hook_preserves_boundary_and_priority_marks():
    config = importlib.import_module("testing.ascend_a2a3.python.language.conftest")
    calls = []
    spec = base.BinaryOpSpec("sample", None, None, ["float32", "int32"], boundary_dtypes=["float16"], low_priority_dtypes=["int32"])
    metafunc = SimpleNamespace(fixturenames=["dtype"], cls=SimpleNamespace(op=spec), parametrize=lambda *args: calls.append(args))
    config.pytest_generate_tests(metafunc)
    [(name, params)] = calls
    assert name == "dtype" and params[0] == "float32"
    assert params[1].values == ("int32",) and [mark.name for mark in params[1].marks] == ["low_priority"]
    calls.clear()
    metafunc.cls._dtype_source = "boundary_dtypes"
    config.pytest_generate_tests(metafunc)
    assert calls == [("dtype", ["float16"])]


@pytest.mark.a2a3_native
@pytest.mark.parametrize("kind", ["binary", "one_dimensional", "scalar", "unary"])
def test_original_factory_builds_complete_ir(kind):
    import tilelang.ascend.a2a3.language as T
    from tilelang import tvm

    if kind == "binary":
        func = binary_op.make_binary_kernel(T.tile.add)(64, 64, 64, 64, "float32")
    elif kind == "one_dimensional":
        func = binary_op.make_1d_kernel(T.tile.max)(64, "float32")
    elif kind == "scalar":
        func = binary_op.make_scalar_kernel(T.tile.add)(64, 64, 64, 64, 2.0, "float32")
    else:
        func = unary_op.make_unary_kernel(T.tile.abs)(64, 64, 64, 64, "float32")
    assert isinstance(func, tvm.tirx.PrimFunc)
    assert len(func.params) == (3 if kind in ("binary", "one_dimensional") else 2)
    assert all(buffer.dtype == "float32" for buffer in func.buffer_map.values())
    restored = tvm.ir.load_json(tvm.ir.save_json(func))
    tvm.ir.assert_structural_equal(func, restored)
