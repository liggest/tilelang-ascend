"""Original numerical regression factories, independently checked without NPU."""

import importlib

import pytest

import tilelang
from tilelang import tvm

pytestmark = pytest.mark.a2a3_native


def _cases():
    cases = []
    for B, N in ((4, 8), (3, 16), (2, 32)):
        cases.append(pytest.param("scalar_index", "scalar_index_2d", (B, N), "ascendc", id=f"scalar-{B}-{N}"))
    for dialect in ("ascendc", "pto"):
        for op in ("max", "min"):
            cases.append(
                pytest.param(
                    "integer_minmax_index", "gm_ub_gm_int_minmax_index", (1024, 256, 128, op, "float"), dialect, id=f"clamp-{op}-{dialect}"
                )
            )
        cases.append(pytest.param("dynamic_fill", "dynamic_fill", (1024, 1024, 128, 256, "float"), dialect, id=f"dynamic-fill-{dialect}"))
        cases.append(pytest.param("dynamic_fill", "dynamic_fill_multirow", (8, 32, "float"), dialect, id=f"multirow-fill-{dialect}"))
    for op in ("sum", "max", "min"):
        for valid in (64, 32, 16):
            cases.append(
                pytest.param("narrow_reduce", "narrow_via_real_shape", (16, 512, valid, op), "ascendc", id=f"reduce-real-{op}-{valid}")
            )
        for offset in (0, 64, 448):
            cases.append(
                pytest.param("narrow_reduce", "narrow_via_slice", (16, 512, 64, op, offset), "ascendc", id=f"reduce-slice-{op}-{offset}")
            )
    cases.append(pytest.param("kernel_driven_mma", "kernel_driven_mma", (), "ascendc", id="kernel-driven-mma"))
    return cases


@pytest.mark.parametrize("module_name,factory_name,args,dialect", _cases())
def test_original_regression_factory_full_source(module_name, factory_name, args, dialect):
    module = importlib.import_module(f"testing.ascend_a2a3.python.language.test_tilelang_ascend_language_{module_name}")
    func = getattr(module, factory_name)(*args)
    configs = getattr(module, "pass_configs", getattr(module, "VEC_PASS_CONFIGS", {}))
    target = {"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": [dialect, "ascend"]}
    with tvm.transform.PassContext(config=configs):
        artifact = tilelang.lower(func, target=target)
    assert artifact.kernel_source
    assert 'extern "C"' in artifact.kernel_source and "void call(" in artifact.kernel_source
    assert artifact.target.attrs["arch"] == "dav-2201"
    assert len(artifact.params) == len(func.params)
