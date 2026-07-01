"""T.tile.fill test suite."""

import pytest
import torch

import tilelang
import tilelang.language as T

from base import DTYPE_MAP, DEFAULT_PASS_CONFIGS, assert_close_npu, make_test_data


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------

def kernel_fill(M, N, block_M, block_N, dtype="float", value=3.14):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.tile.fill(c_ub, value)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FILL_VALUE = {
    "float16": 3.14,
    "float32": 3.14,
    "int16": 42,
    "int32": 42,
}


def _golden_fill(M, N, dtype_str, value):
    torch_dtype = DTYPE_MAP[dtype_str]
    return torch.full((M, N), value, dtype=torch_dtype, device="npu")


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestTileFill:

    # -- compile tests -------------------------------------------------------

    @pytest.mark.l0
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    @pytest.mark.parametrize("dtype", ["float16", "float32"])
    def test_compile(self, dtype, target):
        M, N = 256, 256
        block_M, block_N = 64, 128
        value = _FILL_VALUE[dtype]
        func = kernel_fill(M, N, block_M, block_N, dtype, value)
        tilelang.compile(
            func,
            out_idx=[-1],
            pass_configs=DEFAULT_PASS_CONFIGS,
            target=target,
        )

    # -- E2E tests -----------------------------------------------------------

    @pytest.mark.l0
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    @pytest.mark.parametrize("dtype", ["float16", "float32"])
    def test_basic(self, dtype, target):
        M, N = 256, 256
        block_M, block_N = 64, 128
        value = _FILL_VALUE[dtype]

        func = kernel_fill(M, N, block_M, block_N, dtype, value)
        func = tilelang.compile(
            func,
            out_idx=[-1],
            pass_configs=DEFAULT_PASS_CONFIGS,
            target=target,
        )

        c = func()
        golden = _golden_fill(M, N, dtype, value)
        assert_close_npu(c, golden, dtype)

    # -- dtype variant tests -------------------------------------------------

    @pytest.mark.l1
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    @pytest.mark.parametrize("dtype", ["int16", "int32"])
    def test_int_dtypes(self, dtype, target):
        M, N = 256, 256
        block_M, block_N = 64, 128
        value = _FILL_VALUE[dtype]

        func = kernel_fill(M, N, block_M, block_N, dtype, value)
        func = tilelang.compile(
            func,
            out_idx=[-1],
            pass_configs=DEFAULT_PASS_CONFIGS,
            target=target,
        )

        c = func()
        golden = _golden_fill(M, N, dtype, value)
        assert_close_npu(c, golden, dtype)

    # -- shape variant tests -------------------------------------------------

    @pytest.mark.l1
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    @pytest.mark.parametrize("shape", [(128, 128), (512, 512)])
    def test_shapes(self, shape, target):
        M, N = shape
        block_M, block_N = 64, 128
        dtype = "float16"
        value = _FILL_VALUE[dtype]

        func = kernel_fill(M, N, block_M, block_N, dtype, value)
        func = tilelang.compile(
            func,
            out_idx=[-1],
            pass_configs=DEFAULT_PASS_CONFIGS,
            target=target,
        )

        c = func()
        golden = _golden_fill(M, N, dtype, value)
        assert_close_npu(c, golden, dtype)

    # -- non-aligned shape tests (L1) ----------------------------------------

    @pytest.mark.l1
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    @pytest.mark.parametrize("M,N", [(100, 200), (107, 145), (255, 513)])
    def test_non_aligned_shapes(self, M, N, target):
        block_M = 64 if M >= 64 else 32
        block_N = 64 if N >= 64 else 32
        M_aligned = (M // block_M) * block_M
        N_aligned = (N // block_N) * block_N
        dtype = "float16"
        value = _FILL_VALUE[dtype]

        func = kernel_fill(M_aligned, N_aligned, block_M, block_N, dtype, value)
        func = tilelang.compile(
            func,
            out_idx=[-1],
            pass_configs=DEFAULT_PASS_CONFIGS,
            target=target,
        )

        c = func()
        golden = _golden_fill(M_aligned, N_aligned, dtype, value)
        assert_close_npu(c, golden, dtype)

    # -- L2 boundary tests ---------------------------------------------------

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    @pytest.mark.parametrize("dtype", ["float16", "float32"])
    def test_fill_zero(self, dtype, target):
        M, N = 256, 256
        block_M, block_N = 64, 128
        value = 0

        func = kernel_fill(M, N, block_M, block_N, dtype, value)
        func = tilelang.compile(
            func,
            out_idx=[-1],
            pass_configs=DEFAULT_PASS_CONFIGS,
            target=target,
        )

        c = func()
        golden = _golden_fill(M, N, dtype, value)
        assert_close_npu(c, golden, dtype)

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_fill_negative(self, target):
        M, N = 256, 256
        block_M, block_N = 64, 128

        dtype_f = "float16"
        value_f = -1.0
        func_f = kernel_fill(M, N, block_M, block_N, dtype_f, value_f)
        func_f = tilelang.compile(
            func_f,
            out_idx=[-1],
            pass_configs=DEFAULT_PASS_CONFIGS,
            target=target,
        )
        c_f = func_f()
        golden_f = _golden_fill(M, N, dtype_f, value_f)
        assert_close_npu(c_f, golden_f, dtype_f)

        dtype_i = "int32"
        value_i = -1
        func_i = kernel_fill(M, N, block_M, block_N, dtype_i, value_i)
        func_i = tilelang.compile(
            func_i,
            out_idx=[-1],
            pass_configs=DEFAULT_PASS_CONFIGS,
            target=target,
        )
        c_i = func_i()
        golden_i = _golden_fill(M, N, dtype_i, value_i)
        assert_close_npu(c_i, golden_i, dtype_i)

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_fill_large(self, target):
        M, N = 256, 256
        block_M, block_N = 64, 128

        dtype_f = "float32"
        value_f = 1e4
        func_f = kernel_fill(M, N, block_M, block_N, dtype_f, value_f)
        func_f = tilelang.compile(
            func_f,
            out_idx=[-1],
            pass_configs=DEFAULT_PASS_CONFIGS,
            target=target,
        )
        c_f = func_f()
        golden_f = _golden_fill(M, N, dtype_f, value_f)
        assert_close_npu(c_f, golden_f, dtype_f)

        dtype_i = "int32"
        value_i = 30000
        func_i = kernel_fill(M, N, block_M, block_N, dtype_i, value_i)
        func_i = tilelang.compile(
            func_i,
            out_idx=[-1],
            pass_configs=DEFAULT_PASS_CONFIGS,
            target=target,
        )
        c_i = func_i()
        golden_i = _golden_fill(M, N, dtype_i, value_i)
        assert_close_npu(c_i, golden_i, dtype_i)

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    @pytest.mark.parametrize("dtype", ["float16", "float32"])
    def test_minimum_shape(self, dtype, target):
        M, N = 64, 64
        block_M, block_N = 64, 64
        value = _FILL_VALUE[dtype]

        func = kernel_fill(M, N, block_M, block_N, dtype, value)
        func = tilelang.compile(
            func,
            out_idx=[-1],
            pass_configs=DEFAULT_PASS_CONFIGS,
            target=target,
        )

        c = func()
        golden = _golden_fill(M, N, dtype, value)
        assert_close_npu(c, golden, dtype)
