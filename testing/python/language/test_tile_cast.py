"""T.tile.cast test suite.

API: T.tile.cast(dst, src, mode, count)
     Element-wise data type conversion with rounding mode.

Note: Ascend C Cast intrinsic requires cross-dtype conversion (e.g. float16->float32).
Same-type cast (e.g. float32->float32) is not a supported hardware overload.
"""

import math

import pytest
import torch

import tilelang
import tilelang.language as T

from base import DTYPE_MAP, DEFAULT_PASS_CONFIGS, assert_close_npu, make_test_data

SRC_DTYPE = "float16"
DST_DTYPE = "float32"
CAST_MODE = "CAST_NONE"


# ---------------------------------------------------------------------------
# Count wrapper — auto-infer count from buffer shape
# ---------------------------------------------------------------------------

def _cast_wrapper(dst, src, mode):
    """Wrapper: auto-infer count from dst.shape."""
    count = math.prod(dst.shape)
    return T.tile.cast(dst, src, mode, count)


# ---------------------------------------------------------------------------
# Kernel factory — cross-dtype: float16 -> float32
# ---------------------------------------------------------------------------

def kernel_cast(M, N, block_M, block_N, src_dtype="float16", dst_dtype="float32",
                mode="CAST_NONE"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), src_dtype),  # type: ignore
        B: T.Tensor((M, N), dst_dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), src_dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dst_dtype)
            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            _cast_wrapper(b_ub, a_ub, mode)
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


# ---------------------------------------------------------------------------
# Spec (for conftest dtype parametrization)
# ---------------------------------------------------------------------------

class _CastSpec:
    name = "cast"
    supported_dtypes = [SRC_DTYPE]


_spec = _CastSpec()


# ---------------------------------------------------------------------------
# Golden — float16 -> float32 with CAST_NONE
# ---------------------------------------------------------------------------

def _golden(a):
    return a.to(DTYPE_MAP[DST_DTYPE])


# ---------------------------------------------------------------------------
# Compile tests
# ---------------------------------------------------------------------------

@pytest.mark.compile_time
class TestTileCastCompile:
    op = _spec
    _dtype_source = "supported_dtypes"

    @pytest.mark.l0
    def test_compiles(self, dtype):
        func = kernel_cast(128, 128, 64, 64, dtype, DST_DTYPE, CAST_MODE)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target="ascendc"
        )
        assert callable(compiled)

    @pytest.mark.l0
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_compiles_both_targets(self, dtype, target):
        func = kernel_cast(128, 128, 64, 64, dtype, DST_DTYPE, CAST_MODE)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        assert callable(compiled)

    @pytest.mark.l1
    @pytest.mark.parametrize("shape", [(64, 64), (128, 256), (256, 128)])
    def test_various_shapes_compile(self, shape):
        M, N = shape
        func = kernel_cast(M, N, 64, 64, SRC_DTYPE, DST_DTYPE, CAST_MODE)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target="ascendc"
        )
        assert callable(compiled)


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------

class TestTileCastE2E:
    op = _spec
    _dtype_source = "supported_dtypes"

    @pytest.mark.l0
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_basic(self, dtype, target):
        M, N, block_M, block_N = 1024, 1024, 128, 128
        func = kernel_cast(M, N, block_M, block_N, dtype, DST_DTYPE, CAST_MODE)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        a = make_test_data((M, N), dtype)
        torch.npu.synchronize()
        b = compiled(a)
        assert_close_npu(b, _golden(a), DST_DTYPE)

    @pytest.mark.l1
    @pytest.mark.parametrize("M,N,block_M,block_N", [
        (256, 256, 64, 64),
        (512, 1024, 64, 128),
        (1024, 512, 128, 64),
    ])
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_various_shapes(self, M, N, block_M, block_N, dtype, target):
        func = kernel_cast(M, N, block_M, block_N, dtype, DST_DTYPE, CAST_MODE)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        a = make_test_data((M, N), dtype)
        torch.npu.synchronize()
        b = compiled(a)
        assert_close_npu(b, _golden(a), DST_DTYPE)

    @pytest.mark.l1
    @pytest.mark.parametrize("M,N", [(100, 200), (107, 145), (255, 513)])
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_non_aligned_shapes(self, M, N, dtype, target):
        block_M = 64 if M >= 64 else 32
        block_N = 64 if N >= 64 else 32
        M_aligned = (M // block_M) * block_M
        N_aligned = (N // block_N) * block_N
        func = kernel_cast(M_aligned, N_aligned, block_M, block_N, dtype, DST_DTYPE,
                           CAST_MODE)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        a = make_test_data((M_aligned, N_aligned), dtype)
        torch.npu.synchronize()
        b = compiled(a)
        assert_close_npu(b, _golden(a), DST_DTYPE)


# ---------------------------------------------------------------------------
# Boundary tests
# ---------------------------------------------------------------------------

class TestTileCastBoundary:
    op = _spec
    _dtype_source = "supported_dtypes"

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_zeros(self, dtype, target):
        torch_dtype = DTYPE_MAP[dtype]
        a = torch.zeros(256, 256, dtype=torch_dtype, device="npu")
        func = kernel_cast(256, 256, 64, 64, dtype, DST_DTYPE, CAST_MODE)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        torch.npu.synchronize()
        b = compiled(a)
        assert_close_npu(b, _golden(a), DST_DTYPE)

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_negative_values(self, dtype, target):
        torch_dtype = DTYPE_MAP[dtype]
        a = torch.full((256, 256), -5.0, dtype=torch_dtype, device="npu")
        func = kernel_cast(256, 256, 64, 64, dtype, DST_DTYPE, CAST_MODE)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        torch.npu.synchronize()
        b = compiled(a)
        assert_close_npu(b, _golden(a), DST_DTYPE)

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_large_values(self, dtype, target):
        torch_dtype = DTYPE_MAP[dtype]
        a = torch.full((256, 256), 60000.0, dtype=torch_dtype, device="npu")
        func = kernel_cast(256, 256, 64, 64, dtype, DST_DTYPE, CAST_MODE)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        torch.npu.synchronize()
        b = compiled(a)
        assert_close_npu(b, _golden(a), DST_DTYPE)

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_inf_input(self, target):
        a = torch.full((256, 256), float("inf"), dtype=torch.float16, device="npu")
        func = kernel_cast(256, 256, 64, 64, SRC_DTYPE, DST_DTYPE, CAST_MODE)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        torch.npu.synchronize()
        b = compiled(a)
        assert b.shape == (256, 256)
        assert torch.all(torch.isinf(b))

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_nan_input(self, target):
        a = torch.full((256, 256), float("nan"), dtype=torch.float16, device="npu")
        func = kernel_cast(256, 256, 64, 64, SRC_DTYPE, DST_DTYPE, CAST_MODE)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        torch.npu.synchronize()
        b = compiled(a)
        assert b.shape == (256, 256)
        assert torch.all(torch.isnan(b))

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_minimum_shape(self, dtype, target):
        func = kernel_cast(64, 64, 64, 64, dtype, DST_DTYPE, CAST_MODE)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        a = make_test_data((64, 64), dtype)
        torch.npu.synchronize()
        b = compiled(a)
        assert_close_npu(b, _golden(a), DST_DTYPE)
