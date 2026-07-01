"""T.tile.axpy test suite.

API: T.tile.axpy(dst, src0, scalar_value)
     dst = alpha * src0 + dst  (in-place, dst is both input and output)
"""

import pytest
import torch

import tilelang
import tilelang.language as T

from base import DTYPE_MAP, DEFAULT_PASS_CONFIGS, assert_close_npu, make_test_data

ALPHA = 2.0


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------

def kernel_axpy(M, N, block_M, block_N, dtype="float"):
    """Two GM inputs (A=src0, B=dst_initial), one GM output (C=dst_final)."""
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore  src0
        B: T.Tensor((M, N), dtype),  # type: ignore  dst_initial
        C: T.Tensor((M, N), dtype),  # type: ignore  dst_final
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)
            T.tile.axpy(b_ub, a_ub, ALPHA)
            T.copy(b_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


# ---------------------------------------------------------------------------
# Spec (for conftest dtype parametrization)
# ---------------------------------------------------------------------------

class _AxpySpec:
    name = "axpy"
    supported_dtypes = ["float16", "float32"]


_spec = _AxpySpec()


# ---------------------------------------------------------------------------
# Golden
# ---------------------------------------------------------------------------

def _golden(a, b):
    return ALPHA * a + b


# ---------------------------------------------------------------------------
# Compile tests
# ---------------------------------------------------------------------------

@pytest.mark.compile_time
class TestTileAxpyCompile:
    op = _spec
    _dtype_source = "supported_dtypes"

    @pytest.mark.l0
    def test_compiles(self, dtype):
        func = kernel_axpy(128, 128, 64, 64, dtype)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target="ascendc"
        )
        assert callable(compiled)

    @pytest.mark.l0
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_compiles_both_targets(self, dtype, target):
        func = kernel_axpy(128, 128, 64, 64, dtype)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        assert callable(compiled)

    @pytest.mark.l1
    @pytest.mark.parametrize("shape", [(64, 64), (128, 256), (256, 128)])
    def test_various_shapes_compile(self, shape):
        M, N = shape
        func = kernel_axpy(M, N, 64, 64, _spec.supported_dtypes[0])
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target="ascendc"
        )
        assert callable(compiled)


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------

class TestTileAxpyE2E:
    op = _spec
    _dtype_source = "supported_dtypes"

    @pytest.mark.l0
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_basic(self, dtype, target):
        M, N, block_M, block_N = 1024, 1024, 128, 128
        func = kernel_axpy(M, N, block_M, block_N, dtype=dtype)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        a = make_test_data((M, N), dtype)
        b = make_test_data((M, N), dtype)
        torch.npu.synchronize()
        c = compiled(a, b)
        assert_close_npu(c, _golden(a, b), dtype)

    @pytest.mark.l1
    @pytest.mark.parametrize("M,N,block_M,block_N", [
        (256, 256, 64, 64),
        (512, 1024, 64, 128),
        (1024, 512, 128, 64),
    ])
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_various_shapes(self, M, N, block_M, block_N, dtype, target):
        func = kernel_axpy(M, N, block_M, block_N, dtype=dtype)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        a = make_test_data((M, N), dtype)
        b = make_test_data((M, N), dtype)
        torch.npu.synchronize()
        c = compiled(a, b)
        assert_close_npu(c, _golden(a, b), dtype)

    @pytest.mark.l1
    @pytest.mark.parametrize("M,N", [(100, 200), (107, 145), (255, 513)])
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_non_aligned_shapes(self, M, N, dtype, target):
        block_M = 64 if M >= 64 else 32
        block_N = 64 if N >= 64 else 32
        M_aligned = (M // block_M) * block_M
        N_aligned = (N // block_N) * block_N
        func = kernel_axpy(M_aligned, N_aligned, block_M, block_N, dtype=dtype)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        a = make_test_data((M_aligned, N_aligned), dtype)
        b = make_test_data((M_aligned, N_aligned), dtype)
        torch.npu.synchronize()
        c = compiled(a, b)
        assert_close_npu(c, _golden(a, b), dtype)


# ---------------------------------------------------------------------------
# Boundary tests
# ---------------------------------------------------------------------------

class TestTileAxpyBoundary:
    op = _spec
    _dtype_source = "supported_dtypes"

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_zeros(self, dtype, target):
        torch_dtype = DTYPE_MAP[dtype]
        a = torch.zeros(256, 256, dtype=torch_dtype, device="npu")
        b = torch.zeros(256, 256, dtype=torch_dtype, device="npu")
        func = kernel_axpy(256, 256, 64, 64, dtype)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        torch.npu.synchronize()
        c = compiled(a, b)
        assert_close_npu(c, torch.zeros_like(a), dtype)

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_negative_values(self, dtype, target):
        torch_dtype = DTYPE_MAP[dtype]
        a = torch.full((256, 256), -5.0, dtype=torch_dtype, device="npu")
        b = torch.full((256, 256), 3.0, dtype=torch_dtype, device="npu")
        func = kernel_axpy(256, 256, 64, 64, dtype)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        torch.npu.synchronize()
        c = compiled(a, b)
        assert_close_npu(c, _golden(a, b), dtype)

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_large_values(self, dtype, target):
        torch_dtype = DTYPE_MAP[dtype]
        if dtype == "float32":
            a = torch.full((256, 256), 1e30, dtype=torch_dtype, device="npu")
            b = torch.full((256, 256), 1e30, dtype=torch_dtype, device="npu")
        else:
            a = torch.full((256, 256), 60000.0, dtype=torch_dtype, device="npu")
            b = torch.full((256, 256), 1.0, dtype=torch_dtype, device="npu")
        func = kernel_axpy(256, 256, 64, 64, dtype)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        torch.npu.synchronize()
        c = compiled(a, b)
        assert c.shape == (256, 256)

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_inf_input(self, target):
        a = torch.full((256, 256), float("inf"), dtype=torch.float16, device="npu")
        b = torch.ones(256, 256, dtype=torch.float16, device="npu")
        func = kernel_axpy(256, 256, 64, 64, "float16")
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        torch.npu.synchronize()
        c = compiled(a, b)
        assert c.shape == (256, 256)
        assert torch.all(torch.isinf(c))

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_nan_input(self, target):
        a = torch.full((256, 256), float("nan"), dtype=torch.float16, device="npu")
        b = torch.ones(256, 256, dtype=torch.float16, device="npu")
        func = kernel_axpy(256, 256, 64, 64, "float16")
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        torch.npu.synchronize()
        c = compiled(a, b)
        assert c.shape == (256, 256)
        assert torch.all(torch.isnan(c))

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_minimum_shape(self, dtype, target):
        func = kernel_axpy(64, 64, 64, 64, dtype=dtype)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        a = make_test_data((64, 64), dtype)
        b = make_test_data((64, 64), dtype)
        torch.npu.synchronize()
        c = compiled(a, b)
        assert_close_npu(c, _golden(a, b), dtype)
