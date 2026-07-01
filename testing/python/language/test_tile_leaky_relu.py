"""T.tile.leaky_relu test suite.

API: T.tile.leaky_relu(dst, src0, scalar_value)
     dst = src0 if src0 >= 0 else src0 * alpha
"""

import pytest
import torch

import tilelang
import tilelang.language as T

from base import DTYPE_MAP, DEFAULT_PASS_CONFIGS, assert_close_npu, make_test_data

ALPHA = 0.1


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------

def kernel_leaky_relu(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.tile.leaky_relu(b_ub, a_ub, ALPHA)
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


# ---------------------------------------------------------------------------
# Spec (for conftest dtype parametrization)
# ---------------------------------------------------------------------------

class _LeakyReluSpec:
    name = "leaky_relu"
    supported_dtypes = ["float16", "float32"]


_spec = _LeakyReluSpec()


# ---------------------------------------------------------------------------
# Golden
# ---------------------------------------------------------------------------

def _golden(a):
    return torch.nn.functional.leaky_relu(a, negative_slope=ALPHA)


# ---------------------------------------------------------------------------
# Compile tests
# ---------------------------------------------------------------------------

@pytest.mark.compile_time
class TestTileLeakyReluCompile:
    op = _spec
    _dtype_source = "supported_dtypes"

    @pytest.mark.l0
    def test_compiles(self, dtype):
        func = kernel_leaky_relu(128, 128, 64, 64, dtype)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target="ascendc"
        )
        assert callable(compiled)

    @pytest.mark.l0
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_compiles_both_targets(self, dtype, target):
        func = kernel_leaky_relu(128, 128, 64, 64, dtype)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        assert callable(compiled)

    @pytest.mark.l1
    @pytest.mark.parametrize("shape", [(64, 64), (128, 256), (256, 128)])
    def test_various_shapes_compile(self, shape):
        M, N = shape
        func = kernel_leaky_relu(M, N, 64, 64, _spec.supported_dtypes[0])
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target="ascendc"
        )
        assert callable(compiled)


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------

class TestTileLeakyReluE2E:
    op = _spec
    _dtype_source = "supported_dtypes"

    @pytest.mark.l0
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_basic(self, dtype, target):
        M, N, block_M, block_N = 1024, 1024, 128, 128
        func = kernel_leaky_relu(M, N, block_M, block_N, dtype=dtype)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        a = make_test_data((M, N), dtype)
        torch.npu.synchronize()
        b = compiled(a)
        assert_close_npu(b, _golden(a), dtype)

    @pytest.mark.l1
    @pytest.mark.parametrize("M,N,block_M,block_N", [
        (256, 256, 64, 64),
        (512, 1024, 64, 128),
        (1024, 512, 128, 64),
    ])
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_various_shapes(self, M, N, block_M, block_N, dtype, target):
        func = kernel_leaky_relu(M, N, block_M, block_N, dtype=dtype)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        a = make_test_data((M, N), dtype)
        torch.npu.synchronize()
        b = compiled(a)
        assert_close_npu(b, _golden(a), dtype)

    @pytest.mark.l1
    @pytest.mark.parametrize("M,N", [(100, 200), (107, 145), (255, 513)])
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_non_aligned_shapes(self, M, N, dtype, target):
        block_M = 64 if M >= 64 else 32
        block_N = 64 if N >= 64 else 32
        M_aligned = (M // block_M) * block_M
        N_aligned = (N // block_N) * block_N
        func = kernel_leaky_relu(M_aligned, N_aligned, block_M, block_N, dtype=dtype)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        a = make_test_data((M_aligned, N_aligned), dtype)
        torch.npu.synchronize()
        b = compiled(a)
        assert_close_npu(b, _golden(a), dtype)


# ---------------------------------------------------------------------------
# Boundary tests
# ---------------------------------------------------------------------------

class TestTileLeakyReluBoundary:
    op = _spec
    _dtype_source = "supported_dtypes"

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_zeros(self, dtype, target):
        torch_dtype = DTYPE_MAP[dtype]
        a = torch.zeros(256, 256, dtype=torch_dtype, device="npu")
        func = kernel_leaky_relu(256, 256, 64, 64, dtype)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        torch.npu.synchronize()
        b = compiled(a)
        assert_close_npu(b, _golden(a), dtype)

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_negative_values(self, dtype, target):
        torch_dtype = DTYPE_MAP[dtype]
        a = torch.full((256, 256), -5.0, dtype=torch_dtype, device="npu")
        func = kernel_leaky_relu(256, 256, 64, 64, dtype)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        torch.npu.synchronize()
        b = compiled(a)
        assert_close_npu(b, _golden(a), dtype)

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_large_values(self, dtype, target):
        torch_dtype = DTYPE_MAP[dtype]
        if dtype == "float32":
            a = torch.full((256, 256), 1e30, dtype=torch_dtype, device="npu")
        else:
            a = torch.full((256, 256), 60000.0, dtype=torch_dtype, device="npu")
        func = kernel_leaky_relu(256, 256, 64, 64, dtype)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        torch.npu.synchronize()
        b = compiled(a)
        assert b.shape == (256, 256)

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_inf_input(self, target):
        a = torch.full((256, 256), float("inf"), dtype=torch.float16, device="npu")
        func = kernel_leaky_relu(256, 256, 64, 64, "float16")
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
        func = kernel_leaky_relu(256, 256, 64, 64, "float16")
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
        func = kernel_leaky_relu(64, 64, 64, 64, dtype=dtype)
        compiled = tilelang.compile(
            func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target
        )
        a = make_test_data((64, 64), dtype)
        torch.npu.synchronize()
        b = compiled(a)
        assert_close_npu(b, _golden(a), dtype)
