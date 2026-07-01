"""T.tile.clear test suite."""

import pytest
import torch

import tilelang
import tilelang.language as T

from base import DTYPE_MAP, DEFAULT_PASS_CONFIGS, assert_close_npu, make_test_data


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------

def kernel_clear(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            # Copy random data into UB first to prove clear actually zeros it
            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(a_ub, c_ub)

            # Clear should zero the buffer
            T.tile.clear(c_ub)

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _golden_clear(M, N, dtype_str):
    torch_dtype = DTYPE_MAP[dtype_str]
    return torch.zeros((M, N), dtype=torch_dtype, device="npu")


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestTileClear:

    # -- compile tests -------------------------------------------------------

    @pytest.mark.l0
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    @pytest.mark.parametrize("dtype", ["float16", "float32"])
    def test_compile(self, dtype, target):
        M, N = 256, 256
        block_M, block_N = 64, 128
        func = kernel_clear(M, N, block_M, block_N, dtype)
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
        torch_dtype = DTYPE_MAP[dtype]

        func = kernel_clear(M, N, block_M, block_N, dtype)
        func = tilelang.compile(
            func,
            out_idx=[-1],
            pass_configs=DEFAULT_PASS_CONFIGS,
            target=target,
        )

        # Feed random non-zero data to prove clear actually zeros it
        a = torch.randn(M, N, dtype=torch.float32, device="npu").to(torch_dtype)

        c = func(a)
        golden = _golden_clear(M, N, dtype)
        assert_close_npu(c, golden, dtype)

    # -- dtype variant tests -------------------------------------------------

    @pytest.mark.l1
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    @pytest.mark.parametrize("dtype", ["int16", "int32"])
    def test_int_dtypes(self, dtype, target):
        M, N = 256, 256
        block_M, block_N = 64, 128
        torch_dtype = DTYPE_MAP[dtype]

        func = kernel_clear(M, N, block_M, block_N, dtype)
        func = tilelang.compile(
            func,
            out_idx=[-1],
            pass_configs=DEFAULT_PASS_CONFIGS,
            target=target,
        )

        a = torch.randint(1, 100, (M, N), dtype=torch_dtype, device="npu")

        c = func(a)
        golden = _golden_clear(M, N, dtype)
        assert_close_npu(c, golden, dtype)

    # -- shape variant tests -------------------------------------------------

    @pytest.mark.l1
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    @pytest.mark.parametrize("shape", [(128, 128), (512, 512)])
    def test_shapes(self, shape, target):
        M, N = shape
        block_M, block_N = 64, 128
        dtype = "float16"
        torch_dtype = DTYPE_MAP[dtype]

        func = kernel_clear(M, N, block_M, block_N, dtype)
        func = tilelang.compile(
            func,
            out_idx=[-1],
            pass_configs=DEFAULT_PASS_CONFIGS,
            target=target,
        )

        a = torch.randn(M, N, dtype=torch.float32, device="npu").to(torch_dtype)

        c = func(a)
        golden = _golden_clear(M, N, dtype)
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
        torch_dtype = DTYPE_MAP[dtype]

        func = kernel_clear(M_aligned, N_aligned, block_M, block_N, dtype)
        func = tilelang.compile(
            func,
            out_idx=[-1],
            pass_configs=DEFAULT_PASS_CONFIGS,
            target=target,
        )

        a = torch.randn(
            M_aligned, N_aligned, dtype=torch.float32, device="npu"
        ).to(torch_dtype)

        c = func(a)
        golden = _golden_clear(M_aligned, N_aligned, dtype)
        assert_close_npu(c, golden, dtype)

    # -- L2 boundary tests ---------------------------------------------------

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    @pytest.mark.parametrize("dtype", ["float16", "float32"])
    def test_clear_after_fill(self, dtype, target):
        M, N = 256, 256
        block_M, block_N = 64, 128
        torch_dtype = DTYPE_MAP[dtype]

        func = kernel_clear(M, N, block_M, block_N, dtype)
        func = tilelang.compile(
            func,
            out_idx=[-1],
            pass_configs=DEFAULT_PASS_CONFIGS,
            target=target,
        )

        a = torch.full((M, N), 5.0, dtype=torch_dtype, device="npu")

        torch.npu.synchronize()
        c = func(a)
        golden = _golden_clear(M, N, dtype)
        assert_close_npu(c, golden, dtype)

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    @pytest.mark.parametrize("dtype", ["float16", "float32"])
    def test_minimum_shape(self, dtype, target):
        M, N = 64, 64
        block_M, block_N = 64, 64
        torch_dtype = DTYPE_MAP[dtype]

        func = kernel_clear(M, N, block_M, block_N, dtype)
        func = tilelang.compile(
            func,
            out_idx=[-1],
            pass_configs=DEFAULT_PASS_CONFIGS,
            target=target,
        )

        a = torch.randn(M, N, dtype=torch.float32, device="npu").to(torch_dtype)

        c = func(a)
        golden = _golden_clear(M, N, dtype)
        assert_close_npu(c, golden, dtype)

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    @pytest.mark.parametrize("dtype", ["float16", "float32"])
    def test_large_shape(self, dtype, target):
        M, N = 1024, 1024
        block_M, block_N = 128, 128
        torch_dtype = DTYPE_MAP[dtype]

        func = kernel_clear(M, N, block_M, block_N, dtype)
        func = tilelang.compile(
            func,
            out_idx=[-1],
            pass_configs=DEFAULT_PASS_CONFIGS,
            target=target,
        )

        a = torch.randn(M, N, dtype=torch.float32, device="npu").to(torch_dtype)

        c = func(a)
        golden = _golden_clear(M, N, dtype)
        assert_close_npu(c, golden, dtype)
