"""T.tile.clamp test suite."""

import math

import pytest
import torch

import tilelang
import tilelang.language as T

from base import DTYPE_MAP, DEFAULT_PASS_CONFIGS, assert_close_npu, make_test_data


def _make_clamp():
    def _op(dst, src, min_val, max_val):
        count = math.prod(dst.shape)
        return T.tile.clamp(dst, src, min_val, max_val, count)
    return _op


_clamp_op = _make_clamp()


class _ClampSpec:
    name = "clamp"
    supported_dtypes = ["float16", "float32"]


_spec = _ClampSpec()

SCALAR_RANGES = [
    (-1.0, 1.0),
    (0.0, 0.5),
    (-100.0, 100.0),
    (0.0, 1.0),
]


def kernel_clamp(M, N, block_M, block_N, min_val, max_val, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            _clamp_op(b_ub, a_ub, min_val, max_val)
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


@pytest.mark.compile_time
class TestTileClampCompile:
    op = _spec

    @pytest.mark.l0
    def test_compiles(self, dtype):
        func = kernel_clamp(128, 128, 64, 64, -1.0, 1.0, dtype)
        compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target="ascendc")
        assert callable(compiled)

    @pytest.mark.l0
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_compiles_both_targets(self, dtype, target):
        func = kernel_clamp(128, 128, 64, 64, -1.0, 1.0, dtype)
        compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        assert callable(compiled)


class TestTileClampE2E:
    op = _spec

    @pytest.mark.l0
    @pytest.mark.parametrize("min_val,max_val", SCALAR_RANGES)
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_basic(self, min_val, max_val, dtype, target):
        func = kernel_clamp(1024, 1024, 128, 128, min_val, max_val, dtype)
        compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        a = make_test_data((1024, 1024), dtype)
        torch.npu.synchronize()
        b = compiled(a)
        golden = torch.clamp(a, min_val, max_val)
        assert_close_npu(b, golden, dtype)

    @pytest.mark.l1
    @pytest.mark.parametrize("M,N,block_M,block_N", [
        (256, 256, 64, 64),
        (512, 1024, 64, 128),
        (1024, 512, 128, 64),
    ])
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_various_shapes(self, M, N, block_M, block_N, dtype, target):
        func = kernel_clamp(M, N, block_M, block_N, -1.0, 1.0, dtype)
        compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        a = make_test_data((M, N), dtype)
        torch.npu.synchronize()
        b = compiled(a)
        golden = torch.clamp(a, -1.0, 1.0)
        assert_close_npu(b, golden, dtype)

    @pytest.mark.l1
    @pytest.mark.parametrize("M,N", [(100, 200), (107, 145), (255, 513)])
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_non_aligned_shapes(self, M, N, dtype, target):
        block_M = 64 if M >= 64 else 32
        block_N = 64 if N >= 64 else 32
        M_aligned = (M // block_M) * block_M
        N_aligned = (N // block_N) * block_N
        func = kernel_clamp(M_aligned, N_aligned, block_M, block_N, -1.0, 1.0, dtype)
        compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        a = make_test_data((M_aligned, N_aligned), dtype)
        torch.npu.synchronize()
        b = compiled(a)
        golden = torch.clamp(a, -1.0, 1.0)
        assert_close_npu(b, golden, dtype)


class TestTileClampBoundary:
    op = _spec

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_zeros(self, dtype, target):
        func = kernel_clamp(256, 256, 64, 64, -1.0, 1.0, dtype)
        compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        a = torch.zeros(256, 256, dtype=DTYPE_MAP[dtype], device="npu")
        torch.npu.synchronize()
        b = compiled(a)
        assert_close_npu(b, torch.clamp(a, -1.0, 1.0), dtype)

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_all_out_of_range(self, dtype, target):
        func = kernel_clamp(256, 256, 64, 64, -1.0, 1.0, dtype)
        compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        a = torch.full((256, 256), 100.0, dtype=DTYPE_MAP[dtype], device="npu")
        torch.npu.synchronize()
        b = compiled(a)
        assert_close_npu(b, torch.clamp(a, -1.0, 1.0), dtype)

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_negative_out_of_range(self, dtype, target):
        func = kernel_clamp(256, 256, 64, 64, -1.0, 1.0, dtype)
        compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        a = torch.full((256, 256), -100.0, dtype=DTYPE_MAP[dtype], device="npu")
        torch.npu.synchronize()
        b = compiled(a)
        assert_close_npu(b, torch.clamp(a, -1.0, 1.0), dtype)

    @pytest.mark.l2
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_minimum_shape(self, dtype, target):
        func = kernel_clamp(64, 64, 64, 64, -1.0, 1.0, dtype)
        compiled = tilelang.compile(func, out_idx=[-1], pass_configs=DEFAULT_PASS_CONFIGS, target=target)
        a = make_test_data((64, 64), dtype)
        torch.npu.synchronize()
        b = compiled(a)
        assert_close_npu(b, torch.clamp(a, -1.0, 1.0), dtype)
