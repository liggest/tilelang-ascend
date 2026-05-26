
import pytest
import torch

import tilelang
import tilelang.language as T


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.disable_cache()
    yield


def _compile_expert(program):
    """Compile program with PTO target (NOT ascendc)."""
    return tilelang.compile(program, target="pto")


def _torch_dtype(dtype):
    if dtype == "float16":
        return torch.float16
    return torch.float32


def _copy_vc_experiment_kernel(M=128, N=128, K=128, dtype="float16"):
    """Test kernel: UB→L1→GM round-trip (Vector to Cube communication via TINSERT)."""
    M_half = T.ceildiv(M, 2)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            A_ub = T.alloc_ub((M_half, K), dtype)
            A_ub_nz_tmp = T.alloc_ub((M_half, K), dtype)
            A_l1 = T.alloc_L1((M, K), dtype)
            B_l1 = T.alloc_L1((K, N), dtype)
            C_l0c = T.alloc_L0C((M, N), "float")

            with T.Scope("C"):
                # UB → L1
                T.wait_cross_flag(0, "M")
                # GM → L1
                T.copy(B, B_l1)
                T.set_flag("mte2", "m", 1)
                T.wait_flag("mte2", "m", 1)
                # L1 → GEMM → L0C (Cube computation)
                T.gemm_v0(A_l1, B_l1, C_l0c, init=True)
                T.set_flag("m", "fix", 2)
                T.wait_flag("m", "fix", 2)
                # L0C → GM
                T.copy(C_l0c, C)

            with T.Scope("V"):
                # GM → UB
                T.copy(A[vid * M_half: (vid + 1) * M_half, :], A_ub)
                # UB → L1 (TINSERT)
                T.set_flag("mte2", "v", 3)
                T.wait_flag("mte2", "v", 3)
                T.copy_op.copy_vc_experiment(A_ub, A_l1[vid * M_half, 0], A_ub_nz_tmp)
                T.set_cross_flag("mte3", 0)
    return main


def _copy_cv_experiment_kernel(M=128, N=128, K=128, dtype="float16"):
    """Test kernel: GM→L1→GEMM→L0C→UB→GM (Cube to Vector communication via TMOV)."""
    M_half = T.ceildiv(M, 2)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), "float"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            A_l1 = T.alloc_L1((M, K), dtype)
            B_l1 = T.alloc_L1((K, N), dtype)
            C_l0c = T.alloc_L0C((M, N), "float")
            C_ub = T.alloc_ub((M_half, N), "float")

            with T.Scope("C"):
                # GM → L1
                T.copy(A, A_l1)
                T.copy(B, B_l1)
                T.set_flag("mte2", "m", 1)
                T.wait_flag("mte2", "m", 1)
                # L1 → GEMM → L0C (Cube computation)
                T.gemm_v0(A_l1, B_l1, C_l0c, init=True)
                T.set_flag("m", "fix", 2)
                T.wait_flag("m", "fix", 2)
                # L0C → UB (TMOV)
                T.copy_op.copy_cv_experiment(C_l0c, C_ub, T.copy_op.CopyCVMode.DualSplitM)
                T.set_cross_flag("FIX", 0)

            with T.Scope("V"):
                # L0C → UB
                T.wait_cross_flag(0, "MTE3")
                # UB → GM
                T.copy(C_ub, C[vid * M_half: (vid + 1) * M_half, :])
    return main


@pytest.mark.parametrize("dtype", ["float16"])
def test_copy_vc_experiment(dtype):
    M, N, K = 128, 128, 128

    program = _copy_vc_experiment_kernel(M=M, N=N, K=K, dtype=dtype)

    kernel = _compile_expert(program)

    with open("copy_vc_experiment.cpp", "w", encoding="utf-8") as f:
        print("source code dumped to: copy_vc_experiment.cpp")
        f.write(kernel.get_kernel_source())

    torch_dtype = _torch_dtype(dtype)

    a = torch.randn((M, K), dtype=torch_dtype, device="npu")
    b = torch.randn((K, N), dtype=torch_dtype, device="npu")
    c = torch.empty((M, N), dtype=torch_dtype, device="npu")
    torch.npu.synchronize()

    kernel(a, b, c)
    torch.npu.synchronize()

    ref_c = a @ b
    torch.testing.assert_close(c, ref_c, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", ["float16"])
def test_copy_cv_experiment(dtype):
    M, N, K = 128, 128, 128

    program = _copy_cv_experiment_kernel(M=M, N=N, K=K, dtype=dtype)

    kernel = _compile_expert(program)

    with open("copy_cv_experiment.cpp", "w", encoding="utf-8") as f:
        print("source code dumped to: copy_cv_experiment.cpp")
        f.write(kernel.get_kernel_source())

    torch_dtype = _torch_dtype(dtype)

    a = torch.randn((M, K), dtype=torch_dtype, device="npu")
    b = torch.randn((K, N), dtype=torch_dtype, device="npu")
    c = torch.empty((M, N), dtype=torch.float32, device="npu")
    torch.npu.synchronize()

    kernel(a, b, c)
    torch.npu.synchronize()

    ref_c = a @ b
    torch.testing.assert_close(c, ref_c.to(torch.float32), rtol=1e-3, atol=1e-3)
