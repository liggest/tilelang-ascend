"""Original TA example; runtime effects execute only through main()."""

import os
from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache
import argparse

import tilelang
import tilelang.ascend.a2a3.language as T
import torch


@tilelang.jit(out_idx=[-1])
def vec_add(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

                T.barrier_all()
                T.tile.add(c_ub, a_ub, b_ub)
                T.barrier_all()

                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def main():
    if not os.environ.get("TILELANG_CACHE_DIR"):
        raise RuntimeError("Set a dedicated TILELANG_CACHE_DIR before running this cache-clearing example.")
    from pathlib import Path
    from tilelang import env

    if Path(env.TILELANG_CACHE_DIR).resolve() != Path(os.environ["TILELANG_CACHE_DIR"]).resolve():
        raise RuntimeError("The resolved cache differs from the explicit TILELANG_CACHE_DIR; remove the caller cache override first.")

    import torch_npu  # noqa: F401

    A2A3CythonKernelCache().clear_cache()

    parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
    parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
    parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
    args = parser.parse_args()

    M = args.m
    N = args.n

    func = vec_add(M, N, 128, 256)

    torch.manual_seed(0)

    a = torch.randn(M, N).npu()
    b = torch.randn(M, N).npu()

    torch.npu.synchronize()
    print("init successful!")

    c = func(a, b)

    ref_c = a + b

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Kernel Output Match!")


if __name__ == "__main__":
    main()
