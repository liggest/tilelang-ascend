"""Original TA example; runtime effects execute only through main()."""

import os
from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache
import argparse

import tilelang
import tilelang.ascend.a2a3.language as T
import torch


@tilelang.jit(out_idx=[1])
def reduce_min(M, N, block_M, dtype="float"):
    m_num = M // block_M
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor([M, N], dtype),
        B: T.Tensor([M], dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((sub_block_M, N), dtype)
            b_ub = T.alloc_ub((sub_block_M), dtype)

            row_base = cid * block_M + vid * sub_block_M
            with T.Scope("V"):
                T.copy(A[row_base : row_base + sub_block_M, :], a_ub)

                T.barrier_all()
                T.reduce_min(a_ub, b_ub, dim=-1)
                T.barrier_all()

                T.copy(b_ub, B[row_base : row_base + sub_block_M])

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
    parser.add_argument("--n", type=int, default=256, help="Matrix N dimension")
    args = parser.parse_args()

    M = args.m
    N = args.n

    func = reduce_min(M, N, 128)

    torch.manual_seed(0)

    a = torch.randn(M, N).npu()

    torch.npu.synchronize()
    print("init successful!")

    c = func(a)

    ref_c = torch.min(a, dim=-1).values

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Kernel Output Match!")


if __name__ == "__main__":
    main()
