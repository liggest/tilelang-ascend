"""Original TA developer example; runtime effects execute only through main()."""

import os
from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache
import tilelang
import tilelang.ascend.a2a3.language as T
import torch

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)

            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


test_configs = [
    (32 * 3 + 30, 32 * 2 + 16, 32 * 4 + 31, 32, 32, 32),
    (64 * 8 + 45, 64 * 8, 64 * 8 + 27, 64, 64, 64),
    (128 * 4, 128 * 4 + 99, 128 * 4, 128, 128, 128),
    (1024 + 118, 1024 + 206, 1024 + 55, 128, 256, 64),
]


def main():
    if not os.environ.get("TILELANG_CACHE_DIR"):
        raise RuntimeError("Set a dedicated TILELANG_CACHE_DIR before running this cache-clearing example.")
    from pathlib import Path
    from tilelang import env

    if Path(env.TILELANG_CACHE_DIR).resolve() != Path(os.environ["TILELANG_CACHE_DIR"]).resolve():
        raise RuntimeError("The resolved cache differs from the explicit TILELANG_CACHE_DIR; remove the caller cache override first.")
    import torch_npu  # noqa: F401

    A2A3CythonKernelCache().clear_cache()

    torch.manual_seed(0)

    for idx, (M, N, K, block_M, block_N, block_K) in enumerate(test_configs, 1):
        try:
            func = matmul(M, N, K, block_M, block_N, block_K)
            a = torch.randn(M, K).half().npu()
            b = torch.randn(K, N).half().npu()
            c = torch.empty(M, N).half().npu()
            c = func(a, b)
            ref_c = a @ b
            torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
            print(f"Passed test case {idx}/{len(test_configs)}: M={M}, N={N}, K={K}")
        except Exception as e:
            print("error message:", e)
            print(f"Failed test case {idx}/{len(test_configs)}: M={M}, N={N}, K={K}")
            raise

    print("All test cases passed!")
    print("Kernel Output Match!")


if __name__ == "__main__":
    main()
