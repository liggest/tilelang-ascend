"""Complete original TA example; CLI and device effects execute only in main()."""

import os
from pathlib import Path
from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

"""Minimal tail-block elementwise-add example for the AscendC backend.

M and N are deliberately *not* multiples of the block, so the last row/column
blocks are tails. With the opt-in tail-mask scheme (TL_ASCEND_TAIL_MASK)
``T.tile.add`` is rewritten to a tail-aware helper that computes only over the
valid rectangle; the unused UB gap is pad-filled and never written back.
"""

import tilelang

import tilelang.ascend.a2a3.language as T

import torch

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    # Opt in to the tail-block valid-region scheme (default off).
    tilelang.PassConfigKey.TL_ASCEND_TAIL_MASK: True,
}


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def tail_add(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)
            c_ub = T.alloc_ub((block_M, block_N), dtype)
            T.copy(A[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], a_ub)
            T.copy(B[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], b_ub)
            T.tile.add(c_ub, a_ub, b_ub)
            T.copy(c_ub, C[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N])

    return main


def main():
    if not os.environ.get("TILELANG_CACHE_DIR"):
        raise RuntimeError("Set a dedicated TILELANG_CACHE_DIR before running this cache-clearing example.")
    from tilelang import env

    if Path(env.TILELANG_CACHE_DIR).resolve() != Path(os.environ["TILELANG_CACHE_DIR"]).resolve():
        raise RuntimeError("The resolved cache differs from the explicit TILELANG_CACHE_DIR; remove the caller cache override first.")
    import torch_npu  # noqa: F401

    A2A3CythonKernelCache().clear_cache()

    torch.manual_seed(0)
    for M, N, block_M, block_N, dtype in [
        (34, 130, 32, 32, "float"),
        (34, 130, 32, 32, "float16"),
        (100, 200, 64, 128, "float"),
    ]:
        print(f"tail_add M={M} N={N} block=({block_M},{block_N}) dtype={dtype}")
        func = tail_add(M, N, block_M, block_N, dtype=dtype)
        torch_dtype = torch.float32 if dtype == "float" else torch.float16
        a = torch.randn(M, N, dtype=torch_dtype).npu()
        b = torch.randn(M, N, dtype=torch_dtype).npu()
        c = func(a, b)
        torch.testing.assert_close(c, a + b, rtol=1e-2, atol=1e-2)
        print("  pass")
    print("Kernel Output Match!")


if __name__ == "__main__":
    main()
