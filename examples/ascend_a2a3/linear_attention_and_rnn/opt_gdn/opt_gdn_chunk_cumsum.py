import os
from pathlib import Path
import tilelang
import tilelang.ascend.a2a3.language as T
import torch

"""
Functionality:
Chunkwisely calculate the prefix sum
"""

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def cumsum_ker(B, H, L, C, CC=8, accum_dtype="float"):
    chunk_num = T.ceildiv(L, C * CC)
    VEC_NUM = 2

    @T.prim_func
    def main(
        G: T.Tensor([B, H, L], accum_dtype),
        S: T.Tensor([B, H, L], accum_dtype),
    ):
        with T.Kernel(B * (H // VEC_NUM) * chunk_num, is_npu=True) as (cid, vid):
            bx = cid % chunk_num
            by = (cid // chunk_num) % (H // VEC_NUM) * 2 + vid
            bz = (cid // chunk_num) // (H // VEC_NUM)

            g_ub = T.alloc_ub(
                [
                    C * CC,
                ],
                accum_dtype,
            )
            s_ub = T.alloc_ub(
                [
                    C * CC,
                ],
                accum_dtype,
            )  # Process CC chunks at a time

            with T.Scope("V"):
                T.tile.fill(s_ub, 0.0)
                T.copy(G[bz, by, bx * C * CC], g_ub)
                T.set_flag("mte2", "v", 0)
                T.wait_flag("mte2", "v", 0)
                for ii in range(CC):  # For each chunk
                    ofs = ii * C

                    T.set_flag("v", "s", 0)
                    T.wait_flag("v", "s", 0)

                    s_ub[ofs + 0] = g_ub[ofs + 0]
                    for i in range(1, C):
                        tmp2 = s_ub[ofs + i - 1] + g_ub[ofs + i]
                        s_ub[ofs + i] = tmp2  # Calculate prefix sum
                        # Must use variable tmp2 due to some compiler issue
                T.set_flag("v", "mte3", 0)
                T.wait_flag("v", "mte3", 0)
                T.copy(s_ub, S[bz, by, bx * C * CC])

    return main


def chunk_cumsum(g, C):
    B, H, L = g.shape
    ker = cumsum_ker(B, H, L, C)
    g_sum = ker(g)
    return g_sum


def ref_chunk_cumsum(g, C):
    B, H, L = g.shape
    chunk_num = (L + C - 1) // C
    g = g.view(B, H, chunk_num, C)
    g_sum = torch.cumsum(g, dim=-1)
    g_sum = g_sum.view(B, H, L)
    return g_sum


def _run_original_cli():
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

    A2A3CythonKernelCache().clear_cache()
    torch.manual_seed(0)
    torch.set_printoptions(threshold=float("inf"), sci_mode=True)

    test_configs = [
        (2, 16, 16384, 128),
    ]

    for B, H, L, C in test_configs:  # Ensure that L % (C * CC) = 0
        print(f"Testing cumsum with B={B}, H={H}, L={L}, C={C}")
        g = torch.randn((B, H, L)).npu().to(torch.float)
        g_sum = chunk_cumsum(g, C)
        ref_g_sum = ref_chunk_cumsum(g, C)
        torch.testing.assert_close(g_sum.cpu(), ref_g_sum.cpu(), rtol=1e-5, atol=1e-5)
        print("Test passed!")

    print("Kernel Output Match!")


def main():
    if not os.environ.get("TILELANG_CACHE_DIR"):
        raise RuntimeError("Set a dedicated TILELANG_CACHE_DIR before running this cache-clearing example.")
    from tilelang import env
    from tilelang.env import CacheState

    if Path(env.TILELANG_CACHE_DIR).resolve() != Path(os.environ["TILELANG_CACHE_DIR"]).resolve():
        raise RuntimeError("The resolved cache differs from the explicit TILELANG_CACHE_DIR; remove the caller cache override first.")
    import torch_npu  # noqa: F401
    from torch._tensor_str import PRINT_OPTS
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache
    from tilelang.cache.kernel_cache import KernelCache

    old_enabled = CacheState.is_enabled()
    old_staging_root = KernelCache._last_cleaned_staging_root
    cache = A2A3CythonKernelCache()
    old_memory = cache._memory_cache
    old_print_options = vars(PRINT_OPTS).copy()
    cache._memory_cache = {}
    try:
        _run_original_cli()
    finally:
        cache._memory_cache = old_memory
        KernelCache._last_cleaned_staging_root = old_staging_root
        CacheState._enabled = old_enabled
        torch.set_printoptions(**old_print_options)


if __name__ == "__main__":
    main()
