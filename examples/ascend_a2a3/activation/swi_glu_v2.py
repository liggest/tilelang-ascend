"""Original TA activation; runtime effects execute only through main()."""

import os
from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache
import tilelang
import tilelang.ascend.a2a3.language as T
import torch

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    # tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    # tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def swi_glu(block_M, block_N, split_dim, dtype="bfloat16"):
    M = T.symbolic("M")
    N = T.symbolic("N")

    block_M = min(block_M, 64)
    block_N = min(block_N, 64)

    need_cast = dtype not in ("float", "float32")
    ACC_DTYPE = "float32"

    m_div = 1
    n_div = 2
    if split_dim == 0 or split_dim == -2:
        m_div = 2
        n_div = 1

    m_num = T.ceildiv(M // m_div, block_M)
    n_num = T.ceildiv(N // n_div, block_N)

    num_blocks = m_num * n_num

    VEC_NUM = 2
    rows_per_vec = block_M // VEC_NUM
    elem_num = rows_per_vec * block_N

    NUM_CORES = 48
    stages = 2
    num_iters = T.ceildiv(num_blocks, NUM_CORES)

    @T.prim_func
    def swiglu(A: T.Tensor((M, N), dtype), B: T.Tensor((M // m_div, N // n_div), dtype)):
        m_offset = 0
        n_offset = N // 2
        if split_dim == 0 or split_dim == -2:
            m_offset = M // 2
            n_offset = 0

        with T.Kernel(NUM_CORES, is_npu=True) as (cid, vid):
            # UB buffer
            a0_ub = T.alloc_ub((stages, rows_per_vec, block_N), ACC_DTYPE)
            a1_ub = T.alloc_ub((stages, rows_per_vec, block_N), ACC_DTYPE)
            b_ub = T.alloc_ub((stages, rows_per_vec, block_N), ACC_DTYPE)
            temp_ub = T.alloc_ub((rows_per_vec, block_N), ACC_DTYPE)
            T.tile.fill(temp_ub, 0.0)

            zero_ub = T.alloc_ub((rows_per_vec, block_N), ACC_DTYPE)
            T.tile.fill(zero_ub, 0.0)

            tmp_bf16_1 = T.alloc_ub((rows_per_vec, block_N), dtype)
            tmp_bf16_2 = T.alloc_ub((rows_per_vec, block_N), dtype)
            tmp_bf16_3 = T.alloc_ub((rows_per_vec, block_N), dtype)

            for i in T.serial(num_iters):
                cur = i % stages

                block_id = cid + i * NUM_CORES
                if block_id < num_blocks:
                    bx = block_id // n_num
                    by = block_id % n_num
                    row = bx * block_M + vid * rows_per_vec
                    col = by * block_N
                    row2 = row + m_offset
                    col2 = col + n_offset

                    if need_cast:
                        T.copy(A[row, col], tmp_bf16_1)
                        T.copy(A[row2, col2], tmp_bf16_2)
                        T.tile.cast(a0_ub[cur, :, :], tmp_bf16_1, "CAST_NONE", elem_num)
                        T.tile.cast(a1_ub[cur, :, :], tmp_bf16_2, "CAST_NONE", elem_num)
                    else:
                        T.copy(A[row, col], a0_ub[cur, :, :])
                        T.copy(A[row2, col2], a1_ub[cur, :, :])

                    # compute
                    T.tile.sub(temp_ub, zero_ub, a0_ub[cur, :, :])
                    T.tile.exp(temp_ub, temp_ub)
                    T.tile.add(temp_ub, temp_ub, 1.0)
                    T.tile.div(temp_ub, a0_ub[cur, :, :], temp_ub)
                    T.tile.mul(b_ub[cur, :, :], temp_ub, a1_ub[cur, :, :])

                    out_row = bx * block_M + vid * rows_per_vec
                    out_col = by * block_N

                    if need_cast:
                        T.tile.cast(tmp_bf16_3, b_ub[cur, :, :], "CAST_RINT", elem_num)
                        T.copy(tmp_bf16_3, B[out_row, out_col])
                    else:
                        T.copy(b_ub[cur, :, :], B[out_row, out_col])

    return swiglu


# Tests
test_configs = [
    (1024, 18432, 64, 64, 1, "bfloat16"),
    (1024, 18432, 64, 64, 1, "float"),
    (1024, 1024, 64, 64, 1, "bfloat16"),
    (1024, 1024, 64, 64, 1, "float"),
    (256, 256, 64, 64, 1, "bfloat16"),
    (256, 256, 64, 64, 1, "float"),
    (128, 128, 16, 16, 1, "float"),
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

    with torch.device("npu"):
        torch.manual_seed(0)
        for i in range(1):
            print(f"\nRun {i + 1}/10")
            for M, N, block_M, block_N, split_dim, dtype in test_configs:
                print(
                    f"Testing SwiGLU backward with M={M}, N={N}, block_M={block_M}, block_N={block_N}, split_dim={split_dim}, dtype={dtype}"
                )

                func = swi_glu(block_M, block_N, split_dim, dtype=dtype)
                # print(func.get_kernel_source())
                print("Init successful!")
                if dtype == "bfloat16":
                    a = torch.randn((M, N), dtype=torch.bfloat16)
                else:
                    a = torch.randn((M, N), dtype=torch.float32)
                b = func(a)
                torch.npu.synchronize()

                # ascendc ref
                ascendc_b = torch.ops.npu.npu_swiglu(a)
                torch.npu.synchronize()

                torch.testing.assert_close(b.cpu(), ascendc_b.cpu(), rtol=1e-2, atol=1e-2)
                print("Test passed!")

        print("Kernel Output Match!")


if __name__ == "__main__":
    main()
