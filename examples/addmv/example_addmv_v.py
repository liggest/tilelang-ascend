import argparse
import torch

import tilelang as tl
import tilelang.language as T

tl.cache.clear_cache()


@tl.jit(
    out_idx=[-1],
    pass_configs={
        tl.PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    }
)
def addmv_v(
    N: int,
    K: int,
    block_N: int,
    block_K: int,
    dtype: str = "float16",
    accum_dtype: str = "float32"
):
    """Vector core addmv: y = A @ x + y"""
    VEC_NUM = 2
    TEMP_DTYPE = "uint8"
    CAST_MODE = "CAST_NONE"

    n_num = T.ceildiv(N, block_N)
    k_num = T.ceildiv(K, block_K)
    kernel_num = T.ceildiv(n_num, VEC_NUM)

    not_same_dtype = dtype != accum_dtype

    def cast_or_copy(dst, src, mode, count):
        if not_same_dtype:
            return T.tile.cast(dst, src, mode, count)
        else:
            return T.copy(src, dst)

    @T.prim_func
    def main(
        x: T.Tensor((K,), dtype),
        A: T.Tensor((N, K), dtype),
        y: T.Tensor((N,), dtype),
    ):
        with T.Kernel(kernel_num, is_npu=True) as (cid, vid):
            bn = (cid * VEC_NUM + vid) % n_num

            x_ub = T.alloc_ub((1, block_K), dtype)
            x_32_ub = T.alloc_ub((1, block_K), accum_dtype)
            A_ub = T.alloc_ub((block_N, block_K), dtype)
            temp_ub = T.alloc_ub((block_N, block_K), TEMP_DTYPE)
            A_32_ub = T.alloc_ub((block_N, block_K), accum_dtype)
            y_single_32_ub = T.alloc_ub((block_N,), accum_dtype)
            y_total_32_ub = T.alloc_ub((block_N,), accum_dtype)
            y_orig_ub = T.alloc_ub((block_N,), dtype)
            y_orig_32_ub = T.alloc_ub((block_N,), accum_dtype)
            y_ub = T.alloc_ub((block_N,), dtype)

            # Initialize accumulator to 0
            T.tile.fill(y_total_32_ub, 0.0)

            # Load and cast original y as accumulator
            T.copy(y[bn * block_N], y_orig_ub)
            cast_or_copy(y_orig_32_ub, y_orig_ub, CAST_MODE, block_N)

            # GEMV: compute A @ x
            for bk in T.serial(k_num):
                T.copy(x[bk * block_K], x_ub)
                T.copy(A[bn * block_N, bk * block_K], A_ub)
                cast_or_copy(x_32_ub, x_ub, CAST_MODE, block_K)
                cast_or_copy(A_32_ub, A_ub, CAST_MODE, block_N * block_K)

                # A_32_ub[i, :] = A_32_ub[i, :] * x_32_ub
                for i in T.serial(block_N):
                    T.tile.mul(A_32_ub[i, :], A_32_ub[i, :], x_32_ub)

                # Reduce sum across K dimension
                T.reduce_sum(A_32_ub, y_single_32_ub, temp_ub, dim=-1)

                # Accumulate: y_total = y_total + A @ x
                T.tile.add(y_total_32_ub, y_single_32_ub, y_total_32_ub)

            # Add original y: y_total = y_total + y_orig
            T.tile.add(y_total_32_ub, y_orig_32_ub, y_total_32_ub)

            # Cast back and write
            cast_or_copy(y_ub, y_total_32_ub, CAST_MODE, block_N)
            T.copy(y_ub, y[bn * block_N])

    return main


def ref_program(x, A, y):
    return y + (A @ x)


def check_case(N: int, K: int, block_N: int = 64, block_K: int = 128, dtype: str = "float16"):
    torch_dtype_map = {
        "float16": torch.half,
        "float32": torch.float32,
        "float": torch.float32
    }

    x = torch.randn(K).to(torch_dtype_map[dtype]).npu()
    A = torch.randn(N, K).to(torch_dtype_map[dtype]).npu()
    y_orig = torch.randn(N).to(torch_dtype_map[dtype]).npu()

    kernel = addmv_v(N, K, block_N, block_K, dtype=dtype)

    y_result = kernel(x, A, y_orig)
    ref_y = ref_program(x, A, y_orig)

    torch.testing.assert_close(y_result, ref_y, rtol=1e-2, atol=1e-2)


def main(custom_args=None):
    parser = argparse.ArgumentParser(description="addmv Example (Vector Core)")
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    args, remains = parser.parse_known_args(custom_args)
    N, K = args.n, args.k

    torch.manual_seed(0)

    check_case(N, K, 128, 128)
    print("addmv (Vector Core) example passed!")
    print("Kernel Output Match!")


if __name__ == "__main__":
    main()
