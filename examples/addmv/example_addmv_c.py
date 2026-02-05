import argparse
import torch

import tilelang as tl
import tilelang.language as T

tl.cache.clear_cache()


@tl.jit(
    out_idx=[3],
    workspace_idx=[4],
    pass_configs={
        tl.PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    }
)
def addmv_c(
    N: int,
    K: int,
    block_N: int,
    block_K: int,
    dtype: str = "float16",
    accum_dtype: str = "float32"
):
    """Cube core addmv: y = A @ x + y"""
    FRACTAL_SIZE = 16
    n_num = T.ceildiv(N, block_N)
    k_num = T.ceildiv(K, block_K)

    @T.prim_func
    def main(
        x: T.Tensor((K,), dtype),
        A: T.Tensor((N, K), dtype),
        y: T.Tensor((N,), dtype),
        y_out: T.Tensor((N,), dtype),
        workspace: T.Tensor((N,), dtype),
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            bn = cid % n_num

            A_L1 = T.alloc_L1((block_N, block_K), dtype)
            x_L1 = T.alloc_L1((FRACTAL_SIZE, block_K), dtype)
            C_L0 = T.alloc_L0C((FRACTAL_SIZE, block_N), accum_dtype)
            y_ub = T.alloc_ub((block_N,), dtype)
            y_orig_ub = T.alloc_ub((block_N,), dtype)

            with T.Scope("C"):
                # Calculate A @ x
                for bk in T.serial(k_num):
                    T.copy(x[bk * block_K], x_L1)
                    T.copy(A[bn * block_N, bk * block_K], A_L1)
                    T.gemm_v0(x_L1, A_L1, C_L0, transpose_B=True, init=(bk == 0))

                T.copy(C_L0, workspace[bn * block_N])

            with T.Scope("V"):
                T.copy(workspace[bn * block_N], y_ub)
                T.copy(y[bn * block_N], y_orig_ub)

                # y_out = y_orig + (A @ x)
                T.tile.add(y_ub, y_orig_ub, y_ub)

                T.copy(y_ub, y_out[bn * block_N])

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

    kernel = addmv_c(N, K, block_N, block_K, dtype=dtype)

    # With workspace_idx, workspace and y_out are auto-allocated
    y_result = kernel(x, A, y_orig)
    ref_y = ref_program(x, A, y_orig)

    torch.testing.assert_close(y_result, ref_y, rtol=1e-2, atol=1e-2)


def main(custom_args=None):
    parser = argparse.ArgumentParser(description="addmv Example (Cube Core)")
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    args, remains = parser.parse_known_args(custom_args)
    N, K = args.n, args.k

    torch.manual_seed(0)

    check_case(N, K, 128, 128)
    print("addmv (Cube Core) example passed!")
    print("Kernel Output Match!")


if __name__ == "__main__":
    main()
