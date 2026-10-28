"""Original TA AOT GEMM, using the frozen A2/A3 compiler protocol."""

import argparse
import os
from pathlib import Path
import shutil


def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    import tilelang.ascend.a2a3.language as T

    m_num = M // block_M
    n_num = N // block_N

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
                    T.copy(A[bx * block_M : (bx + 1) * block_M, k * K_L1 : (k + 1) * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)

                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

                    T.barrier_all()

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


def _argument_parser():
    parser = argparse.ArgumentParser(description="NPU Kernel AOT Compilation")
    parser.add_argument("--m", type=int, default=8192, help="Matrix M dimension")
    parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
    parser.add_argument("--k", type=int, default=8192, help="Matrix K dimension")
    parser.add_argument(
        "--target",
        type=str,
        default="ascendc",
        choices=["ascendc", "pto"],
        help="Codegen/compile backend",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="auto",
        help="Hardware platform (auto/A2/A3; retained A5 is outside this namespace)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./kernel_lib.so",
        help="Output shared library path",
    )
    return parser


def build(args):
    import tilelang
    from tilelang.ascend.a2a3.compiler import LibraryGenerator, resolve_compile_config
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache
    from tilelang.backend import create_backend_context

    platform = str(args.platform).strip().upper()
    if platform == "A5":
        raise ValueError("Retained original A5 AOT selection is outside the A2/A3 namespace")
    if platform not in {"AUTO", "A2", "A3"}:
        raise ValueError("A2/A3 AOT platform must be auto, A2, or A3")
    if not os.environ.get("TILELANG_CACHE_DIR"):
        raise RuntimeError("Set a dedicated TILELANG_CACHE_DIR before running this cache-clearing CLI")
    if Path(tilelang.env.TILELANG_CACHE_DIR).resolve() != Path(os.environ["TILELANG_CACHE_DIR"]).resolve():
        raise RuntimeError("TILELANG_CACHE_DIR conflicts with the active cache-directory override; clear was refused")
    A2A3CythonKernelCache().clear_cache()

    M = args.m
    N = args.n
    K = args.k
    func = matmul(M, N, K, 128, 256, 64)
    requested = {"kind": "ascend", "arch": "dav-2201", "keys": [args.target, "ascend"]}
    if platform != "AUTO":
        requested["model"] = platform
    context = create_backend_context(requested, execution_backend="cython")
    # This is the original raw AOT lower/driver path, with no public-JIT
    # pass-config normalization. In particular, bare PTO flags still use O2.
    config = resolve_compile_config(context.target)
    artifact = tilelang.lower(func, target=context.target)
    lib_generator = LibraryGenerator(config)
    lib_generator.update_lib_code(artifact.kernel_source)
    lib_generator.compile_lib()
    shutil.copy(lib_generator.get_lib_path(), args.output)
    print(f"Built {args.output} (target={args.target}, platform={config.platform})")


def main(argv=None):
    args = _argument_parser().parse_args(argv)
    build(args)


if __name__ == "__main__":
    main()
