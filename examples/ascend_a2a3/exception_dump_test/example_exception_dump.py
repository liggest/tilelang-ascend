"""Original exception-dump CLI; importing does not initialize an NPU runtime."""

import argparse
from functools import lru_cache
import os
from pathlib import Path


@lru_cache(maxsize=1)
def _make_vec_add():
    import tilelang
    import tilelang.ascend.a2a3.language as T

    @tilelang.jit(
        out_idx=[2],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
            tilelang.PassConfigKey.TL_ASCEND_EXCEPTION_DUMP: True,
        },
    )
    def vec_add(M, N, dtype="float16"):
        @T.prim_func
        def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                a_ub = T.alloc_ub((M, N), dtype)
                b_ub = T.alloc_ub((M, N), dtype)
                c_ub = T.alloc_ub((M, N), dtype)
                T.copy(A[:, :], a_ub)
                T.copy(B[:, :], b_ub)
                for i, j in T.Parallel(M, N):
                    c_ub[i, j] = a_ub[i, j] + b_ub[i, j]
                T.copy(c_ub[:, :], C[:, :])

        return main

    return vec_add


def vec_add(M, N, dtype="float16"):
    return _make_vec_add()(M, N, dtype)


def launch_kernel_with_hw_exception_triggered(kernel, x, y):
    """Trigger an AI Core hardware exception by launching the kernel with a
    null output pointer.

    This forces the NPU to attempt writing to address 0x0 — an unmapped
    address — which reliably triggers an MTE (Memory Transfer Engine) error
    and invokes the exception dump callback.

    Raises
    ------
    RuntimeError
        Propagated from stream.synchronize() when the hardware exception fires.
    """
    import ctypes
    import torch

    so_path = kernel.adapter.libpath
    lib = ctypes.CDLL(so_path)
    call_fn = lib.call
    call_fn.restype = None
    call_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]

    z_null = ctypes.c_void_p(0)
    exc_stream = torch.npu.Stream()
    print(f"  Launching on stream {exc_stream.npu_stream}")
    call_fn(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(y.data_ptr()),
        z_null,
        ctypes.c_void_p(exc_stream.npu_stream),
    )
    exc_stream.synchronize()
    print("  Stream sync completed without error (unexpected)")


def main(argv=None):
    parser = argparse.ArgumentParser(description="TileLang Exception Dump Example")
    parser.add_argument("--m", type=int, default=128, help="Matrix M dimension")
    parser.add_argument("--n", type=int, default=128, help="Matrix N dimension")
    parser.add_argument(
        "--dump-dir",
        type=str,
        default="/tmp/tilelang_exc_dump",
        help="Directory for CANN exception dump files (ASCEND_DUMP_PATH)",
    )
    args = parser.parse_args(argv)
    if not os.environ.get("TILELANG_CACHE_DIR"):
        raise RuntimeError("Set a dedicated TILELANG_CACHE_DIR before running this cache-clearing CLI")

    # CANN reads these during runtime initialization. Keep both assignments
    # before any torch/torch_npu/tilelang import on the explicit CLI path.
    os.environ["ASCEND_DUMP_PATH"] = args.dump_dir
    os.environ["ASCEND_DUMP_SCENE"] = "aic_err_brief_dump"
    os.makedirs(args.dump_dir, exist_ok=True)

    import tilelang
    import torch
    import torch_npu  # noqa: F401
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache
    from tilelang.ascend.a2a3.tools.parse_exception_dump import parse_exception_dump

    if Path(tilelang.env.TILELANG_CACHE_DIR).resolve() != Path(os.environ["TILELANG_CACHE_DIR"]).resolve():
        raise RuntimeError("TILELANG_CACHE_DIR conflicts with the active cache-directory override; clear was refused")
    A2A3CythonKernelCache().clear_cache()
    M = args.m
    N = args.n

    print("Compiling kernel...")
    func = vec_add(M, N)

    a = torch.randn(M, N, dtype=torch.float16).npu()
    b = torch.randn(M, N, dtype=torch.float16).npu()

    print("Normal execution...")
    c = func(a, b)
    ref_c = a + b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Kernel Output Match!")

    print("\n--- Triggering AI Core exception (null output pointer) ---")
    try:
        launch_kernel_with_hw_exception_triggered(func, a, b)
    except Exception as e:
        print(f"  Exception caught: {type(e).__name__}")
        tensors = parse_exception_dump(args.dump_dir, kernel_name="main_kernel")
        print(f"  Dump file parsed, {len(tensors)} tensor(s) recovered:")
        for t in tensors:
            data = t["data"].reshape(M, N)
            print(f"    {t['type']}[{t['index']}] dtype={t['dtype']}, shape={data.shape}, min={data.min():.4f}, max={data.max():.4f}")

    print("--- Exception test done ---")


if __name__ == "__main__":
    main()
