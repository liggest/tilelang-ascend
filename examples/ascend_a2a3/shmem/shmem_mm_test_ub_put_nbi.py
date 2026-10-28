import argparse
import importlib.util
import multiprocessing as mp
from multiprocessing import Barrier
import os
from pathlib import Path

import tilelang
import tilelang.ascend.a2a3.language as T
import torch

M = 1
N = 352
num_processes = 8
g_ash_size = 1024 * 1024 * 1024
G_IP_PORT = "tcp://100.102.180.145:8666"

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


@tilelang.jit(pass_configs=pass_configs)
def shmem_ub_put_nbi(M, N, nelems, newPe, dtype="int8"):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub_tensor = T.alloc_ub((1, nelems), dtype)
            with T.Scope("V"):
                if vid == 0:
                    T.copy(A, ub_tensor)
                    # Copy from the local UB to the newPe GM
                    T.shmem_ub_put_nbi(ub_tensor, B, nelems, newPe)

    return main


def worker(rank, barrier):
    import torch_npu  # noqa: F401
    import shmem as aclshmem_module

    print(f"Rank {rank}: Setting device")
    torch.npu.set_device(rank)
    ret = aclshmem_module.set_conf_store_tls(False, "")
    if ret != 0:
        raise ValueError("[ERROR] set_conf_store_tls failed")
    # Create initialization attribute object
    attributes = aclshmem_module.InitAttr()
    npu_num = 8
    attributes.my_rank = rank
    attributes.n_ranks = npu_num
    attributes.local_mem_size = g_ash_size
    attributes.ip_port = G_IP_PORT
    attributes.option_attr.data_op_engine_type = aclshmem_module.OpEngineType.MTE
    # Initialize aclshmem
    ret = aclshmem_module.aclshmem_init(attributes)
    if ret == 0:
        print(f"Rank {rank}: Initialization successful")
        torch.manual_seed(0)
        # Create shared memory tensor
        tensor = aclshmem_module.aclshmem_create_tensor([M, 2 * N], dtype=torch.int8, device_id=rank)
        a = tensor[0:1, 0:N].fill_(2)
        b = tensor[0:1, N : 2 * N].fill_(0)
        torch.npu.synchronize()
        nelems = M * N
        # Put the data from this rank onto another rank; here it is set to the next rank.
        newPe = (rank + 1) % npu_num

        func = shmem_ub_put_nbi(M, N, nelems, newPe)
        func(a, b)
        barrier.wait()
        if torch.equal(a, b):
            print(f"Rank {rank}: Test passed!")
        else:
            print("[ERROR] Kernel Output Not Match!")
        aclshmem_module.aclshmem_free_tensor(tensor)

    else:
        print(f"Rank {rank}: Initialization failed with code {ret}")
    # Clean
    aclshmem_module.aclshmem_finialize()
    print(f"Rank {rank}: Finalized")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
    parser.add_argument("--m", type=int, default=1, help="Matrix M dimension")
    parser.add_argument("--n", type=int, default=352, help="Matrix N dimension")
    parser.add_argument("--num_processes", type=int, default=8, help="number of processes")
    return parser.parse_args(argv)


def _run_original():
    print(f"Number of processes: {num_processes}")
    # Synchronize a specified number of processes
    barrier = Barrier(num_processes)
    processes = []

    for rank in range(num_processes):
        p = mp.Process(target=worker, args=(rank, barrier))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Kernel Output Match!")


def main(argv=None):
    args = parse_args(argv)
    requested = os.environ.get("TILELANG_CACHE_DIR")
    if not requested:
        raise RuntimeError("Set a dedicated TILELANG_CACHE_DIR before running this cache-clearing example")
    if Path(tilelang.env.TILELANG_CACHE_DIR).resolve() != Path(requested).resolve():
        raise RuntimeError("TILELANG_CACHE_DIR conflicts with the active cache-directory override; execution was refused")
    if importlib.util.find_spec("shmem") is None:
        raise ModuleNotFoundError(
            "Install the pinned shmem Python extension from 3rdparty/shmem/src/python before running these multi-NPU workers"
        )
    import torch_npu  # noqa: F401
    import shmem  # noqa: F401 - validate the real extension before clearing/spawning
    from tilelang.env import CacheState
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache
    from tilelang.cache.kernel_cache import KernelCache

    global M, N, num_processes
    caller_globals = M, N, num_processes
    caller_raw = CacheState._enabled
    caller_staging = KernelCache._last_cleaned_staging_root
    cache = A2A3CythonKernelCache()
    caller_memory = cache._memory_cache
    cache._memory_cache = {}
    try:
        M, N, num_processes = args.m, args.n, args.num_processes
        cache.clear_cache()
        _run_original()
    finally:
        cache._memory_cache = caller_memory
        KernelCache._last_cleaned_staging_root = caller_staging
        CacheState._enabled = caller_raw
        M, N, num_processes = caller_globals


if __name__ == "__main__":
    main()
