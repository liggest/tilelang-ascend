"""Preserved TA Ascend Carver recommendation workflow, pending evaluation.

Source: 0deb8b93da11b7f0becbcded732d9633b7ca4e34:examples/autotune/example_gemm_carver.py
Source SHA256: 7626ce7a9b7b5e7fdceb8c46671c4c79083aa3b668ee32634ca5bf2a045292f3
The kernel and hint-to-config mapping are retained. No static configuration
stands in for the unavailable Ascend Carver architecture/policies.
"""

import argparse

import tilelang
import tilelang.ascend.a2a3.language as T
import torch


tilelang.disable_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=1024, help="Matrix K dimension")
args = parser.parse_args()

M = args.m
N = args.n
K = args.k

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def get_config(M, N, K) -> list[dict]:
    raise NotImplementedError(
        "A2/A3 Ascend Carver architecture and policy migration is deferred for "
        "later evaluation. Legacy source files do not enable this capability. "
        "Use example_gemm_autotune.py for the migrated shared autotuner."
    )


def _original_recommend_configs(M, N, K) -> list[dict]:
    """Retain the original recommendation flow as inactive porting source."""
    from tilelang import carver
    from tilelang.carver.arch.ascend import Ascend

    arch = Ascend()
    carver_template = carver.MatmulTemplate(
        M=M,
        N=N,
        K=K,
        in_dtype="float16",
        accum_dtype="float16",
        out_dtype="float16",
    ).with_arch(arch)

    hints = carver_template.recommend_hints(topk=20)
    configs = []
    for hint in hints:
        config = {
            "block_M": hint.block[0],
            "block_N": hint.block[1],
            "K_L1": hint.rstep[0],
        }
        configs.append(config)

    return configs


def ref_prog(A, B):
    return A @ B


def supply_prog(params):
    M, K = (int(dim) for dim in params[0].shape)
    _, N = (int(dim) for dim in params[1].shape)
    torch.manual_seed(0)
    return [torch.randn(M, K).half().npu(), torch.randn(K, N).half().npu()]


def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
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

            A_L1 = T.alloc_shared((block_M, K_L1), dtype)
            B_L1 = T.alloc_shared((K_L1, block_N), dtype)

            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            loop_k = T.ceildiv(K, K_L1)
            for k in T.serial(loop_k):
                T.copy(A[bx * block_M, k * K_L1], A_L1)
                T.copy(B[k * K_L1, by * block_N], B_L1)

                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

            T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


def make_matmul(M, N, K):
    # Keep genuine recommend_hints output as the sole search-space producer.
    configs = get_config(M, N, K)
    return tilelang.autotune(configs=configs, ref_prog=ref_prog, supply_prog=supply_prog, atol=1e-2, rtol=1e-2)(
        tilelang.jit(out_idx=[-1], pass_configs=pass_configs)(matmul)
    )


# Ascend Carver migration remains explicitly deferred by the agreed scope.
tuned = make_matmul(M, N, K)
func = tuned(M, N, K)
print("Best Config:", func.get_tuner_result())
print("Test passed!")
