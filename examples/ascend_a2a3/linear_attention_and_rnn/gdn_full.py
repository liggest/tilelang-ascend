import importlib.util
import os
from pathlib import Path

import tilelang  # noqa: F401 - retain original module dependency
import torch
import torch.nn.functional as F


def _load_local_gdn_leaf(name):
    path = Path(__file__).resolve().with_name("gdn") / (name + ".py")
    spec = importlib.util.spec_from_file_location("_a2a3_gdn_full_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_gdn_leaves = {
    "gdn_chunk_cumsum": _load_local_gdn_leaf("gdn_chunk_cumsum"),
    "gdn_chunk_h": _load_local_gdn_leaf("gdn_chunk_h"),
    "gdn_chunk_o": _load_local_gdn_leaf("gdn_chunk_o"),
    "gdn_chunk_scaled_dot_kkt": _load_local_gdn_leaf("gdn_chunk_scaled_dot_kkt"),
    "gdn_solve_tril": _load_local_gdn_leaf("gdn_solve_tril"),
    "gdn_wy_fast": _load_local_gdn_leaf("gdn_wy_fast"),
}
chunk_cumsum = _gdn_leaves["gdn_chunk_cumsum"].chunk_cumsum
ref_chunk_cumsum = _gdn_leaves["gdn_chunk_cumsum"].ref_chunk_cumsum
chunk_h = _gdn_leaves["gdn_chunk_h"].chunk_h
ref_chunk_h = _gdn_leaves["gdn_chunk_h"].ref_chunk_h
chunk_o = _gdn_leaves["gdn_chunk_o"].chunk_o
ref_chunk_o = _gdn_leaves["gdn_chunk_o"].ref_chunk_o
kkt = _gdn_leaves["gdn_chunk_scaled_dot_kkt"].kkt
ref_kkt = _gdn_leaves["gdn_chunk_scaled_dot_kkt"].ref_kkt
solve_tril = _gdn_leaves["gdn_solve_tril"].solve_tril
ref_solve_tril = _gdn_leaves["gdn_solve_tril"].ref_solve_tril
wy_fast = _gdn_leaves["gdn_wy_fast"].wy_fast
ref_wy_fast = _gdn_leaves["gdn_wy_fast"].ref_wy_fast


def ref_seq_gdn(q, k, v, g, beta):
    g = torch.exp(g)
    q = q.float()
    k = k.float()
    v = v.float()
    beta = beta.float()
    Batch, H, L, DK = q.shape
    DV = v.shape[-1]
    S = torch.zeros((Batch, H, DV, DK)).npu().to(torch.float)
    o = torch.empty((Batch, H, L, DV)).npu().to(torch.float)
    I = torch.eye(DK).npu().to(torch.float).view(1, 1, DK, DK)
    for i in range(0, L):
        q_i = q[:, :, i, :]
        k_i = k[:, :, i, :]
        v_i = v[:, :, i, :]
        beta_i = beta[:, :, i].view(Batch, H, 1, 1)
        g_i = g[:, :, i].view(Batch, H, 1, 1)
        kkt = k_i.unsqueeze(-1) * k_i.unsqueeze(-2)
        vkt = v_i.unsqueeze(-1) * k_i.unsqueeze(-2)
        A_i = g_i * (I - beta_i * kkt)
        term_1 = torch.matmul(S, A_i)
        term_2 = beta_i * vkt
        S = term_1 + term_2
        o[:, :, i, :] = torch.einsum("bhpq,bhq->bhp", S, q_i)
    return o.to(torch.float16)


def ref_chunk_gdn(q, k, v, g, beta, C):
    g = ref_chunk_cumsum(g, C)
    a = ref_kkt(k, beta, g, C)
    a = ref_solve_tril(a)
    w, u = ref_wy_fast(k, v, beta, g, a, C)
    s, nv, fs = ref_chunk_h(k, w, u, g, C)
    o = ref_chunk_o(q, k, nv, s, g, C)
    return o


def kernel_chunk_gdn(q, k, v, g, beta, C, BK, BV):
    g = chunk_cumsum(g, C)
    a = kkt(k, beta, g, C, BK)
    a = solve_tril(a)
    w, u = wy_fast(k, v, beta, g, a, C, BK, BV)
    s, nv, fs = chunk_h(k, w, u, g, C, BK, BV)
    o = chunk_o(q, k, nv, s, g, C, BK, BV)
    return o


def _run_original_cli():
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

    A2A3CythonKernelCache().clear_cache()
    torch.manual_seed(0)
    torch.set_printoptions(threshold=float("inf"), sci_mode=True)

    test_configs = [
        (2, 32, 1024, 64, 64, 32, 32, 32),
    ]

    for Batch, H, L, DK, DV, C, BK, BV in test_configs:
        print(f"Testing GDN with Batch={Batch}, H={H}, L={L}, DK={DK}, DV={DV}, C={C}, BK={BK}, BV={BV}")
        q = torch.randn((Batch, H, L, DK)).npu().to(torch.float16)
        k = torch.randn((Batch, H, L, DK)).npu().to(torch.float16)
        v = torch.randn((Batch, H, L, DV)).npu().to(torch.float16)
        q, k = F.normalize(q, dim=-1, p=2), F.normalize(k, dim=-1, p=2)
        g = torch.randn((Batch, H, L)).npu().to(torch.float)
        g = F.logsigmoid(g)
        beta = torch.rand((Batch, H, L)).npu().to(torch.float16)
        ref_o = ref_seq_gdn(q, k, v, g, beta)
        ref_chunk_o = ref_chunk_gdn(q, k, v, g, beta, C)
        ker_o = kernel_chunk_gdn(q, k, v, g, beta, C, BK, BV)
        torch.testing.assert_close(ref_chunk_o.cpu(), ref_o.cpu(), rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(ker_o.cpu(), ref_chunk_o.cpu(), rtol=1e-3, atol=1e-3)
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
