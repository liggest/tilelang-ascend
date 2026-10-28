import importlib.util
import os
from pathlib import Path

import tilelang  # noqa: F401 - retain original module dependency
import torch
import torch.nn.functional as F


def _load_local_opt_gdn_leaf(name):
    path = Path(__file__).resolve().with_name("opt_gdn") / (name + ".py")
    spec = importlib.util.spec_from_file_location("_a2a3_opt_gdn_full_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_opt_gdn_leaves = {
    "opt_gdn_chunk_cumsum": _load_local_opt_gdn_leaf("opt_gdn_chunk_cumsum"),
    "opt_gdn_chunk_h": _load_local_opt_gdn_leaf("opt_gdn_chunk_h"),
    "opt_gdn_chunk_o": _load_local_opt_gdn_leaf("opt_gdn_chunk_o"),
    "opt_gdn_chunk_scaled_dot_kkt": _load_local_opt_gdn_leaf("opt_gdn_chunk_scaled_dot_kkt"),
    "opt_gdn_solve_tril": _load_local_opt_gdn_leaf("opt_gdn_solve_tril"),
    "opt_gdn_wy_fast": _load_local_opt_gdn_leaf("opt_gdn_wy_fast"),
}
cumsum_ker = _opt_gdn_leaves["opt_gdn_chunk_cumsum"].cumsum_ker
chunk_h_ker = _opt_gdn_leaves["opt_gdn_chunk_h"].chunk_h_ker
chunk_o_ker = _opt_gdn_leaves["opt_gdn_chunk_o"].chunk_o_ker
kkt_ker = _opt_gdn_leaves["opt_gdn_chunk_scaled_dot_kkt"].kkt_ker
solve_tril_ker = _opt_gdn_leaves["opt_gdn_solve_tril"].solve_tril_ker
solve_tril_64_ker = _opt_gdn_leaves["opt_gdn_solve_tril"].solve_tril_64_ker
solve_tril_128_ker = _opt_gdn_leaves["opt_gdn_solve_tril"].solve_tril_128_ker
wy_fast_ker = _opt_gdn_leaves["opt_gdn_wy_fast"].wy_fast_ker


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


def _run_original_cli():
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

    A2A3CythonKernelCache().clear_cache()
    torch.manual_seed(0)
    torch.set_printoptions(threshold=float("inf"), sci_mode=True)

    test_configs = [
        (1, 2, 1024, 128, 128, 128, 128, 128),
    ]

    for B, H, L, DK, DV, C, BK, BV in test_configs:
        print(f"Testing GDN with B={B}, H={H}, L={L}, DK={DK}, DV={DV}, C={C}, BK={BK}, BV={BV}")
        q = torch.randn((B, H, L, DK)).npu().to(torch.float16)
        k = torch.randn((B, H, L, DK)).npu().to(torch.float16)
        v = torch.randn((B, H, L, DV)).npu().to(torch.float16)
        q, k = F.normalize(q, dim=-1, p=2), F.normalize(k, dim=-1, p=2)
        g = torch.randn((B, H, L)).npu().to(torch.float)
        g = F.logsigmoid(g)
        beta = torch.rand((B, H, L)).npu().to(torch.float16)

        ker1 = cumsum_ker(B, H, L, C)
        ker2 = kkt_ker(B, H, L, DK, C, BK)
        if C == 32:
            ker3 = solve_tril_ker(B, H, L, C)
        elif C == 64:
            ker3 = solve_tril_64_ker(B, H, L)
        elif C == 128:
            ker3 = solve_tril_128_ker(B, H, L)
        ker4 = wy_fast_ker(B, H, L, DK, DV, C, BK, BV)
        ker5 = chunk_h_ker(B, H, L, DK, DV, C, BK, BV)
        ker6 = chunk_o_ker(B, H, L, DK, DV, C, BK, BV)

        idt = torch.eye(C).npu().to(torch.float)
        msk1 = torch.tril(torch.ones((C, C)), diagonal=-1).npu().to(torch.float)
        msk2 = torch.tril(torch.ones((C, C)), diagonal=0).npu().to(torch.float)
        workspace = torch.zeros((B * H * ((DV + BV - 1) // BV), DK, BV)).npu().to(torch.float16)
        s = torch.zeros((B, H, (L + C - 1) // C, DK, DV)).npu().to(torch.float16)

        g_sum = ker1(g)
        a = ker2(k, beta, g_sum, msk1)
        a = ker3(a, idt)
        w, u = ker4(k, v, beta, g_sum, a)
        nv, fs = ker5(k, w, u, g_sum, workspace, s)
        o = ker6(q, k, nv, s, g_sum, msk2)
        ref_o = ref_seq_gdn(q, k, v, g, beta)
        torch.testing.assert_close(o.cpu(), ref_o.cpu(), rtol=1e-3, atol=1e-3)
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
