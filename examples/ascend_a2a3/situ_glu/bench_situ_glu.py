"""SituGlu benchmark: TileLang kernel (flag-pipelined) vs the unfused torch baseline.

Baseline note: aclnnSituGlu ships with CANN 9.2.0 and is unavailable under
CANN 9.0.0; the kernel-only reference numbers come from the official AscendC
kernel built standalone from the ops-nn source (verbatim op_kernel +
learning-hub direct-launch harness + ported op_host tiling, timed with
device events -- the same protocol as the device columns below). The torch
column is the pre-fusion form this operator replaces. Full data and tuning
records: benchmark.md / benchmark_zh.md.

Timing columns (device time, aclrtEvent-equivalent protocol):
- symbolic: M/N stay symbolic (one compile serves any shape)
- static  : M/N as concrete ints (static specialization, lighter dispatch)
- unfused : chunk + tanh + sigmoid + mul as separate torch ops

Usage: python bench_situ_glu.py [--iters N]
"""

import argparse
import importlib.util
from pathlib import Path
import sys
import time

import torch


def _load_sibling(name):
    source = Path(__file__).with_name(name + ".py").resolve()
    key = "_a2a3_situ_glu_" + name
    existing = sys.modules.get(key)
    if existing is not None:
        if Path(existing.__file__).resolve() != source:
            raise ImportError(f"Conflicting sibling module: {key}")
        return existing
    spec = importlib.util.spec_from_file_location(key, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(key, None)
        raise
    return module


situ_glu = _load_sibling("situ_glu").situ_glu
_tests = _load_sibling("test_situ_glu")
SUB_M, TORCH_TO_TL_DTYPE = _tests.SUB_M, _tests.TORCH_TO_TL_DTYPE
_pick_blocks, golden_situ_glu, run_situ_glu = _tests._pick_blocks, _tests.golden_situ_glu, _tests.run_situ_glu


def torch_situ_glu_unfused(x):
    """Unfused baseline (defaults beta=1, linear_beta=0): y = tanh(g) * sigmoid(g) * up."""
    g, u = torch.chunk(x, 2, dim=-1)
    return torch.tanh(g) * torch.sigmoid(g) * u


def bench_wall(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / iters * 1e3  # ms


def bench_event(fn, iters, warmup=10, rounds=5):
    """Device-time benchmark (same event protocol as the official harness): median us/iter."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    results = []
    for _ in range(rounds):
        start = torch.npu.Event(enable_timing=True)
        stop = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        stop.record()
        torch.npu.synchronize()
        results.append(start.elapsed_time(stop) * 1e3 / iters)  # us
    results.sort()
    n = len(results)
    return results[n // 2] if n % 2 == 1 else 0.5 * (results[n // 2 - 1] + results[n // 2])


def correctness_spot_check():
    """One small end-to-end case vs the CPU golden.

    Doubles as the CI legacy-runner success marker: the runner requires the
    output to contain "Kernel Output Match" / "Test Passed!".
    """
    x = torch.randn(256, 256, dtype=torch.bfloat16)
    ref = golden_situ_glu(x)
    out = run_situ_glu(x.npu())
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)


def _original_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    torch.manual_seed(0)

    correctness_spot_check()

    cases = [
        ((1024, 2048), torch.bfloat16),
        ((4096, 18432), torch.bfloat16),
        ((8192, 18432), torch.bfloat16),
        ((4096, 18432), torch.float16),
        ((1024, 2048), torch.float32),
        ((4096, 18432), torch.float32),
    ]

    print(f"{'shape':>16} {'dtype':>9} {'sym wall(ms)':>12} {'sym dev(us)':>12} {'sta dev(us)':>12} {'unfused(us)':>12} {'dev GB/s':>9}")
    for shape, dt in cases:
        x = torch.randn(*shape, dtype=dt).npu()
        dtype_size = x.element_size()
        m, n = shape

        bm, bn = _pick_blocks(m, n // 2)
        func_sym = situ_glu(bm, bn, 1.0, 0.0, True, dtype=TORCH_TO_TL_DTYPE[dt], sub_M=SUB_M)
        func_sta = situ_glu(bm, bn, 1.0, 0.0, True, dtype=TORCH_TO_TL_DTYPE[dt], sub_M=SUB_M, M=m, N=n)
        x_2d = x.reshape(m, n)

        t_wall = bench_wall(lambda x=x: run_situ_glu(x), args.iters)
        t_sym = bench_event(lambda f=func_sym, x=x_2d: f(x), args.iters)
        t_sta = bench_event(lambda f=func_sta, x=x_2d: f(x), args.iters)
        t_ref = bench_event(lambda x=x: torch_situ_glu_unfused(x), args.iters)

        bytes_per_iter = (m * n + m * n // 2) * dtype_size
        gbps = bytes_per_iter / (t_sta * 1e-6) / 1e9
        print(f"{str(shape):>16} {str(dt).split('.')[-1]:>9} {t_wall:>12.4f} {t_sym:>12.2f} {t_sta:>12.2f} {t_ref:>12.2f} {gbps:>9.1f}")

    print("Kernel Output Match!")


def main():
    import torch_npu  # noqa: F401

    return _original_main()


if __name__ == "__main__":
    main()
