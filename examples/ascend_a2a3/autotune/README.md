# A2/A3 autotune examples

`example_gemm_autotune.py` uses the shared mainline autotuner. It retains the
original GEMM, default `M=N=K=1024`, three default configurations, optional
18-entry search space, seed and `rtol=atol=1e-2`.

```bash
python examples/ascend_a2a3/autotune/example_gemm_autotune.py
```

`--m`, `--n` and `--k` retain their original CLI meaning. Original cache-clearing
calls use `tilelang.disable_cache()` on the modern framework.

`example_gemm_carver.py` preserves the original kernel and the genuine
`Ascend -> MatmulTemplate -> recommend_hints(topk=20)` workflow. Ascend Carver
architecture/policy migration remains deferred for later evaluation, as agreed.
Requesting recommendations raises an explicit diagnostic.
