# A2/A3 autotune examples

`example_gemm_autotune.py` keeps the original TA GEMM, default
`M=N=K=1024`, three default search configurations, and `rtol=atol=1e-2`.
The original 18-entry alternative search-space function is also retained.
It uses the shared mainline autotuner and Cython execution.

Run on an A2/A3 NPU with the Ascend SDK and Torch-NPU configured. The example
clears its kernel cache as the original did, while keeping tuning-result
persistence enabled. Give it a dedicated cache directory:

```bash
export TILELANG_CACHE_DIR="$(mktemp -d -t a2a3-autotune.XXXXXX)"
python examples/ascend_a2a3/autotune/example_gemm_autotune.py
```

`--m`, `--n`, and `--k` keep their original CLI meaning. The original kernel's
full-tile assumptions also remain; this migration does not add a tail algorithm.
The input supplier keeps seed 0 and the original `randn → half → npu` sequence,
using the supplied logical parameter shapes instead of import-time CLI globals.

`example_gemm_carver.py` retains the complete original kernel and genuine
`Ascend → MatmulTemplate → recommend_hints(topk=20)` configuration workflow.
The required Ascend Carver architecture/policies are deferred for later
evaluation. Requesting recommendations gives an explicit diagnostic. No static
search space replaces Carver, and this example is not counted as validated.

The fixed source pin and hashes are recorded in [SOURCE.md](SOURCE.md).
