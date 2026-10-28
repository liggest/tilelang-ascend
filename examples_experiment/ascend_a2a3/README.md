# A2/A3 experimental examples

This namespace preserves the five original active TA example/test pairs for
block-sparse GEMM, convolution, convolution autotuning and the two dequantized
GEMM variants. Algorithms, decorators, configurations, references and precision
thresholds remain intact. These examples do not depend on Carver.

The full original pytest domain contains 28 items: one CPU unpack test and
27 NPU tests. Source and SDK checks cover all 93 original CI factory/config
mappings. Device validation is pending; source or compilation success does not
establish numerical correctness.

```bash
python -m pytest examples_experiment/ascend_a2a3/
```

A build without A2/A3 native components runs the CPU unpack test and skips only
the marked device cases. Each test has a temporary disk/memory cache. The two
autotune companions also isolate the tuner result dictionary, restoring the
caller's original object afterward. Importing modules neither clears caches nor
runs their CLI. For direct CLI execution, provide a dedicated `TILELANG_CACHE_DIR`
because the original CLI cache-clear behavior is retained there.

The original block-sparse standalone tuning list retains all 18 candidates. Its
two largest `block_M=block_N=256` candidates require 262144 bytes of L0C against
131072 available: the original TA lowering crashes; the migrated lowering
reports OOM. They are not part of the four original block-sparse CI cases and
are not counted as successful configurations. No candidate was silently removed.

Pinned source/destination hashes and the validation scope are recorded in
`maint/ascend_a2a3/ecosystem_ports.json`.
