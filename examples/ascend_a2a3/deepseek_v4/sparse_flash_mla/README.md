# DeepSeek-v4 sparse MLA (A2/A3)

The three Python files preserve the complete fixed TA `0deb8b93` SWA, HCA,
CSA factories, API, reference helpers, FAST scenarios, thresholds and exits.
Source provenance is in
`maint/ascend_a2a3/ecosystem_deepseek_v4_sparse_mla_ports.json`.

Run the original scripts from this directory in an existing paired
Torch/TorchNPU/CANN environment. Importing the modules does not compile, run a
kernel or disable the caller's cache. The two original script entry points
still disable compilation caching during their execution and restore the
caller's raw cache state on return or failure. Public constructors respect
caller cache settings.

```bash
export TILELANG_CACHE_DIR="$(mktemp -d /tmp/tilelang-a2a3-sparse-mla.XXXXXX)"
python sparse_flash_mla_api.py
python sparse_flash_mla_kernel.py
```

The API script retains all three original bf16 FAST numerical scenarios and
prints `Kernel Output Match!` only after the original checks. The standalone
kernel script retains its original SWA compile-only constructor and
`TEST PASSED!` print; that print does not establish a numerical result. Neither
script parses `--help`; supplying it still runs the original main body.

Validation currently covers complete AST preservation, six original/modern
source requests (three original bf16 and three supplementary fp16), actual SDK
compilation, finite-domain IR effects and emitted C++ signed condition checks.
Real public JIT cold compilation and a new-process cache restore check typed
parameters, output/workspace indices, sources and loaded libraries without
calling a kernel. Explicit source sampling uses `core_num=24`, not a live
hardware-property query. Device numerical, complete CLI and performance
validation remain pending; the supplementary dtype requests are not additional
original FAST scenarios.

The CSA emitted C++ splits an original floor expression into signed arithmetic
with truncation toward zero. Some intermediate values differ even on active
branches. Both actual tile-count consumers match in the tested finite domain;
this is not a claim of unrestricted algebraic equivalence. The migration does
not change `FloorDiv`, synchronization, flags, shapes or algorithms.
