# Original simple A2/A3 examples

This package preserves 14 original TA Python assets: five activation examples,
three elementwise examples, LayerNorm, RMSNorm forward/backward, and four reductions.
The complete original matrices and CLI defaults produce 36 factory requests; large,
partial-tile and BF16 cases remain present. Original seed, golden, tolerances,
decorators, flags and algorithms are retained.

Source provenance and per-file validation state live in
[`ecosystem_simple_ports.json`](../../maint/ascend_a2a3/ecosystem_simple_ports.json).
The new files use `tilelang.ascend.a2a3.language`; A5 and main CUDA examples are
independent. This package does not change the original CI registry or classify
pending examples as passed merely because files or tests exist.

Importing an example defines its factories and constants without parsing CLI,
clearing a cache, changing Torch device/RNG state, compiling, allocating a tensor
or launching a kernel. Runtime code is available through `main()` and the ordinary
script entry. The four original compile-every-call examples retain that behavior
while running and restore caller cache/default-device state even on failure.

For examples that originally clear the cache, explicitly select a dedicated cache
before running; the original clear remains effective only in that selected A2/A3
namespace. A missing setting produces a diagnostic before any cache operation.
For example, after preparing the supported environment described in the
[migration guide](../../docs/ascend_a2a3/migration.md):

```bash
export TILELANG_CACHE_DIR="$(mktemp -d)"
python examples/ascend_a2a3/elementwise/elementwise_add.py
```

CLI defaults and all original built-in matrices remain unchanged. In particular,
`setvalue_example.py` retains its original fixed 8192-element input and print-only
success marker; it had no numerical assertion. A successful exit from that example
must not be described as an original golden comparison.

The component tests cover imports, cache isolation on failure and all original
source-only requests. OFF builds skip native lowering explicitly. Actual source,
SDK and device results are distinct fields in the manifest; the presence of the
examples or tests does not establish those results.
