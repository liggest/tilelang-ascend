# Remaining experimental source ports

`maint/ascend_a2a3/ecosystem_experiment_rest_ports.json` records fixed original
TA sources at `0deb8b93da11b7f0becbcded732d9633b7ca4e34`.
The initial scope is 33 remaining experimental assets, excluding the 22 already
mapped experiment assets. Other batches remain pending until individually
recorded. Source/SDK validation is separate from numerical device acceptance.

## E1: cumulative sum, random, Hadamard and split-K

Five complete original files retain all factories, helpers, configurations,
references, CLI flags, seeds, thresholds and exit behavior. Their original
runtime matrix is preserved:

- GDN cumulative sum: all 20 original rows, both reverse directions, fragment
  selection, layout, aligned/tail and odd-head paths.
- KDA cumulative sum: all 28 scalar/vector local/global rows, four scaled rows
  and four variable-length dispatcher rows, retaining the original sequence
  bounds `[0,45,118,160]` and chunk-index validation.
- Random 1D: original defaults M=1024, block size 128, seed 42, both LCG
  constants and exact integer reference. All CLI flags remain.
- Hadamard: original batch 4, dimension/block size 2048, float default and
  complete alternative dtype and cross-block algorithms.
- Split-K GEMM: both original 128-cubed/split-2 and 1024-cubed/split-4 requests,
  seed 42, zero-initialized accumulation and original 1e-2 thresholds.

The 60 original runtime rows produce 28 unique native constructor requests.
Complete shape/control tracing records 34 native constructor events and
34 original Torch fallback boundaries; four variable-length rows each visit
three sequences. Metadata placeholders never run an original golden comparison
or establish fallback numerical correctness. Fallback paths remain the complete
original implementation and are not counted as native or device coverage.

Imports do not parse arguments, seed RNGs, allocate tensors, clear caches,
compile or launch. Actual clears remain in their original runtime order,
requiring an explicit dedicated `TILELANG_CACHE_DIR` whose resolved cache path
matches. GDN and KDA retain both original clear calls. The original split-K
cache disable is restored to the caller on exit. Device registration occurs
inside main. Random exposes its original parser defaults at import, and the
complete original parser reconfigures the same globals and derived
`TOTAL_BLOCK_SIZE` before executing its factory; a custom CLI regression checks
that contract. The original random CRLF source is tracked by its raw source hash;
the port uses canonical LF without changing its AST.

Every unique native request passed complete lowering and actual SDK compilation
on both sides. Generated source retains pure unused scalar/default-subcore
bindings, lexical loop/core-variable names and independent allocation ordering.
Two KDA requests retain an extra empty C-core control loop. Full remaining
payload, synchronization, operand uses, host launches and allocation maps are
checked separately; additional unused shift/division arithmetic is bounded in
the actual original launch/loop domain. No pass or emitted source is rewritten
to force equality. Numerical device and performance validation remains pending.
