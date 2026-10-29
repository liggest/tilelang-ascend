# Complete generative recommendation example

```bash
TILELANG_CACHE_DIR=/tmp/my-grc-task python examples/ascend_a2a3/generative_recommendation/mtgr_ragged_segment_attention.py
```

Use an explicit task-owned cache directory. The original test disables cache
and clears once per invocation; the modern backend cache replaces only the
missing old clear API. The guard rejects a missing directory or a conflicting
caller descriptor override. The caller's raw cache-enabled bit, memory-cache
object/content and staging marker are restored on success/exception. No kernel,
reference, input, workspace or precision is replaced for this isolation.

The complete original default CLI remains one configuration:
H8/D128, segments `[1600,8,5,1200]`, rules `[0,1,2,2]`, prefix0. It calls the
original test once, one kernel runtime and one original assert_close at1e-2,
then prints its original success marker. Data generation retains its original
seed0. Core24 is an original constant, not an observed hardware property.
The public test/factory/wrapper retain all original branches and signatures.
The three explicit workspaces and explicit int32 `bin_iters` argument remain.

The original `testcase.py` also defines a complete462-entry generated catalog,
including original multi_seed/perf_test flags. It remains complete, but it is
not the default CLI. This source delivery does not claim all462 configurations
have compiled or run. The catalog and both golden functions remain original
complete algorithms. Helpers load lazily from exact adjacent files without
modifying `sys.path` or accepting foreign same-named modules.

Importing the main module does not load the helper catalog, seed, clear cache,
allocate, compile or register TorchNPU. The CLI registers TorchNPU explicitly
only when invoked. Public references/data helpers retain their original CPU
behavior and explicit seed semantics; their bodies are not replaced by mocks.

TA/main each generated the complete original default factory and compiled its
actual source with Bisheng. Whole zN/nZ helper formulas and all four real call
shapes were compared; their BF16 capacities match, including the preserved
catalog's D64/32 shapes in the static layout proof. The real restored Cython
adapter uses14 explicit buffer handles, int32 bin_iters, then ordered dynamic
scalars total_q/batch/max_segs/num_blocks/max_blocks. No extra runtime buffers
or automatic workspace allocation was introduced, and no adapter runtime ran.

Raw C++ differs through scalar/conditional materialization and two integer
initializer positions. The audit only inlines call-free scalar aliases whose
operands remain lexically stable through every use and relevant loop body.
After that explicit audit transformation, include relocation and redundant
parentheses, the complete payload, control, synchronization, ABI and all28
named views match. The original default integer/metadata domains are recorded.
GetWithOffset is not assumed pure; its complete declaration order, view sizes,
offsets/aliases and final pool descriptor are unchanged.

CPU checks retain all462 catalog entries, complete helper algorithms and full
default CPU reference. Prefix/cache/GQA/rule boundary references were tested
without NPU. These checks, frontend/source/SDK and strict OFF validation are
not a numerical kernel or performance pass. Default device execution and the
larger catalog remain pending; no shape, golden, threshold or failure was
removed to obtain source validation.

## Current normal compiler refresh

The complete original default was regenerated using normal compiler SHA256
`a9cefba866baa68644769faf202515a91490ac970b4dccb99078c8a6c40f867f` and
received one fresh successful public Bisheng compile. The previous frozen
source/SDK audit remains historical; its scalar-alias normalization is not
used as the current comparison. Against fixed-source TA, the new whole C++
is identical after only header relocation, moving two zero-initialized int
declarations across fixed-address view declarations, and substituting two
single-use bool comparison RHS values at their immediately following uses.
There is no floating-point propagation or inserted synchronization in these
differences. All28 ordered views and the complete host ABI are identical.

Full configs/source-before/ELF-after and actual680-header snapshots were
checked. Host ON18 and actualOFF17 plus one native skip pass. Default
NPU execution and the462-entry catalog device matrix remain pending.
Evidence: `refactor_main/reports/p3_PA3_GRC3_root_a9_review/`.
