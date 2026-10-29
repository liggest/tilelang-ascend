# DeepSeek V4 assets for A2/A3

`lightning_indexer.py` retains the complete original TA asset from
`0deb8b93da11b7f0becbcded732d9633b7ca4e34`: ten original functions,
the nested JIT decorator, factory parameters, four CLI scenarios, host helpers,
golden checks and original exit conditions. The source ledger is
`maint/ascend_a2a3/ecosystem_deepseek_v4_ports.json`.

The port uses the private A2/A3 language/layout imports and lazy runtime setup.
Two repeated cached-merge chains use an ordinary `@T.macro` with four explicit
inputs. Inlining those two calls reproduces the complete original factory AST.
Runtime scopes preserve the caller's cache state and device context.

The no-sortcache branch retains the original fill, sort, Axpy, MergeSort and
copy operations and synchronization. No artificial barrier, allocator ordering
change, matrix reduction or compiler-flag change is included. The earlier
isolated `c5fada2e` candidate added a conservative V barrier; it was withdrawn
and was never integrated into the main branch.

The source investigation is recorded in
`refactor_main/reports/p3_codegen_regression_D1/README.zh-CN.md` and its immutable
facts index. The original planner itself can choose either of two layouts for
the same input; this is separate from a confirmed modern RemoveNoOp gap that
retains additional control and GM reads. Address overlap alone is not proof of
a hardware race, safety or numerical equivalence.

All eight complete constructor requests have been refreshed through the normal
ROOT NoOp/Simplify native library and actual Bisheng compilation. The recorded
working compiler snapshot and loaded library hashes identify that refresh;
HEAD alone does not attest the then-uncommitted compiler changes. Across 444
actual sampled core traces, complete ordered operation operands, pointer/type/
mask/sync values and metadata/scalar reader multiplicities match the archived
original requests. This finite CPU replay does not compute vector outputs.

The later normal a9 refresh regenerated all eight sources: three offset-only
changes were actually recompiled, and five byte-identical sources reference
the existing 1bf SDK artifacts after identity checks. These are three new
compiles plus five references. Case-specific address-state observations are
recorded in the ledger, including fresh fixed original TA full-chain case7;
they do not establish asynchronous hardware safety. The artificial V candidate
remains permanently withdrawn.

Raw source still differs in variable naming, pure conditional expansion and
original planner address selection. No hardware safety or instruction-byte
equality claim follows from these checks. Historical alias-model claims are
not reused. Full original CLI, device numerical checks and performance remain
pending. The eight source requests sample explicit core controls 24 and 20;
they do not execute the original default device-property query. Their actual
launch grids are recorded separately in the ledger.

Importing the asset performs no compilation, allocation, cache clearing or
launch. Run device CLI work only under an explicitly authorized device window;
this source-only candidate is not a device-validation claim.

The CLI AST hash is normalized: the original body-level
`torch.set_default_device(f"npu:{_DEV}")` statement (with the original `_DEV = 0`) is replaced by the explicit
`_runtime_scope(npu_default=True)` around `main()`. The four original scenarios
and remaining host body are retained; this is not a literal unchanged CLI AST.
The original CLI has no argparse/help entrypoint. Passing `--help` does not
provide a host-safe exit: it runs `main()` and begins with TorchNPU registration.
Use import/AST checks for host inspection instead of invoking the CLI.
