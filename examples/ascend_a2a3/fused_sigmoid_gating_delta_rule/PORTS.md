# Variable-length fused sigmoid gating delta rule

This port keeps the complete `kernel`, CPU `golden` and public per-case `main` from TA
`0deb8b93da11b7f0becbcded732d9633b7ca4e34`. The five original CLI cases,
seeds, 64-token padding, explicit initial-state layout, output slicing, pass configs,
24-block launch and both `rtol=atol=2e-2` comparisons per case are retained.
The uniform public `main` branch and optional normalization/softplus arguments also remain.

Use a dedicated cache directory when running the original CLI:

```bash
TILELANG_CACHE_DIR=/absolute/path/to/owned-vl1-cache \
  python examples/ascend_a2a3/fused_sigmoid_gating_delta_rule/fused_sigmoid_gating_delta_rule_varlen.py
```

`--threads N` retains the original scheduler. Thread exceptions do **not** propagate
through `join`: that original branch can print `All tests passed!` and exit zero
when a worker failed. Numerical acceptance requires the actual ten comparisons,
not the final message or exit code. This port does not change that limitation.

Import and `--help` do not clear caches, register TorchNPU, seed RNGs or execute a
kernel. The original one-time cache clear runs only after CLI parsing and explicit
cache-ownership checks. Its temporary memory cache, the caller's raw cache-enabled
bit and staging-cleanup state are restored after all threads join, including failures.
The cache descriptor is never rewritten by the example.

## Validation scope

The complete five original requests passed source lowering and real Bisheng SDK
compilation with the normal A2/A3 compiler after original-phase Simplify/RemoveNoOp
compatibility and restoration of the original CombineCV phase. The ordered scalar
views return to the original common scope. C-side reads, stores and loops are zero;
V payload/control/synchronization, twelve-parameter ABI, mixed AIC1:V2 kind and
launch24 are accounted for. No synchronization was added to this example.

Every remaining C++ difference was explicitly audited:

- The backend's required header-path relocation.
- The old float32 `x=a_val+dt_val` alias is propagated as the identical typed RHS
  at its three uses by shared Simplify. This is an actual source representation
  difference, not a claim of identical device instructions or floating-point results.
- Case4 may exchange the norm/tmp addresses. Identical pre-planner inputs in both
  fresh fixed-old and current compilers reproduce both allocation states. The
  inherited planner state variation, view descriptors and pool consumption were
  recorded; the example does not force an allocation order.

The five-case device matrix and optional threaded execution remain **pending**.
Host tests and SDK success do not establish numerical or performance equivalence.
No new NPU execution was performed to finish this port.

The exclusive manifest is `maint/ascend_a2a3/ecosystem_fused_varlen_ports.json`.
The immutable source/SDK audit is
`refactor_main/reports/p3_codegen_regression_recovery_a9cefba8/immutable_index.json`
(SHA256 `66d6b4a50ce701911d0bf29580a2004b30f929cfdb7ad62a4776c3889052aae7`).
It contains the five successful VL1 SDK compilations and eleven neighboring
requests (15 fresh compilations plus one exact same-batch reuse in total).
The original 117-fact recovery point and later 1719/108-fact checkpoints remain
unchanged. Final host/OFF checks and provenance are indexed separately in
`refactor_main/reports/p3_fused_varlen_vl1_final/`.
