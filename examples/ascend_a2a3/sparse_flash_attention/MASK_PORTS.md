# Complete masked sparse attention examples

Fixed source: TileLang-Ascend `0deb8b93da11b7f0becbcded732d9633b7ca4e34`.
Both complete factories, both inline references and the eight original
`sfa_golden.py` helpers are retained. Imports do not compile, seed, toggle cache,
query an NPU or allocate tensors. Run each example without arguments with an
independent `TILELANG_CACHE_DIR`. The full original executable body runs in
`main()` under a scoped NPU default-device context, keeps its seed/cache-disable
calls and restores the caller's raw cache state on success or exception. Public
inline references use their input device; their original bodies/signatures stay
intact. PA loads its exact sibling golden, independent of foreign module names.

Original defaults are unchanged:

- Mask: B1/S1024/SKV32768/H128/HKV1/DQK576/DV512/topk2048, bfloat16.
- PA: B1/S512 and the same remaining dimensions; physical KV516 pages ×128,
  all-zero block table, actual_q_len512 and actual_kv_len32768.

Both retain original core24 (a source constant/default, not a hardware property),
seed0, five automatic workspaces, one runtime, one rtol=atol1e-2 assertion and
`Test Passed!`. Original synchronization and kernel-source/array printing remain.
No device validation has run; source/SDK and CPU reference checks are separate.
The original mask's SKV sentinel is still gathered before masking, so a source
comparison is not a proof that every physical GM read is in logical bounds.
PA retains its original -1 guard/zero-fill. Neither case is silently rewritten.

TA/main each lower both complete default factories and compile two real SDK
libraries. Full raw CPP differences are limited to include relocation, unused
pure coordinates, local loop-name alpha conversion, and the ordinary mask's
bounded KV address addend order. All other payload, allocation, alias, control,
synchronization and ABI text matches after these explicit changes. No source
packing is normalized away and no device instruction/numerical equivalence is
inferred from these checks.

Source/blob/AST/CLI provenance lives in
`maint/ascend_a2a3/ecosystem_sparse_mask_ports.json`. Host/native tests reside
under `testing/ascend_a2a3/python/ecosystem`; they include the original NumPy
reference, page/tail/causal/sequence helpers and import/cache state isolation.
