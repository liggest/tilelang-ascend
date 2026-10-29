# Complete pipelined and developer PA API leaves

Both leaves retain all three original functions, factory branches/options,
workspace declarations and host wrapper from TA
`0deb8b93da11b7f0becbcded732d9633b7ca4e34`. Neither leaf has an original
standalone CLI or golden assertion. No synthetic CLI or success marker is added.
Importing a leaf does not seed, disable cache, allocate tensors or modify
`sys.path`; only the unused import-time parent-path append was removed.

The original explicit public `init_test()` is unchanged. It sets the global
NPU default device and disables cache, with seed42 for pipelined and seed0 for
developer. This API is intentionally not changed into a scoped initializer.
The complete factory/reference consumer must call it only in its intended
process. Source, SDK and host checks below did not call it.

The real four-version `bench_sfa.py --file` consumer is a later separate batch.
For `sparse_flash_attn_pa`, the complete original CLI has three topk2048 inputs
(KV_S2560/6400/48000) followed by topk128 and256 at KV_S2560, plus three
rejection checks for0/64/192. Developer has the three original topk2048 inputs.
Each input retains B=T1/H128/D512/rope64/actualKV2560, core24, scale0.5 and the
original1e-2/equal_nan golden comparison. Core24 is an original constant,
not an observed device property. Consumer runtime remains pending.

Six full source requests per framework cover the three pipelined topk values
and developer's three static KV page counts20/50/375. Each framework compiled
four unique complete source/config/ABI identities with actual Bisheng. The
developer50/375 requests only reuse the matching20-page compiled library after
exact identity comparison; all three frontend buffer shapes are recorded.
The original public developer factory default516 pages is still retained;
these source requests represent actual consumer specializations, not exhaustive
validation of every optional API argument.

Both frameworks reject0/64/192 with exactly the same original AssertionError
types/messages. Main host tests also exercise all six frontend metadata/JSON
roundtrips and the three original rejections. Actual OFF runs keep pure import,
signature and provenance tests while skipping only nine native frontend cases.

Full raw C++ diffs are retained. Every difference is explicitly accounted for:
include relocation, unused pure integer/control bindings, lexical condition
naming, two bounded integer multiplication commutations, and moving the indices
view declaration. In developer this move changes the intermediate UB pool
`dataLen` sequence. It is not assumed pure: CANN copies each view's descriptor
and all named view sizes/offsets/aliases plus the final pool descriptor are the
same; complete sources use each raw pool only during initial view declarations.
After those exact audit transformations, all other complete payload, sync,
control and ABI bytes match. No device or instruction equivalence is claimed.

Original unguarded KV gathers are preserved. The consumer uses nonnegative
selected indices below2560 and an identity page table; arbitrary negative
sentinels or dynamic domains are not claimed bounds-safe. Source/SDK validation
uses real driver availability/count with initialization staying false and no
properties/context/tensor/launch. TA native is the existing installation with
the established read-only runtime alias, not a fresh fixed-pin rebuild.
