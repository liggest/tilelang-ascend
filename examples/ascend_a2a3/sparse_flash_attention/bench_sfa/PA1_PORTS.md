# PA baseline and NoCV API leaves

These two complete leaves come from TA commit
`0deb8b93da11b7f0becbcded732d9633b7ca4e34`. Each retains all three original
functions, factory branches/decorator options, explicit five workspaces and
public wrapper. They have no standalone CLI or numerical assertions. Importing
a leaf neither runs the original initializer nor changes `sys.path`.

Call `init_test()` explicitly only in the intended device process. Its original
public behavior is preserved: set the global default device to NPU, seed42 and
disable cache. Import safety does not make this explicit initializer scoped.
No device or numerical validation is claimed for these leaves.

The original `bench_sfa.py --file` consumer is a separate migration batch. It
selects four versions and compares three original default requests with
`torch_npu.npu_sparse_flash_attention` at rtol/atol1e-2, equal_nan=True. These
leaves retain the original hardcoded topk2048 and core24. Core24 is a source
constant, not an observed device property. The consumer's additional topk128/256
and rejection checks belong to its separately imported pipelined implementation,
not these two factories. No synthetic leaf CLI or success marker was added.

NoCV imports the existing backend-local `make_zn_layout`. The complete original
formula, helper branches and its five actual L1 shapes were compared; all five
BF16 shapes have exact logical/padded capacity. The full factory algorithms,
public wrapper signatures and explicit initializers are unchanged.

TA and main each produced two complete default factory sources and two actual
Bisheng libraries. The three original consumer shapes share each dynamic
factory specialization. Main used the normal native/Cython libraries; TA used
its existing native library with the established read-only runtime alias, not a
fresh fixed-pin build. Only real driver count/availability was queried, with
NPU initialization remaining false. No properties/context/tensor/launch ran.

The full C++ review accounts for include relocation, one cross-pool declaration
movement, unused pure integer bindings/loops, lexical loop naming and two
bounded integer multiplication commutations. Every pool's own view declaration
order, offsets/capacity, alias overlaps and last descriptor match. The
`GetWithOffset` descriptor mutation is real and is not assumed pure. All other
complete payload, synchronization, control and ABI bytes match after the
explicit audit transformations; this is not a device-instruction or numerical
equivalence claim.

NoCV's original gather has no negative-index guard. The original benchmark
selects nonnegative indices below2560 with an identity page table; the migrated
leaf preserves that support boundary and does not claim arbitrary indices or
dynamic shapes are bounds-safe. Device/consumer assertions remain pending.
