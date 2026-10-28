# A2/A3 attention-sink examples

The forward-varlen pair preserves the complete sources from TA commit
`0deb8b93da11b7f0becbcded732d9633b7ca4e34`. Source paths, Git blobs,
content hashes, original control rows and constructor requests are recorded in
`maint/ascend_a2a3/ecosystem_attention_sink_ports.json`.

The original standalone example and test CLI remain separate:

```bash
export TILELANG_CACHE_DIR="$(mktemp -d -t a2a3-attention-sink.XXXXXXXX)"
python examples/ascend_a2a3/attention_sink/example_gqa_sink_fwd_varlen/example_gqa_sink_fwd_varlen.py
python examples/ascend_a2a3/attention_sink/example_gqa_sink_fwd_varlen/test_gqa_sink_fwd_varlen.py --level all
```

The test CLI defaults to L0. It also retains L1, L2, boundary and benchmark
levels; its original L2/boundary warnings do not become blocking failures.
The pytest `test_forward` alias runs the original five L0 rows. The full pair
contains 23 original control rows and 19 distinct constructors, including the
original five benchmark requests. Benchmark warmup/repetition and precision
thresholds are unchanged.

Imports use the A2/A3 language/layout namespaces and an exact sibling module.
Importing does not register TorchNPU, change the default device, disable caches,
allocate tensors or compile kernels. Runtime entry points register TorchNPU
lazily and restore the caller's cache state. The original `_setup` callers use a
lexical NPU default-device scope instead of changing the process default.
Original explicit output/workspace arguments and allocations remain intact.

All 19 complete source requests and 38 paired TA/main Bisheng compilations have
passed. Four noncausal main sources retain an additional real AIC
`q_seqlens.GetValue` per active task: 16,436 configured reads across those four
requests. Their fixed original table addresses/values are checked, and the
remaining payload, allocations and manual synchronization match. These reads
remain in the actual compiled sources; their cost is not measured. This is
source/SDK validation, not numerical, deadlock, performance or CI acceptance.
Device execution of the original matrix is pending.

The backward-BHSD pair preserves both complete original sources, Autograd,
CPU references, double precision gates, retry policy and all 25 original control
rows (standalone1, L0 seven, L1 eight, L2 five, boundary4). Its 34 distinct
constructors produce 31 full source pairs and 62 successful Bisheng compilations;
N129/N509 forward and D64 backward retain their three original assertion
rejections. Original L2 N192 is a valid forward constructor; the float32 case is
an input/runtime boundary, not a source rejection. Seven backward requests swap
`delta_ub`/`tmp_ub` within the same exclusive byte range; all other payload,
allocation aliases and synchronization agree. No added scalar reads or PIPE_ALL
are separated here. Numerical execution remains pending.

Backward test cleanup never scans `/tmp/tmp*.so`. It only removes regular
libraries observed from this component's real JIT result or the actual compiler
exception frame, after exact source/output-pair and file-identity checks. Unknown
argv, cached paths, symlinks, replaced files and other components' libraries are
left intact; source `.cpp` files remain. The original three-attempt compile-error
retry, sleep, stderr capture, returned objects and exceptions remain unchanged.
