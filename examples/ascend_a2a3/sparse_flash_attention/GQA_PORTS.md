# Sparse GQA: three complete original default examples

Source: TileLang-Ascend `0deb8b93da11b7f0becbcded732d9633b7ca4e34`.
The AscendC, PTO and PTO Developer files preserve all six original definitions,
three full executable CLI bodies and decorator options. Run each file without
arguments, using its own `TILELANG_CACHE_DIR`. Imports do not compile, seed, toggle
cache or allocate tensors. `main()` scopes the original NPU default-device
behavior, keeps the original cache-disable call and restores the caller's raw
cache state even on exceptions. The public reference uses its actual `q.device`
while preserving the original body and inspected signature.

All three original defaults remain B=2, S=273, SKV=44444, H_Q=64, H_KV=4,
D=128, topk=2048, query start=28672 and stride=1. Seeds remain 0/42/42.
Each original CLI has one kernel invocation and one `assert_close` with
rtol=atol=1e-2, followed by `Test Passed!`. Device execution is pending;
source/SDK or CPU reference checks do not constitute these three device passes.

The Developer example still supplies four zeroed workspaces with leading
extent 2184. Its independent symbolic `workspace_block_num` is carried by
workspace_1's first dimension; it is neither inferred from Q nor allocated
implicitly. The unused preallocated output and original overwritten output
binding are retained. The real restored Cython adapter and actual native host
prototype agree on eight handles followed by batch, seq_len, seq_len_kv and
workspace_block_num. The other examples have three scalars; their four
workspaces retain automatic allocation.

TA/main each produced all three complete default sources and compiled them
with real Bisheng (one AscendC, two PTO), without device properties, context,
tensor allocation or kernel execution. Three actual SDK libraries were also
restored into real Cython adapters and stopped before runtime. Source changes
are recorded explicitly: include relocation, unused pure coordinate bindings,
bounded commutative integer addresses, and named UB packing/declaration order.
Complete payload, control, synchronization and ABI match after those specific
changes. Named capacities and allocation unions are unchanged; the PTO
non-Developer example retains its intentional L0C alias. AscendC shared UB
GetWithOffset has real dataLen side effects: its final pool descriptor changes
from 2048 to 4096 bytes, while every copied LocalTensor descriptor is unchanged
and no raw pool is used afterward. This is not a claim of byte-identical machine
code or arbitrary-shape numerical equivalence.

`maint/ascend_a2a3/ecosystem_sparse_gqa_ports.json` records each original source
blob, complete AST and CLI provenance. The companion tests live under the real
`testing/ascend_a2a3/python/ecosystem` pytest root, not the example CLI collector.
