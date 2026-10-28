# Original pointwise and related source ports

The independent `maint/ascend_a2a3/ecosystem_pointwise_ports.json` records complete
sources pinned to TA `0deb8b93da11b7f0becbcded732d9633b7ca4e34`.
Source and SDK validation remain separate from numerical device acceptance.

## Broadcast, online softmax and merge sort

| Source | Complete original request set |
|---|---|
| `pad/example_broadcast.py` | CLI `m=1024,n=256`, block 128 |
| `pad/example_broadcast_pipeline.py` | Same CLI shape, block 128/sub-block 64 |
| `softmax/example_online_softmax.py` | Both original shapes `34,130,32,32` and `1024,51200,128,128`, each float/float16/bfloat16 |
| `sort/example_merge_sort.py` | All 2/3/4-way merges, N=64 value-index pairs per input |

Complete factories, decorators, original configurations, references, input
construction and precision/exit behavior remain. Sort retains seed 42 and
its original printed summary; it does not raise or return a failing exit code
when a comparison is false. The other sources retain seed 0. The two broadcast
checks use 1e-2 tolerances; online softmax keeps its original dtype-specific
thresholds. Validation does not strengthen a printed result into a device pass.

Import does not clear caches, seed RNGs, parse arguments, compile or launch.
The complete original runtime executes through `main()`, with explicit
`torch_npu` registration. The original sort main is named `_original_main`
without changing its signature or body, then called by the guarded entry.
Every original cache clear still executes, requiring a dedicated explicit
`TILELANG_CACHE_DIR` and a matching resolved cache path. For an authorized
device run, preserve the original CLI and isolate its cache, for example:

```bash
(
  export TL_PLATFORM=A3
  export TL_RUN_MODE=npu
  TILELANG_CACHE_DIR=$(mktemp -d /tmp/tilelang-a2a3-pointwise.XXXXXXXX) || exit 1
  export TILELANG_CACHE_DIR
  python examples/ascend_a2a3/pad/example_broadcast.py
)
```

Host/source checks cover all 11 original factory requests. SDK and device
states are recorded per source in the ledger. Generated-source differences
are retained and reviewed, including unused scalar/default-subcore bindings,
the core-variable name, and the order/offset of independent UB allocations.
The exact named operand uses and full operation/synchronization payload are
compared separately from allocation maps. No pass or emitted source is edited
to force textual equality. Numerical device and performance validation remain
pending for these sources.

## GEMV, top-k and unsorted segment sum

Four complete sources retain fourteen original runtime requests:

| Source | Original CLI/default matrix |
|---|---|
| `gemv/example_gemv_c.py` | `1024,1024,128,128` float16/float32; `64,64,16,16` float16 |
| `gemv/example_gemv_v.py` | `1024,1024,128,128` float16; `1024,1024,64,64` float32; `64,64,16,16` float16 |
| `topk_selector/example_topk_selector.py` | `B=64,N=4096,top_k=2048`; `B=N=1024,top_k=128` |
| `unsorted_segment_sum/unsorted_segment_sum.py` | All six original float16/bfloat16 rows and original atomic dispatch |

All three segment strategies and their original host dispatch, padding,
sorting, dtype promotion, references and return behavior remain. Two additional
source probes exercise the original non-CLI segment and block-reduce
strategies. Three additional GEMV vector source probes select PTO through the
actual JIT object target. These five probes are explicitly supplementary;
they are not original runtime matrix rows or numerical device validation.

The GEMV vector helper reads the modern `JITImpl.target` field instead of the
old `__jit_impl__.target` indirection. The default field is `None` in main and
`"auto"` in TA; both preserve the original `is_pto=False` branch, while explicit
PTO selects `True` on each side. The original `alloc_temp` definition stays
intact; the factory does not call that local allocation helper. No shared
legacy attribute is introduced. The original GEMV cube docstring's leading
space is normalized by Ruff; algorithm statements remain unchanged.

Existing runtime entries are named `_original_main` with their original
signatures and complete bodies. The outer `main` registers the device backend
and preserves the original clear/disable/neutral cache mode. Clear remains at
its original point inside the top-k host entry. Disabling is restored on exit,
including failure; cache-neutral GEMV cube stays neutral.

Generated sources keep allocation/order differences and original synchronization.
The supplementary segment-reduce probe has two additional read-only C-core GM
reads and two `PIPE_ALL` barriers. For that exact probe, the reads use the same
Offsets address set `0..8` as the original AIV path, with no C-core writes,
flags or vector math. This difference is retained and does not establish a
general segmented-shape, alias, numerical or performance guarantee. Source/SDK
acceptance remains separate from the pending device matrix.
