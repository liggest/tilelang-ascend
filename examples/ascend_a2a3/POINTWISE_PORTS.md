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
