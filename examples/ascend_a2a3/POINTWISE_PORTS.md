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

## Dynamic quantization and int8 matrix multiplication

Four complete sources retain thirteen original runtime requests:

| Source | Complete original request set |
|---|---|
| `add_rms_norm_dynamic_quant/add_rms_norm_dynamic_quant.py` | All eight original shape/dtype rows, including all three unaligned shapes |
| `dynamic_quant/dynamic_quant.py` | CLI `M=1024,N=512,dtype=float16`, using the original adaptive dispatcher |
| `quant_batch_matmul/example_quant_batch_matmul.py` | `Batch=8,M=N=K=1024`, per-tensor fp16 and per-channel bf16 outputs |
| `quant_batch_matmul/example_quant_matmul.py` | `M=N=K=1024`, per-tensor bf16 and per-channel fp16 outputs |

All original tiling, padding, dtype maps, three-output and workspace decorators,
references, precision checks and seed 0 remain. Add/RMS/dynamic-quant keeps
original-shape GM tensors and both aligned and boundary algorithms. Its
original catch-and-collect failure loop and nonzero exit remain intact.
The dynamic-quant CLI keeps its original integer overflow/match-ratio check;
its output values and scales are not replaced by a simpler reference.

Five explicitly supplementary source probes complete the dynamic-quant
three dispatch buckets with both CLI-supported dtypes. They use the original
pure host dispatcher and complete kernel, and are not extra original runtime
matrix rows or device validation. The same original pure `_get_tiling` result
is used for all eight Add/RMS rows; no tile size is changed to pass compilation.

Runtime and actual clears use the same explicit dedicated-cache contract.
Quantized matrix entries preserve their original clear position inside the
renamed original main. The complete source and actual SDK checks cover every
request. Generated source keeps pure unused scalar/declaration-order differences;
all named operands, actual buffer sizes and logical alias relationships are
checked. Original intentional buffer reuse is preserved, rather than assuming
that every allocation must be disjoint for the entire kernel. Numerical device
and performance checks remain pending.

## Causal convolution and cross entropy

Four complete sources retain twenty-nine original runtime requests:

| Source | Complete original request set |
|---|---|
| `causal_conv1d/causal_conv1d.py` | Both original fp16/bf16 prefill inputs; width 4, two batches, 2048 tokens and channels, 24-way splitting |
| `causal_conv1d/causal_conv1d_decode.py` | Both original bf16 configurations: batch 1, channels 2048/3072, seeds 42/43 |
| `causal_conv1d/causal_conv1d_pto.py` | Both original prefill dtype inputs with the original PTO target |
| `cross_entropy_loss/example_cross_entro.py` | All 23 original dtype/large-class/partial-boundary calls |

Prefill retains the original wrapper that casts bf16 input to an fp16 kernel
and converts outputs/state back. Both original cache clears and state handling
remain. Decode preserves the original NPU availability guard and its printed
summary; an unavailable-device skip is not numerical acceptance. Its complete
symbolic dimensions, state/index/bias handling and both output/state reference
checks remain. Cross entropy keeps both original factories, dtype dispatcher,
negative output indices, scalar loss and complete log-probability references.

Every request passed full lowering and actual SDK compilation on each side.
Source/AST checks retain complete factory/helper/host behavior, with only the
private language, guarded runtime and modern cache entry adaptation. The
original symbolic declarations and dependent module constants remain global;
original shebangs remain first and executable where canonical checks require.

Generated convolution sources retain additional C-core read-only metadata
loads. For the original prefill, C reads only the same sequence-length addresses
`0..2` already read by AIV. For both decode inputs, C reads the same original
index/mode address set as AIV, within batch-1 arrays. There are no added C-core
writes, flags or vector operations. Observed integer address reorderings and
cross-entropy index promotion are checked within the original shape/index
domains; all original allocations and aliases remain. These bounded source
witnesses do not establish arbitrary-shape/alias or device correctness.
Numerical device and timing validation remain pending.

## Cumulative minimum, gather and group normalization

Three complete sources retain all sixty original runtime cases:

| Source | Complete original matrix |
|---|---|
| `cann-bench/cummin/example_cummin.py` | All twenty `_CASES`, original input recipes, dtype/axis/tiling dispatch and value/index reference checks |
| `cann-bench/gather/gather.py` | All twenty original cases, including int64 low/high decomposition and recursive axis chunking |
| `cann-bench/group_norm/example_group_norm.py` | All twenty original cases, including serial/pipeline selection, padding and special floating-point inputs |

The original seeds are 0 for cumulative minimum and 42 for gather/group norm.
The sixty runtime cases produce sixty-two distinct native constructor requests:
20 cumulative-minimum, 22 gather and 20 group-norm. Gather's original dictionary
cache, recursion and int64 decomposition produce 51 kernel invocations across
its twenty cases. The source-only trace records shapes and control flow; its
placeholder outputs are never used to run or pass a golden comparison.

Cumulative minimum's data-dependent NaN dispatch is obtained from the complete
original CPU input generator with seed 0 and all twenty cases in order. Only
case 14 has a NaN. CPU RNG hashes and input dtype/shape/byte counts are recorded separately; this trace
is not a device run or a numerical kernel check. All original NaN/Inf behavior,
references, tolerances and exit conditions stay intact. Gather retains its
original printed failure summary without introducing a failing exit code.
Cumulative minimum and group norm retain their original nonzero failure exits.

Group norm uses four explicit scalar-buffer loads in fill arguments:
`total[0]`/`std_val[0]` for the serial and pipeline factories. Actual old and new
frontend IR verifies the same allocated buffer/data identity, shape `[1]` and
index 0 for each of the twenty requests. This adapts the old parser's implicit
scalar load to eager construction without changing the language API or kernel
algorithm. The initial unsupported bare-buffer errors remain in the evidence.

All sixty-two native requests passed complete lowering and SDK compilation on
both sides. Emitted C++ retains unused pure scalar bindings, core-variable naming
and allocation-order differences. Full remaining payload, operands, loops,
synchronization, host launches and logical allocation overlap relations match
within these requests. Additional unused integer expressions are checked within
the actual launch/constant-loop bounds. The raw source is preserved and compiled
as generated. Device numerical and performance validation remains pending.
