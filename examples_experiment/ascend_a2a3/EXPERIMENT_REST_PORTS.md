# Remaining experimental source ports

`maint/ascend_a2a3/ecosystem_experiment_rest_ports.json` records fixed original
TA sources at `0deb8b93da11b7f0becbcded732d9633b7ca4e34`.
The initial scope is 33 remaining experimental assets, excluding the 22 already
mapped experiment assets. Other batches remain pending until individually
recorded. Source/SDK validation is separate from numerical device acceptance.

## E1: cumulative sum, random, Hadamard and split-K

Five complete original files retain all factories, helpers, configurations,
references, CLI flags, seeds, thresholds and exit behavior. Their original
runtime matrix is preserved:

- GDN cumulative sum: all 20 original rows, both reverse directions, fragment
  selection, layout, aligned/tail and odd-head paths.
- KDA cumulative sum: all 28 scalar/vector local/global rows, four scaled rows
  and four variable-length dispatcher rows, retaining the original sequence
  bounds `[0,45,118,160]` and chunk-index validation.
- Random 1D: original defaults M=1024, block size 128, seed 42, both LCG
  constants and exact integer reference. All CLI flags remain.
- Hadamard: original batch 4, dimension/block size 2048, float default and
  complete alternative dtype and cross-block algorithms.
- Split-K GEMM: both original 128-cubed/split-2 and 1024-cubed/split-4 requests,
  seed 42, zero-initialized accumulation and original 1e-2 thresholds.

The 60 original runtime rows produce 28 unique native constructor requests.
Complete shape/control tracing records 34 native constructor events and
34 original Torch fallback boundaries; four variable-length rows each visit
three sequences. Metadata placeholders never run an original golden comparison
or establish fallback numerical correctness. Fallback paths remain the complete
original implementation and are not counted as native or device coverage.

Imports do not parse arguments, seed RNGs, allocate tensors, clear caches,
compile or launch. Actual clears remain in their original runtime order,
requiring an explicit dedicated `TILELANG_CACHE_DIR` whose resolved cache path
matches. GDN and KDA retain both original clear calls. The original split-K
cache disable is restored to the caller on exit. Device registration occurs
inside main. Random exposes its original parser defaults at import, and the
complete original parser reconfigures the same globals and derived
`TOTAL_BLOCK_SIZE` before executing its factory; a custom CLI regression checks
that contract. The original random CRLF source is tracked by its raw source hash;
the port uses canonical LF without changing its AST.

Every unique native request passed complete lowering and actual SDK compilation
on both sides. Generated source retains pure unused scalar/default-subcore
bindings, lexical loop/core-variable names and independent allocation ordering.
Two KDA requests retain an extra empty C-core control loop. Full remaining
payload, synchronization, operand uses, host launches and allocation maps are
checked separately; additional unused shift/division arithmetic is bounded in
the actual original launch/loop domain. No pass or emitted source is rewritten
to force equality. Numerical device and performance validation remains pending.

## E2: grouped GEMM

Three complete original sources retain their original input construction,
metadata generation, references, CLI options and return behavior:

| Entry | Original default CLI |
|---|---|
| Backward | Groups 64/128; M=N=512; block M/N/K=64 |
| Forward | Groups 64/128; K=output dimension=8192; blocks 64/128/64; transposed reference flag remains available |
| Padded forward | Groups 64/128/256; K=N=4096; blocks 64/128/64; optional original profile mode |

All three keep their original seed behavior: no seed is added. The backward
and ordinary forward only print mismatches, without raising a failing exit;
the padded forward retains its original assertion-based verification. Ordinary
forward's original partial-group limitation and unused valid-row field remain
in source. The original five forward and four padded-forward helper test
requests are retained and separately marked as helper entries rather than the
three default CLI runs. Their source/SDK results do not establish tail-case
numerical support.

Backward's original module cache clear requires the dedicated cache contract.
Padded forward's original disable-cache mode is restored to the caller. Runtime
registration and argument parsing occur in main; the backward reference's
original free `block_K` binding is assigned globally by the same CLI before use.
Original tuple arguments are preserved explicitly in the source request ledger.

All twelve default/helper requests passed complete lowering and SDK compilation
on both sides. Raw source retains additional read-only metadata loads on AIV;
ordinary forward also reads the previously unused third metadata column on C.
This third-column address set was not read by the old emitted source. Each
original complete int32 metadata table and every added grid/row/column read is
checked separately, with the original independent input and no kernel writes
or output alias. The added loads are not described as pure scalar operations
or removed from the compiled source. Full remaining payload/synchronization and
physical allocation maps are checked independently. These bounded source checks
are separate from pending numerical device and performance acceptance.

## E3: chunk gated delta rule

Both complete developer and expert sources preserve their original variable-length
wrappers, gating, initial/final state, new-value outputs, workspace decorators,
manual/automatic synchronization, references and 1e-5 precision assertions.
The developer default uses sequence length 2048 with H=8/Hg=4/K=V=128 and
seed 41. The expert default retains all eight original cases and their original
seeds, including the original 2048 case and the seven C++-test-derived rows.
All original CLI flags and failure propagation remain unchanged.

Nine original default runtime rows are distinct from 27 explicitly labelled
source probes of the original `--use_g`/`--use_initial_state` Boolean options.
They produce twelve unique native constructor requests. Complete original test
and wrapper shape/control runs reach the native constructor without executing
an original golden comparison or a device operation. Optional non-CLI helper
modes, such as disabling final-state or new-value storage, remain in the complete
original implementation and are not claimed as validated by this matrix.

All twelve requests passed full lowering and actual SDK compilation on both
sides. The larger expert source differences include repeated pure integer
conditional temporaries. A comparison-only normalization checks unique
single-assignment temporaries and complete branch expressions, preserving
memory-read effect positions and physical allocation maps. Eleven sources
match after these declared name/binding differences. Pure integer branches and
unused arithmetic are bounded in the actual sequence/grid/loop domain.

The remaining developer gating request retains an additional C-side conditional
read of `g_last` and a `PIPE_ALL` barrier. Its 256 actual read points lie within
the original independent float32 `g_pad[8,2112]` and match the old and new AIV
last-value read addresses. The added read and barrier remain in the compiled
source and are not described as pure scalar operations. These source witnesses
are limited to the recorded original requests. Numerical device and performance
validation remains pending.

## E4a: MLA decode and layered tests

The complete original kernel and layered test module retain both smoke requests,
three L0, nine L1, three L2 and five Boundary entries, including the original
132-batch/128-head/8192-KV large request. All references, seeds, retry conditions,
three-attempt limit, precision checks and exit behavior remain unchanged.
Twenty-two original entry rows produce nine distinct successful native
constructors and two original constructor assertions. Both assertions retain
same rejection for 100 heads and zero actual KV length. The original runtime
wrong-input-dtype check is retained and has not been exercised on a device.

Imports do not change the default device, allocate CPU or NPU tensors, or disable
caches. The kernel smoke body uses a scoped NPU default-device context, preserving
the entire original body and restoring its caller's state. The layered test loads
its exact private sibling instead of changing `sys.path`. Its deterministic
float32 column indices remain CPU values 0 through 127, created lazily only at
the two original use sites. A separate real CPU check verifies values, cache
identity and unchanged CPU RNG state. Original runtime cache disabling remains
in the same execution order and restores the caller's cache state on exit.

Nine full source pairs passed actual SDK compilation. All complete physical
allocation maps, including the original overlapping aliases and offsets, match.
After comparison-only pure binding and lexical loop-name normalization, the
remaining memory payload and original synchronization match. Generated source
retains four additional static `PIPE_ALL` waits around pure control regions;
the large request additionally expresses those waits in fourteen-iteration
empty wave loops. The recorded integer and loop domains are bounded. These
real added waits remain in the compiled source, and are not described as
numerical, deadlock or performance equivalence. All device validation is pending.

## E4b: paged MLA decode and complete layered suite

The full paged decode kernel and its original layered test module preserve the
smoke entry, five L0, eleven L1, two L2 and five Boundary rows, including the
original batch-128/8192-KV request. All original host paged-data preparation,
block sorting, defensive index checks, pre-scaling, golden reference, precision
dual gate, nonblocking levels and CLI exit logic remain. The original two L2
runtime dtype/shape rejections remain device-pending; constructing their valid
kernel does not validate those errors. Imports use the private language/layout
and exact sibling module without changing caller search paths; explicit NPU
registration occurs only in main.

The 24 original entry rows produce fourteen distinct complete source pairs,
all compiled by the real SDK. Raw source retains additional AIV reads of the
independent read-only int32 block table and additional `PIPE_ALL` waits. The
large request expands 832 added read declarations; these are real GM effects,
with performance pending. Each original complete identity/sorted table, physical
20-C/40-V grid, active-tile guards, expanded wave/chunk, actual index and scalar
result is checked, including rejected blocks in the final wave. New AIV address
and result sets equal the original C sets; the old AIV source did not read the
table. These witnesses assume the original immutable, independent input and do
not establish arbitrary external alias or mutation safety.

Full physical allocation and original alias maps match. Comparison-only binding
and lexical name normalization separates real added reads and wait-only control;
the remaining payload and non-`PIPE_ALL` synchronization match, and all original
`PIPE_ALL` anchors are retained. The compiled source is never rewritten for
comparison. Numerical correctness, boundary verdicts, deadlock and performance
validation remain pending.

## E4c: persistent MLA decode and complete layered suite

Both original files retain the complete unified kernel, host dtype/device/shape,
contiguity and fractal/L0-budget validation, metadata-only views, references,
optional einops fallback, seed and precision/exit rules. The original smoke,
five L0, ten L1, two blocking L2 and four nonblocking Boundary entries remain.
The full L2 metadata assertions can be checked without allocating device inputs;
that is separate from running the original NPU tests or precision comparisons.

The original runtime queries `torch.npu.get_device_properties("npu")` for its
core count, with exception fallback 20 in the layered suite. That behavior stays
unchanged. Initial source evidence used the original fallback 20 as a compiler control;
that earlier evidence did not measure the real hardware default core count. The 22 original entry shapes/control paths yield
eleven successful native requests in that control and two original early host
metadata rejections. All eleven source pairs passed actual SDK compilation.
A later read-only physical2/visible0 property capture reports 24 Cube and
48 Vector cores. Eleven additional complete source pairs and real SDK compiles
now cover the same original entry matrix with core 24. The fallback-20 evidence
is preserved separately; neither source domain establishes a device verdict.

Original default-device bodies use a scoped NPU context and restore the caller,
while parsing, cache-disable timing and all original body/exit logic remain in
order. Cache state restores on failure or `SystemExit`. Imports are tensor- and
runtime-free, with private language/layout and exact sibling loading. Real core
queries remain in actual runtime; host regression tests separately exercise an
explicit fake property and the original fallback without selecting a device.

Raw source retains four additional static `PIPE_ALL` waits and large unused
scalar/control expansions. There are no added scalar `GetValue` reads in these
pairs. Bounded dead integer/control regions, full physical allocation/alias maps,
original wait anchors and remaining payload/non-`PIPE_ALL` synchronization are
checked in the source control domain. Raw SDK inputs are unchanged. Device
numerical verdicts, actual execution under the measured default, deadlock and
performance remain pending.
