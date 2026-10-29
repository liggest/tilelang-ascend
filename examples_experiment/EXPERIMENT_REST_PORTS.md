> 历史迁移摘要：以下来源、源码/SDK及设备状态属于目录调整前记录，当时目标前缀为 `examples_experiment/ascend_a2a3/`。2026-10-30 后当前入口前缀为 `examples_experiment/`，逐文件搬移及保真证据见 `maint/ascend_a2a3/experiment_layout_move.json`；本次路径调整不把历史状态变成完整 CI 已通过。

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

## E5a: NSA decode and complete layered suite

Both original files preserve the entire factory and host row-index expansion,
complete-block validation, mask/zero-block-count algorithm, five-workspace
contract, references, precision rules and CLI exits. The smoke, three L0,
seventeen L1, three blocking L2 and five nonblocking Boundary rows remain.
This includes the original typical D=128/S=16 and S=8 cases and all padding,
zero-count, special-value and runtime dtype errors. The 29 original rows map
to eleven positive constructors and two original assertions (HQ=8 and HQ=17).
The float32 runtime rejection remains separate and device-pending.

Complete dispatch tracing records constructor parameters and original metadata
options without executing any tensor generator, golden or comparison. Its
recorder cannot convert to a Boolean verdict, so the trace cannot report an
original test pass. Imports use the private language and exact sibling module;
registration and cache disabling run only in main, with caller cache restoration.

All eleven full source pairs passed actual SDK compilation. Remaining payload,
operands and synchronization match after only core-identifier, unused integer
binding and allocation-declaration order normalization. Physical allocation
changes are limited to the order of the block-count matrix and scratch buffer
inside one exclusive UB byte interval; both slots remain mutually disjoint and
outside all other buffers, preserving the entire original alias graph. There
are no added scalar reads or waits. The same AIV row-index scalar read remains
after its original DMA and barrier. Bounds for the new unused arithmetic apply
to the original complete-block or padded-zero row indices. Numerical, runtime
negative, boundary and performance validation remain pending.

## E5b: NSA forward and complete layered suite

The complete original kernel and suite preserve host pre-gathering, all
factories, helpers, flags, precision rules, matrices and exits. Twenty-eight
original rows include the smoke, two L0, seventeen L1, three L2 and five Boundary
entries. The smoke remains a shape/finite check, distinct from the layered
golden. The original `core_num=None` hardware query is unchanged; source
sampling passes the independently measured core24 separately from original
kwargs. The original divisor/single-core selection and explicit `threads=1`
produce the actual `MIX_AIC_1_1` grid, rather than assuming two AIV per C.

Sixteen positive source pairs passed actual SDK compilation. One intended
odd17 request is rejected by both original and current native codegen with the
same CompareScalar 544-element/2176-byte alignment error, before SDK or launch.
The original odd-G `python -c` oracle had a missing-`__file__` setup error
and accepted arbitrary nonzero exits. Its complete child literal is preserved
as inactive `original_odd_gqa_child.txt` beside the test. The corrected check
uses exact sibling paths and accepts only one structured `target_compile`
InternalError for the validated 544-element/2176-byte CompareScalar boundary,
with child status71. Setup/other errors, timeout, signal, malformed markers and
unexpected compile acceptance fail; the child stops before all H2D/launch.
This compile-boundary check makes no runtime hardware rejection claim. The original
float32 runtime negative and short-sequence helper assertion remain distinct.

Generated source retains extra C-side read-only `BlockStarts` loads and actual
`PIPE_ALL` waits. Complete original CPU metadata generation and each actual
selected-grid C/AIV reader are checked, with explicit one-to-one task ratio,
all pipeline iterations, full table values and scalar results. New C address
sets equal the original AIV sets; old C had no such reads. Inputs are independent
read-only metadata, with external aliases/mutation outside these witnesses.
Remaining payload and non-`PIPE_ALL` synchronization match, and original wait
anchors remain. Physical changes only reorder two mutually exclusive UB slots
inside the same exclusive interval, with the original complete alias graph and
capacities preserved. Extra effects stay in the SDK source; device numerical,
oracle runtime, deadlock and performance acceptance remain pending.

## E5c: complete NSA variable-length forward pair

The kernel and layered test retain all original factories, helpers, decorators,
configurations, seeds, precision rules, exits and twenty-two original rows:
one smoke, one L0, fifteen L1, two L2 and three Boundary entries. The original
literal `core_num=20` is unchanged. The runtime wrapper registers TorchNPU and
restores the caller's cache mode; the original smoke's global Torch binding is
retained when its CLI body is moved into `main`.

Complete original CPU preparation, before any native constructor, preserves
all three metadata tables and RNG transitions. `BosPerToken` is `[C]`;
`IsSafe` and `BlockCounts` are `[C,Hkv]`, with `IsSafe` storing actual block
start offsets. Thirteen unique native requests include twelve positive source
pairs and one matching original float64 constructor rejection; all twenty-four
positive generated sources passed actual SDK compilation. The odd-group host
assertion still occurs before seeding or allocation. These are compiler and
CPU-control checks, distinct from the original runtime/golden verdicts.

The original large varlen request exposed native stack exhaustion in the
backend's flat Bind/Assert lexical-tail memory-planning traversal. Its private
iterative adaptation preserves the pre-fix TIRX planned IR byte for byte,
compared with an isolated diagnostic process using a larger stack. The
request, pass sequence and normal production stack limit are unchanged. This
native fix is a prerequisite; the original failure records remain separate.

Raw source retains additional read-only metadata loads and `PIPE_ALL` waits.
For the three large requests, each new source contains 3284 additional unused
metadata-read declarations and 3312 static `PIPE_ALL` calls versus the
original twenty. Actual C20/V40 readers, active guards, complete original CPU
metadata, integer-control ranges, allocation capacities and alias graphs are
checked separately. Remaining payload and non-`PIPE_ALL` synchronization match
after report-only separation of those real extra effects and pure bindings.
Only two disjoint UB slots exchange positions inside the same exclusive byte
interval. No SDK source removes the reads or waits. Device numerical,
concurrent-mutation/alias, deadlock and performance acceptance remain pending;
extra operations are not claimed harmless for arbitrary inputs or timing.

## E5d: complete NSA forward/backward pipeline pair

The backward example and its layered test preserve both complete original
factories (`nsa_fwd` and `nsa_bwd_single`), all five hybrid/tail-mask flags,
helpers, golden, thresholds and exits. Twenty-six original outer rows contain
one smoke, one L0 with all four verdicts, eight L1, five L2 and eleven
nonblocking Boundary entries. The L0 still invokes one forward and two
backward kernels; source-request deduplication does not remove runtime calls.

Twenty-one unique positive factory requests have full old/main source and
actual SDK compilation evidence on each side. Two original BS7/6 forward
constructor assertions are checked separately before lowering; they are not
runtime hardware negatives. Complete original CPU generation, special-value
injection and host validation/mask preparation match in input bytes and RNG
transitions for all twenty-one positive outer rows, stopping before H2D and
reference evaluation. The original host computes and returns `block_mask`,
but the forward kernel actually consumes `BlockIndices`; the source port does
not invent a mask consumer or change this original behavior.

All remaining payload, memory reads/writes and synchronization tokens match
after separating only the actual `GetBlockIdx` local identifier, declaration
order of identical allocation maps, unused call-free bounded integer bindings
and whitespace. Every allocation's type, extent, offset, capacity and complete
overlap graph is unchanged. No additional GM read or `PIPE_ALL` is hidden by
this comparison. Original host shape/empty-sequence assertions remain before
mask allocation. The original NS=1 host gate and all Boundary WARN behavior
remain; dtype runtime checks and every numerical/deadlock/performance verdict
still require the original device matrix. None is claimed from source/SDK.

## DSA sparse finetune backward (E5e)

The full original standalone and layered test retain S=2048, H=16, D=128, topk=64, seed=42, offsets `[0, 1023, 2048]`, automatic fast/generic dispatch, workspace construction, autograd reference, bf16 dual precision gate and exit checks. Importing either file does not register an NPU backend, compile or clear a cache. Running the cache-clearing standalone requires an explicitly dedicated `TILELANG_CACHE_DIR`, as above. Numerical and performance validation remain pending.

The example-local cached launcher consumes the current four-field shape descriptor only for direct shape carriers (`kind=0`, `scale=1`), preserving descriptor order and int64 host slots. Unsupported descriptors or inconsistent carriers use the complete Kernel validator. It keeps original tensors and actual flatten views alive and invalidates a raw id-cache hit when shape, stride, storage offset or data pointer changes. Noncontiguous inputs go to the complete validator as original objects. These checks add CPU work; the class's retained TA timing figures are historical, not measurements for this port. The raw fast path retains the original NPU dtype/device and no-concurrent-mutation preconditions. Stream lookup is delayed until an actual call needs it, preserving the original `None` fallback.

Seven source controls cover both original fast layouts and cast at explicit grid 24 (historical measured hardware property) and 20 (original constructor/fallback), plus the generic factory. No current hardware property query was executed for these samples. The generic/shared layouts are additional public API source probes, not extra original numerical cases. All 14 actual Bisheng compilations pass, alongside eight matching original constructor assertions. Extra raw `PIPE_ALL` waits remain in the migrated source; finite S=2048 reader/integer/capacity/alias witnesses do not establish arbitrary dynamic-input, deadlock or performance equivalence. The original layered test casts after H2D and computes its reference on NPU; CPU metadata checks do not replace that contract.

## Regular sink-forward dependency for E6

The two complete regular files live in `examples/ascend_a2a3/mha_sink_fwd_bhsd/`; they are dependencies of the original experimental sink-backward reference and do not add to the original 33 experimental-asset count. The original Expert factory retains its output/workspace indices, four disabled automatic passes, NUM_CORES=24, manual flags, layouts, stages and cross interval. Standalone and layered/performance entry points retain all original input/reference/precision/exit logic. Default device allocation is scoped and cache mode is restored after the CLI; real pytest entries register the NPU backend only when called.

All 26 original standalone, L0/L1/L2/Boundary and perf-preset rows map to 17 unique positive source requests; both sides compile all 34 SDK outputs, plus three matching original constructor guards. Source comparison separately proves bounded integer/SSA controls and equivalent mask-DMA predicates over each fixed physical/task/KV domain. Original logical allocations/capacities/alias graphs are preserved; physical UB slot permutations have identical exclusive byte unions and are reported explicitly. Actual manual synchronization and memory payload remain ordered identically after those report-only comparisons. The independent CPU mask check is host metadata only. Device precision, original non-blocking warning results, default performance benchmarks and their exit verdicts remain pending.

## GQA forward/backward component (E6a)

The complete `example_gqa_bwd/` pair retains all six native factories, the original Autograd wrapper, original logical-D129 padding/scale and zero-stride-gradient regressions, helpers, references, 169-line precision gate, all original retry/non-blocking logic and CLI exits. The CV factories retain the original three enabled pass keys and vector factories the original two; comments in the source remain historical. Private imports include the regression attention alias and generated msprof child's exact sibling-file loader. Generated-child tensor generation, launch order, profiler flags, warmup count, timeout and result/cleanup logic are retained. Runtime registration is delayed; standalone device scope and caller cache state are restored after execution.

All 29 original standalone/L0/L1/L2/Boundary/regression/warmup/default perf/msprof sampling rows map to 108 positive full-source pairs and two original unaligned-shape constructor assertions. Both sides compile all 216 real SDK outputs. Complete source effect comparisons separate only bounded unused integers, empty controls/core shells and lexical names; original payload/read/write/synchronization is identical. Twelve two-slot UB permutations are recorded with the same exclusive byte union, capacity and alias graph. These checks do not establish numerical/device/performance or arbitrary CLI-parameter support. The L2 float32 runtime-input mismatch remains distinct from a successful float16 constructor; no runtime rejection, atomic retry result, real msprof capture, Autograd gradient or precision verdict is claimed by this source package.

## GQA sink variable-length forward/backward (E6b)

The complete `example_gqa_sink_bwd_varlen/` pair retains both native factories, full host pipeline, original reference/precision, all original special sink and value-range inputs, runtime dtype checks, non-blocking negative/boundary handling, and CLI exits. The original determinism loop still runs three times for each of seven L0 configurations and skips its original large default. Standalone, CLI and directly called layered helper scope their default device/cache state; actual pytest entry registration is delayed.

The 51 original controls include one determinism skip. Fifty runnable controls deduplicate to 44 positive source factory requests; all 88 TA/main SDK compiles pass. The 129/131/empty-length assertions occur in the original host prefix; the original L2 label192 is divisible by64 and is retained as a positive source configuration. Positive constructor results do not prove float64 runtime rejection or any original boundary/precision/determinism result. Actual CPU Cu int32 tables and RNG match, without running original random inputs or references.

Complete source comparisons preserve all nine scalar reads and twelve `PIPE_ALL` calls. Forward `row_pos` aliases retain the exact float32 cast at their scalar compare uses; main also retains an additional unused pure cast, reported with its original metadata/integer bounds. Actual configured C/AIV Cu reader addresses/values, float32 exactness domains and complete capacity/alias graphs are checked. Nine two-slot UB permutations retain the same exclusive byte union. These are source/metadata proofs, not physical core measurements, numerical/deadlock/performance or arbitrary external metadata guarantees.

### MHA sink backward（E6c）

完整 backward 模块与原分层测试保留4个 backward factory、原 Autograd 五核调用链、20个 standalone/L0/L1/L2/Boundary 入口控制、4次原数值重试和原退出合同。forward 复用已迁 regular 私有依赖；固定 TA 不含的 optional perf 模块继续原 WARN-return 边界。61个阳性 full-source 请求及122次实际 SDK 编译通过；3个原未对齐 K1 构造断言两侧一致。此处不代表数值、Autograd、workspace零化或性能已通过。

原导入时全局 `torch.empty` monkeypatch 改为仅原 K1 `_zeroed_npu_workspace` 作用域内的 entry allocator 委托，保原 NPU 分配后 `zero_()` 顺序和线程局部条件；K2/K3/K4/K5 与 golden 不置标志。组件 RLock 支持同组件嵌套/异常/并发，不提供不同模块实例或外部 Torch 替换的全局协调保证；退出只按身份恢复，不覆盖外部新替换。CPU 协议测试不使用 fake kernel/golden 输出。真实 NPU K1 workspace 与完整数值仍待独立设备验收。

### Seer block sparse attention（E6d）

完整原单文件、1个 factory 与2个 topk/threshold mask helpers、两个原 NPU 输入/golden 请求均保留。2对完整 source 和4次实际 SDK 编译通过；相同实读位置的 int8 mask→int32 显式提升与原零比较在完整256有符号值域等价，4个实际 scalar read/5个 PIPE_ALL 及精确物理分配/alias/容量保留。真实 CPU mask helper 的4个检查只验证该 helper，不代替原 NPU随机输入/golden/数值。原 `Test passed!` 字面保留，符合原 bench 的大小写兼容regex；未运行该CLI/设备或性能。

该账本至此候选覆盖原剩余33个实验来源（另有2个regular前向依赖），不代表Root已全部集成或整个生态/CLI/CI/设备验收完成。各来源验证状态仍分别记录。
