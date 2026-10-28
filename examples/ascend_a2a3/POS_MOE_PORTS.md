# Remaining RoPE and MoE source ports

The independent `maint/ascend_a2a3/ecosystem_pos_moe_ports.json` records original
TA sources fixed at `0deb8b93da11b7f0becbcded732d9633b7ca4e34`.
The twelve existing Python ports under `pos_embedding/` and the existing
RotaryPositionEmbedding README are excluded from this ledger.

## Half and interleaved RoPE

The complete `pos_embedding/RotaryPositionEmbedding/rope_half_interleaved.py`
retains its original factory, in-place partial/full rotation, dtype/layout
branches, tiling, original configurations, CANN reference and precision gates.
Original CLI selection remains:

- Default `--level l0`: six blocking cases.
- `--level all`: six L0 and twelve L1 blocking cases, four original nonblocking
  negative cases and six original nonblocking boundary cases.
- `--shape`, `--layout`, `--dtype` and `--perf --side tl|cann`: original single
  case and profiling entries.

The complete original main is retained as `_original_main`. Device registration
and device-object creation occur at runtime. Reference/performance helpers
register their backend lazily when called. Import does not allocate tensors,
compile, seed RNGs, parse arguments or disable caches. Original disable-cache
behavior is restored to the caller's state on both success and failure; no
cache clear is added. All original seeds, thresholds and failure exits remain.

The complete original `bench.sh` retains all thirteen shape/layout/dtype rows,
its 26 TileLang/CANN commands, warm-up 5, launch count 20, Task Duration parsing,
summary collection and exit behavior. Its repository path is adjusted for the
namespaced directory. It requires the original `msprof` environment; the source
checks for that executable before handling `--list`. A mocked command-matrix
test validates orchestration without profiling or passing a kernel comparison.
The script's final printed result is not a numerical acceptance signal.

The 37 original legal/boundary/profile entries produce 35 unique native
constructor requests. All have complete source and SDK validation on both sides;
four original invalid-input requests are rejected by the original host wrapper.
Generated source differences are retained. Six requests contain additional pure
C-core control loops and `PIPE_ALL` barriers, without memory reads/writes, flags
or vector operations. Three requests have additional index/guard differences:
actual launch/loop domains verify the observed modulo distribution and guarded
unreachable tail paths. Full allocation sizes and original logical overlaps
are checked separately from payload. These are bounded source witnesses;
numerical device and performance checks remain pending.

## MoE status

The following independent five-file component records the original MoE token
kernels/classes and shared utility; they are counted separately from RoPE.

## Complete MoE token classes and shared utility

The five original files under `moe_token_permute/` retain all four classes,
all native builders and compile dispatch branches, workspace/output decorators,
manual synchronization, padding, original references and default Torch precision
thresholds. Imports defer NPU registration to the original parameterized tests
and explicit main. Each complete original CLI remains callable directly; no
cache clear or altered cache policy is introduced.

The full original default matrix uses float16, bfloat16 and float32 with seed 42:

| Class | Original scenarios per dtype |
|---|---|
| `MoeTokenPermute` | N=16/H=8/K=4/experts=4, standard and clip-to-10; original PTO and workspace |
| `MoeTokenPermuteGrad` | N=16/H=16/K=4/experts=4, standard and clip-to-10 |
| `MoeTokenUnpermute` | No-probs N=16/H=8/K=4, weighted N=8/H=4/K=2, original round-trip N=16/H=8/K=4 |
| `MoeTokenUnpermuteGrad` | N=8/H=4/K=2, with/without probs, including both original gradient checks |

The 27 original class-constructor scenarios produce 21 unique native builder
requests. The first class keeps explicit PTO; the others keep their original
automatic dialect. Constructors retain their actual padding, dtype and strategy
selection. Every original builder remains in source, including branches not
selected by the default matrix; the ledger explicitly distinguishes those from
validated default requests. `moe_token_utils.py` retains its complete original
CPU self-test and helpers.

The four original test summaries only print precision failures and do not raise
or exit nonzero. This behavior is preserved and tested; exit 0 is not numerical
acceptance. Native request checks intercept only the JIT compile boundary to
return the real PrimFunc, then perform actual complete lowering and SDK compile.
They neither load generated kernels nor compare placeholder outputs to golden.

All 21 default native requests passed source and SDK checks on both sides.
Generated sources retain allocation/name/unused-binding differences. The six PTO
forward requests also retain additional reads of the same local int32 `wp_ub[0]`
for a conditional copy count. The original integer guard `wp < output_rows`
implies `wp <= output_rows - 1`, so the active count is one in both sources;
no intervening local-buffer write is introduced. Additional UB reads remain
explicit in the evidence. These source witnesses do not establish arbitrary
aliasing, device correctness or performance. Numerical device checks remain
pending for the complete original CLI matrix.
