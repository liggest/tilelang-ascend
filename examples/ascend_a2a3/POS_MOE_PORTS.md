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

The four original MoE token kernels/classes and shared utility form the next
five-file closure. They are not counted as ported by this RoPE component.
