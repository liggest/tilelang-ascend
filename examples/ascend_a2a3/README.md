# Ascend A2/A3 examples

These examples use `tilelang.ascend.a2a3.language` and preserve the kernels,
compiler options, and accuracy thresholds from TileLang-Ascend source commit
`0deb8b93da11b7f0becbcded732d9633b7ca4e34`. They are separate from main's A5
examples under `examples/ascend`.

Build TileLang with `USE_ASCEND=ON` and install the compatible CANN and
`torch_npu` dependencies before running on an A2/A3 device. Target detection must
confirm A2/A3; an unknown platform is not assumed to be A3. An explicit
`TL_PLATFORM=A2` or `A3` selects the corresponding known hardware configuration.

Run from the repository root, giving each run a dedicated cache directory:

```bash
TILELANG_CACHE_DIR=/tmp/tilelang-a2a3-gemm \
  python examples/ascend_a2a3/gemm/example_gemm.py
TILELANG_CACHE_DIR=/tmp/tilelang-a2a3-gemm-pto \
  python examples/ascend_a2a3/gemm/example_gemm_pto_developer.py
TILELANG_CACHE_DIR=/tmp/tilelang-a2a3-flags \
  python examples/ascend_a2a3/compile_flags/compile_flags_example.py
```

Both GEMM examples default to `M=N=K=1024`. The compile-flags example defaults
to `M=N=256` and checks both its original custom flags and the subsequent default
flags. Each retains the original `rtol=atol=1e-2` check.

Importing an example does not parse arguments, clear caches, compile kernels, or
run NPU work. Those operations live in `main()`. Running a script retains the
source's explicit cache clearing, adapted to the actual A2A3 cache singleton
because main no longer exports `tilelang.cache.clear_cache()`. Its clearing
operates on the selected cache namespace, so use a dedicated directory as above.
`torch_npu` is imported inside `main()` to support direct script execution even
when PyTorch backend autoload is disabled.

The first adopted tests live under `testing/ascend_a2a3/`:

```bash
python -m pytest testing/ascend_a2a3/ -m 'not a2a3_device'
python -m pytest testing/ascend_a2a3/ -m a2a3_device
```

Source tests require the native A2/A3 builders but no NPU. The pure Python
legacy-flag export test also runs with `USE_ASCEND=OFF`. Device tests require a
confirmed A2/A3 NPU, use a separate temporary cache per test case, and retain both original MMA cases at `8192 x 1024 x 8192`
and `rtol=atol=1e-3`. No source cases are removed to obtain a pass.

See `maint/ascend_a2a3/ecosystem_ports.json` for source hashes, reviewed API
adaptations, and the validation scope. The wider ecology manifest and blocked
candidates do not imply that the remaining examples/tests have been ported.

The raw-source tests restore the prior process cache-enabled state after each
case. MMA cache clearing stays within its per-case temporary directory and
isolated memory dictionary; it does not delete a caller's default cache.

The positional examples under `pos_embedding/` include their six original CI
companion tests. The 17 original cases cover RMSNorm, fused RMSNorm/RoPE,
mask-based RoPE, and forward/backward RoPE for float16, bfloat16, float32,
interleaved/half rotation and TND/BSND layouts:

```bash
python -m pytest examples/ascend_a2a3/pos_embedding/
```

These device tests use a confirmed A2/A3 NPU and a separate temporary cache for
each case. Original shapes, in-place behavior, references and tolerances are
retained. Importing the source modules performs no cache clearing, CLI parsing,
NPU initialization or kernel work. The broader original CI active inventory and
all pending statuses remain separate from this completed subset.

The `xllm_kernels/` and `tile_kernels/{moe,quant}/` companions retain their full
380-case source matrix. Run the original 90-case PR subset separately from the
290 full/nightly cases:

```bash
python -m pytest examples/ascend_a2a3/xllm_kernels/ \
  examples/ascend_a2a3/tile_kernels/ -m 'not low_priority and not ci_skip'
python -m pytest examples/ascend_a2a3/tile_kernels/ -m low_priority
```

The PR subset covers GDN gating, sliced/in-place RoPE, split QKV/RMSNorm/mRoPE,
MoE forward/backward, and plain/packed scale-factor quantization. All 90 original
PR cases passed on A3 without changing shapes, references or tolerances. The
290 retained nightly cases have not been executed as part of this batch.
Imports leave the caller's cache, CLI arguments and local `utils` module binding
unchanged; each test uses an independent temporary disk/memory cache. Builds
without the A2/A3 native components collect these device cases and skip them.

The xllm core-count helper retains its original default of 48 when device
properties do not provide a count. That utility default does not identify the
platform; target validation still requires a confirmed A2/A3 platform. MoE
backward requires the A2A3 flat-Bind/CV cleanup that preserves live resource
consumers while removing the dead vector reads from the automatically split C
branch.

The next source/SDK-reviewed group covers batch GEMM, SiTU GLU gradient,
DeepSeek v4 quantization/HC split/int8 GEMM/sparse attention, BHSD flash
attention and variable-length GQA. Its 16 source/test assets retain the original
matrix, references and tolerances. All 46 original factory inputs have complete
TA/main source and SDK evidence; actual device validation is still pending.

```bash
python -m pytest examples/ascend_a2a3/batch_gemm/ \
  examples/ascend_a2a3/cann-bench/situ_glu_grad/ \
  examples/ascend_a2a3/deepseek_v4/ \
  examples/ascend_a2a3/flash_attention/ \
  examples/ascend_a2a3/gqa_fwd_varlen/ -s
```

GQA L0/L1 originally return a boolean and print `PRECISION_PASS/FAIL` without
asserting that boolean. L2/boundary originally print non-blocking warnings.
A successful pytest exit alone therefore does not establish GQA precision;
inspect those original results separately. The migration retains this contract.
DeepSeek companions restore the caller's default device context after each
case, preserving their original unqualified NPU allocations within the test.
