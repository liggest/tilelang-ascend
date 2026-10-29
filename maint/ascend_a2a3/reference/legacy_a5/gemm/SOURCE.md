# Inactive original A5 FP8 GEMM source

This complete unmodified source is an **inactive historical reference**, not an
A2/A3 capability, active example/CI entry, modern A5 port, feature deferral or
numerical validation. The `.py.txt` suffix excludes it from Python discovery.
Do not execute this archived program as part of migrated A2/A3 validation.

- Source: `tile-ai/tilelang-ascend`
- Fixed commit: `0deb8b93da11b7f0becbcded732d9633b7ca4e34`
- Original path: `examples/gemm/example_gemm_fp8_pto.py`
- Git blob: `d2c3b5ffe8cd70561c74af91ddc0cea4f9fc2e2b`
- Original and archive SHA256: `8ff403eb509735350cfac2ca243a0ec0460859e1c2e6bac84bdc098aab6a0c7a`
- Bytes: `3020`

The original explicitly documents that FP8 TMATMUL (`float8_e4m3` / `float8_e5m2`)
requires the A5 Cube core and is unavailable on A2/A3. Its command description is
`NPU FP8 GEMM Kernel (A5 PTO)`; the factory uses `target="pto"` with
`e4m3_float8` / `e5m2_float8` input types, not an A2/A3 implementation.

Before argument parsing/factory construction, the source checks
`determine_platform() != "A5"`, prints SKIP with an old success-marker phrase,
and exits0. That non-A5 branch is a skip, not a launch or precision pass.
The original also clears cache before that platform gate: no CLI, cache clear,
platform probe, old alias, compiler or A5 device was run for this archive.

Original A5 defaults/branches are preserved in full:1024³, e4m3/e5m2 selector,
128×256×64 tiling, seed0 and the original1e-2 assertion. This records source,
not a promise to run those settings on modern A5. The TA source stays untouched.
Final A2/A3 adoption/CI mapping must retain this asset as inactive, without
routing it through a same-named modern A5 example or counting the old skip green.
