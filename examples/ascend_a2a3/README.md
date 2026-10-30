# Ascend A2/A3 examples

These examples use `tilelang.ascend.a2a3.language`. Build TileLang with
`USE_ASCEND=ON` and configure CANN and Torch-NPU for an A2/A3 device.

Run the original scripts from their migrated directories, for example:

```bash
python examples/ascend_a2a3/gemm/example_gemm.py
python examples/ascend_a2a3/gemm/example_gemm_pto_developer.py
python examples/ascend_a2a3/cross_entropy_loss/example_cross_entro.py
```

Original CLI arguments, inputs, references and pass criteria are preserved.
The original cache-clearing calls use `tilelang.disable_cache()` because the
modern framework no longer provides `tilelang.cache.clear_cache()`.

Run companion tests with pytest, for example:

```bash
python -m pytest examples/ascend_a2a3/pos_embedding/
python -m pytest examples/ascend_a2a3/xllm_kernels/ examples/ascend_a2a3/tile_kernels/ -m 'not ci_skip'
```
