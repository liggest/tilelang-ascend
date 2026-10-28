# Remaining original activation examples

These five complete assets retain the fixed TA source at
`0deb8b93da11b7f0becbcded732d9633b7ca4e34`. Their source hashes, original
requests and validation scope are in
`maint/ascend_a2a3/ecosystem_activation_ports.json`.

| Original file | Original factory requests | Original precision check |
|---|---:|---|
| `gelu_grad.py` | 6 | NPU GELU backward, `rtol=1e-3`, `atol=1e-2` |
| `gelu_mul.py` | 2 | Torch tanh GELU times the second half, `rtol=atol=1e-2` |
| `swi_glu.py` | 4 | Torch SiLU times the other split, `rtol=atol=1e-2` |
| `swi_glu_grad.py` | 7 dynamic-shape requests | NPU SwiGLU backward, `rtol=atol=1e-2` |
| `swi_glu_v2.py` | 7 dynamic-shape requests | NPU SwiGLU forward, `rtol=atol=1e-2` |

The complete original factories, decorators, flags, shapes, split dimensions,
BF16/float branches, references and assertions remain. Dynamic requests retain
their original runtime `M,N`; repeated compilation arguments do not remove
original host cases. The backward script originally has no manual seed call;
none is added. The original single outer iteration and its printed labels are
also retained.

Importing a module does not clear caches, seed RNGs, select a default device,
compile or run the example. The original runtime lives in `main()`, with
explicit `torch_npu` registration. The two original global default-device
setters use a scoped public `torch.device("npu")` context over the same host
segment, preserving the caller context on success or failure.

Source-only lowering, actual Bisheng compilation, host isolation and an actual
`USE_ASCEND=OFF` boundary have been checked. Device numerical execution remains
pending; these checks are not numerical or performance acceptance. Generated
dynamic sources contain additional scalar-only AIC control and local barriers
compared with the original TA output. This difference is retained and recorded
for the later original device matrix, without changing the source algorithm or
removing barriers to obtain matching source text.

After the specific asset is accepted for device validation, run its complete
original matrix from the repository root with a dedicated cache, for example:

```bash
(
  export TL_PLATFORM=A3
  export TL_RUN_MODE=npu
  TILELANG_CACHE_DIR=$(mktemp -d /tmp/tilelang-a2a3-activation.XXXXXXXX) || exit 1
  export TILELANG_CACHE_DIR
  python examples/ascend_a2a3/activation/gelu_grad.py
)
```

Each script really clears only its explicitly selected cache. An existing
caller override pointing at a different resolved cache is rejected. Do not
share this cache with another concurrent example. The original large shapes
and precision thresholds are not replaced with small smoke kernels.
