# Original sources

Fixed TA commit: `0deb8b93da11b7f0becbcded732d9633b7ca4e34`.

| Original path | SHA256 |
|---|---|
| `examples/autotune/example_gemm_autotune.py` | `7eb5b356f3deb660202c381695f0bc9092197d2e813dfd3ab849b83ca0f11da3` |
| `examples/autotune/example_gemm_carver.py` | `7626ce7a9b7b5e7fdceb8c46671c4c79083aa3b668ee32634ca5bf2a045292f3` |

The GEMM factory bodies and original configuration/hint mappings are retained.
The migration uses the A2/A3 language namespace, import-safe CLI entry points,
logical-shape input supply, and the mainline target-local kernel cache API.

Ascend Carver support is a later evaluation item. Its original recommendation
flow is retained in `example_gemm_carver.py`; no Ascend policy is copied and no
static configuration replaces the unavailable recommendation capability.

The experimental blocksparse GEMM and convolution autotune sites use direct
configuration generators, not Carver. Their active test coverage is a separate
migration item and is not included in the Carver deferral.
