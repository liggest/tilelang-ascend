# Original F1 fusion/tail/print sources

These four complete TA assets are migrated from
`0deb8b93da11b7f0becbcded732d9633b7ca4e34`. The precise source/destination
hashes and original factory requests are in
`maint/ascend_a2a3/ecosystem_fusion_ports.json`. Device execution is pending.

| Original entry | Complete no-argument CLI inputs | Original numerical check |
|---|---|---|
| `simple_fusion/matmul_add.py` | M=N=K=1024, tile128×256×64, float16 | matmul+add, rtol/atol1e-2 |
| `simple_fusion/matmul_add_infer_scope.py` | same original input | same original check |
| `tail_mask/example_tail_add.py` | (34,130,32,32,float), (34,130,32,32,float16), (100,200,64,128,float) | original add checks, rtol/atol1e-2 |
| `print/elementwise_print.py` | original fixed2×16/int16, tile2×16 | original bitwise-and check, rtol/atol1e-2 |

Factory bodies, decorators/flags, helpers, original CLI arguments and complete
host loops are preserved. Imports use the A2/A3 dialect. CLI/device operations
run only in `main()`. Each cache-clearing CLI requires its own explicit
`TILELANG_CACHE_DIR`; it rejects a conflicting caller cache override before
clearing. For example:

```sh
TILELANG_CACHE_DIR=/tmp/my-owned-fusion-cache \
python examples/ascend_a2a3/simple_fusion/matmul_add.py
```

All six original factory inputs pass the complete TA/main pipelines and real
Bisheng compilation. The private printf StringImm fix is a required separate
component for the print example: it restores the nine original ASCII print
statements without changing the example strings. Its pre-write uninitialized
UB dump remains in the original source and is not a deterministic data golden.
Original debug flags `-D_DEBUG --cce-enable-print` remain enabled.

The source companion under `testing/ascend_a2a3/python/ecosystem/` protects
import/cache safety, complete original public lowering, parameter order,
operation inventory, named allocations and host ABI. These host/source checks
are distinct from the original device golden and from an actual CI bench run.

The remaining F8 assets (`mhc_post`, `head_compute_mix_kernel`, `moe_aux_fi`,
`moe_topk_gate`) remain planned separately. Their whole generators/full
matrices are retained in inventory; this four-asset component does not claim
their source, SDK or device coverage.
