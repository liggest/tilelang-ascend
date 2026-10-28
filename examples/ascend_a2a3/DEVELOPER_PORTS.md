# Original developer, pipeline and GEMM source ports

The independent `maint/ascend_a2a3/ecosystem_developer_ports.json` ledger records
complete original sources from TA `0deb8b93da11b7f0becbcded732d9633b7ca4e34`.
Source/SDK acceptance and device numerical acceptance remain separate.

## First developer batch

| Source under `developer_mode/` | Original requests | Dialect and original check |
|---|---:|---|
| `gelu_mul_developer.py` | 2 matrix rows | AscendC; tanh GELU times the other half |
| `gemm_developer.py` | 1 original CLI default | AscendC; `a @ b` |
| `matmul_add_developer.py` | 1 original CLI default | AscendC; `a @ b + d` |
| `matmul_add_developer_vec_cast.py` | 1 original CLI default | PTO; `a @ b + d` |

Both GELU shapes, all factories/decorators, explicit/default launch modes,
original pass flags, CLI arguments and seed 0 remain. The original precision
checks use `rtol=atol=1e-2`. GEMM CLI defaults are `m=n=k=1024`, with original
block arguments `128,256,64`; the GELU rows are `256,256,64,64` and
`1024,1024,128,128`. Other user-supplied CLI values remain accepted by the
original interface, but source/SDK validation here covers the original matrix
and defaults, not every possible shape.

Importing a port does not parse arguments, clear or disable caches, seed RNGs,
compile or launch. The original runtime stays in `main()` with explicit
`torch_npu` registration. The two clear-cache scripts require a dedicated
`TILELANG_CACHE_DIR` and reject a conflicting resolved caller override. The two
matmul-add scripts retain their original disable-cache behavior during runtime
and restore the caller's enabled or disabled state afterward.

After a particular asset is accepted for an exclusive device check, its original
CLI can be run from the repository root, for example:

```bash
(
  export TL_PLATFORM=A3
  export TL_RUN_MODE=npu
  TILELANG_CACHE_DIR=$(mktemp -d /tmp/tilelang-a2a3-developer.XXXXXXXX) || exit 1
  export TILELANG_CACHE_DIR
  python examples/ascend_a2a3/developer_mode/gemm_developer.py
)
```

Every original request has passed full source-only lowering and actual Bisheng
compilation. Device execution remains pending. Extra unused scalar bindings
and the shared frontend's unused default-subcore declaration are retained in
the actual generated source; comparisons record these differences rather than
changing passes or source algorithms to obtain identical text.

A5-only FP8, the separately recorded unsafe h16 input, and the historical
AOT/LibTorch assets are outside this port ledger. They are not replaced with
small A2/A3 examples or counted as passed here.

## Pipeline, inferred scope and tail blocks

The next four complete sources retain seven original requests:

| Source | Original source requests |
|---|---:|
| `pipeline/gemm_v0_pipeline.py` | CLI default `8192,1024,8192`, block `128,256,64` |
| `pipeline/matmul_add_pipeline.py` | CLI default `1024,1024,1024`, same block, explicit workspace |
| `gemm/example_gemm_infer_scope.py` | CLI default `1024,1024,1024`, same block |
| `gemm/example_gemm_tail_block_developer.py` | All four original partial M/N/K rows |

The workspace decorator, both original software pipeline stage counts, explicit
C scope, barriers and reference tolerances are retained. The tail-block script
still prints and re-raises each original failing case. The unused `d` allocation
in `gemm_v0_pipeline.py` remains in its original position; a local lint comment
prevents deleting part of the original resident allocation set.

These requests have passed source/SDK checks, with device validation pending.
The two pipeline sources have a different ordering of the same independent
C0 tensor lookup declaration. The actual generated source is retained; no
emitted payload is removed or reordered by this port. The difference is
recorded, not optimized away in the compiler.
