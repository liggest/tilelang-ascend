# TA A2/A3 migration preparation

This tool prepares source candidates from TileLang-Ascend. It preserves the original
implementation and reports adaptations that need review. It does not install files,
assemble a pipeline, or declare a backend complete.

The checked-in `ffi_map.json` records TA commit
`0deb8b93da11b7f0becbcded732d9633b7ca4e34`. TIRX rules are based on mainline commit
`b939fa0137086ee05729be2ce7a066d0e102159d`, especially `frontend_legalize.cc` and
`lower_opaque_block.cc`, and checked against the current TVM headers.

## Reproduce the inventory

Use a fresh output path: every command refuses to overwrite an existing file.
The TA inputs must be clean against their recorded HEAD.

```bash
python maint/ascend_a2a3/migrate.py inventory \
  --ta-root /workspace/baitian/tilelang-ascend \
  --main-root . \
  --output /tmp/a2a3-ffi-map.json
```

The inventory resolves Python wrapper aliases through AST parsing, records active
calls in both engine files, verifies native registrations, and adds the proposal's
explicit 18 exclusive source files and 10 forked passes. It also records optional
and auxiliary registrations. `AscendVectorTerminalCatalog` is a catalog, not a pass.
`AnnotateDeviceRegions` remains in the proposal even though the current phase does
not call it. `LowerDeviceStorageAccessInfo` belongs to engine helper functions and
is not counted as an active A2/A3 phase pass.

There are 36 mapped registrations: 33 pass factories, one auxiliary catalog, and
two build entries. There are 29 distinct active phase factories. Counts of source
files, factories, and calls differ because one source can register multiple
factories. Each entry records its source SHA256, source/target paths, FFI names,
call sites, proposal membership, and scope. Build files go under
`src/ascend/a2a3/codegen/`; pass files go under `src/ascend/a2a3/transform/`.

The scan reports existing destination registrations as collisions. It does not
resolve collisions by replacing or overriding registrations. The checked-in
inventory records the already integrated `A2A3HostLegalize` registration as one
such collision. `excluded_registrations` records unrelated CUDA/HIP and inactive
legacy pass registrations instead of silently dropping them.

## Prepare a candidate

```bash
python maint/ascend_a2a3/migrate.py migrate \
  --input /workspace/baitian/tilelang-ascend/src/transform/ascend_storage_rewrite.cc \
  --output /tmp/ascend_storage_rewrite.candidate.cc \
  --main-root .
```

`migrate` accepts one `.cc`, `.h`, or `.py` input. Repeat it over the reviewed source
inventory for a batch. It renames exact global FFI strings, Python `_ffi_api`
references, TIR headers/namespaces, and known node names. Comments and other string
contents are preserved. No `tl.ascend_*` Op registration is renamed.

The adjacent `.json` file records the mapping's TA commit, actual source repository
and commit when available, its tracked source path, whether bytes match HEAD,
SHA256, and required manual work. A copy outside a Git repository is explicitly
marked `unverified_copy`.

Return codes:

- `0`: a candidate or inventory was generated; this does not mean native validation passed.
- `1`: collision, output already exists, source drift, or another rejected operation.
- `2`: a candidate was generated with mandatory manual review.

C++ candidates requiring review start with `#error`. Python candidates requiring
review raise before executing their body, while retaining legal future imports.
Do not remove these guards until the listed issues have been reviewed. If a known
native source is intentionally being replaced, `--replacing <relative-main-path>`
excludes exactly that file from collision checks; it does not change any source.

## Reproduce the complete HostLegalize calibration

```bash
python maint/ascend_a2a3/migrate.py calibrate-host \
  --ta-root /workspace/baitian/tilelang-ascend \
  --output /tmp/ascend_host.calibrated.cc
cmp /tmp/ascend_host.calibrated.cc src/ascend/a2a3/transform/ascend_host.cc
python -m unittest discover -s maint/ascend_a2a3 -p test_migrate.py -v
```

This reviewed adapter asserts the exact original source hash, then preserves the
complete `HostProcesser` class, branches, visitors, and collectors. Its JSON lists
every interface adaptation. The original `LetStmtNode` visitor only forwards to
the analyzer mutator and never accesses a body; converting this particular visitor
to `BindNode` was reviewed. This exception is not applied to any other pass.

## Manual boundaries

`Bind`, `AllocBuffer`, `DeclBuffer`, assertion scopes, and legacy producer nodes
require structural review: replacing their names does not preserve scope or field
contracts. Moved `s_tir`/TileLang attributes, removed headers, private header paths,
legacy Op name strings, direct/aliased Python TIR symbol imports, and old
`tvm.script.ir_builder.tir` frame paths are explicitly reported when encountered.
Known node spellings include `BlockFrame -> SBlockFrame` and
`block_attr -> sblock_attr`; module path and frame inheritance still need review.

Native candidates always require review of C++ symbol isolation, relative headers,
FFI/container APIs, and target attributes. The generic tool does not infer which
`tir::attr` keys moved to `s_tir` or `tl::attr`, rewrite arbitrary AST bodies,
replace `CHECK` macros, or port codegen base classes. Language/IR entry points such
as TA `tl.Scope` and Op ABI compatibility have their own migration review; this
inventory covers pass and build FFI only. Required dependency `.h`/`.cc` files that
register no FFI must come from the separate source inventory.

## Stage 3 ecology candidates

`ecosystem.py` handles example/test import candidates separately from native IR
migration. Its checked-in `ecosystem_manifest.json` fixes all 632 inventoried TA
assets (340 Python files) to the same source pin, with blob IDs, SHA256, draft
classification, and classification evidence. This is an inventory, not a list of
supported or completed ports. Docs, skills, shell files, gitlinks, A5 assets, and
unresolved IR/runtime APIs remain explicit review work.

The pure `destination_for(original_ta_path: str) -> str` contract is:

- `examples/<relative>` → `examples/ascend_a2a3/<relative>`;
- `examples_experiment/<relative>` → `examples_experiment/<relative>`;
- `testing/<relative>` → `testing/ascend_a2a3/<relative>`.

It rejects absolute paths, non-normalized paths, traversal, and explicit
`ascend_a2a3` wrapper paths. Experimental source and destination paths now
intentionally match because mainline had no experimental tree. New A2/A3 component tests are grouped under
`testing/ascend_a2a3/python/refactor/`; original mainline `testing/ascend/`
assets remain in place. The former experimental `examples_experiment/ascend_a2a3/`
prefix is historical; see `experiment_layout_move.json` for each old/new blob.
Mapping alone does not authorize a migration. CI resolvers may use these
paths only after checking which layout they are running in; they must never fall
back to a same-named main CUDA/A5 file. TA CI workflows/scripts and their dependency
closure are protected for separate review under the user's latest CI requirement.

For example, use a new output directory for each reviewed batch:

```bash
python maint/ascend_a2a3/ecosystem.py \
  --ta-root /workspace/baitian/tilelang-ascend \
  --output-dir /tmp/a2a3-ecology-batch-01 \
  --path examples/gemm/example_gemm.py \
  --path testing/python/language/test_tilelang_ascend_language_src_code.py
python -m unittest discover -s maint/ascend_a2a3 -p 'test_*.py' -v
```

Success returns **2**: it writes only `*.candidate.py` files with an execution
blocker, adjacent provenance/manual-review JSON, and `batch.json`. Return **1**
means rejection. There is no install, overwrite, force, or guard-removal option.
The entire batch is checked before a new output directory is created. The tool
checks TA HEAD, pinned blob/hash, and selected working-tree bytes, without
importing source modules or touching the TA tree. Repeated identical invocations
with fresh directories produce identical candidates and sidecars.

Only explicit aliased language imports and approved layout helper imports are
rewritten. Mixed single-line imports preserve generic names; comments, strings,
kernel bodies, decorators, flags, lists, thresholds, and platform choices stay
unchanged. Bare module imports, unknown submodules, comment-bearing mixed imports,
legacy TIR/FFI/platform/runtime calls, and source side effects are recorded for
manual review. Even an import-only candidate stays blocked until its real factory,
IR/source, SDK, and applicable device checks have been reviewed. In particular,
never import a whole legacy example just to validate the candidate: many execute
cache clearing, argument parsing, and NPU work at module scope.

An uncertain/other classification requires a per-file decision JSON with a
nonempty evidence-based reason, passed via `--decisions`:

```json
{"testing/python/example.py": {"decision": "a2a3", "reason": "record source-specific evidence here"}}
```

Explicit A5 assets cannot be redirected with a decision. The regular
`examples/autotune/` files belong to the autotuner owner and are refused here.
The original Carver example is retained for later evaluation by user scope;
Ascend Carver policy is deferred, without substituting static configurations or
claiming it verified. Other docs/skills assets remain an inventory for later
assigned packages.
