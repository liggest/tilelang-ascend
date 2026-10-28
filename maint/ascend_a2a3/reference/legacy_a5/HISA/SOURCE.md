# Inactive historical A5 source reference

This is an unmodified source archive, **not an active A2/A3 example or CI entry**, a modern A5 API port, a Carver feature deferral, or an A5 numerical claim. The `.py.txt` extension keeps the old program out of Python example discovery. Do not execute it as part of the migrated A2/A3 suite.

- Source repository: `tile-ai/tilelang-ascend`
- Fixed commit: `0deb8b93da11b7f0becbcded732d9633b7ca4e34`
- Original path: `examples/HISA/block_sparse_mqa_attn_expert_test_for_a5.py`
- Original Git blob: `59f2cea5898e6fa50c3d6591901e0198200c55eb`
- Original and archive SHA256: `701b87fb0e3f8fd3440221498a52a01cce8cb8c157312818a567ac8e45aff26b`
- Byte count: `38096`

The source explicitly describes an A5-only PTO CV path. Its factory emits eight `T.copy_op.copy_cv_experiment` calls with SingleVec0/SingleVec1, transferring L0C directly into AIV UB without the regular GM workspace. The original API says PTO A5 only; the corresponding `common.h` helper is inside `#ifdef PTO_PLATFORM_A5` and calls `pto::TMOV` with `AccToVecMode`. The old compiler selects dav-c310/REGISTER_BASE for A5 and dav-c220/MEMORY_BASE otherwise.

The original CLI checks `determine_platform() != "A5"` before device/test setup, prints SKIP and the old `Kernel Output Match!` marker, then exits0. That marker on A2/A3 is an intentional skip, not a kernel launch or precision pass. The original operator manifest does not register this file; legacy bench discovery can find it and accepts that skip marker. No A5 device, original CLI, compiler, old API alias or hardware query was executed to create this archive.

The original TA file remains unchanged. This reference stays inactive in final A2/A3 adoption and CI mapping. Full classification evidence is in the coordinator's `refactor_main/reports/p3_HISA_for_A5_classification/` archive.
