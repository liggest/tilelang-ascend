# 当前 A2/A3 架构与 TA 来源

A2/A3 的当前实现遵循主线 BackendModule 合同。先读 `tilelang/backend/README.md` 和 `.agents/skills/tilelang-backend/SKILL.md`；本文件只补 TA 特有位置，不能据旧全局 phase.py 再造执行后端。

```text
tilelang.ascend.a2a3.language (common eager frontend)
  -> BackendContext: frozen Target arch=dav-2201, model=A2/A3, keys=ascend/pto
  -> tilelang/ascend/a2a3/pipeline.py (PassPipeline)
  -> tilelang/ascend/a2a3/codegen.py (instrumented source codegen)
  -> compiler.py CompileConfig (dialect=ascendc/pto, frozen before cache lookup)
  -> shared cython executor / backend-local ABI, cache, workspace support
```

`src/ascend/a2a3/{op,layout,transform,codegen,tl_templates}` 拥有本地 native；Python pass wrapper在 `tilelang/ascend/a2a3/transform`，transform factories 多数以 `tvm.ffi.get_global_func` 查隔离FFI；language/layout另使用 `_native.require_native` guard。共用原语与配置各有唯一 owner，不能覆盖 A5 或重新注册共享 flag Op。

实际顺序以 `LowerAndLegalize` 与 `OptimizeForTarget` 两函数为准：infer scope -> tmp -> vid -> shape collection -> HostProcesser -> LowerParallel -> LayoutInference -> LowerTileOp；优化阶段包含 CrossCore/CombineCV、软件流水、Flatten/Vectorize/Storage、MemoryPlanning、Sync/SyncVS、ResourceScopeVerify。仅 AscendC A2/A3 使用最后 Selection/MaskLegalize；之后不得再改写 IR。不要把这个摘要替代完整源中的条件、参数与前置 MaterializeKernelLaunch。

新 IR 是 `tvm.tirx`；`s_tir` 仅提供明确共享变换。旧 `vid` 对应 `cthread`，默认/显式 launch 标记不同。`Var()` 创建真实 Var，不表示缺失；依 main IR skill 用 `Optional<Var>` 表达空状态。Buffer/Var/节点身份和结构相等按主线技能区分。

Python `T.Kernel(..., is_npu=True)`、`T.Scope`、`T.tile`、L1/UB/L0 分级语义来自 TA；A5 `tilelang.ascend.language` 的 MixedKernel/SimdVF 不得作为其替代。`T.gemm` 的 generic GPU 分支不是 A2/A3 矩阵原语支持证明；查真实 `T.mma` / `T.tile` 原语和已迁原例。
