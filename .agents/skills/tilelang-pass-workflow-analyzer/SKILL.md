---
name: tilelang-pass-workflow-analyzer
description: 分析A2/A3 AscendC/PTO真实PassPipeline顺序、条件、上下游事实与新pass插入点。
---

# tilelang-pass-workflow-analyzer：现代 A2/A3

仅用于TA A2/A3，先读docs/ascend_a2a3/migration.md与main backend/build/IR技能；已有main技能保持独立。

1. 权威入口是tilelang/ascend/a2a3/pipeline.py。读完整LowerAndLegalize、OptimizeForTarget及前置MaterializeKernelLaunch；旧TA phase文档只作来源背景。
2. 列出每个调用的序号、原参数、config gate、target keys/platform gate和读写事实。model表示A2/A3；keys判AscendC/PTO，不再用model=ascendc/pto识别方言。
3. 先分析producer/consumer，再决定阶段/位置。InferScope/Tmp/Vid、Layout/LowerTile、Flatten/Storage/Memory、CrossCore/CombineCV/Planning/Inject、Sync/SyncVS各有原依赖，不能根据名字重排。
4. default/dev launch markers、cthread、layout_map、workspace、OperationConfig与mask terminal facts逐项核对。AscendC Selection/MaskLegalize为末端结构合同，后面不得再rewrite；PTO不套AscendC mask分支。
5. 新pass插入方案包含上下文、哪些facts被消耗/产生/失效、missing attrs策略、原非法输入诊断与新增回归。先读main backend/dev-guide/tvm-ir技能，保唯一配置/Op owner，不创造新execution mode。
6. 用真实native和原factory验证关键顺序/条件，不能手工移除pass后称完整pipeline通过。输出完整链及条件差异，compile/source/设备分别标明证据。
