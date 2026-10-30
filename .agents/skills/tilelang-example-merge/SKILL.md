---
name: tilelang-example-merge
description: 将TA A2/A3 kernel与测试整理成可审示例，保完整算法/factory/decorator/golden/原case和退出判定；不执行模块顶层kernel、清用户cache或把pending/deferred计pass。
---

# A2/A3 示例整理

本技能只处理用户授权的A2/A3资产；新路径为examples/ascend_a2a3/{relative}。原TA完整整理流程保存在source档案，现代接口与导入安全以本页及migration.md为准。

1. 固定原源码、factory/classes、所有decorator（pass_configs/out_idx/workspace_idx/compile_flags）、shape/dtype、golden及误差门槛。逐项列出被合并的文件和case，不能为了单文件仅留下两个代表例或把失败/边界case删掉。
2. 保完整kernel和host helper算法；必要适配仅语言入口tilelang.ascend.a2a3.language、公开JIT/workspace参数、namespace路径和现代可用cache接口。不要改参数/阈值/数据驻留边界伪造迁移通过。
3. 迁移时保留原执行结构：原文件没有main()就不新增，原有main()保签名、调用和__main__ guard，顶层操作保原位置与顺序。不要为安全导入或迁移审计新增main、guard、torch_npu注册或状态恢复包装；host审计使用AST或审计侧harness，不反改产品入口。
4. 将旧cache.clear_cache()在原位置适配为同一tilelang别名的disable_cache()；不再调用已删除的旧clear_cache API。不要新增专属TILELANG_CACHE_DIR异常、私有cache类或清理/恢复包装；原来没有清cache的地方不增加disable_cache，原有disable_cache保留。
5. 出口继续原判定：真实返回码和Kernel Output Match!/Test Passed!。source/AST通过、设备数值、性能与独立restore分别记录；任何未执行/原失败/pending/Carver deferred都不算pass。
6. 调用纯host/source测试时不会执行NPU；需要原设备case才单独运行。所有实际支持case保原失败传播，不因L1/Boundary/大shape标签把失败自动变warning或skip。
7. 对照完整原source AST/实际lowered source，运行聚焦公开compile/数值，记录cache与native/Cython路径；公共call计时不能改称纯kernel。不换手写小核或绕公开pipeline冒充E2E。

实际已迁写法见examples/ascend_a2a3/gemm/example_gemm.py。未迁资产不得算作已迁能力，不fallback到main同名CUDA文件。
