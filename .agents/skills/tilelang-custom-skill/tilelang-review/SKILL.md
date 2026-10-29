---
name: tilelang-review
description: 对 TA A2/A3 改动做与当前主线 canonical 工具和 TA 原 CI 范围一致的格式审查。用于已指定的代码审查/格式任务；保持用户已授权范围，不自动改环境或覆盖主线已有技能。
---

> 适用范围：TA A2/A3 资产；主线 backend/build/IR/A5 技能保持独立。先读 docs/ascend_a2a3/migration.md，已有用户授权、模式与原配置优先；不重复请求同一批准，不因搬入本 skill 自动授权安装/外发/push/定时任务。

# A2/A3 格式与提交前审查

先读主线 `tilelang-dev-guide`，提交/PR任务再读 `tilelang-submit-pr`。用户已明确修复范围或授权格式化时直接完成；仅当目标范围确实不明才澄清，不反复要求确认。

1. 从实际 diff列出文件和源码owner，避免对整个CUDA/A5树做无关格式修改。现有 main backend/IR/build技能保持不变。
2. 读取主线 `.pre-commit-config.yaml` canonical exact pins。当前clang-format为23.1.2；`requirements-lint.txt`仍21.1.8时明确记录不一致，不能假设18/21/23输出等价。TA legacy format job18保持；modern TA CI 使用canonical pin的独立工具环境。
3. 使用主线 `format.sh --files <exact files>`。若工具缺失，先记录真实限制；需要准备工具时用隔离环境和准确pin，不自动替换用户环境。旧TA `format.sh`的yapf流程仅适用于原TA布局，不假称新主线全门禁已跑。
4. 读取检查输出后修复授权范围内的真实问题。格式检查、源码编译、行为测试和设备验收分别列证据，不能把`git diff --check`当Ruff/clang通过。
5. 新A2/A3原生态路径为examples/ascend_a2a3、examples_experiment、testing/ascend_a2a3；新增A2/A3组件测试在testing/ascend_a2a3/python/refactor；main原有testing/ascend保留。TA CI仍143登记/24active/119pending来源，不以pending当pass；Carver用户deferred单列。
6. 报告已运行命令、退出码、未跑项目和具体依赖；不因没有设备就伪报完整CI，线上runner结果须真实触发后才存在。push/外部消息只在用户明确授权范围内执行。

完整原TA格式技能保存在对应非活动来源文件，包含旧工具与交互流程，不能覆盖这里的现代工具/授权合同。
