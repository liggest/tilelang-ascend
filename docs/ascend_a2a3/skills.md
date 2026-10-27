# TA 技能与 agents 的迁移状态

固定来源 `0deb8b93da11b7f0becbcded732d9633b7ca4e34` 包含 122 项 `.agents` 资产、21 个 SKILL 和 5 个 OpenCode agents 文件。完整路径、blob、原始 hash、目标 hash、必要适配及原文存档见[资产清单](../../maint/ascend_a2a3/docs_manifest.json)。现有主线 22 个技能文件另列受保护 hash，未被此包覆盖。

这里的“适配”表示文档、目录/API、构建与工作流合同经过迁移审查，不代表已经调用外部模型逐个完成 skill 全流程，也不代表其所有代码片段均做了设备执行。不能把复制、AST 或链接检查当 runtime 验收。

| 技能 | 当前适用范围与审查结果 |
|---|---|
| skill-creator | 完整保留通用元工具与资源；TA 来源独立登记，未运行外部 eval/benchmark，不覆盖 main skill |
| tilelang-a5-sim-convert | 原 A5 来源参考完整保留，明确非 A2/A3 sim 能力，旧模板未作为现代 A5 可执行承诺 |
| tilelang-ascend-tile-api | A2/A3 language/op/codegen/templates/OperationConfig 位置，保端到端语义与验证要求 |
| tilelang-api-best-practices | 保 TA 内存/搬运/同步语义，改 A2/A3 入口与本地来源；GPU/A5写法不互换 |
| tilelang-debug-helper | 现代 Debug CMake、build/lib/loader、实际模块与Cython provenance；不清用户构建/缓存 |
| tilelang-env-check | 从旧自动安装改为只读 metadata/pin/tool检查；现代build与配对SDK、native/设备检查分开，旧全文存档 |
| tilelang-error-fixer | 本地TIRX/pass/codegen定位与真实失败证据，保AOT/IR/source分析流程，不旁路pipeline刷绿 |
| tilelang-github-operations | 通用GitHub资料保留；既有用户目标/授权优先，不自动push/消息或重问已知分支 |
| tilelang-programming-model-guide | 保Developer/Expert/混合及原配置；新任务才澄清，已有factory不因推荐默认而重写 |
| tilelang-review | 现代canonical工具与TA原CI区别，准确pin与隔离工具环境；缺工具/未执行单列，旧全文存档 |
| tilelang-submodule-pull | 按gitlink pin恢复而非自动追远端；保完整原定时pull工具作为来源，不自动启用 |
| tilelang-example-merge | 完整保原factory/decorator/golden/所有case；导入无清cache/编译/launch，main使用显式独立cache |
| tilelang-op-design | A2/A3目录/API与原硬件设计约束，已有模式保留，不用A5/CUDA能力补缺 |
| tilelang-op-develop | 保完整算法、host禁作弊及支持域设计，现代路径与导入安全；原case不被新推荐默认改写 |
| tilelang-op-test-design | 原golden/shape/dtype/退出语义与完整case；原pending/未执行/deferred不算pass |
| tilelang-pass-analyzer | 当前source-first查询；旧技术设计/伪IR注明TA版本，不把旧节点模板当现代API |
| tilelang-pass-design | 当前BackendModule/TIRX、attrs/facts/支持域与隔离注册设计，原模板全文存档 |
| tilelang-pass-generate | 完整实现、真实factory/FFI接线和必要回归，已授权设计不反复停骨架批准；不省测试 |
| tilelang-pass-workflow-analyzer | 实读当前两阶段和前置Materialize，保方言/平台/配置门控及mask末端，旧链只作来源 |
| tilelang-perf-optimization | 保实际调优和性能风险资料；publiccall/kernel、冷热/cache/驻留边界与原样本分开 |
| tilelang-skill-review | 仅在授权范围审TA资产，不因skill存在就自动覆盖main技能；来源/修改/评测状态明确 |

OpenCode 原5文件保完整角色、工件和状态机；状态目录改为 `examples/ascend_a2a3/{op}`，环境/语言/编译器路径按本包适配。它们服务明确的 A2/A3 kernel 任务，不能自动接管主线所有编译器工作。已选模式、已有用户授权与已请求性能工作不重复确认；外部动作仍须本次授权。

[AGENTS 合并候选](AGENTS.candidate.md)及原文仅供专审，未写入根 AGENTS；现行主线 backend/build/dev-guide/tvm-ir/A5 技能继续生效。对旧算法参考中的 LetStmt/Allocate/CustomizedCodeNode、旧 `_ffi_api`/注册语法和来源行号，应查询当前实现和迁移指南，不能将旧材料机械改名后编译。
