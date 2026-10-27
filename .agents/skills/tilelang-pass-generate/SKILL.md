---
name: tilelang-pass-generate
description: 按已授权设计实现完整A2/A3本地pass、注册/封装/配置/pipeline接线和必要行为验证，不用旧全局TIR模板或另起执行模式。
---

# tilelang-pass-generate：现代 A2/A3

仅用于TA A2/A3，先读docs/ascend_a2a3/migration.md与main backend/build/IR技能；已有main技能保持独立。

1. 读取设计、真实pipeline与最接近的本地组件。没有单独design文件时可以从已明确用户任务整理可审设计并推进，不强制停下来重复批准。
2. 使用templates/pass-impl-skeleton-template.md核对完整算法、输入/输出attrs、facts、所有target/config分支、header/catalog/OperationConfig依赖。范围变化才给具体差异；不要用精简示例替代完整来源组件。
3. 在src/ascend/a2a3/transform实现完整算法，按main tvm-ir/dev-guide处理handle/Optional/COW/identity；同步维护本地CMake gate。新native注册用TVM_FFI_STATIC_INIT_BLOCK和reflection::GlobalDef().def，不覆盖主线A5同名入口。
4. 本地Python wrapper通过_native.require_native查询隔离FFI；本地pipeline消费wrapper。配置enum沿共享唯一owner，本地pass_config只归一化/默认；编译flags属于compiler resolver，不重复读环境。
5. 保原时序/默认/dev marker/cthread/workspace/layout与mask事实。Let/Bind、AllocBuffer等必要表示适配需真实作用域/临时声明证据；原非法输出改诊断单列，不能吞异常变NoOp。
6. 完成native实际构建、确认fresh loader、same-input来源对照和必要聚焦回归。实现任务必须覆盖真实新行为/边界，不因旧技能说UT/ST后补而省略必要验证；对低风险纯格式无需制造镜像测试。
7. 只在真实运行后写设备数值/性能通过。报告来源pin、所有完整组件与必要修正、实际命令/退出/差异、剩余风险；没有注册或环境时明确缺口，不能给示例核冒充完整通过。

接入的现行签名/模板见references/integration-points.md；原TA全文流程和旧分阶段限制在非活动source档案中保留，不覆盖主线当前授权/质量要求。
