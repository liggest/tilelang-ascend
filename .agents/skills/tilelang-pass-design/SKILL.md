---
name: tilelang-pass-design
description: 为A2/A3 pass形成可实施设计，包含TIRX变换、完整源支持域、配置与pipeline依赖、失败边界和必要验证。
---

# tilelang-pass-design：现代 A2/A3

仅用于TA A2/A3，先读docs/ascend_a2a3/migration.md与main backend/build/IR技能；已有main技能保持独立。

1. 从用户目标/现有设计提取问题、来源组件、支持域和约束；缺信息时先完成不依赖它的调查。用户已授权实现或已确定设计时不重复要求同一批准。
2. 先用workflow分析当前真实pipeline与attrs/facts，再选LowerAndLegalize/OptimizeForTarget中的具体producer/consumer位置。当前源码优于归档参考，AscendC/PTO条件独立，mask legalization后无rewrite。
3. 设计输入/输出IR、每个loop/buffer/scope分支、target/config门控、analyzer/identity maps、COW策略、可缺状态和声明/词法范围。保TA完整有效算法；原缺陷或新必要适配列差异证据，不缩成小核。
4. 按references/pass-impl-patterns.md与main IR/dev-guide技能确定实际handle/Optional/visitor接口；不能从旧LetStmt/Allocate/CustomizedCodeNode模板构造新API。
5. 指定本地native TU、CMake gate、隔离FFI、require_native wrapper和本地pipeline。共享配置enum/Op/catalog只有一个owner；新增config先证明默认，不另造全局ffi_api覆盖A5。
6. 测试计划覆盖真实行为与失败条件，并区分same-input TA对照、native freshness、public compile/source与设备数值/性能。旧来源参考完整保留，但归档模板不是当前可编译代码。
7. 产出pass-design.md：目标/非目标、来源/支持域、位置与依赖、变换伪代码、实现文件、配置/attrs表、回归方案、风险/未决项；不把未决设计或待测清单当完成。
