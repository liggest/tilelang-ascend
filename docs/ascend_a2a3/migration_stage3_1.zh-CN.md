# 阶段 3.1：迁移后续遗留任务

总体目标继续是：将 TA 的 A2/A3 能力迁入现代 TileLang，保持原有算法、pass/codegen 行为及 TA 的测试方式。兼容适配以现代 eager/TIRX 为基础；生成代码中的同步、内存布局、别名与整数语义变化须追溯来源，不用人工补同步或放宽原断言绕过。

**阶段 3 仍待验收，未签署完成。阶段 3.1 是单独登记的后续遗留工作；完成这些事项不自动构成阶段 3 验收。** 原资产映射、编译通过和局部设备通过分别记录，不能相互替代。

## 分支与来源基线

- 主开发线：TileLang 仓库 `refactor_main`，本次登记前为 `68bf79912509bfbb19112f0b468e35cc5f823ca3`。后续迁移、修复和阶段提交优先集成到这里。
- 迁移期间，TA 仓库 `refactor_main` 同步主开发线的阶段 HEAD。完整 TA 仓库的 CI/安装/旧资产处置组装在 TileLang 迁移完成后形成。
- 先前 TA 组装快照 `fef14ff437a235ecf7c7c7d97a77a75598bfa087` 保留历史引用，不把它当作当前已验收的最终交付。
- 原迁移来源为 `0deb8b93da11b7f0becbcded732d9633b7ca4e34`。已从官方远端读取并将本地 `ascendc_pto` 快进到 `463c234ce3087205c092e931207ca87b4cd5494c`，新增 7 个提交。已有 TVM 子模块修改、未跟踪资料和 venv 保留。
- 既有最终安装包设备证据针对功能基线 `4312b1a2099797ed65a92acb8124622409823d0c`，后继 fef 仅加验收文档；不能据此声称最新来源或未来 HEAD 已测试。

## 任务登记

| 编号 | 事项 | 当前状态 | 核对与留痕 |
| --- | --- | --- | --- |
| 3.1-01 | 两仓分支文件去向、修改、新增、删除和旧副本对比 | 审查工具与固定基线报告已生成，正在补核历史 retirement 中的活动 owner 映射 | 完整 Git 树、来源→目标 diff、CSV/离线 HTML；保留 0deb/68bf/fef 快照，不随分支移动改写历史 |
| 3.1-02 | 核验原 TA CI 跑法、迁移后路径与实际结果 | 原控制流与局部结果已核，完整 CI 队列未运行 | 原 bench 177→174 任务；另 24 operator pytest 文件；原 pytest 已用 forked，增量 n4、全量 n8；局部串行节点不当全量 |
| 3.1-03 | 维护 TileLang `refactor_main` HEAD | 持续执行 | 子 Agent 实现，统筹 Agent 审查、测试、集成；沿用 `[Ascend][A2A3]` 阶段提交 |
| 3.1-04 | 两仓阶段 HEAD 同步，完整 TA 组装后置 | 已确认范围，待保存旧 TA 组装快照并同步 | 不覆盖用户工作区或 venv；保留旧 fef 历史和引用变动记录 |
| 3.1-05 | 审计最新 `ascendc_pto` 的新增提交 | 已获取 7 个提交，逐项核查中 | 文件/算法/pass/codegen/CI 影响与当前主开发线覆盖情况分别登记 |
| 3.1-06 | 补迁新增代码、例子、测试和 skill | 待逐包实现和验收 | W4A4 正确性修复优先；保持新上游原算法、完整原测试矩阵和失败边界 |

最新来源增量为：Dilation2D（2f22c6ac）、situ_glu（43b0b697）、Chunk Scaled Dot KKT（cf640944）、W4A4 GEMM 正确性（9da73cd3）、GLM-5.3 kpool compress（f2e4658c）、depthwise conv2d（24b69276）、op-profiling skill（463c234c）。W4A4 修复涉及两个 codegen、AscendC template 和 memory planning，当前主开发线尚未等价覆盖；这是旧来源之后的上游新增修复，须单独迁入并验证。

## 阶段 3 尚待验收的范围

继续依原阶段 3 计划复核完整迁移资产、现代前端适配、原 pass/codegen 与生成代码保真、TA CI 测试路径，以及文档/skills/Agent 资产。现有两个 GEMM、activation 10 CLI、选定 23 pytest 节点的设备证据只证明各自范围。完整 bench、operator pytest、testing 并行矩阵与性能没有因这些结果自动通过。

CUDA 完整回归按用户范围暂不阻塞；无 A5 硬件不等待 A5 实机回归。仍要求 USE_ASCEND 成功编译、当前环境的 A2/A3 功能正常，并用可运行测试和隔离审查保护 A5。Carver 保持已明确的后续评估范围，普通 autotune 不继承该豁免。

详细执行日志与大证据保留在工作区 `refactor_main/` 台账和 reports 中；这些忽略目录不是随仓分发的永久证据。此页作为阶段 3.1 的版本化登记入口，后续每项完成时追加具体提交、验证范围与未验事项。
