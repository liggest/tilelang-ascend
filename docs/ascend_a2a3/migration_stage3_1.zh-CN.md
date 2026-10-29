# 阶段 3.1：迁移后续遗留任务

总体目标继续是：将 TA 的 A2/A3 能力迁入现代 TileLang，保持原有算法、pass/codegen 行为及 TA 的测试方式。兼容适配以现代 eager/TIRX 为基础；生成代码中的同步、内存布局、别名与整数语义变化须追溯来源，不用人工补同步或放宽原断言绕过。

**阶段 3 仍待验收，未签署完成。最新来源的七组增量代码资产已集成；这不等于设备、完整 CI 或性能验收通过。** 本页记录 2026-10-30 的集成快照，后续设备结果以带准确 HEAD、native/Cython 身份的独立报告为准。

## 分支与来源基线

- 主开发线：TileLang 仓库 `refactor_main`。本轮代码资产汇总基线为 `d891d8b79371dc47d57828b045c05095960c081e`；其后的 runtime 修复与设备结果分别留痕，不把本页统计当作后继 HEAD 的验收证据。
- 迁移期间，TA 仓库 `refactor_main` 同步主开发线的阶段 HEAD。完整 TA 仓库的 CI/安装/旧资产处置组装在 TileLang 迁移完成后形成。
- 先前 TA 组装快照 `fef14ff437a235ecf7c7c7d97a77a75598bfa087` 保留历史引用，不把它当作当前已验收的最终交付。固定旧基线审查包保留 0deb/68bf/fef 完整树、逐文件 diff、CSV 和离线 HTML。
- 原迁移来源为 `0deb8b93da11b7f0becbcded732d9633b7ca4e34`。官方最新来源为 `463c234ce3087205c092e931207ca87b4cd5494c`，新增 7 个提交、23 新增 / 6 修改 / 0 删除。原用户 TVM 修改和 venv 均保留。
- 既有最终安装包设备证据针对功能基线 `4312b1a2099797ed65a92acb8124622409823d0c`，不能据此声称最新来源或未来 HEAD 已测试。

## 七组增量的集成和验证

| 编号 | 资产 | 已完成范围 | 仍待验收 |
| --- | --- | --- | --- |
| I1 | Dilation2D，2 个原 Python 文件 | 完整算法、factory、原测试与 CLI 已迁移；与 I3/I6 合计 42 项 host/source、14 个 factory 请求生成 16 份源码并通过 16 次真实 SDK 编译 | 原 CLI 和全部设备测试矩阵 |
| I2 | SiTU GLU，5 个原资产（3 Python + 2 Markdown） | 与 I5 合计 7 项 host/source、56 组纯 CPU reference 对照及主线 25/25 次真实 SDK 编译 | 设备矩阵与完整 CI |
| I3 | Chunk Scaled Dot KKT，2 个原 Python 文件 | 完整原算法及 `run_*` CLI 保留；源码/SDK 计入 I1/I6 汇总 | 原默认 L0 CLI 及其余层级设备测试 |
| I4 | W4A4，4 个核心文件、2 个 Python 文件及 guide 增量 | 原变更 token 保真核对；USE_ASCEND 原生编译通过；原 TA 15 个设备节点及默认 CLI 通过 | 主线设备验收仍 pending：首次尝试在 dtype wrapper 发 kernel 前被拒绝，独立 runtime 修复和重新构建正在单独验证 |
| I5 | GLM-5.3 kpool compress，4 个原 Python 文件 | 完整原资产、CPU reference 和真实 SDK 编译，计入 I2 汇总 | 设备矩阵与完整 CI |
| I6 | Depthwise Conv2D，2 个原 Python 文件 | 保留 NCHW 算法及四个原 pytest 聚合层级；源码/SDK 计入 I1/I3 汇总 | 原 operator pytest 与设备矩阵 |
| I7 | op-profiling skill，6 个资产及版本化 guide | 原始来源归档、完整 parser AST 和静态 CSV 对照，6 项 CPU 测试通过；既有主线 skills 保持 | 迁移资产/静态审查已通过；未运行 msprof 或外部 Agent 评测仅说明证据范围，不新增迁移验收门槛 |

逐文件来源、目标、SHA256、owner 和验收边界见 `maint/ascend_a2a3/upstream_463c234c_ports.json` 及其引用的四份 owner manifest。29 项来源变更均已核对固定 Git 来源和当前目标；七组资产集成不替代行为验收。

旧 `ecosystem_manifest.json` 的 394 项来源台账保持原样。新增 17 项生态资产使来源资产盘点达到 411；另有 6 项新增 skill 资产。411 是来源数量，不是设备通过数。文档当前为 231 项映射、22 个 TA skills、41 份原始归档；原 0deb 的 225/21/34 字段明确保留为历史审查范围，旧 guide 的原文和 hash 仍在历史版本中。

## 编译、CPU 与设备证据的边界

本轮 USE_ASCEND=ON 的 3 个 C++ TU 增量构建成功，实际 libtilelang SHA256 为 `376afdbe0739cad1bf13d00862a70b94c96a2332ae4e347ea289b3b650b3fdbc`。149 项旧回归、54 项新增 native/source 检查通过；54 项包含 I1/I3/I6 的 42 项，不能重复相加。I2/I5 的 7 项 host/source、56 组纯 CPU reference、主线 25/25 次真实 SDK 编译，以及 I7 的 6 项 CPU 文档/parser 检查分别记录。I2/I5 的原 TA 默认 host source 有 6 项 SDK 失败，按实际 NPU storage 路径受控生成的原 TA source 为 25/25 SDK 通过；该受控源码比较不等于原 TA 默认 host 编译或设备通过。

原 TA 最新来源已用 `install_ascend.sh` 重新构建并验证加载。原 TA W4A4 的 15 个设备节点及默认 CLI 通过；主线首次尝试停在 dtype wrapper，尚未发 kernel。主线 runtime 修复后的 Cython/native 身份和设备结果须另报，不能把源码、SDK、CPU reference 或原 TA 的通过写成主线设备通过。

此前 I1/I3/I6 的 19-case/端最小 L0 方案仅保留为未执行的历史参考。当前下一步是核对并提交完整 TA CI 等效命令，等待用户确认后运行迁移队列的 180 个 bench 任务、27 个 operator pytest 文件以及原 TA testing 范围；不再以 L0 子集代替这项完整验收。执行时固定新 HEAD，保持原 CLI/pytest、并发、marker、case、golden 和退出判定；设备窗口、独立 cache 和 owned-process 清理由统筹控制。

## TA CI 的原跑法与迁移后路径

本轮原 TA legacy bench 登记为 177→183，新增六个脚本任务；active operator source/test 对为 24→27，新增 `depthwise_conv_2d`、`kpool`、`situ_glu`。原始最新 145 项 registry 及其差异保存在 `maint/ascend_a2a3/upstream_463c234c/`，属于受控来源数据，尚未激活成当前主线 workflow。保留 TA 原 runner、容器、benchmark 收集和测试控制流；迁移后的安装方式和文件路径按主线适配，完整 TA 组装后置。

当前主仓尚无 TA 的 `ci_cd.yml`、`bench_test.sh` 和 `resolve_operator_tests.py` 活动入口。独立算子 CLI/pytest 已可按迁移路径调用；完整批跑还需准备并核对等效入口，不能引用旧 fef 组装路径冒充已集成。用户已要求确认完整 TA CI 等效命令，完整设备批跑等待其确认。新增六个 legacy 任务分别为 Dilation 源脚本和测试脚本、SiTU bench、W4A4 源脚本、KKT 源脚本和测试脚本；完整源路径列于中央 manifest。

- legacy bench 按 `bench_test.sh` 的原脚本枚举、CLI、退出码和成功标志运行，路径映到 `examples/ascend_a2a3/`、`examples_experiment/ascend_a2a3/`。Dilation 保持原 legacy CLI 路由；KKT 保持默认 L0、`run_*` 非 pytest 协议。
- active operator 保持原 source/test 配对及 `pytest --forked` 的 n4 增量/n8 全量范围。Depthwise 保留四个原 pytest 聚合节点，不把单个 L0 结果扩写成全文件通过。
- 原 `testing/` 的测试范围单独登记，不能用 example 的 source 编译替代。当前未完整运行最新 183 bench、27 operator 对及 testing 并行矩阵。
- 原上游 simple_fusion 两项登记合并后指向源树不存在的 `example_simple_fusion.py`，保留为源问题，不能凭空创造算子或计为通过。

旧 0deb bench 177→174 候选收集、activation 10 CLI 和选定 23 pytest 节点的设备证据均保留为历史范围。它们不证明本轮新增任务、完整 CI、性能或最新安装包验收。

## 阶段 3 尚待验收的范围

继续复核完整迁移资产、现代前端适配、原 pass/codegen 与生成代码保真、TA CI 测试路径，以及文档/skills/Agent 资产。当前阻塞和后续结果在工作区 `refactor_main/` 的 reports 中追加；这些忽略目录不是随仓分发的永久证据。

CUDA 完整回归按用户范围暂不阻塞；无 A5 硬件不等待 A5 实机回归。仍要求 USE_ASCEND 成功编译、当前环境的 A2/A3 功能正常，并用可运行测试和隔离审查保护 A5。Carver 保持后续评估范围，普通 autotune 不继承该豁免。
