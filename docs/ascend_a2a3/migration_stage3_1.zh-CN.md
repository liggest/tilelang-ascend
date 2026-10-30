# 阶段 3.1：迁移后续遗留任务

## 当前安排：迁移保真纠正（2026-10-30）

整体目标不变：将 TA A2/A3 能力迁入现代 TileLang，保留原算法、pass/codegen 行为、示例和测试逻辑，只做必要兼容适配。阶段 3 仍未验收。本轮从 `045edc59` 检查最终文件树，来源固定为 TA `ascendc_pto@463c234c`；先逐文件对照，再追溯引入额外改写的提交，以纠正提交修复当前树，不重写已发布历史。

此前 `69975dd1` 的完整 CI 和后继修复的针对性验证见 [完整基线](validation/full_ci_20261030.zh-CN.md)与[修复记录](validation/ci_repairs_20261030.zh-CN.md)。这些结果保留原范围，本轮不重复批跑。以下任务完成并不自动代表阶段 3 验收通过。

用户本轮裁定：

- 原例子没有 `main()` 就不新增；原来有则保留原签名、调用方式和顶层结构。不得为安全导入或审计隔离重组产品示例。
- 原 `tilelang.cache.clear_cache()` 可在原调用位置改成相应根模块别名的 `tilelang.disable_cache()`。这项语义差异已明确允许，不增加缓存目录要求、缓存实例清理或新的兼容 API。
- 保留必要的语言命名空间、路径、现代 JIT/eager/宏/自动调优接口适配，并逐项说明依据；不改 kernel、golden、参数、seed、测试分级或成功判定，不顺手修原有逻辑问题。
- 原文档不附加迁移声明或审计标签；原测试不添加迁移专用 marker、隔离 fixture 要求。审查材料集中留在迁移台账，原有 `ci_skip`、`low_priority` 等内容保留。
- 验证以完整来源差异、定义和顶层结构对照、语法检查与独立复核为主。只有静态证据不足时才做少量针对检查；当前不运行 NPU、SDK 大矩阵或完整 daily/manual，最终批跑后置。
- 所有阶段性材料都须退出最终交付，不限于 `maint/ascend_a2a3/`，也不按 `.json`、`.md`、`.gz` 等扩展名一刀切。迁移期间有核对需要的可暂留 Git；先登记用途、消费者和最终去向，待整体稳定后统一清理，不立即扩展为全仓删除。

| 编号 | 事项与负责人 | 交付与验收 |
|---|---|---|
| 3.1-07 | 来源差异与例外清单：统筹、pi 独立审计 | 固定原/新路径，区分必要适配与额外改写；不能只用 factory AST 或 rename 标签代表整份文件保真 |
| 3.1-08 | `examples/ascend_a2a3/` 示例、测试与关联脚本纠正：子 Agent | 逐源恢复入口、原测试与互操作脚本；保必要现代 API 接线，撤除额外 guard、包装和输入限制 |
| 3.1-09 | `examples_experiment/` 示例与测试纠正：子 Agent | 保原目录、原控制流和测试协议；自动调优/launcher 必要适配单列复核 |
| 3.1-10 | 两个目录的原 Markdown 恢复：pi `cannbot/deepseek-v4-flash` | 删除原文没有的声明与审计标签，仅保必要路径适配；不重写原设计、命令、结果与说明 |
| 3.1-11 | 审查加载器、测试辅助与台账同步：独立子 Agent | 将审查隔离留在辅助工具，更新受影响的来源 hash/适配说明；不让工具反过来要求改产品入口 |
| 3.1-12 | 最终文件独立复核：pi `cannbot/glm-5.3-flash`、统筹 | 检查整文件、顶层入口、测试、文档及剩余适配；发现未解释逻辑变化则交回原负责人 |
| 3.1-13 | 阶段提交与分支同步：统筹 | 按既有提交格式记录，保持两仓本地 `refactor_main` 一致，完成后更新 bridge 两个快照与现有对比 PR |
| 3.1-14 | 所有迁移问题处理后的完整回归：后置 | 保持 TA 原 daily/manual 逻辑；本轮不执行，不以静态纠正结果替代批跑结果 |
| 3.1-15 | 阶段性材料退出清单：独立子 Agent、pi 审计、统筹 | 盘点迁移新增台账、说明、压缩快照、工具及临时测试的用途和引用；整体稳定后归档至本地 `refactor_main/` 并从交付树移除，保留原 TA 资源与真实长期功能 |

主要实现交由子 Agent 和 pi Agent；统筹负责范围、文件所有权、独立复核和集成。下方原集成快照保留为历史，当前执行范围以上述裁定为准。

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
| I5 | GLM-5.3 kpool compress，4 个原资产（2 Python + 2 Markdown） | 完整原资产、CPU reference 和真实 SDK 编译，计入 I2 汇总 | 设备矩阵与完整 CI |
| I6 | Depthwise Conv2D，2 个原 Python 文件 | 保留 NCHW 算法及四个原 pytest 聚合层级；源码/SDK 计入 I1/I3 汇总 | 原 operator pytest 与设备矩阵 |
| I7 | op-profiling skill，6 个资产及版本化 guide | 原始来源归档、完整 parser AST 和静态 CSV 对照，6 项 CPU 测试通过；既有主线 skills 保持 | 迁移资产/静态审查已通过；未运行 msprof 或外部 Agent 评测仅说明证据范围，不新增迁移验收门槛 |

逐文件来源、目标、SHA256、owner 和验收边界见 `maint/ascend_a2a3/upstream_463c234c_ports.json` 及其引用的四份 owner manifest。29 项来源变更均已核对固定 Git 来源和当前目标；七组资产集成不替代行为验收。

原 0deb 的 394 项 A2/A3 生态来源台账保持原样。新增 17 项生态资产使来源资产盘点达到 411；另有 6 项新增 skill 资产。411 是来源数量，不是设备通过数。文档当前为 231 项映射、22 个 TA skills、41 份原始归档；原 0deb 的 225/21/34 字段明确保留为历史审查范围，旧 guide 的原文和 hash 仍在历史版本中。

## 编译、CPU 与设备证据的边界

本轮 USE_ASCEND=ON 的 3 个 C++ TU 增量构建成功，实际 libtilelang SHA256 为 `376afdbe0739cad1bf13d00862a70b94c96a2332ae4e347ea289b3b650b3fdbc`。149 项旧回归、54 项新增 native/source 检查通过；54 项包含 I1/I3/I6 的 42 项，不能重复相加。I2/I5 的 7 项 host/source、56 组纯 CPU reference、主线 25/25 次真实 SDK 编译，以及 I7 的 6 项 CPU 文档/parser 检查分别记录。I2/I5 的原 TA 默认 host source 有 6 项 SDK 失败，按实际 NPU storage 路径受控生成的原 TA source 为 25/25 SDK 通过；该受控源码比较不等于原 TA 默认 host 编译或设备通过。

原 TA 最新来源已用 `install_ascend.sh` 重新构建并验证加载。原 TA W4A4 的 15 个设备节点及默认 CLI 通过；主线首次尝试停在 dtype wrapper，尚未发 kernel。主线 runtime 修复后的 Cython/native 身份和设备结果须另报，不能把源码、SDK、CPU reference 或原 TA 的通过写成主线设备通过。

此前 I1/I3/I6 的 19-case/端最小 L0 方案仅保留为未执行的历史参考。用户已批准本轮目录调整、等效入口准备及完整 CI 四步工作。61 个实验资产已恢复 `examples_experiment/` 原层级；92 个 A2/A3 专属测试及配套资产已移到 `testing/ascend_a2a3/python/refactor/`，原子目录保留，原 `testing/ascend/` 的 108 个主线文件在本轮搬移中未改动。完整批跑入口已集成并完成 5 项真实 shell/host 检查。

批跑范围是 180 个 bench 任务、27 个 operator pytest 文件，以及整个 `testing/ascend_a2a3/python/`。后者包括原 TA 测试和新增迁移适配测试，报告分别统计来源。保持原 `--forked -n 8 -m 'not ci_skip'`，包含 low_priority，保留原参数、golden 和成功判定。驱动按 bench、testing、operator 顺序分开留存退出码和完整日志；普通失败继续后续队列，总时间预算沿用原 daily/manual 的 360 分钟。

启动前实际 `npu-smi` 显示当前 16 个 NPU 均有外部进程，显存占用约 97%，因此尚未启动完整设备批跑。所需现代环境已补齐 xdist、execnet、einops、scipy，既有 55 个分发包版本保持。设备释放后由统筹复核空闲状态并启动；不将准备、host 检查或设备占用停止计作完整 CI 通过。

## TA CI 的原跑法与迁移后路径

本轮原 TA legacy bench 登记为 177→183，新增六个脚本任务；active operator source/test 对为 24→27，新增 `depthwise_conv_2d`、`kpool`、`situ_glu`。原始最新 145 项 registry 及其差异保存在 `maint/ascend_a2a3/upstream_463c234c/`，属于受控来源数据，尚未激活成当前主线 workflow。保留 TA 原 runner、容器、benchmark 收集和测试控制流；迁移后的安装方式和文件路径按主线适配，完整 TA 组装后置。

当前主仓已具备 `examples/bench_test.sh`、`scripts/ci/{path_layout,ci_scope,resolve_operator_tests}.py`、`ci/pytest-ascend.ini` 和最新 145 项 operator registry。本地全量入口为 `maint/ascend_a2a3/ci_batch_driver.py`；其 plan 模式仅收集清单，执行模式需要统筹核验冻结 HEAD、实际环境及空闲设备。主线 workflow 保持原样，完整 TA 组装仍后置。用户已批准完整流程，当前等待可用设备，不再等待重复执行确认。新增六个 legacy 任务分别为 Dilation 源脚本和测试脚本、SiTU bench、W4A4 源脚本、KKT 源脚本和测试脚本；完整源路径列于中央 manifest。

- legacy bench 按 `bench_test.sh` 的原脚本枚举、CLI、退出码和成功标志运行，路径映到 `examples/ascend_a2a3/`、`examples_experiment/`。Dilation 保持原 legacy CLI 路由；KKT 保持默认 L0、`run_*` 非 pytest 协议。
- active operator 保持原 source/test 配对及 `pytest --forked` 的 n4 增量/n8 全量范围。Depthwise 保留四个原 pytest 聚合节点，不把单个 L0 结果扩写成全文件通过。
- 原 `testing/` 的测试范围单独登记，不能用 example 的 source 编译替代。原始 source 登记为 183 bench；当前迁移目标队列为 180 bench、27 operator 文件和原 testing 范围，完整并行矩阵尚未启动。
- 原上游 simple_fusion 两项登记合并后指向源树不存在的 `example_simple_fusion.py`，保留为源问题，不能凭空创造算子或计为通过。

[TA issue #1950](https://github.com/tile-ai/tilelang-ascend/issues/1950) 的已核对历史 C 基线标题为 2903＝180 bench＋2723 testing 通过，另有 4 testing skip；443 operator 通过独立统计，不计入该标题。新批跑须分别报告 legacy bench、operator pytest 和 testing 三段，不用标题总数代替覆盖范围。

旧 0deb bench 177→174 候选收集、activation 10 CLI 和选定 23 pytest 节点的设备证据均保留为历史范围。它们不证明本轮新增任务、完整 CI、性能或最新安装包验收。

## 阶段 3 尚待验收的范围

继续复核完整迁移资产、现代前端适配、原 pass/codegen 与生成代码保真、TA CI 测试路径，以及文档/skills/Agent 资产。当前阻塞和后续结果在工作区 `refactor_main/` 的 reports 中追加；这些忽略目录不是随仓分发的永久证据。

CUDA 完整回归按用户范围暂不阻塞；无 A5 硬件不等待 A5 实机回归。仍要求 USE_ASCEND 成功编译、当前环境的 A2/A3 功能正常，并用可运行测试和隔离审查保护 A5。Carver 保持后续评估范围，普通 autotune 不继承该豁免。
