# A2/A3 Pass 工作流与角色

本指南是TA独有工作流在现代主线的接线说明；原全文完整存档。main backend/build/dev-guide/tvm-ir技能继续为架构/实现规范，用户已有授权/设计与会话范围优先，不强制重复批准或拆出不存在的工具。

## 角色与交付

| 技能 | 职责 |
|---|---|
| tilelang-pass-analyzer | 当前pass行为、完整算法/支持域、输入输出facts与来源对照 |
| tilelang-pass-workflow-analyzer | 真实完整pipeline顺序/条件、依赖与插入边界 |
| tilelang-pass-design | 可实施设计、TIRX结构/状态、唯一owner与必要验证方案 |
| tilelang-pass-generate | 完整本地实现、FFI/Python/CMake/pipeline接线及必要聚焦回归 |

有明确任务时可以连续完成这些活动，不因没有独立design.md或旧流程要求等待骨架批准而停止已授权工作。需要委派时沿用户明确的协作范围，保持文件owner和原算法source fidelity；未明确的外部动作不由技能名称自动授权。

## 权威来源与边界

当前源码为tilelang/ascend/a2a3/pipeline.py、transform wrapper与src/ascend/a2a3本地native。旧references/pass-designs记录固定TA算法、行号与伪IR，只是来源背景，不能直接复制旧TIR/TVM_REGISTER_GLOBAL/旧全局_ffi_api模板。

TIRX可缺Var用Optional、identity map和结构相等分清、COW与attrs变化有证据。原launch标记/cthread、layout_map、workspace、OperationConfig与terminal mask facts依据真实producer/consumer保持，不能靠默认分支猜。

AscendC最后InstructionSelection/MaskLegalize之后不得rewrite；PTO有独立条件和ABI，不能据model字符串选方言。共享Op/配置有唯一owner，本地新增FFI隔离不覆盖A5。USE_ASCEND=OFF不编本地TU。

## 验证与报告

完整同输入TA/current算法和源码证据优先；机械归一化仅消除已证明表示差异，原非法输入的新诊断单列。实际native构建/loader freshness、聚焦行为测试、公开compile和设备数值/性能各自记录，不能把注册查询当算法通过或把必要UT/ST留空。

使用原factory/decorator/config/shape/golden，不绕pipeline/禁pass/放阈值/换手写核刷绿。只保任务独立cache，不清用户默认cache；原失败与所有原样本保留。最后报告来源pin、完整组件、必要适配、真实命令和未完成项。
