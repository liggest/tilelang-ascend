---
name: tilelang-error-fixer
description: TileLang-Ascend 错误诊断、调试与修复技能。融合了 GDB 故障定位、IR Dump 分析、目标代码倒推（AOT）的诊断能力。提供从环境检查、错误定位、AOT验证到 C++ Pass 修复的端到端工作流。
---

> 适用范围：TA A2/A3 资产；主线 backend/build/IR/A5 技能保持独立。先读 docs/ascend_a2a3/migration.md，已有用户授权、模式与原配置优先；不重复请求同一批准，不因搬入本 skill 自动授权安装/外发/push/定时任务。

# TileLang-Ascend Pass 错误诊断与修复技能

本技能提供 TileLang-Ascend 框架中 Pass 层面错误的完整诊断和修复能力，支持从前端报错、Core Dump 崩溃追溯到后端 C++ Pass 的修复，并集成 AOT 目标代码验证机制。

## 使用场景
- **遇到 Core Dump 或 Segmentation Fault 等严重崩溃，且无其他明确错误信息**
- TileLang 编译报错或生成的 Ascend 目标代码结果不符合预期（精度/性能问题）
- 编写或修改 `src/ascend/a2a3/transform/*.cc` 下的新 Pass 时出现逻辑异常
- 需要使用 AOT 提前编译模式进行算子调试与倒推

## 触发机制
当用户输入包含以下关键字时自动触发：
- **"Core Dump"**、**"Segment Fault"**、**"段错误"**
- **"TileLang 编译报错"**、**"Pass 生成代码错误"**
- **"排查精度问题，分析 IR"**、**"使用 AOT 调试这个算子"**
- **"修复 TileLang Pass"**

## 工作流程

### 步骤 1：环境与状态检查
在进行任何 Pass 调试前，必须确保底层环境正常。
1. 执行环境检查脚本验证 `TL_ROOT`, `PYTHONPATH`, `ASCEND_HOME_PATH`, `LD_LIBRARY_PATH` 等变量。
2. 确认 C++ 工程是否处于最新编译状态。

### 步骤 2：多维度错误信息获取 (Dump & Catch)
根据报错类型，采取不同的信息获取策略：

1. **严重崩溃 (Core Dump / Segment Fault)**:
   - 现象：脚本运行直接崩溃，没有任何 Python 异常堆栈。
   - 动作：使用 GDB 挂载 Python 进程进行排查。
     ```bash
     gdb --args python xxx.py
     # 在 GDB 命令行输入：
     (gdb) run
     # 崩溃后输入 bt 查看堆栈，精确定位是 C++ 源码中哪个函数/Pass 引发了内存违规
     ```

2. **C++ 编译报错 (Build Error)**:
   - 读取本次编译异常/driver记录的实际source路径、完整argv和stderr定位语法或底层API；不要按 `/tmp` 最新文件猜测或读取其它任务的产物。

3. **逻辑/精度异常 (IR Dump)**:
   - 从已解析的 BackendContext/PassPipeline 追踪 `tilelang/ascend/a2a3/pipeline.py` 的真实 producer/consumer。复用 `tilelang/instrumentation/` 的 pass instrumentation 或当前 PassContext instruments 保存前后 IR；不要在共享 engine.lower 恢复旧 target dispatch 或复制已不存在的 shared OptimizeForTarget 调用。

4. **目标代码分析 (Target Dump)**:
   - 通过 `@tilelang.jit(debug_root_path=...)` 或 `.get_kernel_source()` 抓取最终生成的 C++ 核函数。

### 步骤 3：核心错误原因分析 (TileLang 专属模式)
分析获取的 IR、目标代码或 GDB 堆栈，重点排查 TileLang-Ascend 特有的雷区：

1. **循环嵌套与次数匹配**：对比 Cube 和 Vec 部分的循环 `extent`，诊断类似 "Vec loop times is not enough" 的报错。
2. **跨核同步点 (Sync Points)**：追踪 `ascend_auto_set_cross_flag` / `wait_flag`，排查死锁或数据竞争。
3. **工作空间 (Workspace) 访问**：检查张量切片范围，确认访问是否越界。
4. **数据依赖与流水线 (Pipeline)**：分析 `T.Pipelined` 的 `num_stages`，检查读写重叠冲突。

### 步骤 4：修复方案制定（必要时使用 AOT 隔离）

根据步骤 3 的证据提出最小修复。需要区分 TileLang lowering 与目标代码行为时，可用 AOT
构造隔离实验：

**方案 A：直接修复 C++ Pass（适用于 GDB 已明确指出问题代码行的情况）**
- 直接定位 `src/ascend/a2a3/transform/` 或 `src/ascend/a2a3/codegen/` 下的 `.cc` 文件，提供代码修改建议。

**方案 B：AOT counterfactual**

- 固定 TA `examples/gemm_aot` 是后续待迁的 source-only 参考；当前缺失的 `examples/ascend_a2a3/gemm_aot` 不是可执行现代流程。使用实际公开 compile/JIT 的目标代码及当前共享 LibraryGenerator/冻结 compiler config 合同设计隔离实验，不直接套旧 ctypes wrapper，也不手写另一套 Bisheng include/flags。
- 对生成 source 的实验性修改只能放在 disposable 副本中。实验通过只能定位问题层，最终
  修复仍应落到产生该 source 的 frontend/pass/codegen，并用原始重现验证。

### 步骤 5：执行修复与 C++ 重新编译
1. 展示可审 C++ Diff；按会话已有授权实施，不为重复批准停工。超出既有范围的外部动作另行处理。
2. 在隔离工作树保留原始失败、源码版本和本次补丁，保护其它改动。
3. 修改 native 后按 build 技能对实际配置的目录执行 `cmake --build <build-dir> --parallel <jobs>`，兼容 Ninja/Make；确认新的库路径/hash后再验结果，不拿旧 `.so` 通过当修复生效。

### 步骤 6：修复验证与回滚
1. 重新执行用户的重现脚本。
2. 对比修复前后的 IR 图（`diff before.tir after.tir`）或目标代码，确认修改符合预期。
3. **失败处理**：保留失败日志/IR/source，调整本次自有补丁后继续定位；不得自动回滚他人改动或清空 `/tmp`、用户默认 cache。必要回退仅针对本次隔离变更。

---

## 输出格式

### TileLang Pass 诊断与修复报告

```markdown
## TileLang Pass 分析报告

### 基本信息
- **触发报错模块**：[如：Segmentation Fault / CombineCV Pass]
- **涉及算子/脚本**：`example_flash_attention.py`
- **获取的调试信息**：[如：GDB Backtrace 指向 `src/ascend/a2a3/transform/ascend_combinecv.cc` / `lower.py` 打印的 IR 图]

### 错误原因分析
- **错误分类**：[如：空指针解引用 / 循环迭代不匹配]
- **详细分析**：通过 GDB 堆栈发现，在处理 `T.Pipelined` 节点时，试图访问不存在的 `extent` 属性，导致 Core Dump。
- **定位代码**：`src/ascend/a2a3/transform/cross_core_pipeline.cc: 89行`

### 修复方案建议（按已有授权范围）
**建议直接修复 Pass C++ 源码**
1. 修改 `src/ascend/a2a3/transform/cross_core_pipeline.cc`，在访问节点前增加类型检查...
2. 代码 Diff:
   ```cpp
   // Before
   auto extent = op->extent.as<IntImmNode>()->value;
   // After
   if (const auto* imm = op->extent.as<IntImmNode>()) {
       auto extent = imm->value;
       // ...
   }
   ```
3. 修复后在实际配置目录执行 `cmake --build <build-dir> --parallel <jobs>` 并核验装载来源。

*(原 TA gemm_aot 为待迁来源参考；当前现代隔离实验须沿实际 compile/compiler 合同，不得把生成 source 当作长期修复。)*

### 执行与验证结果
- 编译状态：✅ 成功
- 测试脚本验证：✅ 通过
- Core Dump 状态：已解决
```
