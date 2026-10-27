# {Pass 名称} Pass 设计文档

## 1. 概述

### 1.1 Pass 名称

{Pass 名称}

### 1.2 功能描述

{一句话描述 Pass 的功能}

### 1.3 解决的问题

{描述 Pass 要解决的具体问题，如：现有编译流程中缺少某种优化、某些 IR 结构不合法、需要收集特定信息等}

### 1.4 Pass 类型

| 类型 | 说明 |
|------|------|
| **IR 变换类型** | {修改 IR / 收集信息 / 验证 IR} |
| **平台范围** | {平台无关 / Ascend 特定} |
| **优化类别** | {Lowering / 合法化 / 内存 / 流水线 / 同步 / 信息收集 / 其他} |

---

## 2. Pass 定位

### 2.1 阶段归属

**阶段**: {Phase 1: LowerAndLegalize / Phase 2: OptimizeForTarget}

### 2.2 选型理由

{基于以下四个原则的分析结果}

| 原则 | 分析结果 |
|------|----------|
| **功能归属** | {如：属于硬件优化，应放在 Phase 2} |
| **数据依赖** | {如：需要 buffer_shapess（来自 Phase 1），可放在 Phase 2} |
| **输出供给** | {如：产生 address_map，供 AscendSyncInsert 使用，必须在 Phase 2} |
| **语义范围** | {如：Ascend 特定优化，放在 Phase 2} |

### 2.3 Pipeline 位置

**具体位置**: {如：Phase 2，步骤 13，在 `AscendStorageRewrite` 后，`tir.transform.UnrollLoop` 前}

**位置代码示例**:
```python
# tilelang/ascend/a2a3/pipeline.py
def OptimizeForTarget(mod, target, platform):
    # ... (步骤 1-12)
    mod = tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    # ... (步骤 2-12)
    mod = AscendStorageRewrite(is_npu)(mod)  # 步骤 13
    # ===== 新增 Pass =====
    mod = {Pass名称}()(mod)  # 步骤 X
    # =====
    mod = tir.transform.UnrollLoop()(mod)  # 步骤 14
    # ... (后续 Pass)
```

### 2.4 依赖关系分析

#### 上游依赖（输入数据）

| 数据名称 | 产生 Pass | 阶段 | 获取方式 |
|----------|-----------|------|----------|
| {buffer scope} | `AscendInferBufferScope` | Phase 1 步骤 1 | `f->GetAttr<Map<...>>(...)` |
| {buffer_shapess} | `CollectBufferShapes` | Phase 1 步骤 8 | `f->GetAttr<Map<Var, Array<PrimExpr>>>(...)` |
| {address_map} | `AscendMemoryPlanning` | Phase 2 步骤 20 | `f->GetAttr<Map<...>>(...)` |
| ... | ... | ... | ... |

#### 下游供给（输出数据）

| 数据名称 | 使用 Pass | 阶段 | 传递方式 |
|----------|-----------|------|----------|
| {output_attr} | {PassA} | Phase X | `f->attrs[...]` |
| ... | ... | ... | ... |

### 2.5 数据流图

```
Phase 1 / Phase 2 前序 Pass
    ↓
    输出: {数据A}, {数据B}
    ↓
[{Pass名称}] ← 本 Pass
    ↓
    输出: {数据C}, {数据D}
    ↓
Phase X 后续 Pass
```

---

## 3. IR 变换设计

### 3.1 输入 IR 结构

**伪 IR 示例**（本示例为伪 IR 格式，真实 TIR 结构见 ir-examples.md）:

```
# 变换前的 IR 结构
PrimFunc {
  attrs: {
    {已有 attrs}
  }
  body: {
    # {关键 IR 结构描述}
    ForNode {
      loop_var: i
      body: {
        BufferStoreNode { ... }
      }
    }
  }
}
```

### 3.2 输出 IR 结构

**伪 IR 示例**:

```
# 变换后的 IR 结构
PrimFunc {
  attrs: {
    {新增 attrs ←}
    {已有 attrs}
  }
  body: {
    # {变换后的 IR 结构 ←}
    ForNode {
      loop_var: i
      body: {
        BufferStoreNode { ... }  ← {变化说明}
        EvaluateNode { ... }     ← {新增节点}
      }
    }
  }
}
```

### 3.3 变换逻辑伪代码

```python
# 伪代码示例 (精简版)

输入: PrimFunc
处理流程:
  1. 遍历 IR 结构 → VisitStmt_()           # [备注: 深度优先遍历特定节点]
  2. 匹配目标模式 → MatchPattern()         # [备注: 检查是否为特定 IR 结构]
  3. 执行变换操作 → TransformStmt()        # [备注: 生成新 IR 或收集信息]
  4. 更新 attrs → UpdateAttrs()            # [备注: 如需输出数据，设置 attrs]
输出: 变换后的 PrimFunc

# 关键点备注
- VisitStmt_: 继承 IRMutatorWithAnalyzer/StmtExprVisitor
- MatchPattern: 检查 buffer scope、op type 等属性
- TransformStmt: 根据规则生成新语句或收集信息
```

### 3.4 变换要点

- **变化点1**: {如：新增 EvaluateNode 插入同步指令}
- **变化点2**: {如：修改 BufferStoreNode 的访问模式}
- **保持语义**: {如：变换后保持计算语义不变，仅优化执行顺序}

---

## 4. 实现方案

### 4.1 C++ 类设计

**核心类**: `{类名}` (继承 `{父类名}`)

**父类选择依据**:
| 父类 | 适用场景 | 本 Pass 选择 |
|------|----------|--------------|
| `IRMutatorWithAnalyzer` | 修改 IR 结构 | {选择理由} |
| `StmtExprVisitor` | 收集信息、不修改 IR | {选择理由} |
| `StmtExprMutator` | 简单 IR 变换 | {选择理由} |

### 4.2 核心方法

| 方法名 | 功能 | 关键逻辑 |
|--------|------|----------|
| `Substitute()` | Pass 入口 | 读取 attrs → 执行变换 → 返回 PrimFunc |
| `VisitStmt_(NodeType)` | 处理特定节点 | {匹配 → 变换逻辑} |
| `MatchPattern()` | 模式匹配 | {检查条件} |
| `TransformStmt()` | 执行变换 | {生成新 IR} |
| `UpdateAttrs()` | 更新 attrs | {设置输出数据} |

### 4.3 当前实现接入

填写完整native类与辅助函数、隔离FFI、Python require_native wrapper、CMake gate和必要测试。具体TIRX/注册占位模板见pass-generate/templates/pass-impl-skeleton-template.md；不能复制旧TVM_REGISTER_GLOBAL或_ffi_api调用。

### 4.4 参数与配置

记录每个参数的真实类型/来源/default；配置enum唯一owner为tilelang/transform/pass_config.py，本地归一化在tilelang/ascend/a2a3/pass_config.py。native配置注册及默认值先查实际GetConfig和目标默认，不预设False。

### 4.5 Pipeline上下文

给出当前tilelang/ascend/a2a3/pipeline.py中前后调用、条件和完整参数，并证明producer/consumer事实仍成立；model=A2/A3，keys决定方言。AscendC MaskLegalize后不得再rewrite，不用旧phase步骤数字猜插入位置。

## 5. 测试方案

### 5.1 功能测试

| 测试项 | 测试内容 | 验证方法 |
|--------|----------|----------|
| {基础功能} | {变换后的 IR 是否正确} | {检查 attrs / IR 结构} |
| {输入依赖} | {能否正确读取上游 attrs} | {设置 mock attrs 测试} |
| {输出供给} | {能否正确设置下游 attrs} | {检查 attrs 是否可被后续 Pass 读取} |

### 5.2 依赖测试

| 测试项 | 测试内容 |
|--------|----------|
| **上游缺失** | 当上游 attrs 缺失时，Pass 是否正确处理（报错 / 跳过） |
| **顺序错误** | 当 Pass 执行顺序错误时，编译是否失败 |

### 5.3 边界测试

| 测试项 | 测试内容 |
|--------|----------|
| **空 IR** | 当 IR 为空或无目标节点时，Pass 是否正确处理 |
| **极端数据** | 当 attrs 包含极端值（空 map、超大 size）时，Pass 是否正确处理 |

### 5.4 性能测试（可选）

| 测试项 | 测试内容 | 指标 |
|--------|----------|------|
| {编译时间} | Pass 是否显著增加编译时间 | {时间阈值} |
| {生成代码性能} | Pass 是否改善生成代码性能 | {吞吐量/延迟} |

---

## 6. 风险点与注意事项

### 6.1 已知约束

{列出本 Pass 在 TileLang-Ascend 上的已知限制}

### 6.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| {attrs 缺失} | {上游 Pass 未执行} | {编译失败} | {检查 Pass 顺序} |
| {IR 结构不符} | {输入 IR 不符合预期} | {变换失败} | {添加前置检查} |
| ... | ... | ... | ... |

### 6.3 与其他 Pass 的交互影响

| Pass | 交互关系 | 需要注意的点 |
|------|----------|--------------|
| {PassA} | {上游依赖} | {必须在本 Pass 前执行} |
| {PassB} | {下游供给} | {本 Pass 输出必须符合 PassB 预期} |
| {PassC} | {功能冲突} | {不能同时启用} |

---

## 7. 交付清单

### 7.1 目录结构

```
src/ascend/a2a3/transform/
├── {pass_name}.cc           # C++ 实现
└── common/                  # 公共工具（如需新增）

tilelang/ascend/a2a3/transform/
├── __init__.py              # Python Wrapper
└── pass_config.py           # 配置键（如需新增）

tilelang/engine/
└── phase.py                 # Pipeline 集成
```

### 7.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `src/ascend/a2a3/transform/{pass_name}.cc` | {待实现} | C++ 实现 |
| `tilelang/ascend/a2a3/transform/__init__.py` | {待修改} | 添加 Python Wrapper |
| `tilelang/ascend/a2a3/pass_config.py` | {待修改（可选）} | 添加配置键 |
| `tilelang/ascend/a2a3/pipeline.py` | {待修改} | Pipeline 集成 |
| `pass-design.md` | {已完成} | 本设计文档 |

### 7.3 实现顺序

1. ✅ 设计文档（pass-design.md）
2. ⬜ C++ 实现（src/ascend/a2a3/transform/{pass_name}.cc）
3. ⬜ Python Wrapper（tilelang/ascend/a2a3/transform/__init__.py）
4. ⬜ 配置键（可选）
5. ⬜ Pipeline 集成（tilelang/ascend/a2a3/pipeline.py）
6. ⬜ 功能测试
7. ⬜ 依赖测试

### 7.4 后续步骤

完成本设计文档后，建议：
1. 使用 **tilelang-pass-workflow-analyzer** skill 查看详细 Pass 工作流
2. 使用 **tilelang-pass-analyzer** skill 查看类似 Pass 的实现细节
3. 开始 C++ 实现前，参考 `src/ascend/a2a3/transform/` 中相似 Pass 的代码结构
