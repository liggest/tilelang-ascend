# pass_configs 转换规则

## 概述

`pass_configs` 控制编译器的自动化行为。Developer 模式依赖这些自动化开关，Expert 模式通常手动控制。

## Developer 模式标准配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动 Cube/Vector 分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,          # 自动核内同步插入
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,    # 自动内存规划/复用
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,       # 自动核间同步插入
}
```

## Expert 模式标准配置

```python
# Expert 模式通常不设 pass_configs，或只保留基础配置
@tilelang.jit(out_idx=[-1])  # 无 pass_configs
```

## 各开关详解

### TL_ASCEND_AUTO_CV_COMBINE

**功能**：自动将 kernel 中的 Cube 操作和 Vector 操作分离到 `T.Scope("C")` 和 `T.Scope("V")`。

| 模式 | 设置 | 原因 |
|------|------|------|
| Developer | `True` | 无需手写 T.Scope |
| Expert | `False` 或不设 | 手写 T.Scope |

### TL_ASCEND_AUTO_SYNC

**功能**：自动在数据搬运和计算之间插入 `T.barrier_all()` 等同步指令。

| 模式 | 设置 | 原因 |
|------|------|------|
| Developer | `True` | 无需手写 barrier |
| Expert | `False` 或不设 | 手写 barrier/set_flag/wait_flag |

### TL_ASCEND_MEMORY_PLANNING

**功能**：自动分析 buffer 生命周期，实现内存复用。

| 模式 | 设置 | 原因 |
|------|------|------|
| Developer | `True` | 自动复用 buffer 空间 |
| Expert | 可选 `True` | Expert 模式也可受益；或使用 `T.annotate_address` 手动规划 |

### TL_ASCEND_AUTO_CV_SYNC

**功能**：自动在 Cube Scope 和 Vector Scope 之间插入 `T.set_cross_flag` / `T.wait_cross_flag`。

| 模式 | 设置 | 原因 |
|------|------|------|
| Developer | `True` | 无需手写核间同步 |
| Expert | `False` 或不设 | 手写 set_cross_flag/wait_cross_flag |

## 转换步骤

### Developer → Expert

1. 移除 `TL_ASCEND_AUTO_CV_COMBINE: True`
2. 移除 `TL_ASCEND_AUTO_SYNC: True`
3. 移除 `TL_ASCEND_AUTO_CV_SYNC: True`
4. 可保留 `TL_ASCEND_MEMORY_PLANNING: True`（或改用 `T.annotate_address` 手动规划）
5. 同时在代码中添加显式 `T.Scope`、`T.barrier_all`、`T.set_cross_flag` 等

### Expert → Developer

1. 添加 `TL_ASCEND_AUTO_CV_COMBINE: True`
2. 添加 `TL_ASCEND_AUTO_SYNC: True`
3. 添加 `TL_ASCEND_MEMORY_PLANNING: True`
4. 添加 `TL_ASCEND_AUTO_CV_SYNC: True`
5. 同时从代码中移除显式 `T.Scope`、`T.barrier_all`、`T.set_cross_flag` 等

## 混合模式 pass_configs

当 Developer 模式中混用少量 Expert API 时（如 `T.tile.fill`），仍然使用 Developer 的完整 pass_configs：

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}
```
