---
name: tilelang-mode-conversion
description: TileLang Ascend Developer/Expert 模式互转。将 Developer 模式代码转为 Expert 模式，或将 Expert 模式代码转为 Developer 模式。触发：用户要求在两种编程模式间转换，或需要理解两种模式的差异时。
---

# TileLang Ascend Developer/Expert 模式互转

---

## 转换规则索引

| 转换类别 | 核心文档 | 典型场景 |
|---------|---------|---------|
| **模式概览与对比** | [mode-overview.md](references/mode-overview.md) | 理解两种模式的定位、差异、适用场景 |
| **内存分配转换** | [convert-memory.md](references/convert-memory.md) | alloc_shared/fragment ↔ alloc_ub/L1/L0A/L0B/L0C |
| **计算原语转换** | [convert-compute.md](references/convert-compute.md) | T.Parallel+符号API ↔ T.tile.xxx |
| **同步与作用域转换** | [convert-sync-scope.md](references/convert-sync-scope.md) | 自动同步 ↔ 手动 T.Scope + set_flag/wait_flag/barrier |
| **pass_configs 转换** | [convert-passconfigs.md](references/convert-passconfigs.md) | 自动化开关的开启/关闭 |
| **完整转换示例** | [convert-examples.md](references/convert-examples.md) | GEMM、ElementWise Add、MatMul+Add、FlashAttention 完整对照 |

---

## 快速判断模式

| 特征 | Developer 模式 | Expert 模式 |
|------|---------------|-------------|
| 内存分配 | `T.alloc_shared` / `T.alloc_fragment` | `T.alloc_ub` / `T.alloc_L1` / `T.alloc_L0A/L0B/L0C` |
| 计算方式 | `T.Parallel` + `+/-/*/÷/T.exp` 等 | `T.tile.add/sub/mul/exp` 等 |
| 作用域 | 无需 `T.Scope`，编译器自动分离 | 显式 `T.Scope("C")` / `T.Scope("V")` |
| 同步 | 自动（pass_configs 开启） | 手动 `T.barrier_all` / `T.set_flag` / `T.wait_flag` |
| pass_configs | 需开启 AUTO_SYNC/MEMORY_PLANNING/AUTO_CV_COMBINE/AUTO_CV_SYNC | 通常不需要（手动控制） |
| 适用场景 | 快速开发、跨平台兼容 | 极致性能优化、精确硬件控制 |

---

## 转换方向

### Developer → Expert

当需要更精细的性能控制时，将 Developer 模式代码转为 Expert 模式：
1. 替换内存分配 API → [convert-memory.md](references/convert-memory.md)
2. 替换计算原语 → [convert-compute.md](references/convert-compute.md)
3. 添加 T.Scope 和手动同步 → [convert-sync-scope.md](references/convert-sync-scope.md)
4. 移除自动化 pass_configs → [convert-passconfigs.md](references/convert-passconfigs.md)

### Expert → Developer

当需要简化代码或提升可维护性时，将 Expert 模式代码转为 Developer 模式：
1. 替换内存分配 API → [convert-memory.md](references/convert-memory.md)
2. 替换计算原语 → [convert-compute.md](references/convert-compute.md)
3. 移除 T.Scope 和手动同步 → [convert-sync-scope.md](references/convert-sync-scope.md)
4. 启用自动化 pass_configs → [convert-passconfigs.md](references/convert-passconfigs.md)
