# 计算原语转换规则

## 总体原则

- **Developer 模式**：使用 `T.Parallel` 循环 + 符号 API（`+`, `-`, `*`, `/`, `T.exp` 等）
- **Expert 模式**：使用 `T.tile.xxx` 函数式调用（`T.tile.add`, `T.tile.exp` 等）
- **GEMM 和 Reduce 操作**：两种模式共用相同的 API（`T.gemm_v0`, `T.reduce_max` 等）

## 双目运算转换

### Developer → Expert

```python
# Developer 模式
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = a_ub[i, j] + b_ub[i, j]

# Expert 模式
T.tile.add(c_ub, a_ub, b_ub)
```

| Developer (T.Parallel 内) | Expert | 功能 |
|---------------------------|--------|------|
| `c[i,j] = a[i,j] + b[i,j]` | `T.tile.add(c, a, b)` | 加法 |
| `c[i,j] = a[i,j] - b[i,j]` | `T.tile.sub(c, a, b)` | 减法 |
| `c[i,j] = a[i,j] * b[i,j]` | `T.tile.mul(c, a, b)` | 乘法 |
| `c[i,j] = a[i,j] / b[i,j]` | `T.tile.div(c, a, b)` | 除法 |
| `c[i] = T.max(a[i], b[i])` | `T.tile.max(c, a, b)` | 最大值 |
| `c[i] = T.min(a[i], b[i])` | `T.tile.min(c, a, b)` | 最小值 |

### 标量运算

```python
# Developer 模式
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = a_ub[i, j] * sm_scale

# Expert 模式
T.tile.mul(c_ub, a_ub, sm_scale)
```

## 单目运算转换

```python
# Developer 模式
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = T.exp(a_ub[i, j])

# Expert 模式
T.tile.exp(c_ub, a_ub)
```

| Developer (T.Parallel 内) | Expert | 功能 |
|---------------------------|--------|------|
| `T.exp(x)` | `T.tile.exp(dst, src)` | 指数 |
| `T.log(x)` | `T.tile.ln(dst, src)` | 对数 |
| `T.abs(x)` | `T.tile.abs(dst, src)` | 绝对值 |
| `T.sqrt(x)` | `T.tile.sqrt(dst, src)` | 平方根 |
| `T.rsqrt(x)` | `T.tile.rsqrt(dst, src)` | 平方根倒数 |
| `T.max(x, 0)` | `T.tile.relu(dst, src)` | ReLU |

## 行广播运算转换

```python
# Developer 模式 — 自动广播
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = a_ub[i, j] - m_i[i]

# Expert 模式 — 需要手动逐行循环
for h_i in range(block_M // VEC_NUM):
    T.tile.sub(c_ub[h_i, :], a_ub[h_i, :], m_i[h_i])
```

> **重要区别**：Developer 模式的 `T.Parallel` 原生支持行广播（2D buffer 与 1D buffer 运算），Expert 模式需要手动编写 `for h_i in range(...)` 循环逐行处理。

## 数据填充转换

```python
# Developer 模式 — 使用 T.Parallel 赋值
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = 0.0

# Expert 模式
T.tile.fill(c_ub, 0.0)
```

> `T.tile.fill` 在两种模式中都可以使用，Developer 模式也常用它来初始化 buffer。

## GEMM — 两种模式共用

```python
# 两种模式完全相同
T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=True)
```

## Reduce — 两种模式共用

```python
# 两种模式完全相同
T.reduce_max(acc_s_ub, m_i, tmp_ub, dim=-1)
T.reduce_sum(acc_s_ub, sumexp_i_ub, tmp_ub, dim=-1)
```

## Expert 独有操作

以下操作只在 Expert 模式中提供，Developer 模式无等价表达：

| Expert API | 功能 |
|-----------|------|
| `T.tile.cast(dst, src, mode, count)` | 精度转换 |
| `T.tile.compare(dst, src0, src1, mode)` | 逐元素比较 |
| `T.tile.select(dst, mask, src0, src1, mode)` | 条件选择 |
| `T.tile.sort/merge_sort/topk` | 排序操作 |
| `T.tile.gather/gatherb` | 数据收集 |
| `T.tile.transpose(dst, src)` | 16×16 转置 |
| `T.tile.createvecindex(dst, value)` | 创建向量索引 |
| `T.tile.arith_progression(buf, first, diff, count)` | 等差数列 |
| `T.tile.sin/cos(dst, src, tmp)` | 三角函数 |
| `T.tile.leaky_relu(dst, src, scalar)` | Leaky ReLU |
| `T.tile.axpy(dst, src, scalar)` | AXPY 运算 |
| `T.tile.broadcast(dst, src, tmp)` | 广播操作 |

> 在 Developer 模式中如需使用这些操作，可以直接调用（混合编程）。

## 转换注意事项

1. **T.Parallel 会自动分配临时 buffer**：复杂表达式会被拆解，而 `T.tile.xxx` 需要手动管理中间结果
2. **T.Parallel 支持 break/continue**：Expert 模式的 `T.tile.xxx` 不支持
3. **T.Parallel 中条件语句**：支持 `if/else`，转为 Expert 模式时需要用 `T.tile.compare` + `T.tile.select`
4. **性能差异**：两种方式最终生成的硬件指令相同，性能理论上一致
