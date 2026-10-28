> 固定 TA 来源资料（资源迁移，不是内核/入口已迁或当前性能验收）：[examples/developer_mode/README.md](https://github.com/tile-ai/tilelang-ascend/blob/0deb8b93da11b7f0becbcded732d9633b7ca4e34/examples/developer_mode/README.md)。原 benchmark 数字、设计与代码片段保留；原命令和绝对路径属于当时实验记录，未迁执行资产不能在当前仓直接照跑。当前入口/构建先读[迁移指南](../../../docs/ascend_a2a3/migration.md)，逐项状态见[资源清单](../../../maint/ascend_a2a3/ecosystem_resource_ports.json)及 Python ports。

<!-- TA RESOURCE BODY -->
# Flash Attention for Ascend NPU using TileLang-Ascend

## Overview

This repository provides an implementation of **Flash Attention** forward pass using **TileLang-Ascend** DSL for Huawei Ascend NPU. The implementation leverages the hardware characteristics of Ascend NPU through block-wise computation, pipeline optimization, and automatic memory planning to achieve high-performance attention computation.

## Flash Attention Mathematical Formulation

### Standard Attention Computation

The standard attention computation can be expressed as:

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

Where:
- `Q, K, V ∈ ℝ^{batch×heads×seq_len×dim}` are input tensors
- `d_k = dim` is the dimension of the key vectors
- `softmax` is applied along the last dimension

### Online Softmax Algorithm for Block-Wise Computation

For long sequences, computing the full attention matrix `S = Q·K^T ∈ ℝ^{seq_len×seq_len}` is memory-intensive. Flash Attention uses a block-wise algorithm with online softmax updates:

#### State Variables for Row i:
- `m_i`: maximum value encountered so far for row i
- `ℓ_i`: sum of exponentials for row i (scaled by previous max)

#### Online Update Rules:

For each new block `S_new ∈ ℝ^{block_M×block_N}` of attention scores:

1. **Update maximum**:
   ```
   m_i_new = max(m_i_old, max(S_new[i, :]))
   ```

2. **Update exponential sum**:
   ```
   scale = exp(m_i_old - m_i_new)
   ℓ_i_new = ℓ_i_old × scale + sum(exp(S_new[i, :] - m_i_new))
   ```

3. **Update output**:
   ```
   O_i_new = O_i_old × scale + exp(S_new[i, :] - m_i_new) · V_block
   ```

4. **Final normalization**:
   ```
   O_i_final = O_i / ℓ_i
   ```

This algorithm allows processing the attention computation in blocks while maintaining numerical stability.
