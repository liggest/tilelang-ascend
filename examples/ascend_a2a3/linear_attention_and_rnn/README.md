> 固定 TA 来源资料（资源迁移，不是内核/入口已迁或当前性能验收）：[examples/linear_attention_and_rnn/README.md](https://github.com/tile-ai/tilelang-ascend/blob/0deb8b93da11b7f0becbcded732d9633b7ca4e34/examples/linear_attention_and_rnn/README.md)。原 benchmark 数字、设计与代码片段保留；原命令和绝对路径属于当时实验记录，未迁执行资产不能在当前仓直接照跑。当前入口/构建先读[迁移指南](../../../docs/ascend_a2a3/migration.md)，逐项状态见[资源清单](../../../maint/ascend_a2a3/ecosystem_resource_ports.json)及 Python ports。

<!-- TA RESOURCE BODY -->
### Gated Delta Network (GDN)

[Gated Delta Network (GDN)](https://arxiv.org/pdf/2412.06464) is a novel recurrent-style sequence model. It keeps a hidden state $\mathbf S_t$ with update rule:

$$\mathbf S_t=\mathbf S_{t-1}\alpha_t(\mathbf I-\beta_t\mathbf k_t\mathbf k_t^T)+\beta_t\mathbf v_t\mathbf k_t^T.$$

You can find the simplest implementation for this formula in `ref_seq_gdn` in `gdn_full.py`

To speed up the calculation, GDN adopts **chunkwise parallelism**. The sequence with total length $L$ is divided into $L/C$ chunks with length $C$, we first calculate the hidden state at the start point of each chunk (i.e. $\mathbf S_{i\cdot C}$), then use this "partial hidden state" to calculate the output. You can find more technical details in the original paper.

Our implementation of chunkwise parallelism resembles that of [Flash Linear Attention (FLA)](https://arxiv.org/pdf/2412.06464). You can find reference [here](https://sustcsonglin.github.io/blog/2024/deltanet-2/). Specifically, the forward pass of GDN is divided into six processes:

- `chunk_cumsum`: Calculate

  $$\gamma_{i\cdot C+j}=\sum_{k\leq j}g_{i\cdot C+k}\ (i<L/C,j<C),$$

  where $g_i=\ln \alpha_i$. This chunkwise cumsum will be used in several later processes.

- `chunk_scaled_dot_kkt`: Chunkwisely calculate

  $$\mathbf L=\text{strictLower}(\text{diag}(\beta)\cdot (\Gamma\odot \mathbf K\mathbf K^T)),$$

  where $\Gamma_{i,j}=\exp(\gamma_i-\gamma_j)$. $\mathbf L$ is then used in UT transform in the next step.

- `solve_tril`: Chunkwisely calculate

  $$\mathbf A=(\mathbf I+\mathbf L)^{-1}.$$

- `wy_fast`: Implement the UT transform chunkwisely:

  $$\begin{aligned}\mathbf U&=\mathbf A\cdot\text{diag}(\beta)\cdot\mathbf V,\\
  \mathbf W&=\mathbf A\cdot\text{diag}(\exp(\gamma)\odot \beta)\cdot\mathbf K.\end{aligned}$$

- `chunk_h`: Calculate the hidden state at the start point of each chunk. It satisfies the following recurrent formula in each chunk:

  $$\mathbf S_{\text{next}}=\exp(\gamma_{C-1})\mathbf S+(\mathbf U-\mathbf W\mathbf S^T)^T\tilde{\mathbf K},$$

  where $\tilde{\mathbf k_i}=\exp(\gamma_{C-1}-\gamma_i)\cdot \mathbf k_i$

- `chunk_o`: Calculate the output using hidden state at the start point of each chunk. It satisfies the following formula in each chunk:

  $$\mathbf O=\text{diag}(\exp(\gamma))\mathbf Q\mathbf S^T+(\Gamma\odot\mathbf M\odot\mathbf Q\mathbf K^T)(\mathbf U-\mathbf W\mathbf S^T),$$

  where $\mathbf M$ is the causal mask.

---

### Optimize Results

Shape: $(B,H,L,DK,DV,C)=(16,16,16384,128,128,128)$.

|        Kernel        | Latency (ms) |    #ops (approx)     |  TFLOPS  |
| :------------------: | :----------: | :------------------: | :------: |
|     chunk_cumsum     |    $1.93$    |  $4.19\times 10^6$   | $0.0021$ |
| chunk_scaled_dot_kkt |    $8.76$    | $6.87\times 10^{10}$ |  $7.84$  |
|      solve_tril      |   $24.89$    | $2.29\times 10^{10}$ |  $0.92$  |
|       wy_fast        |    $9.92$    | $1.37\times 10^{11}$ | $13.85$  |
|       chunk_h        |    $9.38$    | $2.75\times 10^{11}$ | $29.30$  |
|       chunk_o        |   $13.19$    | $3.44\times 10^{11}$ | $26.04$  |
|        total         |   $68.07$    | $8.48\times 10^{11}$ | $12.45$  |
