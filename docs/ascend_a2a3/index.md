> 固定TA来源中的通用GPU/CPU或伪IR教学资料：示例仍属于其原方言，不是A2/A3支持承诺。A2/A3当前入口/构建/缓存与边界先读迁移指南。

# TileLang-Ascend A2/A3 文档与来源资料

> 本目录来自固定TA版本，A2/A3当前使用先读迁移指南。原通用GPU/CPU与伪IR页面保持其原方言，不代表A2/A3支持承诺。

:::{toctree}
:maxdepth: 2
:caption: A2/A3 MIGRATION

migration
:::

[GitHub](https://github.com/tile-ai/tilelang)

Tile Language (tile-lang) is a concise domain-specific language designed to streamline
the development of high-performance GPU/CPU kernels (e.g., GEMM, Dequant GEMM, FlashAttention, LinearAttention).
By employing a Pythonic syntax with an underlying compiler infrastructure on top of TVM,
tile-lang allows developers to focus on productivity without sacrificing the
low-level optimizations necessary for state-of-the-art performance.

:::{toctree}
:maxdepth: 2
:caption: GET STARTED

get_started/Installation
get_started/overview
:::

:::{toctree}
:maxdepth: 1
:caption: TUTORIALS

tutorials/writing_kernels_with_tilelibrary
tutorials/writing_kernels_with_thread_primitives
tutorials/annotate_memory_layout
tutorials/debug_tools_for_tilelang
tutorials/auto_tuning
tutorials/jit_compilation
tutorials/pipelining_computations_and_data_movements
:::

:::{toctree}
:maxdepth: 1
:caption: DEEP LEARNING OPERATORS

deeplearning_operators/elementwise
deeplearning_operators/gemv
deeplearning_operators/matmul
deeplearning_operators/matmul_dequant
deeplearning_operators/flash_attention
deeplearning_operators/flash_linear_attention
deeplearning_operators/deepseek_mla
deeplearning_operators/convolution
deeplearning_operators/tmac_gpu
:::

:::{toctree}
:maxdepth: 2
:caption: LANGUAGE REFERENCE

language_ref/ast
language_ref/primitives
language_ref/tilelibrary
:::

:::{toctree}
:maxdepth: 1
:caption: ASCEND COMPILER

ascend/compiler_managed_vector_mask
ascend/row_reduce
:::

:::{toctree}
:maxdepth: 1
:caption: A2/A3 API REFERENCE

api_docs/T.copy
api_docs/T.gemm_v0
api_docs/T.reduce_sum
api_docs/T.tile.add
:::

:::{toctree}
:maxdepth: 1
:caption: Privacy

privacy
:::
