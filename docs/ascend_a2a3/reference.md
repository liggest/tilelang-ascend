# TA A2/A3 来源文档导航

此目录保留完整来源页面，包含 A2/A3 API 约束、历史编程资料和通用 GPU/CPU 教学。目录可达不代表所有 API、shape 或设备已实测；当前入口、构建与支持边界先读[迁移指南](migration.md)。通用 GPU/CPU 页面继续属于原方言，不改写为 A2/A3 支持承诺。

```{toctree}
:maxdepth: 1
:caption: 指南与运行配套

TileLang-Ascend Programming Guide
coverage_guide
pytest_marker_guide
```

```{toctree}
:maxdepth: 1
:caption: API参考

api_docs/T.reduce_max
api_docs/T.reduce_min
api_docs/T.tile.arith_progression
api_docs/T.tile.atomic_add
api_docs/T.tile.broadcast
api_docs/T.tile.compare
api_docs/T.tile.createvecindex
api_docs/T.tile.div
api_docs/T.tile.gather
api_docs/T.tile.merge_sort
api_docs/T.tile.mul
api_docs/T.tile.select
api_docs/T.tile.sort
api_docs/T.tile.sub
api_docs/T.tile.topk
api_docs/T.tile.transpose
```

```{toctree}
:maxdepth: 1
:caption: 教程

tutorials/ascend_aicore_exception_dump_en
tutorials/ascend_aicore_exception_dump_zh
tutorials/automatic_workspace_allocation
tutorials/print
tutorials/t_parallel
tutorials/t_pipelined
tutorials/vid_reduction_and_auto_cv_ratio
tutorials/workspace_reduction_tutorial
```

```{toctree}
:maxdepth: 1
:caption: 语言示例

language/tile_leaky_relu
language/tile_max
language/tile_min
language/tile_relu
language/tile_sigmoid
language/tile_silu
```
