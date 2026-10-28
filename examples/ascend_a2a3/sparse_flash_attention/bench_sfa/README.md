> 固定 TA 来源资料（资源迁移，不是内核/入口已迁或当前性能验收）：[examples/sparse_flash_attention/bench_sfa/README.md](https://github.com/tile-ai/tilelang-ascend/blob/0deb8b93da11b7f0becbcded732d9633b7ca4e34/examples/sparse_flash_attention/bench_sfa/README.md)。原 benchmark 数字、设计与代码片段保留；原命令和绝对路径属于当时实验记录，未迁执行资产不能在当前仓直接照跑。当前入口/构建先读[迁移指南](../../../../docs/ascend_a2a3/migration.md)，逐项状态见[资源清单](../../../../maint/ascend_a2a3/ecosystem_resource_ports.json)及 Python ports。

<!-- TA RESOURCE BODY -->
**English** | [中文](README_zh.md)

## 🎉 Overview
SparseMLA is the core attention mechanism introduced in DeepSeek v3.2. This directory implements the related use cases of this mechanism based on Tilelang-AscendC, and provides specific optimization solutions for different optimized versions for developers' reference and use.

## 📊 Test Inputs
The inputs are defined in `bench_sfa.py`, with the following common parameters:

- T=1
- B=1
- Q_N=128
- KV_N=1
- D=512
- D_rope=64
- sparse_size=2048
- block_size=128
- act_kv_s=2560

Three sets of shapes differ only in KV_S:

| No. | KV_S | Description |
| --- | ---: | --- |
| shape0 | 2560 | Short sequence |
| shape1 | 6400 | Medium sequence |
| shape2 | 48000 | Long sequence |

`act_kv_s` is fixed at 2560, and `sparse_size` is fixed at 2048. The effective sparse window size actually participating in the computation remains basically unchanged.

## Performance Results

|  | shape0 | shape1 | shape2 | Average Time (us) | AscendC / TileLang |
| --- | ---: | ---: | ---: | ---: | ---: |
| AscendC  | 98.000 | 99.000 | 100.000 | 99.000 | 1.00x |
| TileLang | 109.760 | 109.100 | 108.980 | 109.280 | 0.91x |

*Note: AscendC reference implementation source:* https://gitcode.com/cann/cann-recipes-infer/blob/master/ops/ascendc/torch_ops_extension/custom_ops/converter/npu_sparse_flash_attention.py

## 📖 File Description
```
├── sparse_flash_attn_pa_baseline.py            # Baseline version
├── sparse_flash_attn_pa_developer.py           # Developer mode version
├── sparse_flash_attn_pa.py                     # Version with T.Pipeline, hybrid programming in developer mode and expert mode
├── sparse_flash_attn_pa_no_cv_pipeline.py      # Version without T.Pipeline, implements matrix multiplication in expert mode
├── sparse_mla_performance_optimization.zh.md   # SparseMLA optimization scheme document
└── bench_sfa.py                                # Execution entry script
```

The specific optimization details of each version are shown in the following table:
| File | Fixed Core | s2 Split Size  | kv gather & continuous move-out | Async copy | ping-pong In/Out | CV pipeline   |  Broadcast Opt | AXPY Opt |  Performance Data (A3) |
|----- |:-----------:  | :----------: | :----------------: | :-----------------: | :---------------: | :-----------: | :-----------: | :------: | :------------: |
|[sparse_flash_attn_pa_baseline.py](https://github.com/tile-ai/tilelang-ascend/blob/0deb8b93da11b7f0becbcded732d9633b7ca4e34/examples/sparse_flash_attention/bench_sfa/sparse_flash_attn_pa_baseline.py)             | √ | 64  | × | × | × | × | × | × | 602us |
|[sparse_flash_attn_pa_developer.py](https://github.com/tile-ai/tilelang-ascend/blob/0deb8b93da11b7f0becbcded732d9633b7ca4e34/examples/sparse_flash_attention/bench_sfa/sparse_flash_attn_pa_developer.py)           | √ | 64  | √ | × | × | × | √ | × | 347us |
|[sparse_flash_attn_pa.py](https://github.com/tile-ai/tilelang-ascend/blob/0deb8b93da11b7f0becbcded732d9633b7ca4e34/examples/sparse_flash_attention/bench_sfa/sparse_flash_attn_pa.py)                               | √ | 64  | √ | √ | √ | √ | √ | √ | 127us |
|[sparse_flash_attn_pa_no_cv_pipeline.py](https://github.com/tile-ai/tilelang-ascend/blob/0deb8b93da11b7f0becbcded732d9633b7ca4e34/examples/sparse_flash_attention/bench_sfa/sparse_flash_attn_pa_no_cv_pipeline.py) | √ | 256 | √ | √ | √ | × | √ | √ | 109us |

## 📄 Usage

Run `bench_sfa.py` directly:
```Python
python bench_sfa.py --file="sparse_flash_attn_pa_baseline"
```
- `--file`: Supports `sparse_flash_attn_pa_baseline`, `sparse_flash_attn_pa_developer`, `sparse_flash_attn_pa` and `sparse_flash_attn_pa_no_cv_pipeline`.
