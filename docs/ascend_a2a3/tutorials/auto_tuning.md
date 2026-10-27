# A2/A3 真实自动调优

A2/A3复用主线共享autotuner、公开JIT/compile和cython执行模式；不以静态配置替代真实candidate编译/数值验证/计时。旧通用GPU与Carver教学全文保存在source档案，不能当作当前Ascend Carver策略已迁证明。

## 普通调优

先读原普通[TA GEMM autotune来源](https://github.com/tile-ai/tilelang-ascend/blob/0deb8b93da11b7f0becbcded732d9633b7ca4e34/examples/autotune/example_gemm_autotune.py)与实际共享autotuner入口。保原get_config/itertools搜索空间、factory、decorator、pass_configs、out_idx/workspace_idx、reference和误差阈值。资产是否已经迁入namespace以ecosystem_ports清单为准；未迁时不fallback到main同名CUDA例。

1. 使用tilelang.ascend.a2a3.language编写或移植原完整kernel，预留真正tiling参数。
2. 由原configs生成候选，经完整公开backend pipeline/codegen/driver编译，再用相同输入/原golden验证；失败和不合法配置保留清楚原因，不丢case刷绿。
3. 在固定设备和测量边界下采集原始样本，冷热/编译/firstcall与publiccall计时分开。workspace和dynamic hostshape必须沿原公开ABI，不私造wrapper或从profiler误计workspace为输出。

编译请求在cache前冻结平台、方言、runmode、cwd和flags。candidate派生选项须来自同一次请求的冻结输入，不在每candidate重新读TL_CCE环境；metadata restore不得静默载另一个平台/方言。用户显式flags的尾序与重复-mllvm参数保留。

实际运行时提供独立TILELANG_CACHE_DIR，并沿原CLI参数；导入不应clear_cache、parse_args、编译或launch。任何source/host测试只证明其各自范围，不写设备调优通过。

## Carver deferred范围

用户明确暂缓TA Ascend recommend_hints策略迁移。原example_gemm_carver及策略资料保留为来源资产，不迁策略、不从CUDA策略猜Ascend hint、不用固定configs冒充Carver调优。CI仅对精确adopted Carver例记录deferred原因/独立数量；原143登记与pending状态保留，deferred不计pass。

普通autotune与原blocksparse/convolution/convolution_autotune实验active不是目录级Carver豁免。已有主线GPU Carver功能仍按主线合同，不因本迁移扩大/删除。
