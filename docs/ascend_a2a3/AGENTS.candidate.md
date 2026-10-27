# A2/A3 AGENTS 合并候选（未激活）

这是供受控 adoption 专审的候选，不是当前仓库根指令文件。固定 TA 原文完整保存在 [_source_config/AGENTS.ta.md](_source_config/archives/source-f329cf1fb0f3a57e897f6af58b79817d11479d1ef31dd1d5bddf3c9968c42efe.gz)。主线现有 backend/build/IR/layout/A5 等技能保留原内容，不因本候选自动覆盖。

## 范围与优先级

- 本节仅针对从 TA 迁入的 A2/A3 kernel、后端和生态资产；GPU/CPU/A5 沿其主线规范。
- 用户明确要求、已授权动作、已选编程模式和既有 factory/config 优先。相同授权不重复询问；新 kernel 的模式确实不明时才澄清 Developer / Expert / 混合。
- 对原算法、完整组件与支持域保真。先找原实例和当前实现，不凭记忆猜 API、不把失败用小核/静态配置/放宽 golden 绕过。
- 如实区分源码/IR/编译/设备/性能/缓存与实际安装验证；只有真实执行过才写通过，不能虚构输出或丢弃原失败。

## 当前入口

A2/A3 使用 `tilelang.ascend.a2a3.language`；A5 使用 `tilelang.ascend.language`。PTO 是 A2/A3 当前支持的代码方言，旧“禁止 tilelang/language/pto.py”仅指旧文件，不得解读为禁止新 PTO 后端。内存级别、scope、同步和 pipeline 参数按所选方言及真实原例保持，不将 CUDA/A5 能力套进 A2/A3。

原 examples/experiment/testing 资产在各自 `ascend_a2a3` 命名空间内；原路径不能 fallback 到 main 同名 CUDA 文件。main `testing/ascend` 的现有 A2/A3 组件测试独立保留。

## 编译器与构建

- 现代前端是 eager/TIRX；先读 main backend、dev-guide、tvm-ir技能，再改本地 language/op/pass/codegen。可缺 Var 用 Optional，默认 Var() 不是空。
- 本地 pipeline 位于 `tilelang/ascend/a2a3/pipeline.py`；完整原 pass 算法、条件、producer/consumer facts与 mask 末端边界不能省。
- native 入口与共享 Op/配置各有唯一 owner；改名按来源映射，不能覆盖 A5/global 同名注册或另造 execution mode。
- native 改动必须真实重编并核对 loader/Cython 来源；dev build、安装 wheel和旧 TA 二进制不能混称。
- 构建沿 main CMake/PEP517；保 runner 已配套 Torch/TorchNPU/CANN 和精确子模块 pin，不自动升级 SDK或给 TIRX打旧补丁。

## 用户资产、CI 与外部动作

TA 原 CI 是此迁移的优先基线，保其 runner/container/events/143登记与 active/pending 分流。scope deferred、未执行与真正 pass 分开；Carver Ascend hint 策略由用户明确 deferred，不用静态配置替代真实调优。普通 autotune与实验 active 并不继承 Carver 豁免。

保用户全局 cache 和环境；需要隔离时使用任务独立目录与冻结配置。原脚本中的 clear_cache、自动安装、git pull、定时调度或外部 GitHub消息不因搬迁就获得执行授权。提交、push、外部消息依本次用户明确范围，不由历史技能自动扩张。

现有设计和 source 行为优先于旧参考页的行号/伪IR/旧TVM注册模板。来源完整存档、必要适配与支持边界逐项可追溯；已有错误/硬件限制明确记录。

## 格式与阶段提交

格式和提交检查沿当前仓库canonical配置与现行主线技能，不把旧TA100列规范重新推广为全仓规则；CI工具目录单独100列例外保持局部。提交作者沿用户指定的人类身份，使用命令局部配置而非改全局Git身份；本迁移阶段标题使用`[Ascend][A2A3]`。仅本地提交授权不自动包含push、PR发布或外部消息。

原文gzip档案可用`gzip -dc docs/ascend_a2a3/_source_config/archives/source-f329cf1fb0f3a57e897f6af58b79817d11479d1ef31dd1d5bddf3c9968c42efe.gz`查看；固定来源也可查[原TA AGENTS](https://github.com/tile-ai/tilelang-ascend/blob/0deb8b93da11b7f0becbcded732d9633b7ca4e34/AGENTS.md)。此文件激活到仓库根时，必须相应重写相对链接，不能直接复制保留候选状态提示。
