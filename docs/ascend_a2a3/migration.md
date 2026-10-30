# TileLang-Ascend A2/A3 迁移指南

本目录保存固定 TA 来源 `0deb8b93da11b7f0becbcded732d9633b7ca4e34` 的文档，并说明其在当前主线的使用方式。主线 GPU/CPU 文档与 A5 方言继续独立维护；不要把旧 TA 的整个仓库布局、全局 pass 或构建脚本直接套到主线。

## 选择正确方言

| 场景 | Python 语言入口 | 目标/执行合同 |
|---|---|---|
| TA A2/A3 AscendC | `import tilelang.ascend.a2a3.language as T` | `arch=dav-2201`，keys 包含 `ascend`，共享 `cython` executor |
| TA A2/A3 PTO | 同上，使用 `T.tile` 的 PTO 原语 | 同一硬件 arch，keys 包含 `pto`；PTO 编译器 argv 使用 `dav-c220`，仍是共享 `cython` executor |
| 主线 A5 | `import tilelang.ascend.language as T` | 主线 A5 target/pipeline；见现有 `tilelang-ascend` 技能 |
| GPU/CPU | `tilelang.language` 或相应后端语言包 | 沿主线文档，不把其线程/内存模型当作 A2/A3 |

`ascend_a2a3`、`pto_a2a3` 是 backend module 名称；`ascendc`、`pto` 是代码方言；`cython` 是执行后端。它们不是可互换的名称，也不代表新增执行模式。

A2/A3 的 NPU launch 保留原 TA 两种形式：

```python
import tilelang.ascend.a2a3.language as T

# 在 @T.prim_func 或 @tilelang.jit 的函数体内：
with T.Kernel(core_count, is_npu=True) as (cid, vid):
    ...

with T.Kernel(core_count, threads=2, is_npu=True) as cid:
    vid = T.get_thread_binding()
    ...
```

NPU grid 只有一个维度；显式 `threads` 接受 1 或 2，并不是 CUDA 的 block threads。内部原 `vid` 使用 `cthread` 表示；默认 launch 与显式 developer launch 的标记不同，不可据变量名 `_` 猜测是否分配 subcore。A5 的 `MixedKernel`/`SimdVF` 不是这些写法的替代品。

(a2a3-build-environment)=

## 构建与环境

先准备与设备 CANN 匹配的现有 Torch/TorchNPU 配对。不要为了满足 Python 依赖随意把已安装配对升级或换成另一套 SDK；包版本号相同也不能代替 CANN/driver 兼容核对。

源码构建沿主线的 [安装说明](../get_started/Installation.md) 和 {download}`build 技能 <../../.agents/skills/tilelang-build/SKILL.md>`。A2/A3 需要启用 `USE_ASCEND`；硬件调度不由开启 CUDA/ROCm 决定。可在仓库根目录进行开发构建：

```bash
A2A3_SDK_CONSTRAINTS=$(mktemp)
python - <<'PY_SDK' > "$A2A3_SDK_CONSTRAINTS"
from importlib.metadata import version
for name in ("torch", "torch-npu"):
    print(f"{name}=={version(name)}")
PY_SDK
python -m pip install -c "$A2A3_SDK_CONSTRAINTS" -r requirements-dev.txt
python -m pip check
cmake -S . -B build -DUSE_ASCEND=ON -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_METAL=OFF -DUSE_LLVM=OFF
cmake --build build -j8
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

依赖安装前后的 Torch/TorchNPU metadata 必须与保存的精确 constraints 一致（含 local version）；解析不能满足时停止并修复依赖，不删约束换 SDK。上述构建命令无需设置 `TILELANG_CACHE_DIR`。

这条路径只构建，不替你安装 CANN、TorchNPU 或运行设备测试。需要安装 wheel 或 editable 包时，用主线 `pip install .` / `pip install -e .` 与同样的 CMake 配置参数，按现行 build 文档处理 native/Cython 产物。

旧 TA 的 `install_ascend.sh`、`set_env.sh`、旧 TVM patch/config 属于 TA CI adoption 的受控资产；不能从旧指南复制补丁直接应用到当前 TIRX。迁移 CI 候选保留旧布局分支，并为现代 eager/TIRX 选择新的构建流程。

## 运行原资产与平台身份

原生态相对路径统一放到这些目录，不覆盖主线同名 CUDA 文件：

| 原 TA 路径 | 当前 A2/A3 路径 |
|---|---|
| `examples/<relative>` | `examples/ascend_a2a3/<relative>` |
| `examples_experiment/<relative>` | `examples_experiment/<relative>` |
| `testing/<relative>` | `testing/ascend_a2a3/<relative>` |

迁移新增的 A2/A3 host/source 组件测试统一放在 `testing/ascend_a2a3/python/refactor/{原 testing/ascend 子目录}`；主线原有 `testing/ascend/` 文件保持原位。两者和原 TA 算子资产测试不是同一集合。先确认具体资产已经完成迁移；未完成资产的显式阻断或原已有失败不能算通过。

使用原 factory、decorator、`pass_configs`、`out_idx`、`workspace_idx`、shape/dtype 和原 golden 门槛。例如运行已迁移的原脚本时：

```bash
(
  export TL_PLATFORM=A3
  export TL_RUN_MODE=npu
  python examples/ascend_a2a3/gemm/example_gemm.py
)
```

原脚本中的 `tilelang.cache.clear_cache()` 按约定适配为 `tilelang.disable_cache()`；原有入口结构、输入和判定保持不变。

具体脚本的参数、内存预算和支持范围以该脚本及其迁移记录为准。上面的路径选择不是所有 shape 或设备的通过承诺。

平台来源是明确的 A2/A3 target model、`TL_PLATFORM=A2|A3` 或实际 ACL 设备识别。普通 target/context 在无设备环境可软处理；真正代码生成需要平台时必须解析成功，不能静默猜 A3。已归一化 target 的 model 冻结后优先于后续环境变化，并进入 cache 身份。

编译配置在 cache lookup 前冻结。路径、cwd、平台、方言、run mode、最终 flags 由 cache miss 与 restore 原样消费；恢复缓存不能重新读取环境推导另一套编译参数。模拟器代码生成/编译与真实 simulator 执行应分别记录，不把编译成功当模拟器或设备数值验收。

## API、pass 与 FFI 的迁移位置

| 原实现位置 | 当前实现位置 |
|---|---|
| `tilelang/language/ascend*.py` 等 TA 原语 | `tilelang/ascend/a2a3/language/` |
| `tilelang/engine/phase.py` | `tilelang/ascend/a2a3/pipeline.py` |
| TA Python pass wrapper | `tilelang/ascend/a2a3/transform/` |
| TA pass 配置归一化 | `tilelang/ascend/a2a3/pass_config.py` |
| `src/transform/*` 的 A2/A3 来源组件 | `src/ascend/a2a3/transform/` |
| `src/target/codegen_ascend*.{cc,h}` | `src/ascend/a2a3/codegen/` |
| TA op/layout/template | `src/ascend/a2a3/op/`、`layout/`、`tl_templates/` |

公开 Python pass 名称保留，由本地 wrapper 访问隔离 native FFI；不要把旧 TA 注册覆盖主线 A5/共享入口。设备 build FFI 为 `target.build.tilelang_ascend_a2a3` 和 `target.build.tilelang_pto_a2a3`，native 第三个参数显式传入平台；公共 codegen wrapper 负责解析平台及 instrumentation。原 `tl.Scope` 在新本地 native 入口为 `tl.A2A3Scope`。

新 TVM IR 使用 `tvm.tirx`、`SBlock`/`SBlockRealize`、`Bind` 与 `AllocBuffer`。旧 `LetStmt`、`Allocate`、`CustomizedCodeNode` 不是把名字全局替换就能构造的新 API。修改 pass 前读主线 {download}`IR 技能 <../../.agents/skills/tilelang-tvm-ir/SKILL.md>`。原 `_src_code` 公开调用已通过本地语言入口承接，不能因此宣称为被删除的通用旧节点创建了全局等价别名。

Python 手动 `T.set_flag(src, dst, event_id)`/`wait_flag` 保留原 TA 语言写法；内部共享 native 入口按 hard-event 组合串适配，不能再传旧三参数 native ABI。`pipe_barrier` 规范化 `PIPE_` 前缀，但 TA 向量 barrier 的源语义仍保留。自动 flag 原有两参数入口不要误改。

旧 `tir.*` pass key 的兼容只在 A2/A3 归一化中处理；新 pass 和 native 配置有唯一 owner。AscendC 与 PTO 的默认值不同，PTO 默认 VS 同步开启且影响默认编译选项；显式用户 flags 的顺序与覆盖不能在 pipeline/driver/cache 各层重复合并。

## 支持范围与验收口径

已迁完整 AscendC/PTO native、公开 JIT/compile、显式 workspace、Cython ABI、cache 和打包链路。A3 代表族的数值、原 TA 实际输出对照、冷热公开 callable 计时和独立进程 cache restore 已分别验收。AscendC 代表包括 GEMM、matmul_add、静态/动态 indexer；PTO 包括原 HISA、静态/动态 sparse attention 与 MoE 各 dtype/类变体。最终 native wheel 在独立环境和非 repo cwd 验证过来源/资源/native 装载与代表数值。

这些证据不意味着所有原资产、A2 实机、模拟器或所有 shape 均已实测。性能样本测量的是声明的 public call + sync 边界，不称为纯 kernel 加速。原 TA 已有的非法源码、debug SDK 兼容边界和共享输出 cacheline 竞态保留为来源问题，不通过改 golden、丢样本或换小核隐藏。

普通 autotuner 与原普通 GEMM autotune 走真实调优链路。TA Ascend Carver hint policy 是用户明确 deferred 范围；保留 `example_gemm_carver.py` 与相关资产，但不迁策略、不用静态配置冒充 recommend_hints，也不将 deferred 计作 pass。blocksparse/convolution 等原 active 实验测试不是 Carver 目录级豁免。

## 文档与技能的来源边界

本目录保留原 TA 文档中的通用 GPU/CPU 教学资料；这些页面的 CUDA/线程示例仍属于其原方言，不是 A2/A3 API 示例。

现有 main backend/build/IR/A5 技能保持原内容；TA 独有技能逐项适配并记录来源，不把旧全局 `AGENTS` 或自动 push/pull/装环境的操作当成本次迁移授权。

```{toctree}
:maxdepth: 1
:caption: 完整来源参考

reference
```
