# 现代 A2/A3 Pass 接入点

本文件以当前主线结构为准；原TA完整参考保存在非活动来源中。先读 `tilelang/backend/README.md` 和 main backend/dev-guide/tvm-ir技能。以下使用 `{PassName}` 占位，不是可直接复制编译的完整pass。

## Native与Python入口

实现放 `src/ascend/a2a3/transform/<pass_name>.cc`，namespace为 `tvm::tl::a2a3`，IR类型来自 `tvm::tirx`。用真实handle传递ObjectRef；可缺的Var/Buffer用Optional，不把默认构造当null。

```cpp
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.A2A3{PassName}", {PassName});
}
```

当前 transform factories 多数直接调用 `tvm.ffi.get_global_func`；`_native.require_native` 在语言/layout入口提供明确缺Op诊断。新pass可以复用该guard（以下为推荐模板，不是对所有现有wrapper的描述），也可沿现有直接lookup风格；两者都不能覆盖旧全局 `_ffi_api`：

```python
from .._native import require_native

def {PassName}():
    return require_native("tl.transform.A2A3{PassName}", "{PassName}")()
```

具体命名先查当前ffi_map及现有wrapper，避免对已带Ascend/A2A3的名字重复加前缀。带参函数按原完整构造和真实配置传参，不随意加`target=None`或推默认值。

## Pipeline和配置

实际顺序/条件以 `tilelang/ascend/a2a3/pipeline.py` 的 LowerAndLegalize/OptimizeForTarget为准；前置共用MaterializeKernelLaunch仍不可省。调用本地 `ascend_transform`；`tirx.transform`与`s_tir.transform`只在当前明确共享步骤使用。不能根据旧target.model的ascendc/pto字符串分支；model是A2/A3，keys决定方言。

现有9配置enum定义在共享 `tilelang/transform/pass_config.py`，后端本地 `tilelang/ascend/a2a3/pass_config.py`只归一化/提供目标默认，不再定义第二套enum。新配置native注册有唯一owner；默认值查GetConfig与target defaults后文档化。编译器flags不放pass normalizer重复解析环境。

AscendC管理的Selection/MaskLegalize后不得再添加IR rewrite；MemoryPlanning/ResourceScopeVerify等事实由真实producer/consumer约束。PTO不假套AscendC mask分支。

## 构建与验证

CMake源清单只加入后端目录；USE_ASCEND=OFF不得编入本地TU。按mainbuild技能构建native，命名查询/IR行为/原source同输入对照分别验证。注册字符串存在不等于算法保真；单纯grep不代替native构建或行为测试。

组件测试位于main `testing/ascend` 的明确A2A3文件；原资产测试位于 `testing/ascend_a2a3`。任务授权实现与验证时无需另问同一骨架批准；设计范围实际改变时再给可审差异。具体实现和必要回归完成后才能宣称交付，不能只列待测清单当pass。
