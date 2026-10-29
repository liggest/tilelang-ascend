# {PassName} 实现骨架

这是设计模板，花括号表示需填入的设计事实，不能当可编译C++直接执行。用户已授权按明确设计实现时直接落实；范围变更才提供差异，不反复请求同一骨架批准。

## 0. 任务与来源

填写任务类型、设计来源、完整原组件/来源pin、目标平台和方言、行为支持域、上游producer与下游consumer、原有效/非法输入边界。不能把改注册表或一个代表用例当完整迁移。

## 1. 文件清单

| 文件 | 责任 |
|---|---|
| src/ascend/a2a3/transform/{pass_name}.cc | 本地完整算法与native factory |
| src/ascend/a2a3/CMakeLists.txt | USE_ASCEND gated来源TU |
| tilelang/ascend/a2a3/transform/__init__.py | require_native wrapper |
| tilelang/ascend/a2a3/pipeline.py | 原序列中的真实插入点/条件 |
| tilelang/transform/pass_config.py | 只有新增公共enum时改唯一owner |
| tilelang/ascend/a2a3/pass_config.py | 仅本地归一化/default所需更改 |
| testing/ascend_a2a3/python/refactor/.../test_a2a3_*.py | 真实新行为与失败边界；原算子资产仍在testing/ascend_a2a3 |

## 2. IR算法与数据合同

类的父类、每个visitor override、helper、identity map/set、analyzer与COW修改位置逐项说明；函数传值/const handle/裸Node参数遵主线tvm-ir技能，不凭旧nullptr或Var()猜空状态。

列出输入attrs的类型/来源/缺失策略，输出attrs的消费者，结构事实的读取/产生/失效，所有原Loop/Buffer/scope分支和临时buffer声明位置。TIRX使用SBlock、Bind、AllocBuffer，不能把旧LetStmt/Allocate机械改名而忽略词法范围或声明。

## 3. Pass factory与隔离注册

参考现有src/ascend/a2a3/transform/ascend_host.cc和ascend_collect_buffer_shape.cc的真实factory。占位形态：

```cpp
namespace tvm::tl::a2a3 {
tvm::transform::Pass {PassName}() {
  auto pass_func = [](tirx::PrimFunc f, IRModule mod,
                      tvm::transform::PassContext ctx) {
    // 完整实现读取配置/attrs，保原算法与必要IR适配。
    return {PassName}Impl::Substitute(std::move(f), ctx);
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                            "tl.A2A3{PassName}", {});
}
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.A2A3{PassName}", {PassName});
}
}  // namespace tvm::tl::a2a3
```

具体名字以已登记FFI为准，不能对现有名字盲加两次前缀。配置只注册一次；默认值须证明来自真实GetConfig和target defaults，不据backend名字推false。

## 4. Python与pipeline

```python
from .._native import require_native

def {PassName}():
    return require_native("tl.transform.A2A3{PassName}", "{PassName}")()
```

不另建全局_ffi_api覆盖A5。带参函数保真实参数，不凭模板改为None或删平台参数。Pipeline使用本地ascend_transform；model表示A2/A3，keys决定AscendC/PTO。列出插入处完整上下文及必须保持的顺序/条件，AscendC MaskLegalize后不可再rewrite。

## 5. 验证与报告

- 原源码同输入IR/source对照，归一化只能消除已证明的表示差异；原非法输出的新诊断单列。
- 新native实际编译并确认装载路径；注册查询只证明名字，不证明算法。
- 聚焦行为/失败边界，包含状态/nullable/COW/launch marker等本改动真正触及的合同。
- 只有设备执行可写数值/性能通过；host/source/compile-only各自记录范围。
- 报告完整改动、来源pin、必要适配、真实命令/退出码及未完成项，不把待补清单算通过。
