# 当前 A2/A3 Pass 代码生成核对

- 完整原算法/所有有效分支及实际依赖都有归属；原缺口或必要修正单列，不换小核。
- src/ascend/a2a3本地TU及USE_ASCEND CMake gate正确，OFF不编入；现行main native与Cython实际重编/装载。
- TIRX handle/visitor/Optional/COW/identity按main IR技能；默认Var()不是null，Bind/AllocBuffer词法范围正确。
- native用TVM_FFI_STATIC_INIT_BLOCK与reflection::GlobalDef().def；隔离名称和Python查询一致，不用旧set_body_typed或全局_ffi_api。
- 现有transform多数直接tvm.ffi.get_global_func，require_native是可复用guard；不把推荐模板假称所有wrapper都如此。
- 公共PassConfigKey只在共享唯一owner定义；本地pass_config归一化与目标默认，compiler flags不在此重复解析。
- 真实pipeline前后facts、config/target/platform条件和完整参数保留；model不是AscendC/PTO方言，MaskLegalize后无rewrite。
- 同输入来源对照/当前IR/源码与聚焦正反行为验证完成；必要测试不是交付后可省的事项。
- public compile/source与实际设备数值/性能分开写；注册名、grep或未运行清单不算完整验证。
- 保用户cache/SDK与工作区，外部动作依已有授权。报告文件、命令、结果、来源、必要适配及未完成项。

具体占位模板见同目录integration-points与templates/pass-impl-skeleton-template.md；原完整checklist保留在source档案。
