# 当前 A2/A3 Pass 实现模式

原TA完整模式代码保存在对应非活动source文件，旧TIR/ObjectRef/注册语法不可直接编译。当前实现必须先读main IR/dev-guide技能与本地源码；不能从旧示例推出新IR支持。

## 分析与改写

`src/ascend/a2a3/transform/ascend_host.cc`保留完整TA HostProcesser算法，并显示TIRX Bind与WithAttr接口；`ascend_collect_buffer_shape.cc`给出visitor/collector与WithAttrs返回。按实际算法选择分析器/visitor/mutator，而不是把所有旧类删成通用IRMutator。

Visitor framework overrides仍使用其规定的裸Node指针；自定义helper参数优先用真实handle；identity map/set与结构相等明确区分。可缺Var用Optional，Var()不是空；WithAttr/WithAttrs是IR函数，不是旧伪代码里的任意成员调用。

## 注册与封装

采用TVM_FFI_STATIC_INIT_BLOCK内reflection::GlobalDef().def，native名称隔离到tl.transform.A2A3前缀；本地Python通过_native.require_native查询。完整占位骨架见pass-generate/templates/pass-impl-skeleton-template.md。不要复制旧TVM_REGISTER_GLOBAL/.set_body_typed或新建全局ffi_api。

## 参数、配置与事实

保原target/platform参数、原配置门控、完整所有分支和Op/catalog/header依赖。现有公共PassConfigKey唯一owner是tilelang/transform/pass_config.py，本地pass_config.py只归一化与提供已证明的默认。共享flags/autoflag ABI不得重注册。

有上游attrs时列出类型/身份键/缺失策略，并证明修改后下游读到同样事实；尤其cthread默认/dev marker、layout_map、workspace与mask terminal事实不可擅自丢弃或扩大保留。

## 故障边界

源非法free-var、无消费者GPU模板、空dtype模板等必须以真实来源证据分类；不把原rawIR未经必要pass的失败扩大成公共API不支持。对原可用算法保真，必要诊断单列，不能用样例allowlist缩小组件。

构建native、原source同输入、聚焦行为和必要设备验证各有独立证据。只列grep命中或构造名字不是完成。
