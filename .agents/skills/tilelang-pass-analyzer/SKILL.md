---
name: tilelang-pass-analyzer
description: 分析A2/A3 pass真实行为、支持域、原算法与TIRX适配，比较来源和当前实现并给出可定位的IR/源码证据。
---

# tilelang-pass-analyzer：现代 A2/A3

仅用于TA A2/A3，先读docs/ascend_a2a3/migration.md与main backend/build/IR技能；已有main技能保持独立。

1. 先确认所问对象是A2/A3本地pass、主线共享TVM变换还是A5行为；不要根据Ascend单词猜归属。当前本地native在src/ascend/a2a3/transform，Python wrapper和pipeline在tilelang/ascend/a2a3。
2. 从现行wrapper/FFI映射确定名字，再读实际函数、配置门控、target/platform条件和完整算法分支。references/pass-designs保留固定TA来源的设计/伪IR，不能把其行号、旧节点或声明当当前事实。
3. 输出由浅入深的说明：解决问题、输入/输出结构、读写attrs/facts、算法阶段、所有有效支持分支与错误边界、调用者/消费者。支持精确/模糊名称与多个pass比较，但不为了省事只展示一个示例allowlist。
4. IR例子注明是真实当前TIRX还是来源伪IR。比较ObjectRef身份/结构相等时遵main IR技能；旧nullptr、Var()、Let/Allocate与新Bind/AllocBuffer不机械等价。
5. 注册存在只证明名字；需要验证的假设用实际public pass同输入行为/source对照，不以grep代替编译，不把未经必要pass的rawIR失败扩大成公共API不支持。
6. 报告来源pin、当前文件/函数、已运行命令、推断与未验证项。源码不明确时写具体缺口，不编造性能/设备支持。
