---
name: tilelang-env-check
description: 验证 TA A2/A3 在现代 TileLang 主线的源码、固定依赖、Torch/TorchNPU/CANN 配对、native/Cython 装载和可选设备冒烟。用于明确请求的 A2/A3 环境检查；不自动更换 SDK、升级 Torch 或执行旧 TVM 补丁。
---

> 适用范围：TA A2/A3 资产；主线 backend/build/IR/A5 技能保持独立。先读 docs/ascend_a2a3/migration.md，已有用户授权、模式与原配置优先；不重复请求同一批准，不因搬入本 skill 自动授权安装/外发/push/定时任务。

# A2/A3 环境检查

先读主线 `tilelang-build` 技能与 `docs/ascend_a2a3/migration.md`。原 TA 环境检查全文及脚本来源可由文档 manifest 查到；当前检查针对现代 eager/TIRX，不能沿用旧 `build/*.so` 或无条件 `bash install_ascend.sh`。

1. 记录 Python、仓库 commit、CANN/Bisheng 和已装包 metadata；不 import SDK 就能先读取 torch/torch-npu 版本与声明依赖。保留 local version，如 `+cpu`。对明确声明的不兼容版本报错；旧包无声明时报告无法推断，不能捏造兼容性结论。
2. 只读 `git submodule status --recursive`，核对主仓库记录的 pin。A2/A3依赖 TVM/TIRX、catlass、shmem、PTO-ISA；其它共享依赖按构建配置检查，不把 GPU-only 模块硬设为 NPU 运行条件。不要运行定时 pull 或追远端最新版本来“修复”pin。
3. 分开检查源码开发构建与已安装 wheel。dev native位于 `build/lib`/`build/tvm`，安装包位于其 site-package 资源目录；最终以实际 import 模块路径、`/proc/self/maps` 或 loader路径证明，不凭文件名猜装载来源。Cython扩展也要记录实际路径/版本。
4. 检查 CANN 环境与 Bisheng 可发现性，再根据用户目标决定是否 probe NPU。普通 context/source检查可无设备；设备数值或模拟器执行须真实运行才能写通过。不要把 TIRX import 成功当完整设备通过。
5. 需要重建时按用户已授权范围执行主线 cmake/PEP517流程；不得自动换全局环境或已装Torch/NPU配对。TA CI adoption 的modern installer负责metadata精确pin及前后核验；旧 TA branch才使用旧 TVM patch/config。
6. 环境检查可调用相邻 `scripts/check_env.sh` 只读检查；设备冒烟 `scripts/quick_verify.py` 必须单独明确运行，并保留源码/输入/结果。它不是所有形状或所有原资产支持证明。

输出表逐项写“通过 / 失败 / 未执行 / 无法推断”、实际命令和路径。失败只定位具体缺项，不能通过吞异常、换小核、清空用户缓存或自动pip升级把环境刷绿。修复后只重跑受影响检查。
