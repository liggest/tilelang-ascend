---
name: tilelang-submodule-pull
description: 处理明确请求的 TA A2/A3 固定三方依赖初始化、读取状态与失败诊断。现代主线按gitlink pin恢复，不自动定时pull或追远端；原TA自动pull工具作为可追溯资料保留。
---

> 适用范围：TA A2/A3 资产；主线 backend/build/IR/A5 技能保持独立。先读 docs/ascend_a2a3/migration.md，已有用户授权、模式与原配置优先；不重复请求同一批准，不因搬入本 skill 自动授权安装/外发/push/定时任务。

# A2/A3 固定子模块恢复

1. 先记录主仓库commit、`git submodule status --recursive`及已有修改。原TA工作区/venv只读约束必须保留，不用reset覆盖用户工作。
2. 需要初始化时，用当前主仓库已记录的gitlink执行 `git submodule update --init --recursive`。不要通过 `git pull --recurse-submodules` 或 `--remote`把依赖换成另一个版本；PTO-ISA/catlass/shmem必须与迁移pin一致。
3. 网络/认证失败记录原stderr和具体模块；按既有授权准备公开依赖或离线本地clone，不能绕过审批、把源码/凭证外发、反复重试到丢失失败证据。
4. 完成后对比gitlink及dirty状态，再依主线build技能构建。现代TVM是TIRX；旧TA的slice/logging补丁不应用到现代pin。
5. 原定时pull脚本与其全文工作流保留为来源资料；不自动创建cron/后台循环。只有用户明确请求该行为且清楚其版本影响时才单独讨论适用性。

缺依赖不等于代码错误；不能临时替换源码分支或删CMake组件伪造通过。报告实际取得的pin、命令、修改和未完成项。
