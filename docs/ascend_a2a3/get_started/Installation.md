# A2/A3 安装与开发构建

A2/A3当前采用现代主线CMake/PEP517与TIRX，不使用旧TA页面中的CUDA镜像、旧TVM config或旧patch。先读{ref}`完整迁移指南 <a2a3-build-environment>`和主线安装文档；本页原通用GPU安装全文在非活动source档案保留。

1. 准备与当前设备/CANN匹配的Torch和TorchNPU；从已装metadata记录精确版本（含local suffix）。安装开发依赖时使用该pair的pip constraints，解析不满足就报错，不自动升级或换栈。
2. 按主线build技能准备构建依赖。开启USE_ASCEND，CUDA/ROCm/Metal按实际需要配置；仅A2/A3开发可明确关闭其它GPU后端。当前原生构建产物及Cython由同一个构建生成，不能拿TA旧二进制替代。
3. dev loader实际使用build/lib与build/tvm；安装wheel实际使用包内native资源。分别检查模块/Cython/maps来源；`import tilelang`成功不等于设备可执行或功能完整。
4. 原代表例从仓库根执行，必须设置专属TILELANG_CACHE_DIR；保真脚本的main会真实clear所选namespace。不要裸运行使其清调用者默认cache。
5. source/compile-only、模拟器编译、模拟器执行、NPU数值/性能分别验证。A3代表族已有阶段验收，不代表A2实机或所有原生态资产均已跑过。

TA CI adoption的旧布局保持原installer，现代分支需明确eager/TIRX检测、SDK配对约束与每次重配置；这不授权当前用户直接给新TVM打旧补丁。旧conf/Makefile/CNAME是来源资料，不参与主站build。

精确命令和当前支持范围以迁移指南与当前主线build技能为准，避免维护另一套容易失真的安装脚本。
