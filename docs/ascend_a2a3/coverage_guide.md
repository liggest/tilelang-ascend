# A2/A3 覆盖率检查

原TA完整coverage流程保存在source档案；当前main构建与TA受控CI adoption是两种明确布局，不能从旧教程直接复制TVM config或假设根bench脚本已存在。

现代源码构建按主线build文档使用独立目录、USE_ASCEND=ON和编译/链接coverage flags，例如C/CXX使用`--coverage -fprofile-arcs -ftest-coverage`、shared/executable linker使用`--coverage`。保现有SDK配对，不自动替换环境；真实重建后再收集。

原TA CI保留`--coverage`和`--enable-cpp-coverage`，modern installer将其适配到当前CMake flags；legacy分支继续原ENABLE_COVERAGE。执行的bench来自examples/ascend_a2a3，pytest目标是testing/ascend_a2a3与明确main A2A3组件，不能拉整个main CUDA/A5测试树。

Python覆盖率依原单独coverage文件/合并逻辑；C++ coverage避免fork并发写gcda。Coverage报告、原operator active tests和legacy pending都要记录scope，不能把报告存在当全部测试通过。main import期间不clear用户cache，所有需要清理的执行使用独立TILELANG_CACHE_DIR。

具体原CI采集脚本由TA CI adoption专审集成；尚未运行线上runner时仅报告本地验证。不要为获得覆盖率数字改变kernel算法、参数、误差阈值或删除失败case。
