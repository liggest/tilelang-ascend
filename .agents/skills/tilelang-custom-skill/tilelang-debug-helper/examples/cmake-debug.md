# 现代 A2/A3 CMake 调试构建

不修改根CMakeLists来硬塞全局`-g -O0`，也不清理用户正在使用的build。按主线build技能准备依赖，在独立目录配置：

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug -DUSE_ASCEND=ON -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_METAL=OFF -DUSE_LLVM=OFF
cmake --build build-debug -j8
```

dev默认loader查build/lib与build/tvm；要调试独立目录，先按实际loader配置选择该构建产物并记录/proc maps、Cython路径，不能从文件存在推断已装载。不要借用原TA旧库冒充当前重编。

GDB/VSCode可保原Python PID等待与断点流程，但断点位置是src/ascend/a2a3的实际pass/codegen，IR为TIRX。修改C++后构建对应产物，再用原factory和配置复现。调试程序不得自动清全局kernel cache或更改SDK环境。
