> 固定 TA 来源资料（资源迁移，不是内核/入口已迁或当前性能验收）：[examples/torch_tl_ascend/demo_libtorch/README.md](https://github.com/tile-ai/tilelang-ascend/blob/0deb8b93da11b7f0becbcded732d9633b7ca4e34/examples/torch_tl_ascend/demo_libtorch/README.md)。原 benchmark 数字、设计与代码片段保留；原命令和绝对路径属于当时实验记录，未迁执行资产不能在当前仓直接照跑。当前入口/构建先读[迁移指南](../../../../docs/ascend_a2a3/migration.md)，逐项状态见[资源清单](../../../../maint/ascend_a2a3/ecosystem_resource_ports.json)及 Python ports。

<!-- TA RESOURCE BODY -->
# Example of integrating TileLang-Ascend operators in C++ (using libtorch)

This directory provides an example of integrating TileLang-Ascend operators into a C++ application using libtorch and libtorch_npu.

At present, the example operator in [`examples/flash_attention/flash_attn_bhsd`](https://github.com/tile-ai/tilelang-ascend/blob/0deb8b93da11b7f0becbcded732d9633b7ca4e34/examples/flash_attention/flash_attn_bhsd.py) is supported.

The example operator is [pre-compiled at building](https://github.com/tile-ai/tilelang-ascend/blob/0deb8b93da11b7f0becbcded732d9633b7ca4e34/examples/torch_tl_ascend/demo_libtorch/prepare.py) and dynamically loaded at runtime using `dlopen`.

## Build

With the following prerequisites:
- [TileLang-Ascend installed](https://github.com/tile-ai/tilelang-ascend/blob/0deb8b93da11b7f0becbcded732d9633b7ca4e34/README.md#tilelang-ascend-installation)
- [libtorch_npu compiled from source](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/building_libtorch_npu.md)

Edit libtorch_npu-related paths in `build.sh`:
```text
...
CMAKE_PREFIX_TORCH_NPU="path/to/pytorch/libtorch_npu"  # root path to compiled libtorch_npu
TORCH_NPU_SOURCE_DIR="path/to/pytorch"  # path to torch_npu source code (where "third_party" locates)
...
```

Then, the `flash_attention_demo` can be built with:

```text
bash build.sh
```

## Test

```text
./build/flash_attention_demo
```
The output should be:
```
init successful!
Loaded libop.so from: ./lib/libop.so
Test Passed!
```

## Basic Usage

Call the integrated operator in C++ with tensors on the NPU device:

```cpp
torch::Tensor output = flash_attention_fwd(q, k, v);
```

See [`flash_attention.cpp`](https://github.com/tile-ai/tilelang-ascend/blob/0deb8b93da11b7f0becbcded732d9633b7ca4e34/examples/torch_tl_ascend/demo_libtorch/flash_attention.cpp) for details.

## Directory Structure

```
demo_libtorch/
├── build.sh             # Script for building the demo
├── prepare.py           # Script for preparing compiled TileLang-Ascend operators in lib/ (Using compile_tl_op/)
├── CMakeLists.txt       # CMake configuration
├── flash_attention.cpp  # Main code of the demo
├── README.md            # This document
└── lib/                 # Directory for compiled operators (.so)

compile_tl_op/           # Utilities for compiling operators
```
