> 固定 TA 来源资料（资源迁移，不是内核/入口已迁或当前性能验收）：[examples/torch_tl_ascend/README.md](https://github.com/tile-ai/tilelang-ascend/blob/0deb8b93da11b7f0becbcded732d9633b7ca4e34/examples/torch_tl_ascend/README.md)。原 benchmark 数字、设计与代码片段保留；原命令和绝对路径属于当时实验记录，未迁执行资产不能在当前仓直接照跑。当前入口/构建先读[迁移指南](../../../docs/ascend_a2a3/migration.md)，逐项状态见[资源清单](../../../maint/ascend_a2a3/ecosystem_resource_ports.json)及 Python ports。

<!-- TA RESOURCE BODY -->
# Example of integrating TileLang-Ascend operators in PyTorch (as a Python C binding)

This directory provides an example of integrating TileLang-Ascend operators into PyTorch in the form of a Python C binding. Specifically, A python package named `torch_tl_ascend` is shipped inside.

At present, the example operator in [`examples/flash_attention/flash_attn_bhsd`](https://github.com/tile-ai/tilelang-ascend/blob/0deb8b93da11b7f0becbcded732d9633b7ca4e34/examples/flash_attention/flash_attn_bhsd.py) is supported.

## Build & Install

With [TileLang-Ascend installed](https://github.com/tile-ai/tilelang-ascend/blob/0deb8b93da11b7f0becbcded732d9633b7ca4e34/README.md#tilelang-ascend-installation), the `torch_tl_ascend` python package can be built and installed with:

```text
python setup.py install
```

For building the `.whl` file only:

```text
python setup.py bdist_wheel
```

## Test

To test the PyTorch operator integration (i.e. `torch.ops.tl_ascend.flash_attention`):

```text
python test_torch.py
```

To test the packaged source code of the integrated TileLang-Ascend operator (i.e. `torch_tl_ascend.op_source.flash_attn_bhsd.flash_attention_fwd`):

```text
python test_source.py
```

## Basic Usage

To call integrated operators in PyTorch, Please refer to [test_torch.py](https://github.com/tile-ai/tilelang-ascend/blob/0deb8b93da11b7f0becbcded732d9633b7ca4e34/examples/torch_tl_ascend/test_torch.py)

```python
import torch
import torch_tl_ascend

...
output = torch.ops.tl_ascend.flash_attention(q, k, v)
```

Source code of integrated operators are also packaged. To call them, please refer to [test_source.py](https://github.com/tile-ai/tilelang-ascend/blob/0deb8b93da11b7f0becbcded732d9633b7ca4e34/examples/torch_tl_ascend/test_source.py)

```python
from torch_tl_ascend.op_source.flash_attn_bhsd import flash_attention_fwd

...
kernel = flash_attention_fwd(B, S, H, D)
output = kernel(q, k, v)
```

## Additional Example: C++ Integration (libtorch)

The `demo_libtorch` subdirectory provides an example of integrating TileLang-Ascend operators into a C++ application using libtorch and libtorch_npu.

See [demo_libtorch/README.md](demo_libtorch/README.md) for details on building and running this example.

## Directory Structure

```
torch_tl_ascend/
├── compile_tl_op        # Utilities for compiling TileLang-Ascend operators and assembling the package
├── demo_libtorch        # A separate C++ example using libtorch and libtorch_npu
├── src
│   ├── torch_tl_ascend  # The torch_tl_ascend package
│   └── _inner.cpp       # The Python C module of the package for wrapping operators and registering to PyTorch
├── README.md            # This document
├── setup.py             # Script for building and installing the package
├── test_source.py       # Script for testing the PyTorch operator integrations
└── test_torch.py        # Script for testing the packaged source code of integrated operators
```
