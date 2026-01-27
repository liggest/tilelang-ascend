# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本代码仓库中进行 [Tilelang-Ascend](https://github.com/tile-ai/tilelang-ascend.git) 算子及框架开发提供指导。

## 项目概述

Tilelang-Ascend 是一个面向 Tile 级编程的领域特定语言（DSL），构建在 TVM (Tensor Virtual Machine) 之上，专为编写开发能在华为昇腾 AI 处理器（NPU）上运行的 AI 计算算子而设计。它提供 Pythonic 的前端，可编译到底层 Ascend C 代码。基于当前目录，本项目提供了使用 Tilelang-Ascend 进行算子开发、测试的完整工作流程。

## 项目目录结构

- `./`：Tilelang-Ascend 代码仓库目录，主要包含如下组件：
   - `3rdparty`：第三方依赖源码，如 `tvm`
   - `docs/tutorials`：少许教程与文档
   - `examples`：各种算子代码样例
   - `src`：C++ 侧源码
   - `testing/python/language`：`pytest` 测试逻辑
   - `tilelang`：Python 侧源码，也即 `tilelang` 包
   - `logs`：用于对算子开发过程的记录日志存档（Agent 无需关注其内部文件）

## 可用资料和工具

### 1. 代码仓库资源
- **项目特性介绍**：`README.md`
- **算子样例**：`examples/`
- **简易教程**：`docs/tutorials/`
- **Python 侧接口源码**：`tilelang/language/ascend.py` & `tilelang/language/ascend_tile.py`
- **C++ 侧接口源码（对接 Ascend C）**：`src/tl_templates/ascend/`
- **检索命令示例**：`grep -r "算子或接口名称" examples`
- **标准参考样例（PagedFlashAttention）**：`examples/flash_attention/paged_flash_attn_bhsd.py`

### 2. 辅助工具
<!-- -  `tl-op`：Tilelang-Ascend 算子开发助手
-  `tl-pass`：Tilelang-Ascend 中间表示（TensorIR）变换 Pass 开发 -->
-  `ascend-docs-search`：查询底层 Ascend C 的相关知识、问题以及 API，本次开发环境在 8.5.0，使用 ascend-docs-search 时必须指定版本号 `8.5.0`，否则结果可能有误，仅在本地目录检索不到相关信息时才使用此技能

### 3. 资料查找方法
  当需要查找 API 或实现方法时，按以下顺序：
  1. **第一步**：检索算子样例中是否有 API 的使用样例
  2. **第二步**：检索 Python、C++ 侧接口源码中是否定义了类似的 API
  3. **第三步**：如果本地代码仓库中没有获得有用的信息，尝试使用 `ascend-docs-search` 工具查找 AscendC 相关的信息
