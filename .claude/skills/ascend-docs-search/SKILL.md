---
name: ascend-docs-search
description: 搜索和获取华为昇腾文档、查询需要了解的url链接，专门用于Ascend C算子开发。提供完整的API参考、示例代码和故障排除指南。**使用优先级：仅在本地asc-devkit/docs/api/context/目录检索不到相关API时才使用此技能**。
---

# Ascend文档搜索技能

## 概述
本技能专门用于搜索华为昇腾社区的官方文档，为Ascend C算子开发提供准确的技术参考。该技能使Claude能够通过三阶段流程搜索和获取华为昇腾文档：搜索、选择和内容获取。

## 技能架构
本技能实现三阶段工作流：
1. **搜索阶段**：使用关键词搜索昇腾社区文档
2. **选择阶段**：分析搜索结果并选择最相关文档
3. **内容获取阶段**：从选定文档中提取详细内容

## 使用时机
**🚨 优先级规则**：
1. **首先**使用本地检索：`asc-devkit/docs/api/context/` 目录（1022个API文档）
2. **其次**参考本地示例：`asc-devkit/examples/` 目录
3. **最后**才使用此技能搜索官网文档

**仅在以下情况使用此技能**：
- 本地context目录检索不到相关API文档
- 需要更详细的官方说明或最新版本信息
- 本地文档版本过旧或不完整

**适用问题类型**：
- Ascend C API函数用法和参数
- 自定义算子开发方法
- 昇腾芯片编程问题
- 编译错误和故障排除

## 详细使用指南

### 推荐使用方式

#### 第一阶段：快速搜索
使用环境变量 `$CLAUDE_PROJECT_DIR` 引用项目根目录：
```bash
python $CLAUDE_PROJECT_DIR/.claude/skills/ascend-docs-search/scripts/ascend_search_client.py "Ascend C 临时内存申请" --lang zh --max_results 5
python $CLAUDE_PROJECT_DIR/.claude/skills/ascend-docs-search/scripts/ascend_search_client.py "AscendC::Add 接口原型" --version "8.3.RC1" --max_results 8
```

或者使用相对路径（从项目根目录执行）：
```bash
python .claude/skills/ascend-docs-search/scripts/ascend_search_client.py "Ascend C 临时内存申请" --lang zh --max_results 5
```

### 参数说明
`search_documents` 方法接受以下参数：

- `keyword` (必需): 搜索关键词，建议使用中文
  - 示例: "Ascend C exp函数"、"算子开发流程"、"内存管理"
  - 注意: 关键词应具体明确，避免过于宽泛

- `lang` (可选): 语言设置，默认为 "zh"
  - 有效值: "zh" (中文), "en" (英文)
  - 建议: 使用 "zh" 搜索中文文档

- `page_size` (可选): 返回结果数量，默认为10
  - 范围: 1-10
  - 建议: 设置为5-8以获得最佳结果

- `page_num` (可选): 页码，默认为1
  - 范围: 1-100
  - 建议: 通常使用默认值1

- `doc_type` (可选): 文档类型，默认为 "DOC"
  - 有效值: "DOC" (文档), "API" (API文档)
  - 建议: 使用 "DOC" 搜索完整文档

- `--version` (可选): 版本过滤字符串，只返回版本信息中包含该字符串的文档
  - 示例: `--version "8.3.RC1"`
  - 建议: 使用 "8.3.RC1"

#### 第二阶段：选择文档
- **相关性分析**: 根据标题和摘要判断文档相关性
- **优先选择**: 优先选择官方华为文档
- **代码示例**: 选择包含代码示例的文档
- **API参考**: 对于API搜索，选择包含函数原型的文档

#### 第三阶段：获取详细内容
```bash
python $CLAUDE_PROJECT_DIR/.claude/skills/ascend-docs-search/scripts/ascend_content_fetcher.py <URL>
```

或使用相对路径（从项目根目录执行）：
```bash
python .claude/skills/ascend-docs-search/scripts/ascend_content_fetcher.py <URL>
```