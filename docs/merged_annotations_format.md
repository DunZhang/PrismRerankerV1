# 合并标注数据格式说明

## 文件概述

合并标注文件由 `scripts/merge_annotations.py` 生成，格式为 **JSONL**（JSON Lines）——每行一个独立的 JSON 对象，可用 `json.loads(line)` 逐行加载。

该文件将多个 LLM 对同一 query-document 对的相关性标注结果合并到一行中，并基于多模型投票生成最终的聚合标签和分数。

### 数据过滤规则

只有当以下 5 个模型**全部**都有标注结果时，该条数据才会被保留；任一模型缺失标注则整条丢弃：

- `deepseek-chat`
- `google/gemini-3-flash-preview`
- `openai/gpt-5.4-mini`
- `qwen3.5-397b-a17b`
- `anthropic/claude-haiku-4.5`

## 字段说明

### 基础字段（来自上游数据）

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `query` | `str` | 查询文本 |
| `document` | `str` | 文档文本 |
| `voyage-rerank-2_and_2.5_score` | `float` | voyage-rerank-2 和 voyage-rerank-2.5 两个重排模型的平均相关性分数 |
| `keywords` | `list[str]` | 从 query 提取的关键词列表（可选，部分数据可能不含此字段） |

### 分数修正字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `revised_score` | `float` | 对 `voyage-rerank-2_and_2.5_score` 做幂次修正后的分数，计算公式：`score ^ 1.609` |

### 各模型标注字段

每个模型对应一个标注字段，值为该模型对 query-document 对的相关性判断：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `deepseek-chat_annotated_label` | `str` | DeepSeek 模型的标注结果（`"yes"` 或 `"no"`） |
| `google/gemini-3-flash-preview_annotated_label` | `str` | Gemini 3 Flash 模型的标注结果 |
| `openai/gpt-5.4-mini_annotated_label` | `str` | GPT-5.4 Mini 模型的标注结果 |
| `qwen3.5-397b-a17b_annotated_label` | `str` | Qwen3.5 模型的标注结果 |
| `anthropic/claude-haiku-4.5_annotated_label` | `str` | Claude Haiku 4.5 模型的标注结果 |

### 聚合标注字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `annotated_label` | `str` | 5 个模型的多数投票结果。`"yes"` 票数 > 2 时为 `"yes"`，否则为 `"no"` |
| `annotated_score` | `float` | 投票得分，计算公式：`yes票数 / 5`。取值范围 `[0.0, 1.0]`，步长 `0.2` |

## 用法

```bash
uv run python scripts/merge_annotations.py \
    --input_path /path/to/annotated.jsonl \
    --save_path /path/to/annotated_merged.jsonl
```

## 示例

一条输出数据示例：

```json
{
  "query": "如何提高代码质量",
  "document": "代码审查是提高代码质量的有效方法...",
  "voyage-rerank-2_and_2.5_score": 0.856432,
  "keywords": ["代码质量", "提高"],
  "revised_score": 0.8106,
  "deepseek-chat_annotated_label": "yes",
  "google/gemini-3-flash-preview_annotated_label": "yes",
  "openai/gpt-5.4-mini_annotated_label": "yes",
  "qwen3.5-397b-a17b_annotated_label": "no",
  "anthropic/claude-haiku-4.5_annotated_label": "yes",
  "annotated_label": "yes",
  "annotated_score": 0.8
}
```
