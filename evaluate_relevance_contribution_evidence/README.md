# evaluate_relevance_contribution_evidence

离线评估 reranker 预测质量的模块：判 label 正确率 + 给 contribution / evidence 的抽取质量打多维度原始分。

## 输入

JSONL，每行至少包含：

| 字段 | 类型 | 含义 |
|---|---|---|
| `query` | `str` | 查询 |
| `document` | `str` | 原文 |
| `annotated_label` | `"yes" \| "no"` | 人工标注的相关性 |
| `pred_text` | `str` | 模型输出，形如 `yes\n<contribution>...</contribution>\n<evidence>...</evidence>` |

## 评估指标清单

一共 **9 个指标**，按评测方式分成 3 类：

### 1) 纯规则（Python 函数）

| 指标 | 分值 | 适用样本 | 评测方式 | 代码 |
|---|---|---|---|---|
| `label_match` | `"yes"` / `"no"` | 全部 | 提 `pred_text` 首 token 与 `annotated_label` 直接比较 | [`_enrich_label_fields`](evaluate.py) 在 [evaluate.py](evaluate.py) |
| `format_score` | 0.0–1.0 | 全部 | 首 token 是 yes/no +0.4；`yes` 分支再 +0.3/+0.3（两个标签齐全且正文 >10）；`no` 分支要求整串就是 "no"，多输出内容不加 0.6 | [`compute_format_score`](evaluate.py) 在 [evaluate.py](evaluate.py) |

### 2) LLM + 规则（deepseek-chat 抽实体 + 正则 + 原文对照）

| 指标 | 分值 | 适用样本 | 评测方式 | 代码 |
|---|---|---|---|---|
| `entity_fidelity` | 0.0–1.0 | `annotated == pred == yes` | `deepseek-chat` 从 evidence 抽专名/术语/代号/URL，正则补抽数字/百分比/日期/时间；合并去重后逐个检查是否在 `document` 中逐字出现；score = present/total | [`compute_entity_fidelity`](entity_fidelity.py) 在 [entity_fidelity.py](entity_fidelity.py) |

### 3) LLM 裁判（deepseek-v4-pro，固定）

以下 6 个维度由**一次裁判调用同时产出**，各自独立打分、**不做综合不加权**。适用样本：`annotated == pred == yes`。裁判 prompt：[templates/judge_contribution_evidence.j2](templates/judge_contribution_evidence.j2)。解析逻辑：[`_parse_judge_output`](evaluate.py) 在 [evaluate.py](evaluate.py)。

| 指标 | 分值 | 评测方式（裁判给分依据） |
|---|---|---|
| `contribution_accuracy` | 1–5 | contribution 是否真实反映文档对 query 的贡献 |
| `contribution_coverage` | 1–5 | 是否一句话涵盖核心贡献点，不遗漏也不冗长 |
| `evidence_faithfulness` | 1–5 | ⭐ 最严重：有没有幻觉，数字/专名是否原样保留 |
| `evidence_self_contained` | 1–5 | 仅凭 evidence 能否完整回答 query |
| `evidence_concision` | 1–5 | 是否去掉无关背景，真的做了提炼 |
| `language_consistency` | **5 或 1** | 叙述性文字语种是否符合规则（单语种同步 / 繁简混排→简体 / 多语种→跟 query 语种或英文），二值 |

### 执行顺序

每行按 `label_match` → `format_score` → （若 yes/yes）`entity_fidelity` + 一次 LLM 裁判调用。非 yes/yes 样本跳过后两步，`entity_fidelity` / `eval_scores` 置 `null`。调度逻辑见 [`process`](evaluate.py) 在 [evaluate.py](evaluate.py)。

## 环境

`DEEPSEEK_API_KEY` 必填，裁判（`deepseek-v4-pro`）和实体抽取（`deepseek-chat`）共用同一把 key。写入项目根 `.env`，`shared.env` 自动加载。

## 用法

`evaluate.py` 没有命令行参数；所有可调项都是文件顶部的大写全局变量，直接改源码：

输出路径全部由 `INPUT_PATH` 自动派生：JSONL 是 `<INPUT_PATH 同目录>/<stem>_eval.jsonl`，xlsx 是 `<...>/<stem>_eval_summary.xlsx`，模型名取 `INPUT_PATH.stem`。

| 全局变量 | 默认值 | 含义 |
|---|---|---|
| `INPUT_PATH` | 见源码 | 输入 JSONL 路径（**唯一需要改的路径**） |
| `JUDGE_MODEL` | `deepseek-v4-pro` | 裁判模型 |
| `JUDGE_BASE_URL` | `https://api.deepseek.com` | 裁判 base_url |
| `JUDGE_API_KEY_ENV` | `DEEPSEEK_API_KEY` | 裁判 api key 的环境变量名 |
| `ENTITY_EXTRACTOR_MODEL` | `deepseek-chat` | 实体抽取模型 |
| `BATCH_SIZE` / `MAX_WORKERS` | 64 / 64 | 批大小 / 并发线程数 |
| `MAX_ROWS` | `30` | 仅处理前 N 行，`None` 表示全量；调试时常用 |
| `ENV_FILE` | `None` | 自定义 `.env` 路径，`None` 走默认 |
| `VERBOSE` | `False` | 打开 DEBUG 日志 |

```bash
# 改完源码后直接运行；评估完成会自动在 SAVE_PATH 旁产出 *_summary.xlsx
uv run python -m evaluate_relevance_contribution_evidence.evaluate

# 单独跑汇总（如需用别的 JSONL 二次汇总）
uv run python -m evaluate_relevance_contribution_evidence.summarize \
    --input /path/to/eval.jsonl
```

### evaluate_annotated（评估标注数据质量）

与 `evaluate` 的区别：输入是 step9 产出的标注数据（`contribution_evidence` 字段，无前导 yes/no），而非模型预测（`pred_text` 字段）。仅对 `annotated_label == "yes"` 的行做裁判评分，`no` 行直接透传。无 `label_match` / `format_score` 指标。

```bash
uv run python -m evaluate_relevance_contribution_evidence.evaluate_annotated \
    --input_path /path/to/step9_output.jsonl \
    --save_path  /path/to/step9_eval.jsonl
```

> 注：`evaluate_annotated.py` 仍保留 CLI 参数（`--provider` / `--judge_model` / `--batch_size` / `--max_workers` / `--max_rows` / `--env_file` / `--enable_thinking` / `-v`），暂未与 `evaluate.py` 同步精简。

## 输出

### 增强 JSONL（逐行追加）

保留原行所有字段，追加：

| 字段 | 类型 | 含义 |
|---|---|---|
| `pred_label` | `"yes" \| "no" \| null` | `pred_text` 首 token |
| `label_match` | `"yes" \| "no"` | `pred_label == annotated_label` |
| `format_score` | `float` | 格式分 0.0–1.0（全部样本） |
| `parsed_contribution` | `str \| null` | `<contribution>` 正文（仅 pred=yes） |
| `parsed_evidence` | `str \| null` | `<evidence>` 正文（仅 pred=yes） |
| `entity_fidelity` | `dict \| null` | `{score, extracted, missing}`（仅 yes/yes） |
| `eval_scores` | `dict \| null` | 6 个维度的整数分 |
| `eval_status` | `scored \| skipped_not_yesyes \| skipped_no_parse \| failed` | 评估状态 |
| `eval_thinking` | `str \| null` | 裁判 `reasoning_content`（thinking 已禁用，恒为 `null`） |
| `eval_raw_content` | `str \| null` | 裁判原始返回文本（未解析前），方便排查解析失败 |
| `judge_model` | `str` | 裁判模型 id（恒为 `deepseek-v4-pro`） |
| `entity_extractor_model` | `str` | 实体抽取模型 id，当前固定 `deepseek-chat` |

### 汇总产物（单文件 xlsx，自动产出，路径为 `<INPUT_PATH 同目录>/<INPUT_PATH.stem>_eval_summary.xlsx`）

单 sheet `metrics`，**只有表头 + 一行汇总**：

- A 列 `model` 写模型名（取 `INPUT_PATH.stem`，比如 `Prism-Qwen3.5-Reranker-0.8B`）
- 其余列依次是各维度在非空单元格上的均值：`accuracy`（即 `label_match` 平均，分类准确率）、`format_score`、`entity_fidelity`、`contribution_accuracy`、`contribution_coverage`、`evidence_faithfulness`、`evidence_self_contained`、`evidence_concision`、`language_consistency`
- 不再写每条样本的逐行明细

## 断点续传

`evaluate.py` 按 `(query, document)` hash 扫描输出文件里已有的行并跳过；`eval_status == "failed"` 的行会被视为未完成、在下次重跑时重新评估（文件会被就地重写以丢掉这些失败行）。
