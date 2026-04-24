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

### 3) LLM 裁判（默认 deepseek-reasoner，可切阿里云百炼 deepseek-v3.2、kimi-k2.5）

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

`DEEPSEEK_API_KEY` **始终必填**（实体保真检查固定走 `deepseek-chat` 抽实体，无论裁判用哪家）。此外按裁判 provider 再加一把 key：

- `--provider deepseek`（默认）：复用上面的 `DEEPSEEK_API_KEY`，无需额外 key
- `--provider bailian`：`BAILIAN_API_KEY`（阿里云百炼 DashScope 的 key）
- `--provider kimi`：`MOONSHOT_API_KEY`

所有 key 写入项目根 `.env`，`shared.env` 自动加载。

## 用法

```bash
# 小样本 dry run（10 行，默认 provider=deepseek，模型 deepseek-reasoner）
uv run python -m evaluate_relevance_contribution_evidence.evaluate \
    --input_path /path/to/pred_res.jsonl \
    --save_path  /tmp/eval_sample.jsonl \
    --max_rows 10 -v

# 全量（默认 deepseek-reasoner）
uv run python -m evaluate_relevance_contribution_evidence.evaluate \
    --input_path /path/to/pred_res.jsonl \
    --save_path  /path/to/eval.jsonl

# 换 Kimi 做裁判
uv run python -m evaluate_relevance_contribution_evidence.evaluate \
    --provider kimi --judge_model kimi-k2.5 \
    --input_path /path/to/pred_res.jsonl --save_path /path/to/eval.jsonl

# 阿里云百炼 deepseek-v3.2（不开思考）
uv run python -m evaluate_relevance_contribution_evidence.evaluate \
    --provider bailian \
    --input_path /path/to/pred_res.jsonl --save_path /path/to/eval.jsonl

# 阿里云百炼 deepseek-v3.2（开思考，走流式聚合 reasoning_content）
uv run python -m evaluate_relevance_contribution_evidence.evaluate \
    --provider bailian --enable_thinking \
    --input_path /path/to/pred_res.jsonl --save_path /path/to/eval.jsonl

# 汇总（output 默认是 <input_parent>/<input_stem>_summary.xlsx）
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

CLI 参数与 `evaluate` 一致：`--provider` / `--judge_model` / `--batch_size` / `--max_workers` / `--max_rows` / `--env_file` / `--enable_thinking` / `-v`。

CLI 参数：`--input_path` / `--save_path` / `--provider {deepseek,kimi,bailian}`（默认 `deepseek`）/ `--judge_model`（默认取 provider 对应的默认模型：deepseek→`deepseek-reasoner`、bailian→`deepseek-v3.2`、kimi→`kimi-k2.5`）/ `--batch_size`（默认 64）/ `--max_workers`（默认 64）/ `--max_rows` / `--env_file` / `--enable_thinking`（仅 `--provider bailian` 生效）/ `-v`。

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
| `eval_reason` | `str \| null` | 裁判理由 |
| `eval_thinking` | `str \| null` | 裁判 `reasoning_content`（deepseek-reasoner / kimi thinking / bailian thinking 流式聚合） |
| `eval_raw_content` | `str \| null` | 裁判原始返回文本（未解析前），方便排查解析失败 |
| `judge_model` | `str` | 裁判模型 id |
| `provider` | `str` | 裁判 provider（`bailian` / `deepseek` / `kimi`） |
| `enable_thinking` | `bool` | 是否开了 bailian 思考模式（其它 provider 恒为 `false`） |
| `entity_extractor_model` | `str` | 实体抽取模型 id，当前固定 `deepseek-chat` |

### 汇总产物（单文件 xlsx，默认 `<input_parent>/<input_stem>_summary.xlsx`）

单 sheet `metrics`，列名即指标名（顺序固定）：`label_match`（1/0）、`format_score`、`entity_fidelity`、`contribution_accuracy`、`contribution_coverage`、`evidence_faithfulness`、`evidence_self_contained`、`evidence_concision`、`language_consistency`。每行对应一条样本（与输入 JSONL 行序一致），缺失值留空；最后一行为各列在非空单元格上的均值（粗体）。

## 断点续传

`evaluate.py` 按 `(query, document)` hash 扫描输出文件里已有的行并跳过；`eval_status == "failed"` 的行会被视为未完成、在下次重跑时重新评估（文件会被就地重写以丢掉这些失败行）。
