# pred_quality_eval

评估 Qwen3.5-2B reranker 在测试集上的预测质量。

## 做了什么

1. **Label 准确率**：从 `pred_text` 提取 yes/no，和 `annotated_label` 对比。
2. **Contribution / Evidence 质检**：对 `annotated_label == pred_label == "yes"` 的行，用 **Kimi k2.5（思考模式开启）** 作为裁判，按 6 个维度（5 分制）给模型抽取的 contribution 和 evidence 打分。

裁判的评分规则写在 [templates/judge_contribution_evidence.j2](templates/judge_contribution_evidence.j2)，评分维度：

| 维度 | 说明 |
|---|---|
| `contribution_accuracy` | contribution 是否真实反映文档对 query 的贡献 |
| `contribution_coverage` | 是否一句话涵盖核心贡献点，不遗漏也不冗长 |
| `evidence_faithfulness` | ⭐ 最严重：有没有幻觉，数字/专名是否原样保留 |
| `evidence_self_contained` | 仅凭 evidence 能否完整回答 query |
| `evidence_concision` | 是否去掉了无关背景，真的做了提炼 |
| `overall` | 综合，非机械平均；任一维出现严重幻觉 overall ≤ 2 |

## 环境

需要设置 `MOONSHOT_API_KEY` 环境变量（在项目根 `.env` 里即可，会被 `shared.env` 自动加载）。

## 用法

```bash
# 小样本 dry run（先跑 10 行看看）
uv run python -m pred_quality_eval.evaluate \
    --max_rows 10 \
    --save_path /tmp/eval_sample.jsonl \
    -v

# 全量
uv run python -m pred_quality_eval.evaluate \
    --input_path /mnt/g/PrismRerankerV1Data/Qwen3.5-2B-samples-4000-result.jsonl \
    --save_path  /mnt/g/PrismRerankerV1Data/Qwen3.5-2B-samples-4000-result.eval.jsonl

# 汇总
uv run python -m pred_quality_eval.summarize \
    --input  /mnt/g/PrismRerankerV1Data/Qwen3.5-2B-samples-4000-result.eval.jsonl \
    --outdir /mnt/g/PrismRerankerV1Data/eval_summary
```

## 输出

### 增强 JSONL

保留原行所有字段，追加：

| 字段 | 类型 | 含义 |
|---|---|---|
| `pred_label` | `"yes" \| "no" \| null` | 从 `pred_text` strip 后提取的首 token |
| `label_match` | `"yes" \| "no"` | `pred_label == annotated_label` |
| `parsed_contribution` | `str \| null` | 从 `pred_text` 抽取的 `<contribution>` |
| `parsed_evidence` | `str \| null` | 从 `pred_text` 抽取的 `<evidence>` |
| `eval_status` | `scored \| skipped_not_yesyes \| skipped_no_parse \| failed` | 评估状态 |
| `eval_scores` | `dict \| null` | 6 个维度的 int 分数 (1–5) |
| `eval_reason` | `str \| null` | 裁判给出的理由 |
| `eval_thinking` | `str \| null` | Kimi thinking 模式的 reasoning_content |
| `judge_model` | `str` | 裁判模型名，默认 `kimi-k2.5` |

### 汇总文件（`outdir/`）

- `metrics.json`：label accuracy、confusion matrix、precision/recall/F1、平均分、分布、低分样例
- `summary.md`：人读友好版本，含 ASCII 分数分布直方图

## 断点续传

`evaluate.py` 会扫描输出 JSONL 里已存在的 `(query, document)` 对并跳过，支持中断后重跑。
