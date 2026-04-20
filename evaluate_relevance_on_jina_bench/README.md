# evaluate_relevance_on_jina_bench

```bash
# BEIR 评测（全部 12 个数据集，结果默认保存到 evaluate_relevance_on_jina_bench/results/）
uv run python -m evaluate_relevance_on_jina_bench beir --batch-size 4

# 自定义输出目录
uv run python -m evaluate_relevance_on_jina_bench beir --batch-size 4 --output-dir /mnt/g/PrismRerankerV1Data/jina_bench_result

# 指定数据集
uv run python -m evaluate_relevance_on_jina_bench beir --batch-size 4 --datasets mteb/scifact mteb/nfcorpus

# 自定义模型和 top-k
uv run python -m evaluate_relevance_on_jina_bench beir --model-name /path/to/model --top-k 200 --batch-size 8

# 查看所有参数
uv run python -m evaluate_relevance_on_jina_bench beir --help


# 评估rerank模型
python -m evaluate_relevance_on_jina_bench rerank_model_test

# 重新计算得分
uv run python -m evaluate_relevance_on_jina_bench eval_topk --results-dir  /mnt/data/PrismRerankerV1Data/jina_bench_result

```
