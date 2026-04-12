# prism_rerank_evaluation

```bash
# BEIR 评测（全部 12 个数据集，结果默认保存到 prism_rerank_evaluation/results/）
uv run python -m prism_rerank_evaluation beir --batch-size 4

# 自定义输出目录
uv run python -m prism_rerank_evaluation beir --batch-size 4 --output-dir /mnt/g/PrismRerankerV1Data/final_rerank_test

# 指定数据集
uv run python -m prism_rerank_evaluation beir --batch-size 4 --datasets mteb/scifact mteb/nfcorpus

# 自定义模型和 top-k
uv run python -m prism_rerank_evaluation beir --model-name /path/to/model --top-k 200 --batch-size 8

# 查看所有参数
uv run python -m prism_rerank_evaluation beir --help
```
