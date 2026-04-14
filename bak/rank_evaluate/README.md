# rank_evaluate

`rank_evaluate` 是一个面向 Reranker 的离线评测模块。它读取一组 JSONL 数据集，每行表示一个 query、对应正例，以及一组 TopK 检索结果中的负例候选；评测时按配置采样负例、调用 Reranker 打分、计算每条 query 的 `NDCG@10`，最后按数据集求平均，并把多模型结果汇总到一个 `xlsx` 表里。

这个目录现在的目标很明确：

- 统一评测入口：`uv run python -m rank_evaluate ...`
- 支持断点续跑：按 query 粒度缓存
- 支持多后端模型：Voyage API、HF、vLLM、Prism 微调模型
- 让开发者能快速扩展：模型注册、数据加载、评测主循环、报表写入都拆成了独立模块

## 1. 快速开始

先安装依赖：

```bash
uv sync
```

然后准备一个数据目录，里面放若干个 `*.jsonl` 文件。每个文件代表一个评测子集。

最常见的运行方式：

```bash
python -m rank_evaluate.report_from_cache --run_tag neg30_seed42

python -m rank_evaluate   --model prism-reranker-0.6b-vllm   --model_path /root/4B-merged  --num_neg 30 --data_dir /mnt/data/PrismRerankerV1Data/dev  --max_queries 100


python -m rank_evaluate   --model qwen3-reranker-0.6b-vllm  --model_path /mnt/data/public_models/Qwen3-Reranker-0.6B   --num_neg 30 --data_dir /mnt/data/PrismRerankerV1Data/dev  --max_queries 100


python -m rank_evaluate   --model voyage-rerank-2.5  --data_dir /mnt/d/data/prism_reranker_test_data   --num_neg 100 --max_queries 100
```

## 2. 输入数据格式

每个 JSONL 文件的一行就是一条 query，格式如下：

```json
{
  "query": "什么是向量检索中的 rerank？",
  "pos_list": [
    "Reranker 用来对召回结果做更精细的相关性排序。"
  ],
  "neg_list": [
    "这是一个和问题无关的文档。",
    "这也是一个负例。",
    "更多候选负例……"
  ]
}
```

字段约束：

- `query`: `str`
- `pos_list`: `list[str]`，不能为空
- `neg_list`: `list[str]`，可以为空

评测时的处理逻辑：

1. 保留该 query 的全部正例。
2. 从 `neg_list` 中按 `--num_neg` 和 `--seed` 采样负例。
3. 将正负例混合并随机打乱。
4. 调用模型对所有候选文档打分。
5. 计算这一条 query 的 `NDCG@10`。

同一个 `seed` 下，负例采样和 query 子采样都是可复现的。

## 3. 输出内容

一次评测会产生两类输出：

### 3.1 Query 级缓存

默认写入：

```text
rank_evaluate/cache/{model_name}/{run_tag}/{dataset_name}.jsonl
```

其中：

- `model_name`: CLI 里的 `--model`
- `run_tag`: `neg{num_neg}_seed{seed}`
- `dataset_name`: 数据集文件名去掉 `.jsonl`

缓存文件每一行格式：

```json
{
  "idx": 0,
  "ndcg": 1.0,
  "scores": [0.98, 0.12, 0.07],
  "relevance": [1, 0, 0]
}
```

这意味着：

- 中途中断后可以自动跳过已完成 query
- 先跑 `--max_queries 50`，后续再跑全量时会复用前 50 条结果
- 可以直接从缓存重建汇总报表，不必重新调用模型

### 3.2 汇总报表

默认输出：

```text
rank_evaluate/evaluation_results.xlsx
```

表格布局：

- 第 1 列：数据集名
- 第 2 列开始：每列一个模型
- 最后一行：`AVERAGE`

同一个模型重复运行时会覆盖对应列；新模型会追加新列。报表会重新计算每一列的平均值，不会因为别的模型新增了数据集而把缺失项按 `0` 算进去。

## 4. 常用命令

### 4.1 查看帮助

```bash
uv run python -m rank_evaluate --help
```

### 4.2 从缓存重建报表

```bash
uv run python -m rank_evaluate.report_from_cache \
  --run_tag neg100_seed42 \
  --output rerank_compare.xlsx
```

只比较指定模型：

```bash
uv run python -m rank_evaluate.report_from_cache \
  --run_tag neg100_seed42 \
  --models voyage-rerank-2 qwen3-reranker-0.6b-vllm
```

### 4.3 查看或清理缓存

```bash
# 查看某个缓存文件
head -3 rank_evaluate/cache/qwen3-reranker-0.6b-vllm/neg100_seed42/my_dataset.jsonl

# 删除某个模型的所有缓存
rm -rf rank_evaluate/cache/qwen3-reranker-0.6b-vllm

# 删除某个 run_tag 的缓存
rm -rf rank_evaluate/cache/*/neg10_seed42
```

## 5. CLI 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | 必填 | 模型名称 |
| `--model_path` | `None` | 本地模型目录或文件路径；Prism 必填 |
| `--data_dir` | `POSIR_DATA_DIR` | JSONL 数据目录 |
| `--num_neg` | `10` | 每条 query 采样的负例数 |
| `--output` | `rank_evaluate/evaluation_results.xlsx` | 输出报表路径 |
| `--cache_dir` | `rank_evaluate/cache` | query 级缓存目录 |
| `--env_file` | 项目根目录 `.env` | API key 等环境变量文件 |
| `--seed` | `42` | 控制负例采样和 query 子采样 |
| `--max_queries` | `None` | 每个数据集最多评测多少条 query |

## 6. 支持的模型

| 模型名 | 后端 | 是否需要 `--model_path` | 说明 |
| --- | --- | --- | --- |
| `voyage-rerank-2-lite` | Voyage API | 否 | Voyage 官方 rerank API |
| `voyage-rerank-2` | Voyage API | 否 | Voyage 官方 rerank API |
| `voyage-rerank-2.5` | Voyage API | 否 | Voyage 官方 rerank API |
| `voyage-rerank-2.5-lite` | Voyage API | 否 | Voyage 官方 rerank API |
| `qwen3-reranker-0.6b-vllm` | vLLM | 是 | 推荐，支持 prefix caching |
| `qwen3-reranker-4b-vllm` | vLLM | 是 | 更大模型 |
| `qwen3-reranker-8b-vllm` | vLLM | 是 | 更大模型 |
| `prism-reranker-0.6b-vllm` | vLLM | 是 | 使用训练时 prompt 模板的微调模型 |

## 7. 环境准备

### 7.1 Voyage API

在 `.env` 里设置：

```bash
VOYAGE_API_KEY=your_key
```

也支持多 key 轮询：

```bash
VOYAGE_API_KEY_1=key_a
VOYAGE_API_KEY_2=key_b
VOYAGE_API_KEY_3=key_c
```

### 7.2 vLLM

这些后端都默认按 GPU 推理设计，建议：

- Linux / WSL2 + CUDA
- 先执行 `uv sync`
- 再用 `uv run ...` 启动，而不是直接用系统 `python`

如果你直接运行 `python -m rank_evaluate`，但当前环境没有装项目依赖，最常见的错误就是 `ModuleNotFoundError`。

## 8. 模块结构

当前目录结构如下：

```text
rank_evaluate/
├── __main__.py
├── __init__.py
├── run.py
├── config.py
├── model_registry.py
├── data_loader.py
├── evaluator.py
├── metrics.py
├── checkpoint.py
├── report.py
├── report_from_cache.py
└── models/
    ├── base.py
    ├── voyage.py
    └── qwen_vllm.py
```

各文件职责：

- `run.py`: CLI 入口，只负责解析参数、组装配置、启动评测
- `config.py`: 默认路径、`.env` 加载、运行配置校验
- `model_registry.py`: 所有模型的注册表和实例化逻辑
- `data_loader.py`: JSONL 校验、负例采样、样本打乱
- `evaluator.py`: 评测主循环、query 子采样、缓存复用、结果汇总
- `metrics.py`: 排序指标实现，目前默认用 `NDCG@10`
- `checkpoint.py`: query 级缓存读写
- `report.py`: 结果写入 xlsx
- `report_from_cache.py`: 不重跑模型，直接从缓存重建报表
- `models/base.py`: 所有模型实现共享的 `BaseReranker`

## 9. 代码执行流程

完整流程如下：

```text
python -m rank_evaluate
  -> run.py 解析参数
  -> config.py 解析默认值 / .env / data_dir
  -> model_registry.py 构建具体模型
  -> evaluator.py 遍历 data_dir 下所有 JSONL
  -> data_loader.py 读取并准备每条 query 的候选文档
  -> models/*.py 返回每个候选文档的相关性分数
  -> metrics.py 计算每条 query 的 NDCG@10
  -> checkpoint.py 写入 query 级缓存
  -> report.py 写入最终 xlsx
```

## 10. 如何扩展

### 10.1 新增一个模型后端

1. 在 `rank_evaluate/models/` 下实现一个新类，继承 `BaseReranker`。
2. 只需要实现：

```python
from .base import BaseReranker


class MyReranker(BaseReranker):
    def rerank(self, query: str, documents: list[str]) -> list[float]:
        ...
```

3. 在 `rank_evaluate/model_registry.py` 里增加一个 `ModelDefinition`。
4. 在 `build_model()` 里补上实例化逻辑。

模型实现只需要遵守一个约定：返回的分数列表必须和输入 `documents` 对齐，分数越大表示越相关。

### 10.2 新增一个评测指标

当前默认指标是 `NDCG@10`。如果你要加别的指标：

1. 在 `rank_evaluate/metrics.py` 里新增函数。
2. 在 `rank_evaluate/evaluator.py` 里接入该指标的计算与聚合。
3. 如果缓存格式需要变化，一并更新 `checkpoint.py` 和 README。

目前仓库里已经把数据加载、评测循环和缓存层拆开了，新增指标时不需要再改模型代码。

## 11. 开发和验证

基础回归测试：

```bash
uv run python -m unittest discover -s tests
```

当前测试覆盖：

- 数据加载和采样是否稳定
- 断点续跑是否真的跳过已缓存 query
- xlsx 报表平均值是否正确
- 模型别名是否能正确解析

如果你改了核心评测逻辑，至少先跑一遍这个命令。

## 12. 常见问题

### `ERROR: --data_dir not specified and POSIR_DATA_DIR environment variable is not set.`

给 `--data_dir`，或者提前设置 `POSIR_DATA_DIR`。

### `xxx requires --model_path`

你选择的是本地模型型后端，例如：

- `prism-reranker-0.6b-vllm`

这些模型不会自动推断路径，必须显式传。

### 运行一半中断了怎么办？

直接重跑同一条命令即可。只要 `--model`、`--num_neg`、`--seed` 和 `--cache_dir` 没变，已完成 query 会从缓存中跳过。

### 为什么推荐 `uv run`？

因为这个目录依赖 `openpyxl`、`tqdm`、`voyageai`、`transformers`、`vllm` 等项目依赖。`uv run` 会自动进入仓库自己的依赖环境，最不容易踩环境错配。
