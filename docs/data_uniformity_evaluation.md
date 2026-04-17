# 数据集分数×长度均匀度评估方法

## 概述

本方法用于评估 Reranker 训练数据在 **分数（score）** 和 **文档长度（document token length）** 两个维度上的分布均匀程度。将数据按分数和长度分别分箱，构成一个 6×8 的网格（48 个格子），然后通过归一化熵等指标量化均匀度。

## 数据要求

JSONL 文件，每行包含：

- `document`：文本内容
- `voyage-rerank-2_and_2.5_score`：原始分数，范围 [0, 1]，由 voyage-rerank-2 和 voyage-rerank-2.5 两个模型的分数取平均得到

## 分箱定义

### 分数分箱（6 档）

对原始分数做修正：`corrected = raw_score ** 1.609`，然后以 `STEP = 1/6` 等宽切分。

| 索引 | 修正后分数区间 |
|------|----------------|
| 0    | [0.000, 0.167) |
| 1    | [0.167, 0.333) |
| 2    | [0.333, 0.500) |
| 3    | [0.500, 0.667) |
| 4    | [0.667, 0.833) |
| 5    | [0.833, 1.000] |

```python
def score_bin(raw):
    return min(int((raw ** 1.609) / (1 / 6)), 5)
```

使用 1.609 次幂的原因：原始分数在高分区聚集严重，做幂变换后各区间的样本量更平衡。

### 长度分箱（8 档）

对 `document` 字段使用 **tiktoken `cl100k_base`** 编码，统计 token 数。编码时需加 `disallowed_special=()`，否则遇到特殊 token（如 `<|endoftext|>`）会报错。

| 索引 | token 数区间    |
|------|-----------------|
| 0    | [0, 64)         |
| 1    | [64, 128)       |
| 2    | [128, 256)      |
| 3    | [256, 512)      |
| 4    | [512, 1024)     |
| 5    | [1024, 2048)    |
| 6    | [2048, 4096)    |
| 7    | [4096, +∞)      |

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
EDGES = [0, 64, 128, 256, 512, 1024, 2048, 4096, 10**9]

def len_bin(n):
    for i in range(len(EDGES) - 1):
        if EDGES[i] <= n < EDGES[i + 1]:
            return i
    return len(EDGES) - 2

token_count = len(enc.encode(document_text, disallowed_special=()))
```

## 均匀度指标

设 48 个格子的计数为 v₁, v₂, ..., v_N（N=48），总量 S = Σvᵢ，概率 pᵢ = vᵢ / S。

### 归一化熵（主指标）

```
H_norm = -Σ(pᵢ × ln(pᵢ)) / ln(N)
```

- 取值范围 [0, 1]，1.0 表示完全均匀
- 这是最核心的指标

### 变异系数（CV）

```
CV = σ / μ
```

其中 μ 为各格均值，σ 为标准差。越接近 0 越均匀。

### max/min 比

```
max/min = max(vᵢ) / min(vᵢ)
```

越接近 1 越均匀，衡量极端偏差。

### 解读参考

| H_norm 范围       | 评价         |
|--------------------|--------------|
| ≥ 0.995            | 接近均匀     |
| [0.97, 0.995)      | 有一定偏斜   |
| < 0.97             | 明显不均衡   |

实际案例参考：

| 数据集 | H_norm | CV | max/min | 评价 |
|--------|--------|----|---------|------|
| step8_annotated_merged（均衡后） | 0.9951 | 0.20 | 2.07 | 接近均匀 |
| step8_expanded2_annotated_merged | 0.9888 | 0.31 | 2.18 | 有一定偏斜 |
| step6_no_medical 原始 | 0.9742 | 0.43 | 10.65 | 明显不均衡 |
| step6_expanded2 原始 | 0.9537 | 0.61 | 11.18 | 明显不均衡 |

## 完整脚本

```python
import json
import math
import tiktoken
from collections import defaultdict

enc = tiktoken.get_encoding("cl100k_base")
STEP = 1 / 6
EDGES = [0, 64, 128, 256, 512, 1024, 2048, 4096, 10**9]
LEN_BINS = list(zip(EDGES[:-1], EDGES[1:]))


def len_bin(n: int) -> int:
    for i, (a, b) in enumerate(LEN_BINS):
        if a <= n < b:
            return i
    return len(LEN_BINS) - 1


def score_bin(raw: float) -> int:
    return min(int((raw**1.609) / STEP), 5)


path = "your_file.jsonl"  # 替换为实际路径
grid: dict[tuple[int, int], int] = defaultdict(int)
total = 0

with open(path, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        sb = score_bin(d["voyage-rerank-2_and_2.5_score"])
        dl = len(enc.encode(d["document"], disallowed_special=()))
        lb = len_bin(dl)
        grid[(sb, lb)] += 1
        total += 1

# 打印表格
headers = [f"[{a},{'+∞' if b >= 10**8 else b})" for a, b in LEN_BINS]
print(f"total rows: {total}\n")
print(f'{"score":<18}' + "".join(f"{h:>12}" for h in headers) + f'{"row_sum":>10}')
col_sums = [0] * len(LEN_BINS)
for sb in range(6):
    lo, hi = sb * STEP, (sb + 1) * STEP
    row_sum = 0
    cells = []
    for lb in range(len(LEN_BINS)):
        v = grid[(sb, lb)]
        cells.append(v)
        row_sum += v
        col_sums[lb] += v
    print(
        f'{f"[{lo:.3f},{hi:.3f})":<18}'
        + "".join(f"{v:>12}" for v in cells)
        + f"{row_sum:>10}"
    )
print(
    f'{"col_sum":<18}'
    + "".join(f"{v:>12}" for v in col_sums)
    + f"{sum(col_sums):>10}"
)

# 均匀度指标
vals = [grid[(sb, lb)] for sb in range(6) for lb in range(len(LEN_BINS))]
N = len(vals)
S = sum(vals)
ps = [v / S for v in vals if v > 0]
H_norm = -sum(p * math.log(p) for p in ps) / math.log(N)
mu = S / N
cv = (sum((v - mu) ** 2 for v in vals) / N) ** 0.5 / mu
mx, mn = max(vals), min(vals)

print(f"\ncells: {N}, nonzero: {len(ps)}")
print(f"mean: {mu:.2f}, min: {mn}, max: {mx}")
print(f"H_norm:  {H_norm:.4f}   (越接近 1 越均匀)")
print(f"CV:      {cv:.4f}   (越接近 0 越均匀)")
print(f"max/min: {mx / max(mn, 1):.2f}")
```

## 均衡化操作

当数据不均匀时，可以通过对每个格子设定上限 T，对超出 T 的格子随机下采样来提升均匀度。

扫描不同 T 值的效果（以 step6_expanded2 为例）：

| 上限 T | 删除量 | 保留量 | H_norm |
|--------|--------|--------|--------|
| 200    | 15,463 | 9,295  | 0.9984 |
| 300    | 11,877 | 12,881 | 0.9940 |
| 400    | 8,946  | 15,812 | 0.9878 |
| 500    | 6,539  | 18,219 | 0.9815 |

根据实际需要权衡数据量和均匀度选择合适的 T。

## 依赖

```bash
uv add tiktoken
```
