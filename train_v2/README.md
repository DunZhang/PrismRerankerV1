# train_v2 训练说明

这个目录是当前多功能重排模型的训练实现。它和旧版训练逻辑最大的区别有两点：

1. 样本不再是 `1 query + 1 pos + 多个 neg` 的 listwise 结构，而是**单条 query-document pair**。
2. loss 不再是 `rank + listwise + pointwise`，而是**`point-wise distillation + SFT` 的混合训练**。

如果你是从旧版 README 或旧版 `train/` 思路切过来，先把这两个前提换掉，再看下面的说明。

## 训练目标

当前实现把一个 Qwen3-Reranker 类模型当作 causal LM 来训练，同时学两件事：

- **point-wise distillation**
  用教师分数 `revised_score` 监督模型的 `yes/no` 倾向，让模型的相关性分数和教师分数对齐。
- **SFT**
  让模型不仅输出 `yes` / `no`，还输出与 query 相关的贡献和证据文本。

同一条样本可以只参与 point-wise、只参与 SFT，或者同时参与两者。

## 目录结构

```text
train_v2/
├── config.py           # dataclass 配置、YAML 解析、旧版平铺配置兼容、配置校验
├── constants.py        # prompt 入口、LoRA 默认 target modules、yes/no token id
├── data.py             # JSONL 样本解析、双数据源混合、训练 collate
├── modeling.py         # tokenizer / model / LoRA / QLoRA 初始化
├── trainer.py          # point-wise loss、SFT loss、训练循环、日志、checkpoint
├── train_config_local.yaml  # 默认配置样例
├── train_v2.py              # 命令行入口
└── merge_lora.py       # LoRA adapter 合并脚本（当前是硬编码路径的一次性脚本）
```

Prompt 模板和训练指令不在本目录，而在：

- [`../shared/prompts.py`](../shared/prompts.py)
- [`../shared/templates/reranker_raw.j2`](../shared/templates/reranker_raw.j2)

## 数据格式

### 总体形态

训练数据是 **JSONL**，每行一条 query-document pair。代码里对应的数据结构是 `FlatSample`。

所有样本都至少需要这三个字段：

- `query`
- `document`
- `loss_type`

其中 `loss_type` 只能是以下三种之一：

- `point-wise`
- `sft`
- `point-wise;sft`

### 三种样本类型

#### 1. `point-wise`

只做分数蒸馏，需要：

- `query`
- `document`
- `loss_type: "point-wise"`
- `revised_score`

示例：

```json
{
  "query": "如何提高代码质量",
  "document": "代码审查和自动化测试都能降低缺陷率。",
  "revised_score": 0.81,
  "loss_type": "point-wise"
}
```

#### 2. `sft`

只做生成监督，需要：

- `query`
- `document`
- `loss_type: "sft"`
- `annotated_label`（通常是 `yes` 或 `no`）
- `contribution_evidence`（可以为空字符串，代码允许缺省后默认为空）

训练时目标文本会被拼成：

```text
{annotated_label}
{contribution_evidence}
```

也就是：

```python
target_text = f"{annotated_label}\n{contribution_evidence}".strip()
```

示例：

```json
{
  "query": "如何提高代码质量",
  "document": "代码审查和自动化测试都能降低缺陷率。",
  "annotated_label": "yes",
  "contribution_evidence": "<contribution>提到了提高代码质量的方法。</contribution>\n<evidence>文档指出代码审查和自动化测试可以减少缺陷。</evidence>",
  "loss_type": "sft"
}
```

#### 3. `point-wise;sft`

同一条样本同时做 point-wise 和 SFT，需要同时带齐两边字段：

- `query`
- `document`
- `loss_type: "point-wise;sft"`
- `revised_score`
- `annotated_label`（通常是 `yes` 或 `no`）
- `contribution_evidence`（可缺省；缺省时目标文本只剩 label）

示例：

```json
{
  "query": "如何提高代码质量",
  "document": "代码审查和自动化测试都能降低缺陷率。",
  "revised_score": 0.81,
  "annotated_label": "yes",
  "contribution_evidence": "<contribution>提到了提高代码质量的方法。</contribution>\n<evidence>文档指出代码审查和自动化测试可以减少缺陷。</evidence>",
  "loss_type": "point-wise;sft"
}
```

### 官方数据生成方式

如果你沿用当前数据处理链路，`step10_get_final_training_data.py` 的规则是：

- `step10_point_wise_data.jsonl`
  从 rerank distill 数据生成，每条样本都写成 `loss_type = "point-wise"`，并把教师分数变成 `revised_score = score ** 1.609`
- `step10_sft_data.jsonl`
  从带多数投票标签和证据的数据生成：
  - 如果 `revised_score > 0.5` 且 `annotated_label == "yes"`，或 `revised_score <= 0.5` 且 `annotated_label == "no"`，写成 `point-wise;sft`
  - 否则写成 `sft`

换句话说，SFT 数据里已经内含两类样本：

- 教师分数和投票标签一致：同时做两种 loss
- 教师分数和投票标签不一致：只做 SFT，不做 point-wise

## Prompt 形式

训练 prompt 使用原始字符串模板，不走 `apply_chat_template()`。

模板来自 [`../shared/templates/reranker_raw.j2`](../shared/templates/reranker_raw.j2)，结构如下：

```text
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
<Instruct>: {instruction}
<Query>: {query}
<Document>: {doc}<|im_end|>
<|im_start|>assistant
<think>

</think>
```

默认 instruction 来自 [`../shared/prompts.py`](../shared/prompts.py)：

- 先回答 `yes` 或 `no`
- 再给出 XML 格式的 `<contribution>` 和 `<evidence>`

这意味着：

- **point-wise 样本**输入的是 `prompt`
- **SFT 样本**输入的是 `prompt + target_text + eos_token`

## Loss 计算

当前总损失是：

```text
L_total = gamma_point * L_point + gamma_sft * L_sft
```

没有旧版的 `rank loss`、`listwise loss`、`temperature`、`hard_neg_scale`。

### 1. Point-wise loss

point-wise 监督的核心分数定义是：

```text
student_z = yes_logit - no_logit
student_score = sigmoid(student_z)
L_point = MSE(student_score, revised_score)
```

其中：

- `YES_TOKEN_ID = 9693`
- `NO_TOKEN_ID = 2152`

这两个 token id 当前写死在 `constants.py` 里，默认假设你用的是 Qwen3-Reranker 系列模型。

#### point-wise logit 取哪个位置

- 对 `point-wise` 样本：
  prompt 后面没有目标文本，所以直接取 `outputs.logits[:, -1, :]`
- 对 `point-wise;sft` 样本：
  需要监督“第一个回答 token 是 yes 还是 no”，所以取 `outputs.logits[:, prompt_length - 1, :]`

这里的 `prompt_length - 1` 不是随便写的。因为 causal LM 在位置 `t-1` 的 logit 预测的是位置 `t` 的 token，所以如果第一个目标 token 是 `yes` / `no`，就必须读 prompt 最后一个位置的 logits。

### 2. SFT loss

SFT 用的是标准自回归交叉熵：

```text
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
L_sft = cross_entropy(shift_logits, shift_labels, ignore_index=-100)
```

构造 `labels` 时，prompt 部分全部被置为 `-100`，只有目标文本部分参与 loss。

### 3. 混合样本如何算

对 `loss_type = "point-wise;sft"` 的样本：

- 先把整条序列 `prompt + target + eos` 跑一遍
- 同时计算：
  - prompt 边界上的 point-wise loss
  - 目标文本上的 SFT loss
- 再按 `gamma_point` 和 `gamma_sft` 加权求和

对纯 `point-wise` 或纯 `sft` 样本，另一项 loss 为 0。

## 数据加载与混合策略

配置里有两个训练文件：

- `data.sft_data_file`
- `data.point_wise_data_file`

至少要提供一个。

加载逻辑是：

1. 两个文件分别读成 `FlatDataset`
2. 用 `InterleavedDataset` 按 `sft_ratio` 混在一起
3. DataLoader 再对这个混合后的 dataset 做 `shuffle=True`

### `sft_ratio` 的真实含义

如果两个数据源都存在，混合数据集的目标长度是：

```text
max(
  ceil(n_pointwise / (1 - sft_ratio)),
  ceil(n_sft / sft_ratio)
)
```

效果是：

- 尽量让整体采样比例接近 `sft_ratio`
- 较小的数据集会被**循环过采样**
- 这个混合映射是按 `seed` 预先生成的，因此可复现

### `train_samples` 的真实行为

`train_samples` 不是随机采样，也不是 reservoir sampling。

当前代码的行为是：

- 对每个输入文件分别只读取前 `N` 行
- 然后再做混合

如果你想要随机子采样，需要自己先把数据打乱，或者改代码。

## 训练流程

1. 读取 YAML 配置，兼容旧版平铺 key
2. 校验配置合法性
3. 加载 tokenizer 和基座模型
4. 按配置决定是全参微调、LoRA，还是 QLoRA
5. 读取 point-wise / SFT 两份 JSONL 并混合
6. 用 `Accelerator` 做梯度累积和分布式训练
7. 每隔若干个 optimizer step 打日志
8. 每当 `global_samples_seen` 达到阈值时保存一个 checkpoint

当前实现**没有 dev 集评估、没有 early stopping、没有 best checkpoint 选择**。这是一个纯训练循环。

## 配置说明

推荐直接看 [`train_config_local.yaml`](./train_config_local.yaml)。当前配置分为 7 组。

### `model`

- `path`: 基座模型路径，必填
- `max_seq_length`: 最大序列长度
- `dtype`: `bfloat16` / `float16` / `float32` / `null`
- `load_in_4bit`: 是否用 4bit 量化加载
- `attn_implementation`: 例如 `flash_attention_2`
- `gradient_checkpointing`: 是否开启 gradient checkpointing

约束：

- `load_in_4bit: true` 时，`lora.enabled` 必须也是 `true`

### `lora`

- `enabled`: 是否启用 LoRA
- `r`
- `alpha`
- `dropout`
- `target_modules`
- `use_rslora`

### `data`

- `sft_data_file`: SFT 数据路径，可为空
- `point_wise_data_file`: point-wise 数据路径，可为空
- `sft_ratio`: 混合采样时 SFT 占比
- `train_samples`: 每个输入文件最多读取多少行
- `num_workers`: DataLoader worker 数
- `pin_memory`: CUDA 环境下是否启用 pin memory

注意：

- 两个数据文件不能同时为空
- 如果两个数据文件都提供了，`sft_ratio` 虽然校验上允许 `0.0` 和 `1.0`，但运行时会因为除零而出错
- 所以当前代码的安全用法是：
  - 同时提供两份数据时，`0 < sft_ratio < 1`
  - 只提供一份数据时，`sft_ratio` 写什么都无所谓

### `training`

- `num_epochs`
- `learning_rate`
- `weight_decay`
- `warmup_steps`
- `lr_scheduler`: `cosine` 或 `linear`
- `grad_accum_steps`
- `max_grad_norm`: 设为 `0` 表示不裁剪
- `seed`

调度步数的计算方式需要注意：

- optimizer step 数按 `grad_accum_steps * world_size` 来折算
- scheduler step 数只按 `grad_accum_steps` 来折算

这是当前代码的显式实现，不是 README 层面的抽象。

### `loss`

- `gamma_point`: point-wise MSE loss 的权重
- `gamma_sft`: SFT cross-entropy 的权重

### `output`

- `dir`: 输出目录
- `run_name`: wandb run 名称
- `save_interval_samples`: 每看到多少全局样本保存一次 checkpoint

### `logging`

- `wandb_project`
- `wandb_mode`: `offline` / `online` / `disabled`
- `log_interval_steps`: 每多少个 optimizer step 记录一次日志

## 运行方式

### 单卡

```bash
uv run python train_v2/train_v2.py --config train_v2/train_config_local.yaml
```

### 多卡

```bash
uv run accelerate launch --num_processes 2 \
  train_v2/train_v2.py --config train_v2/train_config_local.yaml
```

## 输出内容

输出目录下通常会有：

```text
output_dir/
├── train_config.yaml
├── train_log.txt
├── wandb/
└── samples-5000/
    ├── train_config.resolved.yaml
    ├── tokenizer 配置文件
    └── 模型权重
```

需要特别注意几点：

1. 当前只会按 `save_interval_samples` 定期保存 `samples-{N}`。
2. **训练结束时不会自动额外保存一个 `last/` checkpoint。**
3. 如果总训练样本数小于 `save_interval_samples`，那这次训练可能完全不落模型权重，只留下日志和配置。
4. 当前 checkpoint **不保存 optimizer / scheduler state**，所以它不是严格意义上的可恢复断点。
5. LoRA 模式下保存的是 adapter，不是合并后的完整基座模型。

## LoRA 合并

`merge_lora.py` 现在不是通用 CLI，而是一个带硬编码路径的一次性脚本。

如果你要导出完整模型，需要先改里面的：

- `base_model_path`
- `adapter_path`
- `output_path`

再运行脚本。

## 当前实现的约束和坑

这些约束不是“推荐”，而是当前代码已经默认依赖的行为：

1. **训练 batch size 必须是 1。**
   `make_train_collate_fn()` 里直接校验了 `len(batch) == 1`。

2. **point-wise 打分严格依赖 yes/no token id。**
   如果你换的不是同一词表体系的 Qwen3-Reranker 模型，需要先重新确认 token id。

3. **prompt 必须使用原始模板。**
   不能随便改成 `apply_chat_template()`，否则 prompt 边界和 yes/no logit 的位置都会变。

4. **tokenizer 使用 left padding。**
   这已经在 `modeling.py` 里写死。

5. **SFT 样本必须给目标文本留出空间。**
   当前代码没有检查 target 是否被完全截断。如果 `prompt` 已经把 `max_seq_length` 占满，SFT supervision 会失真，甚至可能出现无有效标签的问题。

6. **当前没有评估逻辑。**
   你只能看训练 loss、日志和 wandb，不能直接从这个目录得到 dev 指标。

7. **`train_samples` 是“截前 N 条”，不是随机抽样。**
   这对调试集分布会有影响。

## 一句话总结

`train_v2` 的本质是：把 reranker 训练改成了**单条 query-document pair 的混合监督训练**，其中 `revised_score` 负责 point-wise 蒸馏，`annotated_label + contribution_evidence` 负责 SFT，而 `loss_type` 决定一条样本到底参与哪几项 loss。
