# Train 模块说明

这个目录只负责 reranker 蒸馏训练，目标是让开发者能快速看懂训练链路、定位修改点，并且在不碰业务外围代码的情况下独立维护训练逻辑。

## 为什么要蒸馏

直接的问题是**成本和延迟**。Voyage 等商用 reranker API 排序效果好，但每次调用都有网络延迟和费用，不适合线上高频场景。我们希望用一个 0.6B 的小模型在本地跑推理，同时尽量逼近大模型的效果。

传统做法是拿人工标注的 pos/neg 二元标签直接训练小模型，但这浪费了大量信息——一个只说"相关/不相关"的标签，远不如教师给出的连续分数丰富。教师的分数天然包含了**负例之间的相对难度**：给 0.53 的负例和给 0.19 的负例，对学生来说难度完全不同。Hinton 称这类隐含在软标签中的信息为"dark knowledge"，而蒸馏的核心目的就是把这些知识传递给学生。

数据层面还做了两件事来保证教师信号的质量：

1. **多教师融合**：用 `voyage-rerank-2` 和 `voyage-rerank-2.5` 两个模型的平均分作为最终教师分数（见 [`process_data/kalm_to_list_wise_train_data.py`](/mnt/d/Codes/PrismRerankerV1/process_data/kalm_to_list_wise_train_data.py)），减少单一教师的偏差。
2. **质量门控**：过滤掉任一教师对正例打分低于 0.5、或正例得分低于负例的样本——如果教师自己都拿不准，这条数据就不该拿来教学生。

## Loss 设计

训练使用三个 loss 的加权组合（见 [`trainer.py`](/mnt/d/Codes/PrismRerankerV1/train/trainer.py) 的 `compute_losses()`）：

```
L_total = α · L_rank + β · L_listwise + γ · L_pointwise
```

三者从不同维度指导学生，互不冗余。

### L_rank（排序损失）— 基本排序正确性

保证正例分数高于所有负例，这是 reranker 最基本的要求。

形式上采用 InfoNCE / 交叉熵，比成对 margin loss 更稳定——一次 forward 就能同时处理多个负例。关键设计是**自适应难度加权**：利用教师的正负分差（margin）来衡量每个负例的难度，然后对负例做加权 softmax：

- margin 小 → 教师也觉得这个负例跟正例很像 → 给更大权重
- margin 大 → 明显的负例 → 降低权重

这比均匀对待所有负例更高效。如果一个负例连教师都打了 0.4 的高分，学生应该花更多精力去学会区分它；反之，教师只给 0.02 的负例没太大学习价值。

对应代码：[`trainer.py`](/mnt/d/Codes/PrismRerankerV1/train/trainer.py) 的 `compute_rank_loss()`。

### L_listwise（列表级蒸馏）— 完整排序结构

Rank loss 只关心"正例排第一"，但对负例之间的相对顺序毫不关心。而教师的分数蕴含了丰富的排序结构：给 0.53 的负例确实比给 0.19 的更接近正例，这种信息对学生同样有价值。

做法是将教师分数通过 inverse sigmoid 还原为 logits，和学生 logits 一起除以温度 T，分别做 softmax 得到概率分布，然后计算 KL 散度。温度 T > 1 会把分布"压软"，放大负例之间的差异信号，让学生更容易捕捉到这些微妙的相对关系。最终乘以 T² 是 Hinton 蒸馏的标准做法，用来补偿温度对梯度幅度的缩放。

对应代码：[`trainer.py`](/mnt/d/Codes/PrismRerankerV1/train/trainer.py) 的 `compute_list_loss()`。

### L_pointwise（逐点分数回归）— 绝对分数校准

前两个 loss 都只关心相对顺序，不关心分数的绝对值。但在实际使用中，reranker 经常需要做阈值截断——比如"只保留分数 > 0.5 的文档"。如果模型的分数没有校准（教师给 0.8 的文档学生给 0.3），阈值策略就会失效。

做法很直接：`MSE(sigmoid(student_z), teacher_score)`，让学生的输出概率逼近教师的绝对分数。

对应代码：[`trainer.py`](/mnt/d/Codes/PrismRerankerV1/train/trainer.py) 的 `compute_point_loss()`。

### 三者如何互补

| Loss | 关注维度 | 解决的问题 |
|------|---------|-----------|
| rank | 正例 vs 负例 | 保证正例排在最前面 |
| listwise | 所有文档的相对顺序 | 学到完整的排序结构（dark knowledge） |
| pointwise | 每个文档的绝对分数 | 分数校准，支持阈值截断 |

只用 rank loss，负例之间的顺序是随机的；加上 listwise，排序结构完整了，但分数可能漂移；再加 pointwise，分数也被校准到和教师一致的尺度上。三者覆盖了排序正确性、排序精细度、分数校准三个层面。

## 目录结构

```text
train/
├── constants.py              # prompt 模板、yes/no token id、默认 LoRA target modules
├── config.py                 # 配置 dataclass、YAML 解析、兼容旧版平铺配置、配置校验
├── data.py                   # JSONL 数据集、采样、batch collate
├── modeling.py               # tokenizer / model / LoRA / QLoRA 初始化、logit 提取
├── trainer.py                # loss、eval、日志、checkpoint、主训练器
├── train_v1.py               # 命令行入口
└── train_config.yaml         # 默认配置
```

## 当前训练约束

训练实现围绕以下不变量写死，改其中任何一个都需要同步改 `data.py`、`trainer.py` 和可能的评估逻辑：

1. 一个样本固定是 `1 query + 1 pos + 7 neg`。
2. 模型分数来自最后一个 token 的 `yes_logit - no_logit`。
3. prompt 必须使用原始字符串模板，不能走 `apply_chat_template()`。
4. tokenizer 使用 `left padding`，这样可以直接取 `outputs.logits[:, -1, :]`。
5. 训练 batch 固定为 `1 个 query`，即一个 batch 内有 8 个 document prompt。

代码现在会在加载数据时校验样本形状和 teacher score 范围，数据不对会直接报出文件和行号。

## 技术栈

- `torch`: 张量、优化器、DataLoader
- `transformers`: tokenizer、基础模型、scheduler
- `peft`: LoRA / QLoRA
- `accelerate`: 单卡 / 多卡训练、梯度累积、分布式评估
- `wandb`: 训练指标记录，默认离线模式

## 快速开始

单卡：

```bash
uv run python train/train_v1.py --config train/train_config.yaml
```

多卡：

```bash
uv run accelerate launch --num_processes 2 \
  train/train_v1.py --config train/train_config.yaml
```

## 配置说明

推荐使用分组后的新配置格式，代码仍兼容旧版平铺写法，但新配置更容易维护。

### `model`

- `path`: 基座模型路径。
- `max_seq_length`: prompt 最大长度。
- `dtype`: `bfloat16` / `float16` / `float32` / `null`。
- `load_in_4bit`: 是否启用 4bit 量化加载。
- `attn_implementation`: 例如 `flash_attention_2`；设为 `null` 时使用 transformers 默认实现。
- `gradient_checkpointing`: 是否启用 gradient checkpointing。

### `lora`

- `enabled`: `true` 时 LoRA，`false` 时全参微调。
- `r` / `alpha` / `dropout`: LoRA 超参数。
- `target_modules`: 注入模块列表。
- `use_rslora`: 是否开启 RS-LoRA。

`load_in_4bit: true` 必须和 `lora.enabled: true` 搭配，配置校验会直接拦住不合法组合。

### `data`

- `train_file` / `dev_file`: 训练和开发集 JSONL 路径。
- `train_samples`: 训练集子采样数，`null` 表示全量。
- `eval_samples`: dev 子采样数，`null` 表示全量。
- `num_workers`: DataLoader worker 数。
- `pin_memory`: CUDA 场景下是否启用 pin memory。

数据集子采样使用 reservoir sampling，避免只截前 N 条带来的偏差。

### `training`

- `num_epochs`: 训练轮数。
- `learning_rate`: AdamW 学习率。
- `weight_decay`: 权重衰减。
- `warmup_steps`: scheduler warmup。
- `lr_scheduler`: `cosine` 或 `linear`。
- `grad_accum_steps`: 梯度累积步数。
- `max_grad_norm`: 梯度裁剪阈值，设为 `0` 表示关闭。
- `seed`: 随机种子。

这里的 LR schedule 会按 `accelerate` 的分布式语义计算：

- `optimizer step` 受 `grad_accum_steps * world_size` 影响
- `scheduler step` 只按 `grad_accum_steps` 缩减，不再手动按 `world_size` 缩减

### `loss`

- `alpha_rank`: 排序损失权重。
- `beta_list`: listwise distill 权重。
- `gamma_point`: pointwise distill 权重。
- `temperature`: listwise KL 的温度。
- `hard_neg_scale`: rank loss 的 hard negative 难度缩放。

### `evaluation`

- `interval_samples`: 每看到多少全局训练样本做一次 dev 评估。

这里按 `global_samples_seen` 触发评估，而不是按单卡 step。多卡场景下会按每个 epoch 的真实唯一样本数累加，不把分布式补齐出来的重复样本算进去。

### `output`

- `dir`: 输出目录。
- `run_name`: 本次运行名称。
- `save_every_eval`: 每次 dev 评估时都保存一份 checkpoint 到 `samples-{N}/`，默认开启。
- `save_last_checkpoint`: 训练结束后是否额外保存 `last/`。

保存最优模型（`best/`）和定期保存（`samples-{N}/`）是两个独立逻辑，互不影响。

### `logging`

- `wandb_project`: wandb project 名称。
- `wandb_mode`: `offline` / `online` / `disabled`。
- `log_interval_steps`: 每多少个 optimizer step 记录一次训练日志。

## 训练流程

1. `train_v1.py` 读取 YAML，并在 `config.py` 中完成兼容转换和校验。
2. `modeling.py` 加载 tokenizer / model，并根据配置挂上 LoRA 或 QLoRA。
3. `data.py` 按 query 组装 8 个 prompt，并在 collate 阶段一次性 tokenize。
4. `trainer.py` 计算三种 loss：
   `rank + listwise + pointwise`
5. `accelerate` 负责梯度累积、单卡/多卡统一和分布式 dev MRR 聚合。
6. 只有当 dev MRR 创新高时，才会覆盖 `output_dir/best/`。

## 关键实现细节

### Prompt 和 token id

- prompt 模板在 [`constants.py`](/mnt/d/Codes/PrismRerankerV1/train/constants.py)。
- `yes` / `no` token id 也集中在 [`constants.py`](/mnt/d/Codes/PrismRerankerV1/train/constants.py)。

如果后续换模型族，优先改这一个文件，不要把 prompt 和 token id 分散到训练逻辑里。

### 配置兼容

[`config.py`](/mnt/d/Codes/PrismRerankerV1/train/config.py) 同时支持：

- 旧版平铺配置，例如 `model_path`、`use_lora`
- 新版分组配置，例如 `model.path`、`lora.enabled`

同时会对未知字段直接报错，避免静默忽略拼错的 key。

### 多卡日志

训练 loss 日志现在按所有进程聚合后的平均值记录，不再只看主卡局部 batch。

### 模型保存

保存内容包括：

- 模型权重或 LoRA adapter
- tokenizer
- `train_config.resolved.yaml`

这样后续复现实验不需要再回头翻手工修改过的配置文件。

## 常见改动入口

### 调整 prompt

改 [`constants.py`](/mnt/d/Codes/PrismRerankerV1/train/constants.py)。

### 改 LoRA 注入模块

优先改 [`train_config.yaml`](/mnt/d/Codes/PrismRerankerV1/train/train_config.yaml) 的 `lora.target_modules`；如果你想改默认值，再动 [`config.py`](/mnt/d/Codes/PrismRerankerV1/train/config.py)。

### 新增 loss

改 [`trainer.py`](/mnt/d/Codes/PrismRerankerV1/train/trainer.py) 的 `compute_*_loss()` 和 `compute_losses()`。

### 放宽数据格式

改 [`data.py`](/mnt/d/Codes/PrismRerankerV1/train/data.py) 的样本校验、`EXPECTED_*` 常量，以及 [`trainer.py`](/mnt/d/Codes/PrismRerankerV1/train/trainer.py) 中默认正例 index 的假设。

### 更换保存策略

改 [`trainer.py`](/mnt/d/Codes/PrismRerankerV1/train/trainer.py) 的 `_run_evaluation()` 和 `train()` 结尾逻辑。

## 输出内容

默认输出目录下会有：

- `train_log.txt`: 面向人读的结构化训练日志，时间戳使用北京时间 `UTC+08:00`
- `wandb/`: 离线 wandb 运行目录
- `best/`: 当前 dev MRR 最优 checkpoint
- `samples-{N}/`: 每次评估时保存的 checkpoint（`N` 为全局已见样本数），仅当 `output.save_every_eval: true` 时生成
- `last/`: 仅当 `output.save_last_checkpoint: true` 时生成

## 开发建议

1. 先改配置，再改代码。现在大多数运行策略都能通过 YAML 调整。
2. 先看 [`trainer.py`](/mnt/d/Codes/PrismRerankerV1/train/trainer.py)，再看其他模块。训练主流程已经集中在这里。
3. 如果要改数据形状，不要只改 dataset；评估和 loss 都有隐含假设。
4. 如果环境不支持 `flash_attention_2`，直接把 `model.attn_implementation` 设成 `null`，不要在代码里硬删。
