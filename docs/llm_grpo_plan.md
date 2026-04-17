# LLM 相关性判别 + 证据生成：GRPO 强化学习训练方案

## 背景

训练一个 2B-8B 规模的 LLM，用于检索场景下的 query-document 相关性判别与证据抽取。模型输入为 `(query, document)`,输出三件事：一个 yes/no 标签（并用 `sigmoid(logit_yes - logit_no)` 作为排序分数）、一段 contribution 概括该文档对 query 的贡献、一段 evidence 作为原文的忠实改写用于下游替代原文。

当前已有一版基于某开源 base LLM 的 LoRA SFT 模型，用 point-wise MSE loss（对齐 teacher score）+ SFT 交叉熵双 loss 训练，2 万条数据下排序和文本能力都"基本可看"。可构造 listwise 数据（同一 query + 多个 doc + teacher score）。本文档讨论在此基础上是否加 thinking、是否加 RL，以及方案怎么落地。

**核心结论：不加 thinking，在现有 SFT 基础上接 GRPO 以提升生成质量（contribution/evidence）**。你的 SFT 阶段用 pointwise MSE 拟合顶级闭源 reranker 的连续分数作为 teacher score，dev loss 已到 0.00x——排序能力基本学饱，RL 阶段没空间再优化排序。真正值得 RL 的是生成侧：SFT 的 cross-entropy 只能让 contribution/evidence"长得像参考答案"，没法直接优化忠实度、精简度这些真正想要的属性。GRPO 作为当前显存和稳定性最优的 group-based RL 算法，是默认选择。

---

## 一、为什么不加 thinking

两个原因：

1. **实证不支持**：Lu et al. (arXiv:2510.08985) 和 Jedidi et al. (arXiv:2505.16886) 两篇 2025 年的研究都发现，CoT 会破坏 pointwise reranker 的 calibration，把分数推向极端，NDCG 反而下降。
2. **任务性质不匹配**：三个输出——yes/no 打分、contribution 概括、evidence 忠实改写——都不是"需要推理"的任务。打分是 calibration、概括是抽取压缩、改写要求忠实搬运，加 think 只增加 hallucination 风险和 3-10 倍延迟。

---

## 二、基座选择

**推荐用 Instruct 版本，不用推理（thinking）版本**。2B-8B 规模的主流开源 LLM 通常有两个分支：一个是 Instruct 版本（或 Chat 版本），为通用指令跟随优化；另一个是推理版本（带 thinking 机制，如 Qwen3 thinking 版、DeepSeek-R1 蒸馏版等）。相关性判别 + 证据抽取任务更接近指令跟随，不是推理任务，Instruct 版本更合适。

如果要复用已有推理版 checkpoint，训练和推理时都要关闭 thinking 模式——具体 API 看模型实现（比如 Qwen3 系列用 `enable_thinking=False` 参数，DeepSeek-R1 系列可通过 system prompt 引导跳过 `<think>` 段）。


---

## 三、GRPO 方案

### 为什么必须上 RL

光靠 SFT 解决不了这个任务的核心问题，必须上 RL：

1. **关键失败模式都是全局属性，token-level 监督压不住**。幻觉、遗漏、冗余必须把 evidence 整体对照原文整体才能判断——一句话本身可能没毛病，但组合起来可能漏了关键信息或添了原文没有的推断。SFT 的 cross-entropy 是逐 token 独立的，看不到"这个 token 在整句话里和原文的关系"；必须上 sequence-level reward 才能对这类错误产生有效梯度。

2. **三个目标互相冲突，SFT 无法教模型权衡**。忠实、自包含、精简常常拉扯——想完整覆盖就难精简，想精简可能漏信息，想包全可能忍不住脑补。SFT 只能"模仿参考答案的字面权衡"——参考答案在三者上是什么比例，模型就学什么比例，无法根据具体 query 和 doc 动态调整。RL 的 reward 让三个目标共同参与 advantage 计算，trade-off 由模型在 rollout 中自己学会。

3. **硬约束只能通过 sequence-level reward 挂上去**。evidence 有一条硬约束："宁可不输出也不能瞎编"。SFT 的 token-level loss 没法表达这种不对称优先级——正确 token 和幻觉 token 的惩罚都是 cross-entropy。RL 可以用规则 gate 把含幻觉的 response 的 reward 直接归零，这是 SFT loss 架构上做不到的事。

4. **RL 能让模型对齐到比 SFT 标注更强的 judge**。这条要说清楚，它不是"突破人类天花板"那种玄学收益——RL 本质是让模型向 LLM judge（Claude / GPT-4 级）的判断标准靠拢。如果 judge 比现有训练集的标注质量更高，RL 就相当于把监督源从"静态标注集"换成"一个更强的打分模型"。在 evidence 改写这种"信息等价但表达允许发散"的任务上，judge 能识别的质量上限通常高于静态标注，这部分 SFT 吃不到。

5. **任务天然适合 RL，reward 信号干净可验证**。前四条是"为什么必须"，这一条是"为什么能成"——忠实/自包含/精简都能 rubric 化让 LLM judge 打分，再加规则 gate 做实体核对，reward 噪声低、成本可控，不会遇到某些 RL 场景里 reward 难以定义或高方差的麻烦。

### 为什么选 GRPO

GRPO 是当前 RL 算法里综合上手成本、稳定性、显存开销最好的默认选择。换 RLOO 也能做，但没有明显理由换。

| 算法 | 适合你的场景吗 | 理由 |
|------|--------------|------|
| **GRPO** | ✅ **首选** | 无 critic 模型，4B/8B 显存压力小；TRL 开箱即用；DeepSeek-R1 验证过稳定性 |
| RLOO | ⚠️ 备选 | 与 GRPO 类似但不做标准差归一化，作为 plan B |
| DPO | ❌ 不推荐 | 离线方法，无法灵活组合多个 reward |
| PPO | ❌ 不推荐 | 需要 4 个模型，显存开销大，训练不稳定，已被 GRPO 替代 |



### SFT → RL 衔接

**必须先做 SFT,不能跳过**。2B-8B 小模型探索能力不足，跳过 SFT 直接 GRPO 会崩（GroupRank 实验中 NDCG 从 42.18 骤降到 38.17）。你现有的 SFT 模型就是天然的 RL 起点。

**SFT 不要训到收敛**。最佳 checkpoint 不是 loss 最低的，而是输出多样性（entropy）最高的——过度 SFT 让模型偏离 base 分布太远，RL 探索空间被压缩。监控 output entropy，开始持续下降时取前一个 checkpoint。

### Reward 设计——整个方案的核心

三个 reward 信号，按重要性排序：

**1) 排序 Reward（权重 0.15，低权重设计）**

由于你的 teacher score 来自顶级闭源 reranker 的连续分数，且 SFT 阶段的 MSE 拟合已经做到 dev loss 0.00x（排序能力基本学饱），RL 阶段的 ranking reward 不再以"提升排序"为目标，而是**作为低权重信号防止生成侧的 RL 把打分能力搞崩**。

做法直接复用 SFT 的 pointwise MSE 思路——取模型输出的 `sigmoid(logit_yes - logit_no)` 和 teacher score 算距离，转成"越大越好"的 reward：

```python
def ranking_reward(pred_score, teacher_score):
    """
    pred_score: sigmoid(logit_yes - logit_no)，需从模型 forward 结果中提取
    teacher_score: 顶级闭源 reranker 的连续分数，[0, 1]
    """
    return 1.0 - abs(pred_score - teacher_score)  # [0, 1]
```



**工程实现注意**：TRL 的 GRPOTrainer 默认只把 completion text 传给 reward 函数，不暴露 logits。要拿 `sigmoid(logit_yes - logit_no)`，需要继承 `GRPOTrainer` 并 override 生成逻辑——在 generate 时开 `output_scores=True, return_dict_in_generate=True`，定位 `{LABEL: }` 后那个 token 的位置取 yes/no token logits。因为你的输出格式固定且 label 在最前面，位置定位简单，改动量 50-100 行。**不要**退化成从 completion text parse yes/no——那样 pred 只有 {0, 1} 两个值，teacher 的连续信号被量化丢光。


**2) 格式 Reward（权重 0.20）**

格式 reward 看似简单但极其重要——RL 中格式崩塌是最常见的失败模式，DeepSeek-R1 始终保持 format reward 与正确性 reward 同等地位。

```python
import re

def format_reward(response_text):
    reward = 0.0
    if re.search(r'\{LABEL:\s*(yes|no)\}', response_text):
        reward += 0.4
    contrib = re.search(r'<contribution>(.*?)</contribution>', response_text, re.DOTALL)
    if contrib and len(contrib.group(1).strip()) > 10:
        reward += 0.3
    evidence = re.search(r'<evidence>(.*?)</evidence>', response_text, re.DOTALL)
    if evidence and len(evidence.group(1).strip()) > 10:
        reward += 0.3
    return reward
```

**format reward 权重永远不低于 0.2**。如果训练中格式率降到 85% 以下,立即停训提高 format 权重后继续。

**3) 文本质量 Reward（权重 0.65，主目标）**

采用 **规则 gate + LLM judge** 两层结构：规则 gate 做 0 成本硬约束，抓幻觉的"大头"；LLM judge 做细粒度评分。

**第一层：规则 gate（硬约束，任一违反 → reward = 0）**

- **实体保真**：用正则 + NER 提取 evidence 中的数字、日期、百分比、专名、术语，每一个必须在原文中逐字出现。这一步几乎不花钱，但能抓掉大部分明显幻觉（"12 周"被改成"16 周"、"6.8kg"被改成"6.5kg"、虚构原文没提的研究机构等）。
- **语言一致**：evidence 语种与原文一致（原文混合语种时默认英文）。

```python
import re, spacy
nlp = spacy.load("zh_core_web_sm")  # 或 en_core_web_sm，按语种切

def rule_gate(evidence, source_doc):
    # 实体保真：NER 抽出命名实体 + 正则抽数字/百分比
    doc = nlp(evidence)
    entities = [e.text for e in doc.ents 
                if e.label_ in {"DATE", "CARDINAL", "PERCENT", "ORG", "PERSON", "GPE", "PRODUCT"}]
    numbers = re.findall(r'\d+\.?\d*%?', evidence)
    for token in set(entities + numbers):
        if token not in source_doc:
            return False
    # 语种一致
    if detect_language(evidence) != detect_language(source_doc):
        return False
    return True
```

**第二层：LLM judge 打分**

用比策略模型更强的模型（如 GPT-4 / Claude）做 judge，一次调用同时输出三项分数。三者的相对重要性：**忠实度 ≈ 自包含度 > 精简度**——忠实是最高原则，自包含保证 evidence 能完整替代原文，精简度次之。

- **忠实度**：evidence 每一句是否都能从原文直接得出？有没有原文没说的因果、结论、推断？
- **自包含度**：对照原文，evidence 是否完整保留了所有与 query 相关的信息？有没有遗漏？
- **精简度**：evidence 中有没有与 query 无关的背景或冗余？

```python
judge_prompt = """评估下面的 evidence 是否高质量地替代了原文用于回答 query。

Query: {query}
原文: {source_doc}
Evidence: {evidence}

对以下三项分别打 0-3 分，只返回 JSON，不要解释：
- faithfulness: evidence 每句都能从原文直接得出，无额外推断或编造 → 3 分
- self_contained: evidence 完整保留了原文中所有与 query 相关的信息，无遗漏 → 3 分
- conciseness: evidence 无与 query 无关的冗余 → 3 分

格式: {{"faithfulness": x, "self_contained": x, "conciseness": x}}"""

def text_quality_reward(query, source_doc, evidence):
    # 第一层：规则 gate 硬约束
    if not rule_gate(evidence, source_doc):
        return 0.0
    # 第二层：LLM judge 打分，忠实和自包含优先于精简
    scores = call_llm_judge(judge_prompt.format(
        query=query, source_doc=source_doc, evidence=evidence))
    return (3 * scores['faithfulness'] + 3 * scores['self_contained'] 
            + scores['conciseness']) / 21  # 归一化到 [0, 1]
```


**多 Reward 组合注意事项**：

- 三项权重：text_quality 0.65 + format 0.20 + ranking 0.15，text_quality 是主目标
- 所有 reward 归一化到 [0, 1]
- GRPO advantage 会自动做 z-score 归一化，绝对尺度不影响，但**相对方差要相近**，否则高方差 reward 会主导梯度
- 训练初期可临时把 format 权重提到 0.4、text_quality 降到 0.45，让模型先学稳格式再追求质量
