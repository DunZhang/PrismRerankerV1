# PrismV1 模型训练流程介绍

## 背景信息
我要训练一个LLM.

大小：2B-8B之间的多个尺寸

模型特点: 与只输出一个数字得分的传统 reranker 不同， Prism 在一次推理中同时返回三项结果：校准后的相关度评分、 一句话贡献摘要（说明该页面对查询的核心价值）， 以及去除噪声后的证据段落（可直接送入下游 LLM，无需再读原文）

## 模型输入输出

输入：query+document

输出：一段文本，格式如下：
```
{LABEL}
<contribution>该文档帮 query 解决了什么问题</contribution>
<evidence>基于文档相关内容整合改写的自包含文本</evidence>
```


LABEL：为yes或no，代表文档对回答该 query 是否有实质帮助，此外还可以把 `sigmoid(logit_yes - logit_no)` 作为相关性分数用于排序

contribution：一句话概括该文档帮 query 解决了什么，涵盖所有核心贡献点，但每个点只需点到即可，不要展开。

evidence：将完全替代原文用于辅助回答问题，下游模型只能看到 evidence、永远无法访问原文。忠实是最高原则：是原文的搬运工，只对原文换一种表达方式，绝不添加原文中没有的信息。自包含：文档中所有对回答 query 有用的信息必须保留在 evidence 中，不可遗漏。精简：去除与 query 无直接关联的背景铺垫和冗余内容，只保留对回答 query 有实际贡献的信息。


### 输入输出示例

<query>如何快速减肥</query>
<document>
一项为期12周的随机对照试验（n=200）发现，间歇性禁食组平均减重6.8kg，显著高于传统热量限制组的4.1kg（p&lt;0.01）。研究者认为间歇性禁食通过延长脂肪氧化窗口期来加速减脂。
</document>

对应的输出：
yes
<contribution>用对照实验数据证明了间歇性禁食比传统热量限制减重更快</contribution>
<evidence>一项12周随机对照试验（200人）显示，间歇性禁食组平均减重6.8kg，显著优于传统热量限制组的4.1kg（p&lt;0.01），其机制可能是延长了脂肪氧化窗口期。</evidence>

## 训练数据

一条训练数据格式：
```
prompt, 即模型的输入：

"""
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided.<|im_end|>
<|im_start|>user
<Instruct>: Given a query and a document, judge whether the document is relevant to the query. Answer "yes" or "no", then provide in XML:
1. <contribution>: what the document contributes to the query.
2. <evidence>: a self-contained rewrite of relevant content.
<Query>: {{ query }}
<Document>: {{ doc }}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""


输出，即用来计算交叉熵loss的文本：

"""
{LABEL}
<contribution>该文档帮 query 解决了什么问题</contribution>
<evidence>基于文档相关内容整合改写的自包含文本</evidence>
"""


除此之外还会存在一个相关性分数，即为yes的概率
注意：除了这交叉熵loss, 还会使用sigmoid(logit_yes - logit_no)作为预测得分，来计算MSE Loss
```

## 当前训练流程

对基座 LLM 做 LoRA 微调，使用两个 loss 的加权和：`total = gamma_point * loss_point + gamma_sft * loss_sft`。

Point-wise loss：取 prompt 末位 hidden state 投影得到 yes/no logit，计算 `MSE(sigmoid(logit_yes - logit_no), teacher_score)`，让模型学习相关性打分。teacher_score即为label score.


SFT loss：对 prompt 之后的 target 文本（`yes/no` + `<contribution>` + `<evidence>`）计算自回归交叉熵，让模型学习生成结构化输出。
每条训练样本通过 `loss_type` 字段（`point-wise` / `sft` / `point-wise;sft`）决定参与哪些 loss 的计算。

## 当前训练效果
我在Qwen3.5 4B模型上线进行了2W条数据的训练，经过测试发现模型的排序能力和输出文本能力均基本可看。
备注：
排序能力：即模型输出score的能力如何，这个可以用经典NDCG评估: /mnt/d/codes/PrismRerankerV1/evaluate_relevance_on_jina_bench

文本能力：即模型输出contribution和evidence的能力如何，LLM as Judge: /mnt/d/codes/PrismRerankerV1/evaluate_relevance_contribution_evidence
