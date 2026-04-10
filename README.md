# PrismRerankerV1


## llm打标命令行
supported models,一定要按照这个顺序来，因为价格是由低到高来的，这样可以省很多钱:
deepseek/deepseek-chat
bailian/qwen3.5-397b-a17b
openrouter/google/gemini-3-flash-preview
openrouter/anthropic/claude-haiku-4.5
openrouter/openai/gpt-5.4-mini：没办法关thinking，所以可以跑第二次把最大token数设置成256,然后继续512，1024



zhipu/glm-5
moonshot/kimi-k2.5

### 最终选择：
deepseek: 0.64（最独立）
gpt5.4m: 0.70
gemini: 0.72
claude: 0.73
qwen3.5: 0.76
没有任何一对超过 0.82，抱团问题消除了。>=3/5 yes → yes。
deepseek	deepseek-chat_annotated_label
gemini	google/gemini-3-flash-preview_annotated_label
gpt5.4m	openai/gpt-5.4-mini_annotated_label
qwen3.5	qwen3.5-397b-a17b_annotated_label
claude	anthropic/claude-haiku-4.5_annotated_label


uv run python -m process_data_extend2.step7_annotate_relevance     --input_path /mnt/g/PrismRerankerV1Data/data_extend2/step6_added.jsonl    --save_path /mnt/g/PrismRerankerV1Data/data_extend2/step7_expanded2_web-search_query_document_pairs_annotated.jsonl     --model openrouter/openai/gpt-5.4-mini --max_rows 10000 --batch_size 32

uv run python -m process_data.step7_annotate_relevance     --input_path /mnt/g/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs_for-score-length-balance_no_medical.jsonl   --save_path /mnt/g/PrismRerankerV1Data/step7_kalm_web-search_query_document_pairs_annotated.jsonl     --model openrouter/openai/gpt-5.4-mini --max_rows 6500 --batch_size 32

## 文件含义
**注意：第一波数据没有严格按照这个顺序来，而且也包含了一些实验性的东西，所以没法一一对应，只能保证靠后的文件是标准的格式，后期如果想从某个数据再做操作，需要专门处理**
step1_process_kalm：处理最原始的kalm

step2_**_web_search: 给query搜索topk，放到extra里

step3_reprocess_web_search_data: 重新处理整理topk数据，主要是新增web_search_topk_docs，如何获取topk是个有意思的事情，而且一定要想好，重新做代价大，因为做完之后，后面就要去得到重排模型的得分了

step4_get_rerank_teacher_scores: 获取voyage reranker的得分

step5_query_to_keywords: 随机选取一部分问题，去把它变成关键词组合查询的形式,就可以作为纯蒸馏重排模型的数据了，所以这是一个非常重要的中间点。


==============================================

step6_extract_query_document_pair_from_topk_data: 获取（query,document，score）对，用于后续标注处理

step7_annotate_relevance: 使用不同的llm进行打标，一个大模型会占用一行，所以最终会产生很多冗余的行数

step8_merge_annotations: 因为上面讲有冗余，所以说这个代码是专门做合并的，速度很快

step9_generate_contribution_evidence: 取那些标注为 yes 的 query document pair 对，去生成 contribution 和 evidence

# qwen3.5 vllm

```
vllm serve /root/Qwen3.5-2B-epoch-1 \
  -dp 1 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-prefix-caching
  ```