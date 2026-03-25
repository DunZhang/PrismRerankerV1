# PrismRerankerV1


## llm打标命令行
supported models:
deepseek/deepseek-chat
openrouter/google/gemini-3-flash-preview
moonshot/kimi-k2.5
openrouter/openai/gpt-5.4-mini：没办法关thinking，所以可以跑第二次把最大token数设置成256,然后继续512，1024
bailian/qwen3.5-397b-a17b
openrouter/anthropic/claude-haiku-4.5
zhipu/glm-5

最终选择：
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


uv run python -m process_data.annotate_relevance     --input_path /mnt/g/PrismRerankerV1Data/kalm_web-search_query_document_pairs.jsonl     --save_path /mnt/g/PrismRerankerV1Data/kalm_web-search_query_document_pairs_annotated.jsonl     --model openrouter/anthropic/claude-haiku-4.5     --max_rows 5000 --batch_size 8