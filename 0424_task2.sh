


# python -m evaluate_relevance_on_jina_bench rerank_model_test /root/prism_released_models/Prism-Qwen3.5-Reranker-2B
# sleep 120
# python infer_on_test_data_hf.py /root/prism_released_models/Prism-Qwen3.5-Reranker-2B
# sleep 120


# python -m evaluate_relevance_on_jina_bench rerank_model_test /root/prism_released_models/Prism-Qwen3.5-Reranker-4B
# sleep 120
# python infer_on_test_data_hf.py /root/prism_released_models/Prism-Qwen3.5-Reranker-4B
# sleep 120


# python -m evaluate_relevance_on_jina_bench rerank_model_test /root/prism_released_models/Prism-Qwen3.5-Reranker-9B
# sleep 120
# python infer_on_test_data_hf.py /root/prism_released_models/Prism-Qwen3.5-Reranker-9B
# sleep 120

python -m evaluate_relevance_on_jina_bench rerank_model_test /root/prism_released_models/Prism-Qwen3.5-Reranker-4B
sleep 120
python -m evaluate_relevance_on_jina_bench rerank_model_test /root/prism_released_models/Prism-Qwen3.5-Reranker-9B
sleep 120