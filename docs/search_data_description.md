文件：`/mnt/g/PrismRerankerV1Data/KaLM__all_retrieval_voyage-rerank2_voyage-rerank2.5_web-search-processed.jsonl` 存储了一批搜索数据，格式为json-line, 即one line one json, 每一行使用`json.loads`加载即可，加载后是如下的格式：

```
{
  "query": str,                                      # 查询文本
  "pos_list": list[str],                             # 与该query相关的正例文档列表
  "neg_list": list[str],                             # 与该query不相关的负例文档列表
  "src": str,                                        # 数据来源标识，如"web"
  "voyage-rerank-2_pos_scores": list[float],         # voyage-rerank-2对pos_list的相关性评分，与pos_list一一对应
  "voyage-rerank-2_neg_scores": list[float],         # voyage-rerank-2对neg_list的相关性评分，与neg_list一一对应
  "voyage-rerank-2.5_pos_scores": list[float],       # voyage-rerank-2.5对pos_list的相关性评分，与pos_list一一对应
  "voyage-rerank-2.5_neg_scores": list[float],       # voyage-rerank-2.5对neg_list的相关性评分，与neg_list一一对应
  "web_search_topk_docs": list[str],                 # 以该query进行网页检索返回的top-k文档列表
  "voyage-rerank-2.5_web_search_topk_docs_scores": list[float],  # voyage-rerank-2.5对web_search_topk_docs的相关性评分，与其一一对应
  "voyage-rerank-2_web_search_topk_docs_scores": list[float]     # voyage-rerank-2对web_search_topk_docs的相关性评分，与其一一对应
}

```

voyage-rerank-2和voyage-rerank-2.5是2个重排模型，用于给query-document打分，如果要获取最终分数，可以使用这2个模型的分数的平均值。

