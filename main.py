import json

with open("G:/PrismRerankerV1Data/step5_KaLM__all_retrieval_voyage-rerank2_voyage-rerank2.5_web-search-processed_keywords.jsonl","r",encoding="utf8") as fr:
    for line in fr:
        item = json.loads(line)
        if "keywords" in item:
            print(item["keywords"])