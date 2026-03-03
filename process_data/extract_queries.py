"""从 KaLM__all_retrieval.jsonl 提取 query 并 shuffle 输出。"""

import json
import random

SRC_PATH = "/mnt/g/PrismRerankerV1Data/simple_filtered_kalm/KaLM__all_retrieval.jsonl"
OUT_PATH = "/mnt/g/PrismRerankerV1Data/kalm_queries.jsonl"


def main() -> None:
    queries: list[str] = []
    with open(SRC_PATH, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            query = item.get("query", "").strip()
            if query:
                queries.append(query)

    print(f"总计: {len(queries)} queries")
    random.shuffle(queries)

    with open(OUT_PATH, "w", encoding="utf8") as fw:
        for q in queries:
            fw.write(json.dumps({"query": q}, ensure_ascii=False) + "\n")

    print(f"已保存至 {OUT_PATH}")


if __name__ == "__main__":
    main()
