"""从 mteb__nq_top100.jsonl 中抽取所有正例（relevance > 0）。"""

import json

INPUT_PATH = "/mnt/d/PrismRerankerV1Data/mteb__nq_top100.jsonl"
OUTPUT_PATH = "/mnt/d/PrismRerankerV1Data/mteb__nq_positives.jsonl"


def main() -> None:
    count = 0
    with open(INPUT_PATH, "r", encoding="utf-8") as fin, open(
        OUTPUT_PATH, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            item = json.loads(line)
            query = item["query"]
            for doc in item["documents"]:
                if doc["relevance"] > 0:
                    fout.write(
                        json.dumps(
                            {"query": query, "document": doc["content"]},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    count += 1
    print(f"Done. Extracted {count} positive pairs -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
