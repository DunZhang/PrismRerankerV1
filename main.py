import json
import random
from os.path import join


# 1.7188 是 voyage2.5
# 1.609是2个voyage
# final_score = (0.2 * voyage_rerank_2_score + 0.5 * voyage_rerank_2_5_score + 0.3 * qwen3_reranker_4b_score) ** 2.2846

# def _add_score(item):
#     item["revised_score"] = (item.get("voyage-rerank-2.5_score") * 0.45
#                              + item.get("voyage-rerank-2_score") * 0.1
#                              + item.get("Qwen3-Reranker-4B_score") * 0.45
#                              ) ** 2.8835
#
#     return item


# def _add_score(item):
#     try:
#         item["revised_score"] = item["voyage-rerank-2.5_score"] ** 1.45
#         return item
#     except:
#         return None

def _add_score(item):
    try:
        item["revised_score"] = item["Qwen3-Reranker-4B_score"] ** 0.6425
        return item
    except:
        return None


def main(read_path):
    sft_data = []
    with open(read_path, "r", encoding="utf8") as fr:
        for line in fr:
            item = json.loads(line)
            item = _add_score(item)
            if not item:
                continue
            if (item["revised_score"] > 0.5 and item["annotated_label"] == "yes") or (
                    item["revised_score"] <= 0.5 and item["annotated_label"] == "no"):
                item["loss_type"] = "point-wise;sft"
            else:
                item["loss_type"] = "sft"
            sft_data.append(json.dumps(item, ensure_ascii=False) + "\n")
    return sft_data


if __name__ == "__main__":
    # 这个数据只关注query，document，revised_score，annotated_label，contribution_evidence

    read_path = "G:/PrismRerankerV1Data/final_sft.jsonl"
    save_path = "G:/PrismRerankerV1Data/final_sft_exp.jsonl"

    write_data = main(read_path)
    with open(save_path, "w", encoding="utf8") as fw:
        fw.writelines(write_data)
