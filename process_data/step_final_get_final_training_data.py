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

def _add_score(item):
    try:
        item["revised_score"] = item["voyage-rerank-2.5_score"] ** 1.45
        return item
    except:
        return None


def main(qp_contribution_evidence_path, rerank_distill_path):
    sft_data, sft_qd = [], []
    with open(qp_contribution_evidence_path, "r", encoding="utf8") as fr:
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
            sft_qd.append(item["query"] + item["document"])
            sft_data.append(json.dumps(item, ensure_ascii=False) + "\n")
    random.shuffle(sft_data)
    sft_qd = set(sft_qd)
    ############################################
    rerank_data = []
    keys = sft_qd
    if isinstance(rerank_distill_path, str):
        rerank_distill_paths = [rerank_distill_path]
    else:
        rerank_distill_paths = rerank_distill_path
    for rerank_distill_path in rerank_distill_paths:
        with open(rerank_distill_path, "r", encoding="utf8") as fr:
            for line in fr:
                item = json.loads(line)
                qd = item["query"] + item["document"]
                if qd in keys:
                    continue
                else:
                    keys.add(qd)
                item = _add_score(item)
                if not item:
                    continue
                item["loss_type"] = "point-wise"
                rerank_data.append(json.dumps(item, ensure_ascii=False) + "\n")
    random.shuffle(rerank_data)
    ############################################

    q_list = list(set([json.loads(item)["query"] for item in sft_data]))
    random.shuffle(q_list)
    train_qs, dev_qs = set(q_list[350:]), set(q_list[:350])
    dev_data = [
        item
        for item in sft_data
        if json.loads(item)["query"] in dev_qs
    ]
    sft_data = [
        item
        for item in sft_data
        if json.loads(item)["query"] in train_qs
    ]
    return rerank_data, sft_data, dev_data


if __name__ == "__main__":
    # 这个数据只关注query，document，revised_score，annotated_label，contribution_evidence

    save_dir = "G:/PrismRerankerV1Data"

    rerank1, sft1, dev1 = main(
        qp_contribution_evidence_path="G:/PrismRerankerV1Data/step9_kalm_web-search_query_document_pairs_contribution_evidence.jsonl",
        rerank_distill_path=[
            "G:/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs_balanced.jsonl",
            # "G:/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs.jsonl",
            # "G:/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs_no_medical_length-score-balance.jsonl",
            # "G:/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs_no_medical.jsonl",
        ]

    )
    print("len(rerank1),len(sft1),len(dev1)", len(rerank1), len(sft1), len(dev1))
    rerank2, sft2, dev2 = main(
        qp_contribution_evidence_path="G:/PrismRerankerV1Data/data_extend2/step9_expanded2_web-search_query_document_contribution_evidence.jsonl",
        rerank_distill_path=[
            "G:/PrismRerankerV1Data/data_extend2/step6_expanded2_web-search_query_document_pairs_length-score-balance.jsonl",
            # "G:/PrismRerankerV1Data/data_extend2/step6_expanded2_web-search_query_document_pairs.jsonl",
        ]

    )
    print("len(rerank2),len(sft2),len(dev2)", len(rerank2), len(sft2), len(dev2))
    if len(dev1) > len(dev2):
        dev1, dev2 = dev2, dev1
    dev2 = random.sample(dev2, len(dev1))
    # qp_contribution_evidence_path
    write_data = sft1 + sft2
    # write_data = sft1
    random.shuffle(write_data)
    with open(join(save_dir, "final_sft.jsonl"), "w", encoding="utf8") as fw:
        fw.writelines(write_data)

    # rerank_distill_path
    write_data = rerank1 + rerank2
    # write_data = rerank1
    random.shuffle(write_data)
    with open(join(save_dir, "final_point_wise.jsonl"), "w", encoding="utf8") as fw:
        fw.writelines(write_data)

    # dev data
    write_data = dev1 + dev2
    # write_data = dev1
    random.shuffle(write_data)
    with open(join(save_dir, "final_dev_data.jsonl"), "w", encoding="utf8") as fw:
        fw.writelines(write_data)
