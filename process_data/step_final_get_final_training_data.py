import json
import random
from os.path import join


def main(qp_contribution_evidence_path, rerank_distill_path):
    sft_data = []
    q_list = []
    with open(qp_contribution_evidence_path, "r", encoding="utf8") as fr:
        for line in fr:
            item = json.loads(line)
            if (item["revised_score"] > 0.5 and item["annotated_label"] == "yes") or (
                    item["revised_score"] <= 0.5 and item["annotated_label"] == "no"):
                item["loss_type"] = "point-wise;sft"
            else:
                item["loss_type"] = "sft"
            sft_data.append(json.dumps(item, ensure_ascii=False) + "\n")
            q_list.append(item["query"])

    ############################################
    rerank_data = []
    with open(rerank_distill_path, "r", encoding="utf8") as fr:
        for line in fr:
            item = json.loads(line)
            item["revised_score"] = item.get("voyage-rerank-2_and_2.5_score") ** 1.609
            item["loss_type"] = "point-wise"
            rerank_data.append(json.dumps(item, ensure_ascii=False) + "\n")
    # get dev data
    random.shuffle(rerank_data)
    dev_data = rerank_data[:1000]
    rerank_data = rerank_data[1000:]
    for item in rerank_data:
        q_list.append(json.loads(item)["query"])
    q_set = set(q_list)
    dev_data = [
        item
        for item in dev_data
        if json.loads(item)["query"] not in q_set
    ]

    return rerank_data, sft_data, dev_data


if __name__ == "__main__":
    # 这个数据只关注query，document，revised_score，annotated_label，contribution_evidence

    save_dir = "G:/PrismRerankerV1Data"

    rerank1, sft1, dev1 = main(
        qp_contribution_evidence_path="G:/PrismRerankerV1Data/step9_kalm_web-search_query_document_pairs_contribution_evidence.jsonl",
        rerank_distill_path="G:/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs.jsonl"

    )
    rerank2, sft2, dev2 = main(
        qp_contribution_evidence_path="G:/PrismRerankerV1Data/data_extend2/step9_expanded2_web-search_query_document_contribution_evidence.jsonl",
        rerank_distill_path="G:/PrismRerankerV1Data/data_extend2/step6_expanded2_web-search_query_document_pairs.jsonl"

    )

    # qp_contribution_evidence_path
    write_data = sft1 + sft2
    random.shuffle(write_data)
    with open(join(save_dir, "final_sft.jsonl"), "w", encoding="utf8") as fw:
        fw.writelines(write_data)

    # rerank_distill_path
    write_data = rerank1 + rerank2
    random.shuffle(write_data)
    with open(join(save_dir, "final_point_wise.jsonl"), "w", encoding="utf8") as fw:
        fw.writelines(write_data)

    # dev data
    write_data = dev1 + dev2
    random.shuffle(write_data)
    with open(join(save_dir, "final_dev_data.jsonl"), "w", encoding="utf8") as fw:
        fw.writelines(write_data)
