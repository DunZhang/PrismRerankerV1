import json
from os.path import join

if __name__ == "__main__":
    # 这个数据只关注query，document，revised_score，annotated_label，contribution_evidence
    qp_contribution_evidence_path = "G:/PrismRerankerV1Data/step9_kalm_web-search_query_document_pairs_contribution_evidence.jsonl"
    rerank_distill_path = "G:/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs.jsonl"

    save_dir = "G:/PrismRerankerV1Data"

    # qp_contribution_evidence_path
    write_data = []
    with open(qp_contribution_evidence_path, "r", encoding="utf8") as fr:
        for line in fr:
            item = json.loads(line)
            if (item["revised_score"] > 0.5 and item["annotated_label"] == "yes") or (
                    item["revised_score"] <= 0.5 and item["annotated_label"] == "no"):
                item["loss_type"] = "point-wise;sft"
            else:
                item["loss_type"] = "sft"
            write_data.append(json.dumps(item, ensure_ascii=False) + "\n")

    with open(join(save_dir, "step10_sft_data.jsonl"), "w", encoding="utf8") as fw:
        fw.writelines(write_data)

    # rerank_distill_path
    write_data = []
    with open(rerank_distill_path, "r", encoding="utf8") as fr:
        for line in fr:
            item = json.loads(line)
            item["revised_score"] = item.get("voyage-rerank-2_and_2.5_score") ** 1.609
            item["loss_type"] = "point-wise"
            write_data.append(json.dumps(item, ensure_ascii=False) + "\n")

    with open(join(save_dir, "step10_point_wise_data.jsonl"), "w", encoding="utf8") as fw:
        fw.writelines(write_data)
