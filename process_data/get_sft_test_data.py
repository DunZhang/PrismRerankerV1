import json
import random
from os.path import join

# TODO 这个想用来做测试数据的前提，是没有把之前的 point 数据拿过来做训练。
#
# 说白了，如果只用了 SFT 的数据，那不然的话就测试集泄露了，所以只能临时用一下。
if __name__ == "__main__":
    # 这个数据只关注query，document，revised_score，annotated_label，contribution_evidence
    qp_contribution_evidence_path = "G:/PrismRerankerV1Data/step9_kalm_web-search_query_document_pairs_contribution_evidence.jsonl"
    rerank_distill_path = "G:/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs.jsonl"

    save_dir = "G:/PrismRerankerV1Data"

    # qp_contribution_evidence_path
    q_list = []
    with open(qp_contribution_evidence_path, "r", encoding="utf8") as fr:
        for line in fr:
            item = json.loads(line)
            q_list.append(item["query"])

    q_set = set(q_list)
    # rerank_distill_path
    write_data = []
    with open(rerank_distill_path, "r", encoding="utf8") as fr:
        for line in fr:
            item = json.loads(line)
            if item["query"] not in q_set:
                write_data.append(
                    json.dumps(
                        {"query": item["query"], "document": item["document"],
                         "revised_score": item.get("voyage-rerank-2_and_2.5_score") ** 1.609},
                        ensure_ascii=False
                    ) + "\n"
                )
            else:
                print("in qset")
    random.shuffle(write_data)
    with open(join(save_dir, "test_data_for_only_sft_model.jsonl"), "w", encoding="utf8") as fw:
        fw.writelines(write_data[:1000])
