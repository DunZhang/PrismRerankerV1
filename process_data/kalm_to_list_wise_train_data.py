import json
import random

import tqdm

if __name__ == "__main__":
    read_path = "/mnt/g/PrismRerankerV1Data/KaLM__all_retrieval_voyage-rerank2_voyage-rerank2.5.jsonl"
    all_save_path = "/mnt/g/PrismRerankerV1Data/KaLM__all_retrieval_voyage-rerank2_voyage-rerank2.5_all_filtered.jsonl"
    train_save_path = "/mnt/g/PrismRerankerV1Data/KaLM__all_retrieval_voyage-rerank2_voyage-rerank2.5_train_filtered.jsonl"
    dev_save_path = "/mnt/g/PrismRerankerV1Data/KaLM__all_retrieval_voyage-rerank2_voyage-rerank2.5_dev_filtered.jsonl"

    write_data = []
    with open(read_path, "r", encoding="utf8") as fr:
        for line in tqdm.tqdm(fr):
            item = json.loads(line)
            pos_scores1, neg_scores1 = (
                item["voyage-rerank-2_pos_scores"],
                item["voyage-rerank-2_neg_scores"],
            )
            pos_scores2, neg_scores2 = (
                item["voyage-rerank-2.5_pos_scores"],
                item["voyage-rerank-2.5_neg_scores"],
            )
            merged_pos_scores = [sum(i) / len(i) for i in zip(pos_scores1, pos_scores2)]
            merged_neg_scores = [sum(i) / len(i) for i in zip(neg_scores1, neg_scores2)]
            # 正例的老师得分都要大于0.5
            # 正例的老师得分都要大于负例的老师得分
            keep = True
            for pos_scores, neg_scores in (
                (pos_scores1, neg_scores1),
                (pos_scores2, neg_scores2),
                (merged_pos_scores, merged_neg_scores),
            ):
                if any((i < 0.5 for i in pos_scores)):
                    # print(f"skip, pos_score < 0.5")
                    keep = False
                    break
                if any((p < n for p in pos_scores for n in neg_scores)):
                    # print(f"skip, pos_score < neg_score")
                    keep = False
                    break
                if len(pos_scores) != 1 or len(neg_scores) != 7:
                    # print(f"skip, pos_list and neg_list should be 1 and 7")
                    keep = False
                    break
            if not keep:
                continue
            write_data.append(
                {
                    "query": item["query"],
                    "pos_list": item["pos_list"],
                    "neg_list": item["neg_list"],
                    "teacher_pos_scores": merged_pos_scores,
                    "teacher_neg_scores": merged_neg_scores,
                }
            )
            # if random.random()<0.01:
            #     print(random.choice(write_data[-1]["pos_list"]+write_data[-1]["neg_list"]))
            #     print("="*100)
            #     print("\n\n\n")
    print(f"过滤后的数据量是：{len(write_data)}")
    random.shuffle(write_data)
    write_data = [json.dumps(i, ensure_ascii=False) + "\n" for i in write_data]
    with open(all_save_path, "w", encoding="utf8") as fw:
        fw.writelines(write_data)
    # with open(train_save_path, "w", encoding="utf8") as fw:
    #     fw.writelines(write_data[:-10000])
    # with open(dev_save_path, "w", encoding="utf8") as fw:
    #     fw.writelines(write_data[-10000:])
