import json
import os
import random

import polars as pl
from os.path import join

NAMES = [
    "AdvertiseGen",
    "arxiv_qa",
    "aya_dataset",
    "cCOVID-News",
    "ChatMed_Consult_Dataset",
    "cMedQA-V2.0",
    "cmrc2018",
    "CodeFeedback",
    "csl",
    "dbpedia-entity",
    "DRCD",
    "dureader",
    "dureader_mrc",
    "ELI5_custom",
    "esci",
    "Expertqa",
    "fiqa",
    "GooAQ",
    "hotpot_qa",
    "law-gpt",
    "lawzhidao",
    "lima-chinese",
    "llm_retrieval_short_long",
    "miracl",
    "mldr",
    "mmarco-chinese",
    "mr-tydi",
    "msmarco-passage",
    "msmarco-v2",
    "Multi-CPR",
    "nfcorpus",
    "PAQ_pairs",
    "PubMedQA",
    "rag-dataset-12000",
    "RefGPT",
    "retrieval_data_llm_infgrad",
    "SearchQA",
    "squad_v2",
    "T2Ranking",
    "triviaqa",
    "UMETRIP-QA",
    "WebCPM",
    "webgpt_comparisons",
    "webqa",
    "wikipedia-nq",
    "yahoo-answers",
]


def _process_item(item: dict, src: str = None):
    # query
    query = item.get("query", "").strip()
    ss = query.split("\n Query: ")
    if len(ss) != 2:
        return
    query = ss[1].strip()
    if len(query) < 2:
        return
    # pos list
    pos_list = item.get("pos")
    neg_list = item.get("neg")
    if (
        not pos_list
        or not neg_list
        or not isinstance(pos_list, list)
        or not isinstance(neg_list, list)
    ):
        return
    pos_list = [i.strip() for i in pos_list if i.strip()]
    neg_list = [i.strip() for i in neg_list if i.strip()]
    if not pos_list or not neg_list:
        return
    return (
        json.dumps(
            {"query": query, "pos_list": pos_list, "neg_list": neg_list, "src": src},
            ensure_ascii=False,
        )
        + "\n"
    )


def merge_and_shuffle_all(save_dir: str):
    all_lines = []
    for name in NAMES:
        file_path = join(save_dir, f"KaLM__{name}.jsonl")
        if not os.path.exists(file_path):
            print(f"跳过不存在的文件: {file_path}")
            continue
        with open(file_path, "r", encoding="utf8") as f:
            lines = f.readlines()
        all_lines.extend(lines)
        print(f"读取 {name}: {len(lines)} 条")
    random.shuffle(all_lines)
    out_path = join(save_dir, "KaLM__all_retrieval.jsonl")
    with open(out_path, "w", encoding="utf8") as fw:
        fw.writelines(all_lines)
    print(f"合并完成，共 {len(all_lines)} 条，已保存至 {out_path}")


if __name__ == "__main__":
    read_dir = "/mnt/g/KaLM-embedding-finetuning-data/"
    save_dir = "/mnt/g/PrismRerankerV1Data/simple_filtered_kalm"
    MAX_NUM = 5 * 10000
    ####################
    total_n = 0
    for name in NAMES:
        print(f"process {name}......")
        read_paths = [
            join(read_dir, name, file_name)
            for file_name in os.listdir(join(read_dir, name))
            if file_name.endswith(".parquet")
        ]
        data = []
        for read_path in read_paths:
            data.extend(pl.read_parquet(read_path).rows(named=True))
        data = [_process_item(item, f"KaLM__{name}") for item in data]
        data = [i for i in data if i]
        random.shuffle(data)
        data = data[:MAX_NUM]
        print(f"number of processed {name} is:{len(data)}")
        with open(join(save_dir, f"KaLM__{name}.jsonl"), "w", encoding="utf8") as fw:
            fw.writelines(data)
        total_n += len(data)
    print(f"全部处理完毕, 总数据量：{total_n}")
    merge_and_shuffle_all(save_dir)
