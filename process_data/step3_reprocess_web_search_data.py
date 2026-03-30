"""
这个代码的目的是当处没想好怎么从tavily search的结果获取topk docs
现在想的差不多，从extra中拿原始数据重新搞

我估计很多搜索接口都要这么干
"""
import json
import re
import random

import tqdm


def find_b64_len(text):
    """
    查找由A-Z、a-z、0-9、+、/组成的最大连续子串长度
    """
    # 定义正则表达式模式：匹配A-Za-z0-9+/这些字符
    pattern = r'[A-Za-z0-9+/]+'

    # 找到所有匹配的子串
    matches = re.findall(pattern, text)

    # 如果没有匹配，返回0
    if not matches:
        return 0

    # 计算每个匹配的长度，返回最大值
    max_length = max(len(match) for match in matches)
    # print(max_length)
    return max_length


def add_web_search_topk(item):
    topk_docs = []
    if "tavily_topk" in item:
        item.pop("tavily_topk")
        search_result = item["extra"]["original_tavily_result"]
        if isinstance(search_result["answer"], str) and search_result["answer"].strip():
            topk_docs.append(search_result["answer"].strip())
        for search_item in search_result["results"]:
            title = search_item.get("title") or ""
            content = search_item.get("content") or ""
            raw_content = search_item.get("raw_content") or ""
            cands = []
            if isinstance(content, str) and content.strip():
                # TODO 20260311: 我觉得大家真实用的时候还是会好好清洗的，所以我只会保留很少的一部分raw content
                for _ in range(10):
                    cands.append(content.strip())
            if isinstance(raw_content, str) and raw_content.strip() and find_b64_len(
                    raw_content.strip()) < 1000:
                cands.append(raw_content.strip())
            if cands:
                fc = random.choice(cands)
            else:
                continue
            if random.random() < 0.5 and title.strip():
                if random.random() < 0.5:
                    fc = f"{title.strip()}\n{fc}"
                else:
                    fc = f"title: {title.strip()}\ncontent: {fc}"
            topk_docs.append(fc)
    elif "original_zhipu_result" in item["extra"]:
        for search_item in item["extra"]["original_zhipu_result"]:
            title = search_item.get("title") or ""
            content = search_item.get("content") or ""
            title, content = title.strip(), content.strip()
            if not content:
                continue
            if title and random.random() < 0.5:
                if random.random() < 0.5:
                    content = f"{title}\n{content}"
                else:
                    content = f"title: {title}\ncontent: {content}"
            topk_docs.append(content)
    elif "original_exa_result" in item["extra"]:
        if not isinstance(item["extra"]["original_exa_result"].get("results"), list):
            return
        for search_item in item["extra"]["original_exa_result"]["results"]:
            title = search_item.get("title") or ""
            content = search_item.get("text") or ""
            title, content = title.strip(), content.strip()
            if not content:
                continue
            if title and random.random() < 0.5:
                if random.random() < 0.5:
                    content = f"{title}\n{content}"
                else:
                    content = f"title: {title}\ncontent: {content}"
            topk_docs.append(content)
    else:
        raise Exception()

    item.pop("extra")
    if len(topk_docs) > 10:
        topk_docs = random.sample(topk_docs, 10)
    if not topk_docs:
        return
    item["web_search_topk_docs"] = topk_docs
    return item


# defaultdict(<class 'int'>, {'original_tavily_result': 14962, 'original_exa_result': 17901, 'original_zhipu_result': 5832})
if __name__ == "__main__":
    read_path = "G:/PrismRerankerV1Data/KaLM__all_retrieval_voyage-rerank2_voyage-rerank2.5_web-search.jsonl"
    save_path = "G:/PrismRerankerV1Data/KaLM__all_retrieval_voyage-rerank2_voyage-rerank2.5_web-search-processed.jsonl"
    write_data = []
    with open(read_path, "r", encoding="utf8") as fr:
        for line in tqdm.tqdm(fr, total=38695):
            # print(f"已处理数量：{idx}")
            item = json.loads(line)
            item = add_web_search_topk(item)
            if not item:
                continue

            write_data.append(json.dumps(item, ensure_ascii=False) + "\n")
            # if len(write_data)>10:
            #     break
    random.shuffle(write_data)
    with open(save_path, "w", encoding="utf8") as fw:
        fw.writelines(write_data)
