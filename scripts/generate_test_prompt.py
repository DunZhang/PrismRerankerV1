import collections
import json
import random

from jinja2 import Template

from langdetect import detect
import pycountry


def detect_language(text):
    code = detect(text)
    # langdetect 返回的可能带地区后缀，取前两位
    lang = pycountry.languages.get(alpha_2=code[:2])
    return lang.name if lang else code


MODELS = [
    "deepseek-chat_annotated_label",
    "google/gemini-3-flash-preview_annotated_label",
    "openai/gpt-5.4-mini_annotated_label",
    "qwen3.5-397b-a17b_annotated_label",
    "anthropic/claude-haiku-4.5_annotated_label",
]
if __name__ == "__main__":
    read_path = "D:/PrismRerankerV1Data/kalm_web-search_query_document_pairs_annotated_merged.jsonl"
    with open("../shared/templates/relevance_extract.j2", "r", encoding="utf8") as fr:
        temp = Template(fr.read().strip())

    with open(read_path, "r", encoding="utf8") as fr:
        lines = fr.readlines()
        random.shuffle(lines)
        for line in lines:
            item = json.loads(line)
            label2count = collections.Counter([item[k] for k in MODELS])
            print(label2count)
            if label2count["yes"] > 2:
                query, document = item["query"], item["document"]
                lang = detect_language(text=document)
                # if lang.lower() != "chinese":
                #     continue
                prompt = temp.render(query=query, document=document, lang=lang)
                with open("p.txt", "w", encoding="utf8") as fw:
                    fw.write(prompt)

                x = input()
                if x.strip() == "xxx":
                    break
