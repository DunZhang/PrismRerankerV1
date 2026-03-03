"""
Translate all queries in example.jsonl to Chinese and save locally.
"""
import json
import time
from deep_translator import GoogleTranslator


INPUT_FILE = "/mnt/g/KaLM-embedding-finetuning-data/llm_retrieval_short_long/example.jsonl"
OUTPUT_FILE = "/mnt/g/KaLM-embedding-finetuning-data/llm_retrieval_short_long/example_zh.jsonl"

translator = GoogleTranslator(source="auto", target="zh-CN")


def translate_with_retry(text: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            return translator.translate(text)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1} after error: {e}")
                time.sleep(2)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return text  # fallback to original


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Total entries: {len(lines)}")
    results = []

    for i, line in enumerate(lines):
        data = json.loads(line.strip())
        original_query = data["query"]

        print(f"[{i+1:3d}/{len(lines)}] {original_query[:80]!r}...")
        zh_query = translate_with_retry(original_query)
        print(f"         -> {zh_query[:80]!r}")

        results.append({
            "query_original": original_query,
            "query_zh": zh_query,
        })

        # Avoid rate limiting
        time.sleep(0.3)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nDone! Saved {len(results)} entries to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
