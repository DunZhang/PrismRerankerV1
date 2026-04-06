"""将 instruction 转换为搜索 query，调用 DeepSeek API。

用法:
    uv run python process_data_extend2/instruction2query.py

所有参数通过全局变量配置，直接修改代码即可。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# ============================================================
# 配置
# ============================================================
INPUT_PATH = Path(
    "/mnt/g/PrismRerankerV1Data/data_extend2/instructions_balanced.jsonl"
)
OUTPUT_PATH = Path(
    "/mnt/g/PrismRerankerV1Data/data_extend2/queries_from_instructions.jsonl"
)
ENV_FILE = Path("/mnt/d/Codes/PrismRerankerV1/.env")

MAX_WORKERS = 32
BATCH_SIZE = 64
LIMIT: int | None = 30000  # 只处理前 N 行，None 表示全量
FORMAL_RATIO = 0.5  # 正规 query 的比例，剩余为口语化 query
SEED = 42

# ============================================================
# Prompt
# ============================================================

# 正规化 query prompt：生成标准、规范的搜索引擎 query
SYSTEM_PROMPT_FORMAL = """\
你是一个搜索引擎查询生成助手。用户会给你一条 instruction，\
你需要判断：为了更好地回答这个 instruction，是否需要去互联网上搜索信息。

**判断原则：宽松搜索。** 只要搜索结果对回答有一定帮助，就应该生成 query。\
不要轻易判定为"不需要搜索"。大多数知识性、事实性、技术性问题都值得搜索。\
只有纯粹的数学计算、简单的代码补全、纯创意写作等完全不依赖外部知识的场景才不搜索。

**输出格式（严格遵守，不得违反）：**
- 需要搜索：直接输出一条搜索 query，必须是自然语言问句，不超过 32 个字。\
禁止输出任何解释、前缀、编号或多余文字，你的回复中有且仅有 query 本身。
- 不需要搜索：只输出 none

**语言约束：** 生成的 query 必须使用 {lang} 语言，与 instruction 的语言保持一致。

再次强调：你的输出只能是一条 query 或者 none，绝对不要输出其他任何内容。"""

# 口语化 query prompt：模拟真实用户在搜索框里随手打出的问题
SYSTEM_PROMPT_CASUAL = """\
你是一个模拟真实用户搜索行为的助手。用户会给你一条 instruction，\
你需要想象：一个普通人遇到这个问题时，会在搜索引擎里怎么搜？

**判断原则：宽松搜索。** 只要搜索结果对回答有一定帮助，就应该生成 query。\
不要轻易判定为"不需要搜索"。大多数知识性、事实性、技术性问题都值得搜索。\
只有纯粹的数学计算、简单的代码补全、纯创意写作等完全不依赖外部知识的场景才不搜索。

**风格要求：** 模拟真实用户的搜索习惯——口语化、不完整、带语气词都可以。\
比如用户不会搜"如何解决Python内存溢出问题"，而是搜"python内存爆了怎么办"。\
不要写得像教科书，要像真人随手在搜索框打的那种。

**输出格式（严格遵守，不得违反）：**
- 需要搜索：直接输出一条口语化的搜索 query，不超过 32 个字。\
禁止输出任何解释、前缀、编号或多余文字，你的回复中有且仅有 query 本身。
- 不需要搜索：只输出 none

**语言约束：** 生成的 query 必须使用 {lang} 语言，与 instruction 的语言保持一致。

再次强调：你的输出只能是一条 query 或者 none，绝对不要输出其他任何内容。"""

LANG_DISPLAY: dict[str, str] = {
    "zh-cn": "中文",
    "en": "English",
}

USER_TEMPLATE = "instruction: {instruction}"

# ============================================================
# Logging
# ============================================================
log = logging.getLogger("instruction2query")


def setup_logging() -> None:
    """配置日志。"""
    fmt = "%(asctime)s | %(levelname)-5s | %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt, datefmt="%H:%M:%S", stream=sys.stderr
    )
    for name in ("httpx", "openai", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ============================================================
# 数据加载与缓存
# ============================================================
def load_input(path: Path, limit: int | None = None) -> list[dict]:
    """加载输入 jsonl，可选限制行数。"""
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def instruction_hash(instruction: str) -> str:
    """计算 instruction 的 sha256 摘要（前16位）。"""
    return hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:16]


def load_cache(path: Path) -> set[str]:
    """从已有输出文件加载已处理的 instruction hash 集合。"""
    done: set[str] = set()
    if not path.exists() or path.stat().st_size == 0:
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done.add(row["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    log.info("缓存加载完成: %d 条已处理", len(done))
    return done


# ============================================================
# API 调用
# ============================================================
def call_deepseek(
    client: OpenAI, instruction: str, lang: str, style: str
) -> str:
    """调用 DeepSeek 生成搜索 query。

    Args:
        style: "formal" 或 "casual"
    """
    lang_name = LANG_DISPLAY.get(lang, lang)
    prompt_tpl = (
        SYSTEM_PROMPT_FORMAL if style == "formal" else SYSTEM_PROMPT_CASUAL
    )
    system = prompt_tpl.format(lang=lang_name)
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(instruction=instruction),
            },
        ],
        max_tokens=128,
        temperature=1.0,
    )
    content = (resp.choices[0].message.content or "").strip()
    # 去掉可能的引号包裹
    if content.startswith('"') and content.endswith('"'):
        content = content[1:-1].strip()
    if content.startswith("'") and content.endswith("'"):
        content = content[1:-1].strip()
    # 模型输出 none 表示不需要搜索
    if content.lower() == "none":
        return ""
    return content


def process_one(
    client: OpenAI, record: dict, style: str
) -> dict | None:
    """处理单条记录，返回输出 dict 或 None（query 为空时跳过）。"""
    try:
        query = call_deepseek(
            client, record["instruction"], record["lang"], style
        )
    except Exception as e:
        log.warning("API 调用失败: %s", e)
        return None

    if not query:
        return None

    return {
        "id": instruction_hash(record["instruction"]),
        "query": query,
        "lang": record["lang"],
        "cate": record["cate"],
        "style": style,
    }


# ============================================================
# 主流程
# ============================================================
def run(limit: int | None = None) -> None:
    """主处理流程。"""
    setup_logging()
    load_dotenv(ENV_FILE)

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        log.error("DEEPSEEK_API_KEY 未找到，请检查 .env 文件")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

    # 加载输入和缓存
    records = load_input(INPUT_PATH, limit=limit)
    done = load_cache(OUTPUT_PATH)

    # 过滤已处理的（按 instruction hash 判断）
    todo = [r for r in records if instruction_hash(r["instruction"]) not in done]

    log.info("=" * 60)
    log.info("Instruction → Query 转换")
    log.info("=" * 60)
    log.info("输入: %s", INPUT_PATH)
    log.info("输出: %s", OUTPUT_PATH)
    log.info("总行数: %d, 已处理: %d, 待处理: %d", len(records), len(done), len(todo))
    log.info("线程数: %d, 批大小: %d", MAX_WORKERS, BATCH_SIZE)
    log.info("Formal:Casual 比例: %.0f%%:%.0f%%",
             FORMAL_RATIO * 100, (1 - FORMAL_RATIO) * 100)
    log.info("-" * 60)

    if not todo:
        log.info("没有待处理的数据，退出")
        return

    # 为每条数据预分配 style（确定性，基于索引）
    rng = random.Random(SEED)
    styles = [
        "formal" if rng.random() < FORMAL_RATIO else "casual"
        for _ in todo
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    success = 0
    empty = 0
    errors = 0

    # 分批处理，每批写一次文件
    for batch_start in range(0, len(todo), BATCH_SIZE):
        batch = todo[batch_start : batch_start + BATCH_SIZE]
        batch_styles = styles[batch_start : batch_start + BATCH_SIZE]
        results: list[dict] = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(process_one, client, r, s): r
                for r, s in zip(batch, batch_styles)
            }

            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"批次 {batch_start // BATCH_SIZE + 1}",
                unit="条",
            )
            for future in pbar:
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        success += 1
                    else:
                        empty += 1
                except Exception:
                    errors += 1

        # 追加写入
        if results:
            with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        elapsed = time.monotonic() - t_start
        total_done = success + empty + errors
        log.info(
            "进度: %d/%d | 成功: %d, 空query: %d, 错误: %d | 耗时: %.0fs",
            min(batch_start + BATCH_SIZE, len(todo)),
            len(todo),
            success,
            empty,
            errors,
            elapsed,
        )

    elapsed = time.monotonic() - t_start
    log.info("=" * 60)
    log.info(
        "完成! 成功: %d, 空query跳过: %d, 错误: %d, 总耗时: %.0fs",
        success, empty, errors, elapsed,
    )


if __name__ == "__main__":
    run(limit=LIMIT)
