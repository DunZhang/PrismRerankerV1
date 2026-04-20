"""Entity-fidelity check for generated evidence.

For each yes/yes sample, we want to make sure the key domain entities in
``evidence`` (proper nouns, technical terms, numbers, dates, times, codes,
URLs) appear **verbatim** in the source ``document``. This catches the bulk
of hallucinations that an LLM judge might otherwise tolerate.

The entity list comes from two sources:

1. A DeepSeek-chat call that extracts proper nouns, terminology, model
   codes, etc. from the evidence text.
2. A regex pass that captures numbers, percentages, dates, times.

Both sources are merged, deduped, and each entity is tested for literal
substring presence in the document. The score is ``present / total``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

log = logging.getLogger("evaluate_relevance_contribution_evidence.entity_fidelity")

MAX_RETRIES = 1
MAX_COMPLETION_TOKENS = 1024
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

_REGEX_NUMERIC = re.compile(r"\d+(?:[.,]\d+)*%?")
_REGEX_DATE_TIME = re.compile(
    r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?"
    r"|\d{1,2}[-/月]\d{1,2}日?"
    r"|\d{1,2}[:：]\d{2}(?:[:：]\d{2})?"
)

_EXTRACT_PROMPT = """从下面的文本中抽取需要与原文**逐字一致**的关键实体，包括：
- 专名：人名、机构名、地名、产品名、品牌名、项目/论文/书名
- 专业术语、缩写、代号、型号
- 代码片段、公式、URL、文件路径
- 日期、时间
- 带量词的数字短语（如"1个"、"3款"、"5次"、"两种方法"）；仅抽取整个短语，不要单独抽数字

排除：普通名词、形容词、动词、通用描述词；不带量词的裸数字（裸数字由正则单独处理）。

严格要求：抽取出的每一项必须是文本中**逐字出现**的子串，严禁改写、归纳或补全。

只返回 JSON 数组，每项为一个字符串，不要解释、不要额外文字。若无任何关键实体，返回 []。

文本：
{text}
"""


def build_entity_extractor_client(api_key: str) -> Any:
    """Build an OpenAI-compatible client pointing at DeepSeek."""
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def extract_numeric_temporal(text: str) -> list[str]:
    """Regex-based pass for numeric/date/time tokens.

    Drops bare single-digit integers (0-9) — too noisy and rarely load-bearing.
    Multi-digit numbers, decimals, percentages, and date/time tokens are kept.
    """
    matches: list[str] = []
    matches.extend(_REGEX_NUMERIC.findall(text))
    matches.extend(_REGEX_DATE_TIME.findall(text))
    seen: set[str] = set()
    out: list[str] = []
    for m in matches:
        m = m.strip()
        if not m or m in seen:
            continue
        if len(m) == 1 and m.isdigit():
            continue
        seen.add(m)
        out.append(m)
    return out


def _parse_json_list(raw: str) -> list[str] | None:
    """Extract a JSON array from model output, tolerating code-fence wrapping."""
    if not raw:
        return None
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    out: list[str] = []
    for item in data:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
    return out


def extract_key_terms_via_llm(
    evidence: str,
    client: Any,
    model: str,
) -> list[str]:
    """Ask deepseek-chat to pull proper nouns / terms from evidence.

    Any LLM-returned term that isn't a literal substring of ``evidence`` is
    dropped — models sometimes paraphrase or hallucinate, and such terms can
    never be present in the document either, which would unfairly tank the
    fidelity score.
    """
    if not evidence or not evidence.strip():
        return []
    prompt = _EXTRACT_PROMPT.format(text=evidence)
    last_exc: Exception | None = None
    for attempt in range(1 + MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_COMPLETION_TOKENS,
                temperature=0.0,
            )
            content = response.choices[0].message.content or ""
            terms = _parse_json_list(content)
            if terms is not None:
                return [t for t in terms if t in evidence]
            log.debug(
                "entity extractor returned unparseable output (attempt %d): %s",
                attempt + 1,
                content[:200],
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log.warning(
                "entity extractor call failed (attempt %d): %s: %s",
                attempt + 1,
                type(exc).__name__,
                exc,
            )
    if last_exc is not None:
        log.warning("entity extractor gave up after %d attempts", MAX_RETRIES + 1)
    return []


def compute_entity_fidelity(
    evidence: str,
    document: str,
    client: Any,
    model: str,
) -> dict[str, Any]:
    """Return fidelity dict with ``score``, ``extracted``, ``missing``.

    ``score`` = present / total (1.0 if no entities extracted at all).
    ``extracted`` = deduped list of all entity strings considered.
    ``missing`` = subset of ``extracted`` whose literal string is absent
    from ``document``.
    """
    llm_terms = extract_key_terms_via_llm(evidence, client, model)
    regex_terms = extract_numeric_temporal(evidence)

    seen: set[str] = set()
    extracted: list[str] = []
    for term in list(llm_terms) + list(regex_terms):
        t = term.strip()
        if not t or t in seen:
            continue
        seen.add(t)
        extracted.append(t)

    if not extracted:
        return {"score": 1.0, "extracted": [], "missing": []}

    missing = [t for t in extracted if t not in document]
    present_count = len(extracted) - len(missing)
    score = present_count / len(extracted)
    return {"score": score, "extracted": extracted, "missing": missing}
