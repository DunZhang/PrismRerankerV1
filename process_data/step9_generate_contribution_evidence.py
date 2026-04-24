"""Generate contribution & evidence for query-document pairs via DeepSeek.

Filters rows where annotated_label=="yes", calls DeepSeek to extract
contribution and evidence, and appends the raw LLM output as a new field.
After generation, verifies language consistency and retries mismatches.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import jinja2
import zhconv
from tqdm import tqdm

from shared.env import DEFAULT_PROJECT_ENV_FILE, load_optional_dotenv

log = logging.getLogger("generate_contribution_evidence")

INPUT_PATH = Path(
    "/mnt/g/PrismRerankerV1Data/"
    "step8_kalm_web-search_query_document_pairs_annotated_merged.jsonl"
)
SAVE_PATH = Path(
    "/mnt/g/PrismRerankerV1Data/"
    "step9_kalm_web-search_query_document_pairs_contribution_evidence.jsonl"
)
TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "shared"
    / "templates"
    / "relevance_extract.j2"
)

MODEL = "deepseek-v4-pro"

BATCH_SIZE = 128
MAX_WORKERS = 32
MAX_ROWS: int | None = None
ENV_FILE: Path | None = None
VERBOSE = False
MAX_RETRIES = 2
LANG_FIX_MAX_RETRIES = 3
LANG_CLASSIFY_MONO_THRESHOLD = 0.85
LANG_CLASSIFY_MULTI_THRESHOLD = 0.50
LANG_JUDGE_SNIPPET_CHARS = 2000
# A Chinese document is "simp+trad mixed" when BOTH variants contribute at
# least MIN absolute distinguishing characters AND the minority variant
# covers at least RATIO of all distinguishing characters. When this fires and
# the doc is otherwise monolingual, output is forced to Simplified Chinese.
ZH_MIX_MIN_MINORITY_CHARS = 5
ZH_MIX_MIN_MINORITY_RATIO = 0.10


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-5s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S", stream=sys.stderr)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s"


def _compute_pair_hash(query: str, document: str) -> str:
    """Hash key = sha256(query + document). No model name needed."""
    content = f"{query}\n{document}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Unicode script analysis
# ---------------------------------------------------------------------------


def _count_scripts(text: str) -> dict[str, int]:
    """Count characters by Unicode script category."""
    cjk = latin = hangul = kana = 0
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            cjk += 1
        elif 0x41 <= cp <= 0x5A or 0x61 <= cp <= 0x7A:
            latin += 1
        elif 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
            hangul += 1
        elif 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF:
            kana += 1
    return {"cjk": cjk, "latin": latin, "hangul": hangul, "kana": kana}


def _dominant_script(text: str) -> str:
    """Return the dominant Unicode script category of *text*.

    Returns one of: ``"cjk"``, ``"latin"``, ``"hangul"``, ``"kana"``,
    ``"unknown"``.
    """
    counts = _count_scripts(text)
    total = sum(counts.values())
    if total == 0:
        return "unknown"
    return max(counts, key=counts.get)  # type: ignore[arg-type]


def _chinese_variant_counts(text: str) -> tuple[int, int]:
    """Return ``(simp_chars, trad_chars)`` — counts of variant-distinguishing
    characters present in *text*.

    ``simp_chars`` is the number of characters that change when converting the
    text to Traditional (i.e. simplified-only characters currently in *text*);
    ``trad_chars`` is the symmetric count for the other direction.
    """
    if not text:
        return 0, 0
    simp_chars = sum(1 for a, b in zip(text, zhconv.convert(text, "zh-tw")) if a != b)
    trad_chars = sum(1 for a, b in zip(text, zhconv.convert(text, "zh-cn")) if a != b)
    return simp_chars, trad_chars


def _chinese_variant(text: str) -> str:
    """Return ``"simplified"`` | ``"traditional"`` | ``"unknown"`` for CJK text.

    The variant with more distinguishing characters wins. Ties (including the
    case where no distinguishing character is present) fall back to
    ``"unknown"``.
    """
    simp, trad = _chinese_variant_counts(text)
    if simp > trad:
        return "simplified"
    if trad > simp:
        return "traditional"
    return "unknown"


def _is_chinese_variant_mixed(text: str) -> bool:
    """Return True iff *text* contains substantial Simplified AND Traditional
    content (both variants pass minimum absolute + minority-ratio thresholds).
    """
    simp, trad = _chinese_variant_counts(text)
    minority = min(simp, trad)
    total = simp + trad
    if minority < ZH_MIX_MIN_MINORITY_CHARS or total == 0:
        return False
    return minority / total >= ZH_MIX_MIN_MINORITY_RATIO


def _chinese_with_variant(text: str) -> str:
    """Name Chinese text as ``"Simplified Chinese"`` / ``"Traditional Chinese"``.

    Falls back to plain ``"Chinese"`` when no distinguishing character is
    present in the text.
    """
    variant = _chinese_variant(text)
    if variant == "simplified":
        return "Simplified Chinese"
    if variant == "traditional":
        return "Traditional Chinese"
    return "Chinese"


def _detect_language(text: str) -> str:
    """Detect language from text using Unicode script analysis.

    For CJK-dominant text, distinguishes Chinese/Japanese/Korean by
    script-specific characters, and further splits Chinese into Simplified
    vs Traditional via :func:`_chinese_variant`. For Latin-dominant text,
    falls back to ``langdetect`` for finer distinction (English/French/...).
    """
    if not text:
        return "Simplified Chinese"

    counts = _count_scripts(text)
    total = sum(counts.values())
    if total == 0:
        return "Simplified Chinese"

    cjk, latin = counts["cjk"], counts["latin"]
    hangul, kana = counts["hangul"], counts["kana"]

    # CJK characters dominate → distinguish by Japanese/Korean markers
    if cjk + kana + hangul > latin:
        if kana > 0 and kana / total > 0.05:
            return "Japanese"
        if hangul > 0 and hangul / total > 0.05:
            return "Korean"
        return _chinese_with_variant(text)

    # Latin-dominant → use langdetect for finer distinction
    try:
        import langdetect
        import pycountry

        code = langdetect.detect(text)
        lang = pycountry.languages.get(alpha_2=code[:2])
        return lang.name if lang else code
    except Exception:  # noqa: BLE001
        return "English"


_TAG_RE = re.compile(r"</?(?:contribution|evidence)>", flags=re.IGNORECASE)

_SENT_SPLIT_RE = re.compile(r"[。！？!?\n]+|(?<=\.)\s+")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+")


def _strip_ce_tags(text: str) -> str:
    """Remove <contribution>, </contribution>, <evidence>, </evidence> tags."""
    return _TAG_RE.sub("", text)


def _split_sentences(text: str) -> list[str]:
    """Coarse sentence splitter by CJK/Latin terminators and newlines."""
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if len(p.strip()) >= 3]


def _sentence_bucket(sentence: str) -> str:
    """Classify a sentence as ``"east_asian"`` | ``"latin"`` | ``"unknown"``.

    Latin characters are counted per-word (maximal runs of ``[A-Za-z]``), so a
    single English term embedded in an otherwise CJK sentence contributes only
    one token and does not flip the classification. CJK / Hangul / Kana are
    merged into the east-Asian bucket (different languages sharing the block).
    """
    cjk = hangul = kana = 0
    for ch in sentence:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            cjk += 1
        elif 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
            hangul += 1
        elif 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF:
            kana += 1
    east_asian = cjk + hangul + kana
    latin_tokens = len(_LATIN_WORD_RE.findall(sentence))
    if east_asian == 0 and latin_tokens == 0:
        return "unknown"
    if east_asian >= latin_tokens:
        return "east_asian"
    return "latin"


def _sentence_bucket_ratios(text: str) -> tuple[dict[str, int], int]:
    """Count sentences per bucket (east_asian / latin).

    Returns ``(counts_by_bucket, total_sentences_counted)``. Sentences with no
    recognised characters are skipped.
    """
    counts: dict[str, int] = {}
    total = 0
    for sent in _split_sentences(text):
        bucket = _sentence_bucket(sent)
        if bucket == "unknown":
            continue
        counts[bucket] = counts.get(bucket, 0) + 1
        total += 1
    return counts, total


def _expected_script(lang: str) -> str:
    """Map an output-language name to its expected dominant Unicode script."""
    if "Chinese" in lang or lang == "Japanese":
        return "cjk"
    if lang == "Korean":
        return "hangul"
    return "latin"


def _check_row_mismatch(row: dict[str, Any]) -> bool | None:
    """Return True if the row's CE script violates the expected output lang.

    Expected script is derived from ``row['output_language']`` — which under
    the multilingual → query-language rule may be the query's language (not
    necessarily English). For Simplified / Traditional Chinese output, also
    enforces Chinese variant match. Returns ``None`` when CE is missing,
    script is ``unknown``, or ``output_language`` is absent.
    """
    ce = row.get("contribution_evidence")
    if not ce:
        return None
    output_lang = row.get("output_language", "")
    if not output_lang:
        return None
    ce_stripped = _strip_ce_tags(ce)
    ce_script = _dominant_script(ce_stripped)
    if ce_script == "unknown":
        return None
    expected = _expected_script(output_lang)
    if ce_script != expected:
        return True
    # Same script: for Simp/Trad Chinese output, also enforce variant match.
    if expected == "cjk":
        if "Simplified" in output_lang:
            expected_var = "simplified"
        elif "Traditional" in output_lang:
            expected_var = "traditional"
        else:
            return False  # e.g. Japanese — no variant check
        ce_var = _chinese_variant(ce_stripped)
        if ce_var != "unknown" and expected_var != ce_var:
            return True
    return False


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _load_template() -> jinja2.Template:
    template_text = TEMPLATE_PATH.read_text(encoding="utf-8")
    return jinja2.Template(template_text)


def _load_input_rows(input_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_done_hashes(save_path: Path) -> set[str]:
    """Scan output file, return hashes of all rows already written."""
    done: set[str] = set()
    if not save_path.exists() or save_path.stat().st_size == 0:
        return done
    size_mb = save_path.stat().st_size / (1024 * 1024)
    log.info("Scanning cache from %s (%.1f MB)...", save_path, size_mb)
    with open(save_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                h = _compute_pair_hash(row["query"], row["document"])
                done.add(h)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    log.info("Cache: %d completed entries", len(done))
    return done


def _append_rows(rows: list[dict[str, Any]], save_path: Path) -> None:
    with open(save_path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _write_all_rows(rows: list[dict[str, Any]], save_path: Path) -> None:
    """Atomically overwrite save_path with all rows via tmp file."""
    tmp_path = save_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(save_path)


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------


def _call_deepseek(client: Any, prompt: str, temperature: float = 0.0) -> str | None:
    """Call DeepSeek via OpenAI SDK, return raw content or None."""
    for attempt in range(1 + MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                extra_body={"thinking": {"type": "disabled"}},
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content.strip()
            if attempt < MAX_RETRIES:
                log.debug("DeepSeek returned empty, retrying...")
                continue
            log.warning("DeepSeek returned empty after retries")
            return None
        except Exception as exc:  # noqa: BLE001
            log.warning("DeepSeek call failed: %s: %s", type(exc).__name__, exc)
            if attempt < MAX_RETRIES:
                time.sleep(1)
                continue
            return None
    return None


# ---------------------------------------------------------------------------
# Language classification (rule + LLM fallback)
# ---------------------------------------------------------------------------


_LLM_LANG_JUDGE_PROMPT = """\
You will receive a document. Decide whether it is written primarily in ONE \
language (monolingual) or genuinely mixed across multiple languages \
(multilingual).

Rules:
- Isolated loanwords, technical terms, product names, URLs, or a few foreign \
phrases DO NOT make the document multilingual.
- A document is multilingual only when it contains SUBSTANTIAL content \
(roughly >= 20% of the text, including at least a few complete sentences) in \
each of two or more languages.
- `primary_language` is the main language when monolingual, or the most \
dominant language when multilingual. Return a common English name such as \
"Simplified Chinese", "Traditional Chinese", "English", "Japanese", \
"Korean", "French", "German", "Spanish", ... For Chinese text, ALWAYS \
distinguish Simplified vs Traditional; never return the bare word "Chinese".
- `languages` is the full list of substantial languages in the document, \
using the same naming convention as `primary_language`. For monolingual \
docs it MUST equal [primary_language]. For multilingual docs, include every \
language with substantial content and place `primary_language` first.

Output strictly ONE JSON object. No extra text, no markdown code fence.
Schema:
{{"multilingual": <true|false>, "primary_language": "<language name>", \
"languages": ["<language name>", ...]}}

Document:
<<<
{text}
>>>"""


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _llm_judge_multilingual(
    client: Any, text: str
) -> tuple[bool, str, list[str]] | None:
    """Ask LLM for ``(multilingual, primary_language, languages)``.

    ``languages`` is the deduped list of substantial languages in the doc
    with ``primary_language`` pinned first; for monolingual docs it equals
    ``[primary_language]``. Returns ``None`` on parse/LLM failure.
    """
    snippet = text[:LANG_JUDGE_SNIPPET_CHARS]
    prompt = _LLM_LANG_JUDGE_PROMPT.format(text=snippet)
    raw = _call_deepseek(client, prompt, temperature=0.0)
    if not raw:
        return None
    m = _JSON_OBJ_RE.search(raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    ml = bool(obj.get("multilingual", False))
    lang = str(obj.get("primary_language", "")).strip()
    if not lang:
        return None
    langs: list[str] = [lang]
    seen: set[str] = {lang}
    raw_langs = obj.get("languages")
    if isinstance(raw_langs, list):
        for item in raw_langs:
            s = str(item).strip()
            if s and s not in seen:
                langs.append(s)
                seen.add(s)
    return ml, lang, langs


def _languages_from_buckets(text: str, primary: str) -> list[str]:
    """Detect per-bucket language in a multilingual doc.

    Aggregates sentences per east_asian / latin bucket and runs
    :func:`_detect_language` on each aggregate. ``primary`` is pinned first;
    duplicates are removed.
    """
    east_asian_sents: list[str] = []
    latin_sents: list[str] = []
    for sent in _split_sentences(text):
        bucket = _sentence_bucket(sent)
        if bucket == "east_asian":
            east_asian_sents.append(sent)
        elif bucket == "latin":
            latin_sents.append(sent)
    langs: list[str] = [primary]
    seen: set[str] = {primary}
    for bucket_sents in (east_asian_sents, latin_sents):
        if not bucket_sents:
            continue
        detected = _detect_language(" ".join(bucket_sents))
        if detected and detected not in seen:
            langs.append(detected)
            seen.add(detected)
    return langs


def _classify_doc_language(text: str, client: Any) -> tuple[bool, str, list[str]]:
    """Classify ``text`` into ``(is_multilingual, primary_language, languages)``.

    Three-stage cascade driven by sentence-level buckets (east-Asian / Latin):
      A. Top bucket >= ``LANG_CLASSIFY_MONO_THRESHOLD`` → monolingual.
      B. Top bucket < ``LANG_CLASSIFY_MULTI_THRESHOLD`` → multilingual
         (``languages`` derived via :func:`_languages_from_buckets`).
      C. Ambiguous middle → LLM judge (returns its own ``languages`` array);
         on LLM failure fall back to monolingual.
    ``primary_language`` comes from :func:`_detect_language`, which does the
    finer-grained naming (Chinese/Japanese/Korean/English/...). For
    monolingual results ``languages == [primary_language]``.
    """
    if not text or not text.strip():
        return False, "Simplified Chinese", ["Simplified Chinese"]

    counts, total = _sentence_bucket_ratios(text)
    primary = _detect_language(text)
    if total < 3:
        return False, primary, [primary]

    dominant = max(counts, key=lambda k: counts[k])
    ratio = counts[dominant] / total

    if ratio >= LANG_CLASSIFY_MONO_THRESHOLD:
        return False, primary, [primary]

    if ratio < LANG_CLASSIFY_MULTI_THRESHOLD:
        return True, primary, _languages_from_buckets(text, primary)

    judged = _llm_judge_multilingual(client, text)
    if judged is not None:
        return judged
    return False, primary, [primary]


# ---------------------------------------------------------------------------
# Phase 1: Generate contribution & evidence
# ---------------------------------------------------------------------------


def _classify_and_generate(
    client: Any,
    template: jinja2.Template,
    row: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    """Classify doc language, render prompt, call LLM. Returns ``(out_row, ce)``.

    ``out_row`` always carries ``detected_language``, ``is_multilingual``,
    ``doc_languages`` and ``output_language`` so that Phase 2 can verify /
    fix without re-detecting.

    Language decision:
    - Multilingual doc: detect the query's language; if it matches one of
      ``doc_languages`` output in that language, otherwise fall back to
      English.
    - Monolingually-CJK doc internally mixing Simplified and Traditional
      characters: forced to Simplified Chinese.
    - Otherwise: use the doc's detected primary language.
    """
    is_multilingual, primary, doc_languages = _classify_doc_language(
        row["document"], client
    )
    if is_multilingual:
        query_lang = _detect_language(row["query"])
        output_lang = query_lang if query_lang in doc_languages else "English"
    elif primary in ("Simplified Chinese", "Traditional Chinese") and (
        _is_chinese_variant_mixed(row["document"])
    ):
        output_lang = "Simplified Chinese"
    else:
        output_lang = primary
    prompt = template.render(
        query=row["query"], document=row["document"], lang=output_lang
    )
    output = _call_deepseek(client, prompt, temperature=0.4)

    out_row = dict(row)
    out_row["contribution_evidence"] = output
    out_row["detected_language"] = primary
    out_row["is_multilingual"] = is_multilingual
    out_row["doc_languages"] = doc_languages
    out_row["output_language"] = output_lang
    return out_row, output


def process(client: Any, template: jinja2.Template) -> None:
    input_path = INPUT_PATH
    save_path = SAVE_PATH
    batch_size = BATCH_SIZE
    max_workers = MAX_WORKERS
    max_rows = MAX_ROWS

    # Load input rows
    log.info("Loading input rows from %s ...", input_path)
    all_rows = _load_input_rows(input_path)
    log.info("Loaded %d total rows", len(all_rows))

    # Apply max_rows limit (over all rows)
    if max_rows is not None:
        all_rows = all_rows[:max_rows]

    # Load cache
    done_hashes = _load_done_hashes(save_path)

    # Split into: non-yes passthrough, yes pending, yes already done
    passthrough: list[dict[str, Any]] = []
    pending: list[tuple[int, dict[str, Any]]] = []
    yes_total = 0
    for i, row in enumerate(all_rows):
        h = _compute_pair_hash(row["query"], row["document"])
        if h in done_hashes:
            continue  # already in output file
        if row.get("annotated_label") != "yes":
            passthrough.append(row)
        else:
            yes_total += 1
            pending.append((i, row))

    already_done = len(all_rows) - len(passthrough) - len(pending)

    log.info("=" * 60)
    log.info("Generate Contribution & Evidence")
    log.info("=" * 60)
    log.info("Input:            %s", input_path)
    log.info("Output:           %s", save_path)
    log.info("Model:            %s", MODEL)
    log.info("Batch size:       %d", batch_size)
    log.info("Workers:          %d", max_workers)
    log.info("Total rows:       %d", len(all_rows))
    log.info("Passthrough:      %d (non-yes, to write as-is)", len(passthrough))
    log.info("Yes pending:      %d (need LLM)", len(pending))
    log.info("Already done:     %d", already_done)
    log.info("-" * 60)

    if not passthrough and not pending:
        log.info("Nothing to do; all rows already in output.")
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Write non-yes rows as-is first
    if passthrough:
        _append_rows(passthrough, save_path)
        log.info("Wrote %d non-yes rows as-is", len(passthrough))

    if not pending:
        log.info("No yes rows to process.")
        return

    written = 0
    failed = 0
    t_start = time.monotonic()
    pbar = tqdm(total=len(pending), desc="Extract", unit="row", dynamic_ncols=True)

    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start : batch_start + batch_size]
        batch_rows: dict[int, dict[str, Any]] = {idx: row for idx, row in batch}

        results: dict[int, tuple[dict[str, Any], str | None]] = {}
        worker_count = min(max_workers, len(batch))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            future_to_idx = {
                pool.submit(_classify_and_generate, client, template, row): idx
                for idx, row in batch
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:  # noqa: BLE001
                    log.warning("Unexpected error for row %d: %s", idx, exc)
                    fallback_row = dict(batch_rows[idx])
                    fallback_row["contribution_evidence"] = None
                    results[idx] = (fallback_row, None)

        batch_output: list[dict[str, Any]] = []
        for idx, _row in batch:
            out_row, output = results[idx]
            batch_output.append(out_row)
            if output is not None:
                written += 1
            else:
                failed += 1

        _append_rows(batch_output, save_path)
        pbar.update(len(batch))
        pbar.set_postfix_str(f"ok={written} fail={failed}")

    pbar.close()

    elapsed = time.monotonic() - t_start
    log.info("-" * 60)
    log.info("FINISHED in %s", _fmt_duration(elapsed))
    log.info("  written=%d, failed=%d", written, failed)
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Phase 2: Language verification & retry (per-row, up to 3 attempts each)
# ---------------------------------------------------------------------------


def _fix_one_row(
    row: dict[str, Any],
    client: Any,
    template: jinja2.Template,
) -> tuple[dict[str, Any], bool]:
    """Try up to LANG_FIX_MAX_RETRIES times to fix a mismatched row.

    Each retry uses increasing temperature (0.3 / 0.6 / 0.9) to avoid
    producing the same wrong output repeatedly. The target output language is
    taken from the row's cached ``output_language`` (Phase 1 decision).

    Returns (final_row, fixed). If fixed=False the row keeps its original CE.
    """
    row_with_meta = dict(row)
    prompt = template.render(
        query=row["query"], document=row["document"], lang=row["output_language"]
    )
    for attempt in range(1, LANG_FIX_MAX_RETRIES + 1):
        temp = 0.3 * attempt
        output = _call_deepseek(client, prompt, temperature=temp)
        if output is None:
            log.debug(
                "  LLM returned None on attempt %d (temp=%.1f) for query='%s'",
                attempt,
                temp,
                row.get("query", "")[:50],
            )
            continue
        tentative = dict(row_with_meta)
        tentative["contribution_evidence"] = output
        mismatch = _check_row_mismatch(tentative)
        if mismatch is False or mismatch is None:
            return tentative, True
        log.debug(
            "  Attempt %d (temp=%.1f) still mismatched for query='%s'"
            " (lang=%s, doc=%s, ce=%s)",
            attempt,
            temp,
            row.get("query", "")[:50],
            row["output_language"],
            _dominant_script(row["document"]),
            _dominant_script(_strip_ce_tags(output)),
        )
    return row_with_meta, False


def verify_and_fix_languages(
    save_path: Path,
    client: Any,
    template: jinja2.Template,
    max_workers: int = MAX_WORKERS,
) -> None:
    """Scan save_path, find language-mismatched rows, fix each (up to 3 tries)."""
    log.info("=" * 60)
    log.info("Post-process: Language Verification & Fix")
    log.info("=" * 60)
    log.info("File:             %s", save_path)
    log.info("Max retries/row:  %d", LANG_FIX_MAX_RETRIES)
    log.info("Workers:          %d", max_workers)
    log.info("-" * 60)

    # --- Step 1: scan and find mismatched rows ---
    log.info("Scanning for language mismatches ...")
    all_rows: list[dict[str, Any]] = []
    mismatched_indices: list[int] = []
    total_checked = 0
    skipped_no_ce = 0
    skipped_unknown = 0
    mismatch_detail: dict[str, int] = {}

    with open(save_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            all_rows.append(row)
            ce = row.get("contribution_evidence")
            if not ce:
                skipped_no_ce += 1
                continue
            result = _check_row_mismatch(row)
            if result is None:
                skipped_unknown += 1
                continue
            total_checked += 1
            if result:
                mismatched_indices.append(len(all_rows) - 1)
                ce_stripped = _strip_ce_tags(ce)
                ce_script = _dominant_script(ce_stripped)
                out_lang = row.get("output_language", "?")
                if row.get("is_multilingual") is True:
                    if _expected_script(out_lang) == "cjk" and ce_script == "cjk":
                        if "Simplified" in out_lang:
                            expected_var = "simp"
                        elif "Traditional" in out_lang:
                            expected_var = "trad"
                        else:
                            expected_var = "?"
                        ce_var = _chinese_variant(ce_stripped)[:4]
                        key = f"multi({expected_var})->cjk({ce_var})"
                    else:
                        key = f"multi({out_lang})->{ce_script}"
                else:
                    doc = row.get("document", "")
                    doc_script = _dominant_script(doc)
                    if doc_script == "cjk" and ce_script == "cjk":
                        if "Simplified" in out_lang:
                            expected_var = "simp"
                        elif "Traditional" in out_lang:
                            expected_var = "trad"
                        else:
                            expected_var = "?"
                        ce_var = _chinese_variant(ce_stripped)[:4]
                        key = f"cjk({expected_var})->cjk({ce_var})"
                    else:
                        key = f"{doc_script}->{ce_script}"
                mismatch_detail[key] = mismatch_detail.get(key, 0) + 1

    log.info("  Total rows:        %d", len(all_rows))
    log.info("  Checked:           %d", total_checked)
    log.info("  Skipped (no CE):   %d", skipped_no_ce)
    log.info("  Skipped (unknown): %d", skipped_unknown)
    log.info("  Mismatched:        %d", len(mismatched_indices))
    if mismatch_detail:
        for pair, cnt in sorted(
            mismatch_detail.items(), key=lambda x: x[1], reverse=True
        ):
            log.info("    %-20s %d", pair, cnt)

    if not mismatched_indices:
        log.info("No language mismatch found. All clean!")
        return

    # Log sample mismatched rows
    sample_n = min(5, len(mismatched_indices))
    for i in range(sample_n):
        idx = mismatched_indices[i]
        row = all_rows[idx]
        out_lang = row.get("output_language", "?")
        expected = f"{_expected_script(out_lang)}({out_lang})"
        log.info(
            "  Example #%d: row=%d, expected=%s, ce=%s, query='%s'",
            i + 1,
            idx + 1,
            expected,
            _dominant_script(_strip_ce_tags(row.get("contribution_evidence", ""))),
            row.get("query", "")[:60],
        )

    # --- Step 2: fix each mismatched row (concurrent, each up to 3 tries) ---
    log.info("")
    log.info("Regenerating %d mismatched rows ...", len(mismatched_indices))
    fixed_count = 0
    failed_count = 0
    t_start = time.monotonic()
    pbar = tqdm(
        total=len(mismatched_indices),
        desc="LangFix",
        unit="row",
        dynamic_ncols=True,
    )

    worker_count = min(max_workers, len(mismatched_indices))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        future_to_idx = {
            pool.submit(_fix_one_row, all_rows[idx], client, template): idx
            for idx in mismatched_indices
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result_row, fixed = future.result()
            except Exception as exc:  # noqa: BLE001
                log.warning("Unexpected error for row %d: %s", idx + 1, exc)
                fixed = False
                result_row = all_rows[idx]

            if fixed:
                all_rows[idx] = result_row
                fixed_count += 1
            else:
                failed_count += 1

            pbar.update(1)
            pbar.set_postfix_str(f"fixed={fixed_count} failed={failed_count}")

    pbar.close()
    elapsed = time.monotonic() - t_start

    log.info("-" * 60)
    log.info("Language fix summary:")
    log.info("  Total mismatched:  %d", len(mismatched_indices))
    log.info("  Fixed:             %d", fixed_count)
    log.info("  Failed (3 tries):  %d", failed_count)
    log.info("  Time:              %s", _fmt_duration(elapsed))

    # --- Step 3: write back ---
    if fixed_count > 0:
        _write_all_rows(all_rows, save_path)
        log.info("  Save file updated.")

    # --- Step 4: record failures ---
    if failed_count > 0:
        fail_path = save_path.parent / "lang_mismatch_failures.jsonl"
        with open(fail_path, "w", encoding="utf-8") as f:
            for idx in mismatched_indices:
                row = all_rows[idx]
                if _check_row_mismatch(row):
                    record = {"row_index": idx + 1, **row}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.warning(
            "  %d rows still mismatched after %d retries, saved to %s",
            failed_count,
            LANG_FIX_MAX_RETRIES,
            fail_path,
        )
    else:
        log.info("All language mismatches resolved!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    _setup_logging(verbose=VERBOSE)

    if not INPUT_PATH.exists():
        log.error("input_path not found: %s", INPUT_PATH)
        sys.exit(1)

    load_optional_dotenv(env_file=ENV_FILE, default_env_file=DEFAULT_PROJECT_ENV_FILE)

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        log.error("DEEPSEEK_API_KEY not set")
        sys.exit(1)

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    template = _load_template()

    # Phase 1: generate
    process(client, template)

    # Phase 2: verify & fix language mismatches
    verify_and_fix_languages(SAVE_PATH, client, template)


if __name__ == "__main__":
    main()
