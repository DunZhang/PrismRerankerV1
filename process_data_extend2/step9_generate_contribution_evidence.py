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
from tqdm import tqdm

from shared.env import DEFAULT_PROJECT_ENV_FILE, load_optional_dotenv

log = logging.getLogger("generate_contribution_evidence")

INPUT_PATH = Path(
    "/mnt/g/PrismRerankerV1Data/data_extend2/"
    "step8_expanded2_web-search_query_document_pairs_annotated_merged.jsonl"
)
SAVE_PATH = Path(
    "/mnt/g/PrismRerankerV1Data/data_extend2/"
    "step9_expanded2_web-search_query_document_contribution_evidence.jsonl"
)
TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "shared"
    / "templates"
    / "relevance_extract.j2"
)

BATCH_SIZE = 266
MAX_WORKERS = 128
MAX_ROWS: int | None = None
ENV_FILE: Path | None = None
VERBOSE = False
MAX_RETRIES = 2
LANG_FIX_MAX_RETRIES = 3


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


def _detect_language(text: str) -> str:
    """Detect language from text using Unicode script analysis.

    For CJK-dominant text, distinguishes Chinese/Japanese/Korean by
    script-specific characters. For Latin-dominant text, falls back
    to ``langdetect`` for finer distinction (English/French/German etc.).
    """
    if not text:
        return "Chinese"

    counts = _count_scripts(text)
    total = sum(counts.values())
    if total == 0:
        return "Chinese"

    cjk, latin = counts["cjk"], counts["latin"]
    hangul, kana = counts["hangul"], counts["kana"]

    # CJK characters dominate → distinguish by Japanese/Korean markers
    if cjk + kana + hangul > latin:
        if kana > 0 and kana / total > 0.05:
            return "Japanese"
        if hangul > 0 and hangul / total > 0.05:
            return "Korean"
        return "Chinese"

    # Latin-dominant → use langdetect for finer distinction
    try:
        import langdetect
        import pycountry

        code = langdetect.detect(text)
        lang = pycountry.languages.get(alpha_2=code[:2])
        return lang.name if lang else code
    except Exception:  # noqa: BLE001
        return "English"


_TAG_RE = re.compile(
    r"</?(?:contribution|evidence)>", flags=re.IGNORECASE
)


def _strip_ce_tags(text: str) -> str:
    """Remove <contribution>, </contribution>, <evidence>, </evidence> tags."""
    return _TAG_RE.sub("", text)


def _is_lang_matched(doc_text: str, ce_text: str) -> bool:
    """Check if document and contribution_evidence share the same script."""
    doc_script = _dominant_script(doc_text)
    ce_script = _dominant_script(_strip_ce_tags(ce_text))
    if doc_script == "unknown" or ce_script == "unknown":
        return True  # can't judge, treat as ok
    return doc_script == ce_script


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


def _call_deepseek(
    client: Any, prompt: str, temperature: float = 0.0
) -> str | None:
    """Call DeepSeek via OpenAI SDK, return raw content or None."""
    for attempt in range(1 + MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
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
# Phase 1: Generate contribution & evidence
# ---------------------------------------------------------------------------


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
    log.info("Model:            deepseek-chat")
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

        # Render prompts
        prompts: dict[int, str] = {}
        for idx, row in batch:
            lang = _detect_language(row["document"])
            prompts[idx] = template.render(
                query=row["query"], document=row["document"], lang=lang
            )

        # Concurrent LLM calls
        results: dict[int, str | None] = {}
        worker_count = min(max_workers, len(batch))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            future_to_idx = {
                pool.submit(_call_deepseek, client, prompts[idx]): idx
                for idx, _row in batch
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:  # noqa: BLE001
                    log.warning("Unexpected error for row %d: %s", idx, exc)
                    results[idx] = None

        # Build output rows
        batch_output: list[dict[str, Any]] = []
        for idx, row in batch:
            output = results.get(idx)
            out_row = dict(row)
            out_row["contribution_evidence"] = output
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
    producing the same wrong output repeatedly.

    Returns (final_row, fixed). If fixed=False the row keeps its original CE.
    """
    lang = _detect_language(row["document"])
    prompt = template.render(
        query=row["query"], document=row["document"], lang=lang
    )
    for attempt in range(1, LANG_FIX_MAX_RETRIES + 1):
        temp = 0.3 * attempt
        output = _call_deepseek(client, prompt, temperature=temp)
        if output is None:
            log.debug(
                "  LLM returned None on attempt %d (temp=%.1f)"
                " for query='%s'",
                attempt,
                temp,
                row.get("query", "")[:50],
            )
            continue
        if _is_lang_matched(row["document"], output):
            new_row = dict(row)
            new_row["contribution_evidence"] = output
            return new_row, True
        log.debug(
            "  Attempt %d (temp=%.1f) still mismatched for query='%s'"
            " (doc=%s, ce=%s)",
            attempt,
            temp,
            row.get("query", "")[:50],
            _dominant_script(row["document"]),
            _dominant_script(_strip_ce_tags(output)),
        )
    return row, False


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
            doc_script = _dominant_script(row.get("document", ""))
            ce_script = _dominant_script(_strip_ce_tags(ce))
            if doc_script == "unknown" or ce_script == "unknown":
                skipped_unknown += 1
                continue
            total_checked += 1
            if doc_script != ce_script:
                mismatched_indices.append(len(all_rows) - 1)
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
        log.info(
            "  Example #%d: row=%d, doc=%s, ce=%s, query='%s'",
            i + 1,
            idx + 1,
            _dominant_script(row.get("document", "")),
            _dominant_script(
                _strip_ce_tags(row.get("contribution_evidence", ""))
            ),
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
                ce = row.get("contribution_evidence", "")
                if ce and not _is_lang_matched(row.get("document", ""), ce):
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
