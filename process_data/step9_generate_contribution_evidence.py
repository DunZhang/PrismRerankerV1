"""Generate contribution & evidence for query-document pairs via DeepSeek.

Filters rows where annotated_label=="yes", calls DeepSeek to extract
contribution and evidence, and appends the raw LLM output as a new field.

Usage:
uv run python -m process_data.generate_contribution_evidence \
    --input_path /mnt/g/.../merged.jsonl \
    --save_path /mnt/g/.../output.jsonl \
    --max_rows 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import jinja2
import langdetect
import pycountry
from tqdm import tqdm

from shared.env import DEFAULT_PROJECT_ENV_FILE, load_optional_dotenv

log = logging.getLogger("generate_contribution_evidence")

DEFAULT_INPUT_PATH = Path(
    "/mnt/g/PrismRerankerV1Data/"
    "kalm_web-search_query_document_pairs_annotated_merged.jsonl"
)
TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "shared"
    / "templates"
    / "relevance_extract.j2"
)

MAX_RETRIES = 2


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


def _detect_language(text: str) -> str:
    """Detect language from text, return language name (e.g. 'Chinese')."""
    try:
        code = langdetect.detect(text)
        lang = pycountry.languages.get(alpha_2=code[:2])
        return lang.name if lang else code
    except Exception:  # noqa: BLE001
        return "Chinese"


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


def _call_deepseek(client: Any, prompt: str) -> str | None:
    """Call DeepSeek via OpenAI SDK, return raw content or None."""
    for attempt in range(1 + MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
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


def process(
    input_path: Path,
    save_path: Path,
    batch_size: int = 32,
    max_workers: int = 32,
    max_rows: int | None = None,
    env_file: Path | None = None,
) -> None:
    load_optional_dotenv(env_file=env_file, default_env_file=DEFAULT_PROJECT_ENV_FILE)

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        log.error("DEEPSEEK_API_KEY not set")
        sys.exit(1)

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    template = _load_template()

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate contribution & evidence for query-document pairs."
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input JSONL file (read-only).",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        required=True,
        help="Output JSONL file (append-only).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Rows processed per batch before saving (default: 32).",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=32,
        help="Max parallel LLM requests per batch (default: 32).",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Only process the first N yes-labeled rows.",
    )
    parser.add_argument(
        "--env_file",
        type=Path,
        default=None,
        help="Optional .env override. Defaults to project root .env.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    _setup_logging(verbose=args.verbose)

    if not args.input_path.exists():
        log.error("input_path not found: %s", args.input_path)
        sys.exit(1)

    process(
        input_path=args.input_path,
        save_path=args.save_path,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        max_rows=args.max_rows,
        env_file=args.env_file,
    )


if __name__ == "__main__":
    main()
