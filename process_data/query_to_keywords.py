"""Batch-convert natural language queries to keywords using DeepSeek Chat API.

Usage:
uv run python -m process_data.query_to_keywords \
    --read_path /mnt/g/PrismRerankerV1Data/KaLM__all_retrieval_voyage-rerank2_voyage-rerank2.5_all_filtered.jsonl \
    --save_path /mnt/g/PrismRerankerV1Data/KaLM__all_retrieval_voyage-rerank2_voyage-rerank2.5_all_filtered_keywords.jsonl


Each record is processed exactly once.  Records selected (with probability
``--probability``) get a ``"keywords"`` field added; unselected records are
written as-is.  Supports resumption — already-written rows (identified by
sha256 of query+pos_list+neg_list) are skipped on restart.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from shared.env import DEFAULT_PROJECT_ENV_FILE, load_optional_dotenv

log = logging.getLogger("query_to_keywords")

_KEYWORDS_RE = re.compile(r"<keywords>(.*?)</keywords>", re.DOTALL)

# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------
_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "shared"
    / "templates"
    / "natural_query_to_keywords.j2"
)


def _load_template() -> str:
    """Load the Jinja2 template as raw string (we only have one variable)."""
    return _TEMPLATE_PATH.read_text(encoding="utf-8")


def _render_prompt(template: str, question: str) -> str:
    """Render the prompt template with the given question."""
    return template.replace("{{ question }}", question)


# ---------------------------------------------------------------------------
# Hash / checkpoint
# ---------------------------------------------------------------------------
def _compute_row_hash(row: dict) -> str:
    """Deterministic hash from query + pos_list + neg_list."""
    content = json.dumps(
        {
            "query": row["query"],
            "pos_list": row["pos_list"],
            "neg_list": row["neg_list"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _load_done_hashes(save_path: Path) -> set[str]:
    """Rebuild set of processed row hashes from the output file."""
    done: set[str] = set()
    if not save_path.exists() or save_path.stat().st_size == 0:
        return done

    size_mb = save_path.stat().st_size / (1024 * 1024)
    log.info("Rebuilding checkpoint from output (%.1f MB)...", size_mb)
    with open(save_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done.add(_compute_row_hash(row))
            except (json.JSONDecodeError, KeyError):
                continue
    log.info("Checkpoint rebuilt: %d rows already processed", len(done))
    return done


def _count_lines(path: Path) -> int:
    """Count total lines in a file using raw byte reading."""
    count = 0
    buf_size = 1024 * 1024
    with open(path, "rb") as f:
        while buf := f.read(buf_size):
            count += buf.count(b"\n")
    return count


# ---------------------------------------------------------------------------
# DeepSeek API
# ---------------------------------------------------------------------------
def _extract_keywords(text: str) -> str:
    """Extract keywords from <keywords>...</keywords> tags."""
    m = _KEYWORDS_RE.search(text)
    if m:
        return m.group(1).strip()
    # Fallback: return the whole response stripped
    return text.strip()


def _call_deepseek(client: OpenAI, prompt: str) -> str:
    """Call DeepSeek chat API and return extracted keywords."""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=256,
    )
    content = resp.choices[0].message.content or ""
    return _extract_keywords(content)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-5s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S", stream=sys.stderr)
    # Silence noisy HTTP request logs from openai/httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s"


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------
def process(
    read_path: Path,
    save_path: Path,
    probability: float = 0.3,
    batch_size: int = 128,
    max_workers: int = 32,
    seed: int = 42,
    env_file: Path | None = None,
) -> None:
    """Convert queries to keywords in batches with checkpoint support."""
    load_optional_dotenv(env_file=env_file, default_env_file=DEFAULT_PROJECT_ENV_FILE)

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        log.error("DEEPSEEK_API_KEY not found in environment")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    template = _load_template()
    rng = random.Random(seed)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    done_hashes = _load_done_hashes(save_path)

    total = _count_lines(read_path)
    log.info("=" * 60)
    log.info("Query to Keywords Conversion")
    log.info("=" * 60)
    log.info("Input:       %s", read_path)
    log.info("Output:      %s", save_path)
    log.info("Probability: %.2f", probability)
    log.info("Batch size:  %d", batch_size)
    log.info("Workers:     %d", max_workers)
    log.info("Seed:        %d", seed)
    log.info("Total rows:  %d, checkpoint: %d done", total, len(done_hashes))
    log.info("-" * 60)

    processed = 0
    skipped = 0
    converted = 0
    errors = 0
    t_start = time.monotonic()

    with (
        open(read_path, encoding="utf-8") as fin,
        open(save_path, "a", encoding="utf-8") as fout,
    ):
        pbar = tqdm(
            fin,
            total=total,
            desc="Processing",
            unit="row",
            dynamic_ncols=True,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}] "
                "{postfix}"
            ),
        )
        batch: list[dict] = []

        for line in pbar:
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            row_hash = _compute_row_hash(row)
            if row_hash in done_hashes:
                skipped += 1
                pbar.set_postfix_str(
                    f"ok={processed} conv={converted} skip={skipped} err={errors}"
                )
                continue

            batch.append(row)

            if len(batch) >= batch_size:
                n_conv, n_err = _process_batch(
                    batch, client, template, rng, probability, max_workers, fout
                )
                processed += len(batch)
                converted += n_conv
                errors += n_err
                batch = []
                pbar.set_postfix_str(
                    f"ok={processed} conv={converted} skip={skipped} err={errors}"
                )

        # Process remaining rows
        if batch:
            n_conv, n_err = _process_batch(
                batch, client, template, rng, probability, max_workers, fout
            )
            processed += len(batch)
            converted += n_conv
            errors += n_err
            pbar.set_postfix_str(
                f"ok={processed} conv={converted} skip={skipped} err={errors}"
            )

        pbar.close()

    elapsed = time.monotonic() - t_start
    speed = processed / elapsed if elapsed > 0 else 0
    log.info("-" * 60)
    log.info("FINISHED in %s", _fmt_duration(elapsed))
    log.info(
        "  processed=%d, converted=%d, skipped=%d, errors=%d",
        processed,
        converted,
        skipped,
        errors,
    )
    log.info("  avg speed: %.2f rows/s", speed)
    log.info("=" * 60)


def _process_batch(
    batch: list[dict],
    client: OpenAI,
    template: str,
    rng: random.Random,
    probability: float,
    max_workers: int,
    fout,  # noqa: ANN001
) -> tuple[int, int]:
    """Process a batch: select queries, call API in parallel, write all rows.

    Returns:
        (converted_count, error_count)
    """
    # Decide which rows to convert
    selected_indices: list[int] = [
        i for i in range(len(batch)) if rng.random() < probability
    ]

    converted = 0
    errors = 0

    if selected_indices:
        # Build prompts for selected rows
        prompts: dict[int, str] = {
            i: _render_prompt(template, batch[i]["query"]) for i in selected_indices
        }

        # Call API in parallel
        results: dict[int, str | None] = {}
        with ThreadPoolExecutor(max_workers=min(max_workers, len(prompts))) as pool:
            future_to_idx = {
                pool.submit(_call_deepseek, client, prompt): idx
                for idx, prompt in prompts.items()
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    log.warning(
                        "API error (query=%r...): %s: %s",
                        batch[idx]["query"][:50],
                        type(e).__name__,
                        e,
                    )
                    results[idx] = None
                    errors += 1

        # Attach keywords to selected rows
        for idx, keywords in results.items():
            if keywords is not None:
                batch[idx]["keywords"] = keywords
                converted += 1

    # Write all rows in batch (selected or not)
    for row in batch:
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    fout.flush()

    return converted, errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert natural queries to keywords using DeepSeek Chat API."
    )
    parser.add_argument(
        "--read_path",
        type=Path,
        required=True,
        help="Input JSONL file.",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        required=True,
        help="Output JSONL file (appended incrementally).",
    )
    parser.add_argument(
        "--probability",
        type=float,
        default=0.3,
        help="Probability of converting a query to keywords (default: 0.3).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Number of rows per batch (default: 128).",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=32,
        help="Max threads for API calls (default: 32).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible selection (default: 42).",
    )
    parser.add_argument(
        "--env_file",
        type=Path,
        default=None,
        help="Path to .env file (default: .env in project root).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    _setup_logging(verbose=args.verbose)

    if not args.read_path.exists():
        log.error("read_path not found: %s", args.read_path)
        sys.exit(1)

    process(
        read_path=args.read_path,
        save_path=args.save_path,
        probability=args.probability,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        seed=args.seed,
        env_file=args.env_file,
    )


if __name__ == "__main__":
    main()
