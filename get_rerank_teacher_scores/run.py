"""CLI entry point for batch teacher-score generation.

Usage:
  uv run python -m get_rerank_teacher_scores \\
      --read_path /path/to/input.jsonl \\
      --save_path /path/to/output.jsonl \\
      --model_name voyage-rerank-2

 uv run python -m get_rerank_teacher_scores  --read_path /mnt/g/PrismRerankerV1Data/KaLM__all_retrieval_voyage-rerank2.jsonl --save_path /mnt/g/PrismRerankerV1Data/KaLM__all_retrieval_voyage-rerank2_voyage-rerank2.5.jsonl  --model_name voyage-rerank-2.5
Supported model names:
  voyage-rerank-2          Voyage AI rerank-2
  voyage-rerank-2-lite     Voyage AI rerank-2-lite
  voyage-rerank-2.5        Voyage AI rerank-2.5
  voyage-rerank-2.5-lite   Voyage AI rerank-2.5-lite
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

from tqdm import tqdm

from shared.env import DEFAULT_PROJECT_ENV_FILE, load_optional_dotenv

from .scorer import SUPPORTED_MODELS, create_voyage_scorer

log = logging.getLogger("teacher_scores")


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging with timestamps."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-5s | %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, stream=sys.stderr)


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


def _load_checkpoint(ckpt_path: Path, save_path: Path) -> set[str]:
    """Rebuild checkpoint from output file, then merge with ckpt file.

    The output file is the source of truth — any row present there is
    truly done.  The ckpt file is an optimistic log that may contain
    hashes for rows whose output was never flushed (interrupted between
    ckpt write and output write).  By rebuilding from the output first,
    we guarantee no data is silently skipped on restart.
    """
    done: set[str] = set()

    # Rebuild from output file (source of truth)
    if save_path.exists() and save_path.stat().st_size > 0:
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
        log.info("Checkpoint rebuilt: %d rows already scored", len(done))

    # Rewrite ckpt file to match actual output
    with open(ckpt_path, "w", encoding="utf-8") as f:
        for h in sorted(done):
            f.write(h + "\n")

    return done


def _count_lines(path: Path) -> int:
    """Count total lines in a file using raw byte reading (fast for large files)."""
    count = 0
    buf_size = 1024 * 1024  # 1 MB chunks
    with open(path, "rb") as f:
        while buf := f.read(buf_size):
            count += buf.count(b"\n")
    return count


def _fmt_duration(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s"


def process(
    read_path: Path,
    save_path: Path,
    model_name: str,
    env_file: Path | None = None,
) -> None:
    """Score every row in *read_path* and write results to *save_path*.

    Args:
        read_path: Input JSONL with query/pos_list/neg_list.
        save_path: Output JSONL (appended incrementally).
        model_name: User-facing model identifier.
        env_file: Optional .env file for API keys.
    """
    load_optional_dotenv(
        env_file=env_file,
        default_env_file=DEFAULT_PROJECT_ENV_FILE,
    )

    pos_key = f"{model_name}_pos_scores"
    neg_key = f"{model_name}_neg_scores"

    # Checkpoint file lives next to save_path
    ckpt_path = save_path.parent / f"{save_path.name}.ckpt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    done_hashes = _load_checkpoint(ckpt_path, save_path)

    file_size_mb = read_path.stat().st_size / (1024 * 1024)
    log.info("Counting lines in %.0f MB file...", file_size_mb)
    total = _count_lines(read_path)
    already_done = 0
    processed = 0
    skipped_err = 0
    total_docs = 0
    t_start = time.monotonic()

    log.info("=" * 60)
    log.info("Teacher Score Generation")
    log.info("=" * 60)
    log.info("Input:      %s", read_path)
    log.info("Output:     %s", save_path)
    log.info("Model:      %s", model_name)
    log.info("Total rows: %d, checkpoint: %d done", total, len(done_hashes))
    remaining_est = total - len(done_hashes)
    log.info("Estimated remaining: ~%d rows", max(remaining_est, 0))
    log.info("-" * 60)

    scorer = create_voyage_scorer(model_name)

    try:
        with (
            open(read_path, encoding="utf-8") as fin,
            open(save_path, "a", encoding="utf-8") as fout,
            open(ckpt_path, "a", encoding="utf-8") as fckpt,
        ):
            pbar = tqdm(
                fin,
                total=total,
                desc="Scoring",
                unit="row",
                dynamic_ncols=True,
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}] "
                    "{postfix}"
                ),
            )
            for line in pbar:
                line = line.strip()
                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    skipped_err += 1
                    pbar.set_postfix_str(
                        f"ok={processed} skip={already_done} err={skipped_err}"
                    )
                    continue

                row_hash = _compute_row_hash(row)

                if row_hash in done_hashes:
                    already_done += 1
                    pbar.set_postfix_str(
                        f"ok={processed} skip={already_done} err={skipped_err}"
                    )
                    continue

                query: str = row["query"]
                pos_list: list[str] = row["pos_list"]
                neg_list: list[str] = row["neg_list"]
                all_docs = pos_list + neg_list
                n_pos = len(pos_list)
                n_docs = len(all_docs)

                t_call = time.monotonic()
                try:
                    scores = scorer.rerank(query, all_docs)
                except Exception as e:
                    skipped_err += 1
                    log.warning(
                        "API error (query=%r...): %s: %s — skipping",
                        query[:50],
                        type(e).__name__,
                        e,
                    )
                    pbar.set_postfix_str(
                        f"ok={processed} skip={already_done} err={skipped_err}"
                    )
                    continue
                call_ms = (time.monotonic() - t_call) * 1000

                row[pos_key] = scores[:n_pos]
                row[neg_key] = scores[n_pos:]

                # Write result first, then checkpoint.  If Ctrl+C hits
                # in between, the restart logic in _load_checkpoint()
                # rebuilds from the output file, so no data is lost or
                # duplicated.
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                fckpt.write(row_hash + "\n")
                fckpt.flush()

                done_hashes.add(row_hash)
                processed += 1
                total_docs += n_docs

                # Update progress bar postfix
                elapsed = time.monotonic() - t_start
                speed = processed / elapsed if elapsed > 0 else 0
                pbar.set_postfix_str(
                    f"ok={processed} skip={already_done} err={skipped_err} "
                    f"| {call_ms:.0f}ms/call {speed:.1f}row/s"
                )

                # Periodic detailed log every 100 processed rows
                if processed % 100 == 0:
                    eta = (remaining_est - processed) / speed if speed > 0 else 0
                    log.info(
                        "Progress: %d/%d scored, %d cached, %d errors "
                        "| %.1f row/s | %d docs scored | ETA %s",
                        processed,
                        remaining_est,
                        already_done,
                        skipped_err,
                        speed,
                        total_docs,
                        _fmt_duration(eta),
                    )
            pbar.close()
    finally:
        scorer.close()

    elapsed = time.monotonic() - t_start
    speed = processed / elapsed if elapsed > 0 else 0
    log.info("-" * 60)
    log.info("FINISHED in %s", _fmt_duration(elapsed))
    log.info(
        "  processed=%d, cached=%d, errors=%d, total_docs=%d",
        processed,
        already_done,
        skipped_err,
        total_docs,
    )
    log.info("  avg speed: %.2f rows/s", speed)
    log.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reranker teacher scores for JSONL datasets."
    )
    parser.add_argument(
        "--read_path",
        type=Path,
        required=True,
        help="Input JSONL file (query/pos_list/neg_list per line).",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        required=True,
        help="Output JSONL file (appended incrementally).",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help=f"Model identifier. Supported: {SUPPORTED_MODELS}",
    )
    parser.add_argument(
        "--env_file",
        type=Path,
        default=None,
        help="Path to .env file for API keys (default: .env in project root).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    args = parser.parse_args()

    _setup_logging(verbose=args.verbose)

    if not args.read_path.exists():
        log.error("read_path not found: %s", args.read_path)
        sys.exit(1)

    process(
        read_path=args.read_path,
        save_path=args.save_path,
        model_name=args.model_name,
        env_file=args.env_file,
    )


if __name__ == "__main__":
    main()
