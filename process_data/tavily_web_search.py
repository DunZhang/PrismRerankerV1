"""Augment KaLM JSONL rows with Tavily web search results.

Usage:
uv run python -m process_data.tavily_web_search \
    --read_path /mnt/g/PrismRerankerV1Data/KaLM__all_retrieval_voyage-rerank2_voyage-rerank2.5.jsonl \
    --save_path /mnt/g/PrismRerankerV1Data/KaLM__all_retrieval_voyage-rerank2_voyage-rerank2.5_web_search.jsonl

Behavior:
- Reads the input JSONL line by line.
- Uses ``query`` to call Tavily search.
- Adds ``tavily_topk`` and ``extra.original_tavily_result`` to each written row.
- Appends to the output file incrementally and supports resume via row hash.
- Rotates across all ``TAVILY_API_KEY_{idx}`` keys from ``.env``.
- Each key is reserved at most 980 times across resumptions.

Note:
The current ``.env`` may not have enough total Tavily quota for the whole input.
When all keys reach the configured per-key cap, the script stops cleanly and can
resume later after more keys are added to ``.env``.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from shared.env import DEFAULT_PROJECT_ENV_FILE, load_optional_dotenv

log = logging.getLogger("tavily_web_search")

DEFAULT_READ_PATH = Path(
    "/mnt/g/PrismRerankerV1Data/"
    "KaLM__all_retrieval_voyage-rerank2_voyage-rerank2.5.jsonl"
)
DEFAULT_SAVE_PATH = Path(
    "/mnt/g/PrismRerankerV1Data/"
    "KaLM__all_retrieval_voyage-rerank2_voyage-rerank2.5_web_search.jsonl"
)

KEY_LIMIT = 100
KEY_RPM = 100
TAVILY_MAX_QUERY_LEN = 395
TAVILY_ENV_RE = re.compile(r"^TAVILY_API_KEY_(\d+)$")


@dataclass(frozen=True)
class TavilyKey:
    index: int
    name: str
    value: str


class TavilyKeyAllocator:
    """Round-robin allocator with a hard per-key reservation cap."""

    def __init__(
        self,
        keys: list[TavilyKey],
        key_usage: dict[str, int],
        next_cursor: int = 0,
    ) -> None:
        if not keys:
            raise ValueError("At least one Tavily key is required.")
        self.keys = keys
        self.key_usage = {key.name: int(key_usage.get(key.name, 0)) for key in keys}
        self.next_cursor = next_cursor % len(keys)
        self.runtime_disabled_keys: set[str] = set()

    def reserve(self) -> TavilyKey | None:
        for offset in range(len(self.keys)):
            idx = (self.next_cursor + offset) % len(self.keys)
            key = self.keys[idx]
            if key.name in self.runtime_disabled_keys:
                continue
            used = self.key_usage[key.name]
            if used >= KEY_LIMIT:
                continue
            self.key_usage[key.name] = used + 1
            self.next_cursor = (idx + 1) % len(self.keys)
            return key
        return None

    def remaining_capacity(self) -> int:
        return sum(max(KEY_LIMIT - self.key_usage[key.name], 0) for key in self.keys)

    def remaining_usable_capacity(self) -> int:
        return sum(
            max(KEY_LIMIT - self.key_usage[key.name], 0)
            for key in self.keys
            if key.name not in self.runtime_disabled_keys
        )

    def total_reserved(self) -> int:
        return sum(self.key_usage.values())

    def disable_for_current_run(self, key_name: str) -> bool:
        if key_name in self.runtime_disabled_keys:
            return False
        self.runtime_disabled_keys.add(key_name)
        return True

    def has_usable_key(self) -> bool:
        return any(
            key.name not in self.runtime_disabled_keys
            and self.key_usage[key.name] < KEY_LIMIT
            for key in self.keys
        )

class _PerKeyRateLimiter:
    """Thread-safe sliding-window rate limiter, one window per API key."""

    def __init__(self, rpm: int) -> None:
        self.rpm = rpm
        self.window = 60.0
        self._lock = threading.Lock()
        self._timestamps: dict[str, collections.deque[float]] = {}

    def acquire(self, key: str) -> None:
        """Block until a request slot is available for *key*."""
        while True:
            with self._lock:
                now = time.monotonic()
                if key not in self._timestamps:
                    self._timestamps[key] = collections.deque()
                dq = self._timestamps[key]
                cutoff = now - self.window
                while dq and dq[0] < cutoff:
                    dq.popleft()
                if len(dq) < self.rpm:
                    dq.append(now)
                    return
                sleep_dur = dq[0] + self.window - now
            time.sleep(max(sleep_dur, 0.05))


_rate_limiter = _PerKeyRateLimiter(rpm=KEY_RPM)


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


def _compute_row_hash(row: dict[str, Any]) -> str:
    content = json.dumps(
        {
            "query": row["query"],
            "pos_list": row["pos_list"],
            "neg_list": row["neg_list"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _count_lines(path: Path) -> int:
    count = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            count += chunk.count(b"\n")
    return count


def _load_done_hashes(save_path: Path) -> set[str]:
    done: set[str] = set()
    if not save_path.exists() or save_path.stat().st_size == 0:
        return done

    size_mb = save_path.stat().st_size / (1024 * 1024)
    log.info("Rebuilding resume checkpoint from output (%.1f MB)...", size_mb)
    with open(save_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done.add(_compute_row_hash(row))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    log.info("Resume checkpoint rebuilt: %d rows already written", len(done))
    return done


def _load_tavily_keys() -> list[TavilyKey]:
    keys: list[TavilyKey] = []
    for env_name, env_value in os.environ.items():
        match = TAVILY_ENV_RE.fullmatch(env_name)
        if not match or not env_value:
            continue
        keys.append(
            TavilyKey(index=int(match.group(1)), name=env_name, value=env_value.strip())
        )
    keys.sort(key=lambda item: item.index)
    return keys


def _new_allocator(keys: list[TavilyKey]) -> TavilyKeyAllocator:
    """Create a fresh allocator with zero usage (per-run counting)."""
    return TavilyKeyAllocator(
        keys=keys,
        key_usage={key.name: 0 for key in keys},
    )


def _search_with_tavily(api_key: str, query: str) -> dict[str, Any]:
    try:
        from tavily import TavilyClient
    except ImportError as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(
            "Missing dependency: tavily-python. Install it with "
            "`uv pip install tavily-python` or refresh the project environment."
        ) from exc

    if len(query) > TAVILY_MAX_QUERY_LEN:
        query = query[:TAVILY_MAX_QUERY_LEN]

    _rate_limiter.acquire(api_key)

    client = TavilyClient(api_key)
    response = client.search(
        query=query,
        include_answer="advanced",
        search_depth="basic",
        max_results=20,
        include_raw_content="markdown",
        chunks_per_source=1,
    )
    if not isinstance(response, dict):
        raise TypeError(
            f"Tavily returned {type(response).__name__}, expected a dict response."
        )
    return response


def _is_usage_limit_error(exc: Exception) -> bool:
    if exc.__class__.__name__ == "UsageLimitExceededError":
        return True

    message = str(exc).lower()
    has_quota_hint = any(
        token in message
        for token in ("usage limit", "quota", "credits", "credit balance")
    )
    has_exhausted_hint = any(
        token in message for token in ("exceeded", "exhausted", "depleted", "limit")
    )
    return has_quota_hint and has_exhausted_hint


def _choose_doc(search_item: dict[str, Any], rng: random.Random) -> str:
    candidates: list[str] = []
    for field in ("content", "raw_content"):
        value = search_item.get(field)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    if candidates:
        doc = rng.choice(candidates)
    else:
        doc = str(search_item.get("title") or search_item.get("url") or "").strip()

    title = str(search_item.get("title") or "").strip()
    if title and doc and rng.random() < 0.5:
        doc = f"{title}\n{doc}"
    return doc


def _build_tavily_topk(row_hash: str, response: dict[str, Any]) -> dict[str, Any]:
    rng = random.Random(row_hash)
    topk_docs: list[str] = []

    answer = response.get("answer")
    if isinstance(answer, str) and answer.strip():
        topk_docs.append(answer.strip())

    results = response.get("results")
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            doc = _choose_doc(item, rng)
            if not doc:
                continue
            topk_docs.append(doc)

    return {"topk_docs": topk_docs}


def _augment_row(
    row: dict[str, Any],
    row_hash: str,
    response: dict[str, Any],
) -> dict[str, Any]:
    out_row = dict(row)

    extra = out_row.get("extra")
    if extra is None:
        extra = {}
    elif not isinstance(extra, dict):
        extra = {"legacy_extra": extra}
    else:
        extra = dict(extra)

    extra["original_tavily_result"] = response
    out_row["extra"] = extra
    out_row["tavily_topk"] = _build_tavily_topk(row_hash, response)
    return out_row


def _process_batch(
    batch: list[tuple[str, dict[str, Any]]],
    allocator: TavilyKeyAllocator,
    max_workers: int,
    fout,  # noqa: ANN001
    done_hashes: set[str],
) -> tuple[int, int, bool]:
    reservations: list[tuple[str, dict[str, Any], TavilyKey]] = []
    exhausted = False

    for row_hash, row in batch:
        key = allocator.reserve()
        if key is None:
            exhausted = True
            break
        reservations.append((row_hash, row, key))

    if not reservations:
        return 0, 0, True

    results: dict[str, dict[str, Any] | Exception] = {}
    worker_count = min(max_workers, len(reservations))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        future_to_request = {
            pool.submit(_search_with_tavily, key.value, row["query"]): (row_hash, key)
            for row_hash, row, key in reservations
        }
        for future in as_completed(future_to_request):
            row_hash, key = future_to_request[future]
            try:
                results[row_hash] = future.result()
            except Exception as exc:  # noqa: BLE001
                results[row_hash] = exc
                if _is_usage_limit_error(exc):
                    if allocator.disable_for_current_run(key.name):
                        log.warning(
                            "Tavily quota exhausted for key=%s; disabling it for the "
                            "rest of this run.",
                            key.name,
                        )

    written = 0
    failed = 0
    for row_hash, row, key in reservations:
        result = results[row_hash]
        if isinstance(result, Exception):
            failed += 1
            log.warning(
                "Tavily search failed for key=%s query=%r: %s: %s",
                key.name,
                row.get("query", "")[:80],
                type(result).__name__,
                result,
            )
            continue

        out_row = _augment_row(row=row, row_hash=row_hash, response=result)
        fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        done_hashes.add(row_hash)
        written += 1

    fout.flush()
    os.fsync(fout.fileno())

    exhausted = exhausted or not allocator.has_usable_key()
    return written, failed, exhausted


def process(
    read_path: Path,
    save_path: Path,
    batch_size: int = 16,
    max_workers: int = 16,
    env_file: Path | None = None,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive.")

    load_optional_dotenv(env_file=env_file, default_env_file=DEFAULT_PROJECT_ENV_FILE)

    keys = _load_tavily_keys()
    if not keys:
        log.error("No TAVILY_API_KEY_{idx} keys found in environment or .env.")
        sys.exit(1)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    done_hashes = _load_done_hashes(save_path)
    allocator = _new_allocator(keys)
    total_rows = _count_lines(read_path)

    remaining_capacity = allocator.remaining_capacity()
    remaining_rows = max(total_rows - len(done_hashes), 0)

    log.info("=" * 60)
    log.info("Tavily Web Search Augmentation")
    log.info("=" * 60)
    log.info("Input:            %s", read_path)
    log.info("Output:           %s", save_path)
    log.info("Batch size:       %d", batch_size)
    log.info("Workers:          %d", max_workers)
    log.info("Tavily keys:      %d", len(keys))
    log.info("Key limit:        %d searches/key", KEY_LIMIT)
    log.info("Already written:  %d", len(done_hashes))
    log.info("Total rows:       %d", total_rows)
    log.info("Remaining rows:   %d", remaining_rows)
    log.info("Remaining quota:  %d", remaining_capacity)
    log.info("-" * 60)

    if remaining_rows == 0:
        log.info("Nothing to do; output already covers all input rows.")
        return

    if remaining_capacity == 0:
        log.warning("No remaining Tavily quota. Add more keys or reset quota, then rerun.")
        return

    if remaining_rows > remaining_capacity:
        log.warning(
            "Remaining rows (%d) exceed remaining quota (%d). This run will stop when "
            "all keys reach the configured per-key cap.",
            remaining_rows,
            remaining_capacity,
        )

    written = 0
    skipped = 0
    failed = 0
    exhausted = False
    t_start = time.monotonic()

    with (
        open(read_path, encoding="utf-8") as fin,
        open(save_path, "a", encoding="utf-8") as fout,
    ):
        pbar = tqdm(
            fin,
            total=total_rows,
            desc="Tavily",
            unit="row",
            dynamic_ncols=True,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}] "
                "{postfix}"
            ),
        )

        batch: list[tuple[str, dict[str, Any]]] = []
        for line in pbar:
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                failed += 1
                pbar.set_postfix_str(
                    f"written={written} skip={skipped} fail={failed} quota={allocator.remaining_usable_capacity()}"
                )
                continue

            row_hash = _compute_row_hash(row)
            if row_hash in done_hashes:
                skipped += 1
                pbar.set_postfix_str(
                    f"written={written} skip={skipped} fail={failed} quota={allocator.remaining_usable_capacity()}"
                )
                continue

            batch.append((row_hash, row))
            if len(batch) < batch_size:
                continue

            batch_written, batch_failed, batch_exhausted = _process_batch(
                batch=batch,
                allocator=allocator,
                max_workers=max_workers,
                fout=fout,
                done_hashes=done_hashes,
            )
            written += batch_written
            failed += batch_failed
            exhausted = batch_exhausted
            batch = []
            pbar.set_postfix_str(
                f"written={written} skip={skipped} fail={failed} quota={allocator.remaining_usable_capacity()}"
            )
            if exhausted:
                break

        if batch and not exhausted:
            batch_written, batch_failed, batch_exhausted = _process_batch(
                batch=batch,
                allocator=allocator,
                max_workers=max_workers,
                fout=fout,
                done_hashes=done_hashes,
            )
            written += batch_written
            failed += batch_failed
            exhausted = batch_exhausted
            pbar.set_postfix_str(
                f"written={written} skip={skipped} fail={failed} quota={allocator.remaining_usable_capacity()}"
            )

        pbar.close()

    elapsed = time.monotonic() - t_start
    log.info("-" * 60)
    log.info("FINISHED in %s", _fmt_duration(elapsed))
    log.info("  written=%d, skipped=%d, failed=%d", written, skipped, failed)
    log.info("  remaining quota=%d", allocator.remaining_usable_capacity())
    if exhausted:
        log.warning(
            "Stopped because all currently usable Tavily keys are exhausted or have "
            "hit the configured per-key cap."
        )
    log.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment KaLM JSONL with Tavily web search results."
    )
    parser.add_argument(
        "--read_path",
        type=Path,
        default=DEFAULT_READ_PATH,
        help="Input JSONL file.",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        default=DEFAULT_SAVE_PATH,
        help="Output JSONL file written incrementally in append mode.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Rows reserved and written per batch (default: 16).",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=16,
        help="Max parallel Tavily requests per batch (default: 16).",
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

    if not args.read_path.exists():
        log.error("read_path not found: %s", args.read_path)
        sys.exit(1)

    process(
        read_path=args.read_path,
        save_path=args.save_path,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        env_file=args.env_file,
    )


if __name__ == "__main__":
    main()
