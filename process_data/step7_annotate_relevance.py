"""Annotate query-document pairs with LLM relevance labels.

Usage:
uv run python -m process_data.annotate_relevance \
    --model deepseek/deepseek-chat --max_rows 5 --batch_size 5

Output format (long table): each line is one model's annotation for one pair.
Multiple models scoring the same pair produce multiple lines. Append-only writes.
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
from tqdm import tqdm

from shared.env import DEFAULT_PROJECT_ENV_FILE, load_optional_dotenv

log = logging.getLogger("annotate_relevance")

DEFAULT_INPUT_PATH = Path(
    "/mnt/g/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs.jsonl"
)
DEFAULT_SAVE_PATH = Path(
    "/mnt/g/PrismRerankerV1Data/step7_kalm_web-search_query_document_pairs_annotated.jsonl"
)
TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1] / "shared" / "templates" / "relevance_judge.j2"
)

MAX_RETRIES = 1


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-5s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S", stream=sys.stderr)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s"


def _compute_pair_hash(query: str, document: str, model_name: str) -> str:
    """Hash key = sha256(query + document + model_name)."""
    content = f"{query}\n{document}\n{model_name}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _compute_pair_only_hash(query: str, document: str) -> str:
    """Hash key = sha256(query + document), ignoring model_name."""
    return hashlib.sha256(f"{query}\n{document}".encode("utf-8")).hexdigest()


def _derive_model_name(model: str) -> str:
    """Strip provider prefix: 'deepseek/deepseek-chat' -> 'deepseek-chat'."""
    if "/" in model:
        return model.split("/", 1)[1]
    return model


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


def _load_done_hashes(save_path: Path, model_name: str) -> set[str]:
    """Scan output file, return set of hashes that have a non-None label."""
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
                if row.get("model_name") != model_name:
                    continue
                if row.get("annotated_label") is not None:
                    h = _compute_pair_hash(
                        row["query"], row["document"], model_name
                    )
                    done.add(h)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    log.info("Cache: %d completed entries for model %s", len(done), model_name)
    return done


def _load_vote_counts(save_path: Path) -> dict[str, tuple[int, int]]:
    """Scan output file, return per-pair (yes_count, no_count) across all models."""
    counts: dict[str, list[int]] = {}  # hash -> [yes, no]
    if not save_path.exists() or save_path.stat().st_size == 0:
        return {}
    with open(save_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                label = row.get("annotated_label")
                if label not in ("yes", "no"):
                    continue
                h = _compute_pair_only_hash(row["query"], row["document"])
                if h not in counts:
                    counts[h] = [0, 0]
                if label == "yes":
                    counts[h][0] += 1
                else:
                    counts[h][1] += 1
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return {h: (v[0], v[1]) for h, v in counts.items()}


def _append_rows(rows: list[dict[str, Any]], save_path: Path) -> None:
    """Append rows to the output file."""
    with open(save_path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# LLM calling
# ---------------------------------------------------------------------------

def _extract_label(text: str | None) -> str | None:
    """Extract 'yes' or 'no' from the first token of text."""
    if not text or not text.strip():
        return None
    first_token = text.strip().split()[0].lower().rstrip(".,;:!?")
    if first_token in ("yes", "no"):
        return first_token
    return None


# Per-model extra kwargs for litellm.completion.
_MODEL_EXTRA_KWARGS: dict[str, dict[str, Any]] = {
    "openrouter/google/gemini-3-flash-preview": {
        "max_tokens": 5,  # minimum that reliably returns content
        "reasoning_effort": "minimal",
    },
    "openrouter/openai/gpt-5.4-mini": {
        "max_tokens": 512,  # 128,256,512,1024的去尝试
        "reasoning_effort": "low",
    },
}

_DEFAULT_KWARGS: dict[str, Any] = {
    "max_tokens": 1,
}


def _call_llm_litellm(model: str, prompt: str) -> str | None:
    """Call LLM via litellm and return 'yes', 'no', or None."""
    import litellm

    extra = _MODEL_EXTRA_KWARGS.get(model, _DEFAULT_KWARGS)

    for attempt in range(1 + MAX_RETRIES):
        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                **extra,
            )
            label = _extract_label(response.choices[0].message.content)
            if label is not None:
                return label
            if attempt < MAX_RETRIES:
                log.debug(
                    "LLM returned %r, retrying...", response.choices[0].message.content
                )
                continue
            log.warning("LLM returned unexpected token after retries, setting None")
            return None
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM call failed: %s: %s", type(exc).__name__, exc)
            return None
    return None


def _call_llm_bailian(
    model_name: str, prompt: str, api_key: str
) -> str | None:
    """Call Bailian (Aliyun DashScope) API via openai SDK. Thinking disabled."""
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    for attempt in range(1 + MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                temperature=0.0,
                extra_body={"enable_thinking": False},
            )
            content = response.choices[0].message.content
            label = _extract_label(content)
            if label is not None:
                return label
            if attempt < MAX_RETRIES:
                log.debug("Bailian returned %r, retrying...", content)
                continue
            log.warning("Bailian returned unexpected token after retries, setting None")
            return None
        except Exception as exc:  # noqa: BLE001
            log.warning("Bailian call failed: %s: %s", type(exc).__name__, exc)
            return None
    return None


def _call_llm_zhipu(
    model_name: str, prompt: str, api_key: str
) -> str | None:
    """Call Zhipu API via zai-sdk. Thinking disabled, max 1 token."""
    from zai import ZhipuAiClient

    client = ZhipuAiClient(api_key=api_key)
    for attempt in range(1 + MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                temperature=0.0,
                thinking={"type": "disabled"},
            )
            content = response.choices[0].message.content
            label = _extract_label(content)
            if label is not None:
                return label
            if attempt < MAX_RETRIES:
                log.debug("Zhipu returned %r, retrying...", content)
                continue
            log.warning("Zhipu returned unexpected token after retries, setting None")
            return None
        except Exception as exc:  # noqa: BLE001
            log.warning("Zhipu call failed: %s: %s", type(exc).__name__, exc)
            return None
    return None


def _call_llm_moonshot(
    model_name: str, prompt: str, api_key: str
) -> str | None:
    """Call Moonshot API via openai SDK. Thinking disabled, max 1 token."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.moonshot.ai/v1")
    for attempt in range(1 + MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1,
                extra_body={"thinking": {"type": "disabled"}},
            )
            content = response.choices[0].message.content
            if content is None:
                if attempt < MAX_RETRIES:
                    log.debug("Moonshot returned empty content, retrying...")
                    continue
                log.warning("Moonshot returned empty content after retries")
                return None
            label = _extract_label(content)
            if label is not None:
                return label
            if attempt < MAX_RETRIES:
                log.debug("Moonshot returned %r, retrying...", content)
                continue
            log.warning("Moonshot returned unexpected token after retries, setting None")
            return None
        except Exception as exc:  # noqa: BLE001
            log.warning("Moonshot call failed: %s: %s", type(exc).__name__, exc)
            return None
    return None


# ---------------------------------------------------------------------------
# Provider routing
# ---------------------------------------------------------------------------

_PROVIDER_ENV_KEYS: dict[str, str] = {
    "moonshot": "MOONSHOT_API_KEY",
    "bailian": "BAILIAN_API_KEY",
    "zhipu": "ZHIPU_API_KEY_1",
}


def _make_caller(model: str) -> tuple[str, Any]:
    """Return (model_name, callable(prompt) -> label) based on provider prefix.

    Provider routing:
    - ``moonshot/kimi-k2.5`` → openai SDK via Moonshot API
    - ``bailian/qwen3.5-397b-a17b`` → openai SDK via Bailian (DashScope)
    - ``zhipu/glm-5`` → zai-sdk via Zhipu API
    - everything else → litellm (default)
    """
    provider, _, model_name = model.partition("/")
    if not model_name:
        provider, model_name = "", model

    if provider in _PROVIDER_ENV_KEYS:
        api_key = os.environ.get(_PROVIDER_ENV_KEYS[provider], "")
        if not api_key:
            raise RuntimeError(
                f"Provider {provider!r} requires env var "
                f"{_PROVIDER_ENV_KEYS[provider]} to be set."
            )

        if provider == "moonshot":

            def caller(prompt: str) -> str | None:
                return _call_llm_moonshot(model_name, prompt, api_key)

        elif provider == "bailian":

            def caller(prompt: str) -> str | None:
                return _call_llm_bailian(model_name, prompt, api_key)

        elif provider == "zhipu":

            def caller(prompt: str) -> str | None:
                return _call_llm_zhipu(model_name, prompt, api_key)

        else:
            raise RuntimeError(f"Unknown provider: {provider!r}")

        return model_name, caller

    # Default: litellm
    def caller(prompt: str) -> str | None:
        return _call_llm_litellm(model, prompt)

    return model_name, caller


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def process(
    input_path: Path,
    save_path: Path,
    model: str,
    batch_size: int = 32,
    max_workers: int = 32,
    max_rows: int | None = None,
    env_file: Path | None = None,
) -> None:
    load_optional_dotenv(env_file=env_file, default_env_file=DEFAULT_PROJECT_ENV_FILE)

    model_name, call_fn = _make_caller(model)
    template = _load_template()

    # Load input rows (read-only)
    log.info("Loading input rows from %s ...", input_path)
    input_rows = _load_input_rows(input_path)
    log.info("Loaded %d input rows", len(input_rows))

    # Determine scan range
    scan_limit = min(max_rows, len(input_rows)) if max_rows is not None else len(input_rows)

    # Load cache: hashes of already-completed annotations
    done_hashes = _load_done_hashes(save_path, model_name)

    # Load vote counts across all models for early-majority skip
    vote_counts = _load_vote_counts(save_path)

    # Find pending rows
    pending: list[int] = []
    majority_decided = 0
    for idx in range(scan_limit):
        row = input_rows[idx]
        h = _compute_pair_hash(row["query"], row["document"], model_name)
        if h in done_hashes:
            continue
        # Skip if majority already decided (>=3 yes or >=3 no)
        ph = _compute_pair_only_hash(row["query"], row["document"])
        votes = vote_counts.get(ph)
        if votes is not None and (votes[0] >= 3 or votes[1] >= 3):
            majority_decided += 1
            continue
        pending.append(idx)

    already_done = scan_limit - len(pending) - majority_decided

    log.info("=" * 60)
    log.info("Relevance Annotation")
    log.info("=" * 60)
    log.info("Input:            %s", input_path)
    log.info("Output:           %s", save_path)
    log.info("Model:            %s (%s)", model, model_name)
    log.info("Batch size:       %d", batch_size)
    log.info("Workers:          %d", max_workers)
    log.info("Total input rows: %d", len(input_rows))
    log.info("Scan limit:       %s", scan_limit if max_rows else "all")
    log.info("Already done:     %d", already_done)
    log.info("Majority decided: %d (skipped)", majority_decided)
    log.info("Pending:          %d", len(pending))
    log.info("-" * 60)

    if not pending:
        log.info("Nothing to do; all scanned rows already have labels.")
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    failed = 0
    t_start = time.monotonic()
    pbar = tqdm(total=len(pending), desc="Annotate", unit="row", dynamic_ncols=True)

    for batch_start in range(0, len(pending), batch_size):
        batch_indices = pending[batch_start : batch_start + batch_size]

        # Render prompts
        prompts: dict[int, str] = {}
        for idx in batch_indices:
            row = input_rows[idx]
            prompts[idx] = template.render(query=row["query"], document=row["document"])

        # Concurrent LLM calls
        results: dict[int, str | None] = {}
        worker_count = min(max_workers, len(batch_indices))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            future_to_idx = {
                pool.submit(call_fn, prompts[idx]): idx
                for idx in batch_indices
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:  # noqa: BLE001
                    log.warning("Unexpected error for row %d: %s", idx, exc)
                    results[idx] = None

        # Build output rows for this batch and append
        batch_output: list[dict[str, Any]] = []
        for idx in batch_indices:
            label = results.get(idx)
            out_row = dict(input_rows[idx])  # copy original fields
            out_row["model_name"] = model_name
            out_row["annotated_label"] = label
            batch_output.append(out_row)
            if label is not None:
                written += 1
            else:
                failed += 1

        _append_rows(batch_output, save_path)

        pbar.update(len(batch_indices))
        pbar.set_postfix_str(f"ok={written} fail={failed}")

    pbar.close()

    elapsed = time.monotonic() - t_start
    log.info("-" * 60)
    log.info("FINISHED in %s", _fmt_duration(elapsed))
    log.info("  labeled=%d, failed=%d", written, failed)
    log.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate query-document pairs with LLM relevance labels."
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
        default=DEFAULT_SAVE_PATH,
        help="Output JSONL file (append-only).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek/deepseek-chat",
        help="litellm model name (default: deepseek/deepseek-chat).",
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
        help="Only process the first N rows of the input file.",
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
        model=args.model,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        max_rows=args.max_rows,
        env_file=args.env_file,
    )


if __name__ == "__main__":
    main()
