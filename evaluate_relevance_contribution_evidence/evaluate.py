"""Evaluate Qwen reranker predictions with an LLM judge.

Two tasks:
1. Extract yes/no from ``pred_text`` and compare to ``annotated_label``.
2. For rows where both labels are ``yes``, ask a reasoning model
   (Kimi k2.5 or DeepSeek deepseek-reasoner) to score the model-generated
   contribution/evidence against the original ``relevance_extract.j2`` rules.

Output: a new JSONL file with the original fields plus evaluation fields.

Usage:
    # Kimi k2.5 (default)
    uv run python -m evaluate_relevance_contribution_evidence.evaluate \\
        --input_path /mnt/g/PrismRerankerV1Data/Qwen3.5-2B-samples-4000-result.jsonl \\
        --save_path  /mnt/g/PrismRerankerV1Data/Qwen3.5-2B-samples-4000-result.eval.jsonl

    # DeepSeek deepseek-reasoner
    uv run python -m evaluate_relevance_contribution_evidence.evaluate --provider deepseek ...
"""

from __future__ import annotations

import argparse
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

from evaluate_relevance_contribution_evidence.entity_fidelity import (
    build_entity_extractor_client,
    compute_entity_fidelity,
)
from shared.env import DEFAULT_PROJECT_ENV_FILE, load_optional_dotenv

log = logging.getLogger("evaluate_relevance_contribution_evidence")

DEFAULT_INPUT_PATH = Path(
    "/mnt/g/PrismRerankerV1Data/relevance_contribution_evidence_evaluate_result/" \
    "prism_reranker_v1_4B_sft_samples-35001.jsonl"
)
DEFAULT_SAVE_PATH = Path(
    "/mnt/g/PrismRerankerV1Data/relevance_contribution_evidence_evaluate_result/" \
    f"{os.path.basename(DEFAULT_INPUT_PATH)[:-6]}_eval.jsonl"
)
TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "templates" / "judge_contribution_evidence.j2"
)
MAX_RETRIES = 2
MAX_COMPLETION_TOKENS = 8192

# Provider-specific config: default model, base_url, env var for api key.
PROVIDER_CONFIGS: dict[str, dict[str, str]] = {
    "kimi": {
        "default_model": "kimi-k2.5",
        "base_url": "https://api.moonshot.ai/v1",
        "api_key_env": "MOONSHOT_API_KEY",
    },
    "deepseek": {
        "default_model": "deepseek-reasoner",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "bailian": {
        "default_model": "deepseek-v3.2",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "BAILIAN_API_KEY",
    },
}
DEFAULT_PROVIDER = "deepseek"
ENTITY_EXTRACTOR_MODEL = "deepseek-chat"
ENTITY_EXTRACTOR_API_KEY_ENV = "DEEPSEEK_API_KEY"

_CONTRIB_RE = re.compile(r"<contribution>(.*?)</contribution>", re.DOTALL)
_EVIDENCE_RE = re.compile(r"<evidence>(.*?)</evidence>", re.DOTALL)
_SCORES_RE = re.compile(r"<scores>(.*?)</scores>", re.DOTALL)
_REASON_RE = re.compile(r"<reason>(.*?)</reason>", re.DOTALL)

SCORE_FIELDS: tuple[str, ...] = (
    "contribution_accuracy",
    "contribution_coverage",
    "evidence_faithfulness",
    "evidence_self_contained",
    "evidence_concision",
    "language_consistency",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-5s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S", stream=sys.stderr)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s"


def _pair_hash(query: str, document: str) -> str:
    return hashlib.sha256(f"{query}\n{document}".encode("utf-8")).hexdigest()


def _extract_label(text: str | None) -> str | None:
    """Return 'yes' / 'no' from the first token of text, else None."""
    if not text or not text.strip():
        return None
    first_token = text.strip().split()[0].lower().rstrip(".,;:!?")
    if first_token in ("yes", "no"):
        return first_token
    return None


def compute_format_score(pred_text: str | None) -> float:
    """Rule-based format reward in [0, 1].

    Scoring contract:
    - +0.4 if first token is ``yes`` / ``no`` (label parseable)
    - If label == "yes": +0.3 for each of ``<contribution>`` / ``<evidence>``
      tags present with body > 10 chars (both required for a full 1.0)
    - If label == "no": +0.6 iff the entire output is just "no" (optional
      trailing punctuation); any extra content — including ``<contribution>``
      or ``<evidence>`` tags, or free-form commentary — forfeits the bonus
    """
    if not pred_text:
        return 0.0
    label = _extract_label(pred_text)
    if label is None:
        return 0.0
    score = 0.4
    if label == "yes":
        contrib = _CONTRIB_RE.search(pred_text)
        if contrib and len(contrib.group(1).strip()) > 10:
            score += 0.3
        evidence = _EVIDENCE_RE.search(pred_text)
        if evidence and len(evidence.group(1).strip()) > 10:
            score += 0.3
    else:
        cleaned = pred_text.strip().lower().rstrip(".,;:!?")
        if cleaned == "no":
            score += 0.6
    return score


def _parse_contribution_evidence(pred_text: str) -> tuple[str | None, str | None]:
    """Extract <contribution> and <evidence> bodies from pred_text, if present."""
    contribution = None
    evidence = None
    m = _CONTRIB_RE.search(pred_text)
    if m:
        contribution = m.group(1).strip() or None
    m = _EVIDENCE_RE.search(pred_text)
    if m:
        evidence = m.group(1).strip() or None
    return contribution, evidence


def _parse_judge_output(raw: str) -> tuple[dict[str, int] | None, str | None]:
    """Parse <scores> and <reason> blocks from the judge response."""
    if not raw:
        return None, None
    scores_block = _SCORES_RE.search(raw)
    reason_block = _REASON_RE.search(raw)
    reason = reason_block.group(1).strip() if reason_block else None

    if not scores_block:
        return None, reason

    body = scores_block.group(1)
    scores: dict[str, int] = {}
    for field in SCORE_FIELDS:
        m = re.search(rf"<{field}>\s*([1-5])\s*</{field}>", body)
        if not m:
            return None, reason
        scores[field] = int(m.group(1))
    return scores, reason


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
    """Hashes of (query, document) rows already written to save_path.

    Rows with ``eval_status == "failed"`` are treated as NOT done so they get
    retried on resume. To keep the file consistent with the returned set, the
    save file is rewritten in place to drop those failed rows.
    """
    done: set[str] = set()
    if not save_path.exists() or save_path.stat().st_size == 0:
        return done

    kept_lines: list[str] = []
    dropped_failed = 0
    with open(save_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if row.get("eval_status") == "failed":
                dropped_failed += 1
                continue
            try:
                done.add(_pair_hash(row["query"], row["document"]))
            except KeyError:
                continue
            kept_lines.append(stripped)

    if dropped_failed > 0:
        tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for line in kept_lines:
                f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(save_path)
        log.info(
            "Dropped %d failed rows from %s; they will be retried.",
            dropped_failed,
            save_path,
        )

    return done


def _append_rows(rows: list[dict[str, Any]], save_path: Path) -> None:
    with open(save_path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _load_template() -> jinja2.Template:
    return jinja2.Template(TEMPLATE_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Kimi judge call
# ---------------------------------------------------------------------------


def _call_judge(
    client: Any,
    provider: str,
    model_name: str,
    prompt: str,
    enable_thinking: bool = False,
) -> tuple[str | None, str | None]:
    """Call the judge model via an OpenAI-compatible client.

    Returns ``(content, reasoning_content)``. Both may be ``None`` on failure.

    - ``kimi``: k2.5 thinking mode is on by default; temperature is fixed at
      1.0 when thinking is enabled.
    - ``deepseek``: ``deepseek-reasoner`` silently ignores sampling params,
      so we don't pass ``temperature``.
    - ``bailian``: DashScope OpenAI-compatible endpoint; ``enable_thinking``
      toggles DeepSeek reasoning. When True, reasoning is delivered via
      streamed ``delta.reasoning_content`` chunks, so we switch to streaming.
    """
    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_COMPLETION_TOKENS,
    }
    if provider == "kimi":
        kwargs["temperature"] = 1.0
    if provider == "bailian":
        kwargs["extra_body"] = {"enable_thinking": enable_thinking}

    use_stream = provider == "bailian" and enable_thinking

    last_exc: Exception | None = None
    for attempt in range(1 + MAX_RETRIES):
        try:
            if use_stream:
                content, reasoning = _consume_stream(client, kwargs)
            else:
                response = client.chat.completions.create(**kwargs)
                message = response.choices[0].message
                content = message.content
                reasoning = getattr(message, "reasoning_content", None)
            if content and content.strip():
                reasoning = reasoning.strip() if reasoning else None
                return content.strip(), reasoning
            log.debug("%s returned empty content (attempt %d)", provider, attempt + 1)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log.warning(
                "%s call failed (attempt %d): %s: %s",
                provider,
                attempt + 1,
                type(exc).__name__,
                exc,
            )
    if last_exc is not None:
        log.warning("%s call gave up after %d retries", provider, MAX_RETRIES + 1)
    return None, None


def _consume_stream(
    client: Any, kwargs: dict[str, Any]
) -> tuple[str | None, str | None]:
    """Run a streaming completion and accumulate content + reasoning_content."""
    stream = client.chat.completions.create(**kwargs, stream=True)
    content_buf: list[str] = []
    reasoning_buf: list[str] = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        chunk_reasoning = getattr(delta, "reasoning_content", None)
        if chunk_reasoning:
            reasoning_buf.append(chunk_reasoning)
        chunk_content = getattr(delta, "content", None)
        if chunk_content:
            content_buf.append(chunk_content)
    content = "".join(content_buf) if content_buf else None
    reasoning = "".join(reasoning_buf) if reasoning_buf else None
    return content, reasoning


def _build_client(provider: str, api_key: str) -> Any:
    from openai import OpenAI

    cfg = PROVIDER_CONFIGS[provider]
    return OpenAI(api_key=api_key, base_url=cfg["base_url"])


# ---------------------------------------------------------------------------
# Row-level evaluation
# ---------------------------------------------------------------------------


def _enrich_label_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Add pred_label / label_match / parsed_* fields to a copy of row."""
    out = dict(row)
    pred_text_raw: str = row.get("pred_text") or ""
    pred_text = pred_text_raw.strip()
    pred_label = _extract_label(pred_text)
    annotated = row.get("annotated_label")

    out["pred_label"] = pred_label
    out["label_match"] = "yes" if pred_label == annotated else "no"
    out["format_score"] = compute_format_score(pred_text_raw)

    if pred_label == "yes":
        contribution, evidence = _parse_contribution_evidence(pred_text)
    else:
        contribution, evidence = None, None
    out["parsed_contribution"] = contribution
    out["parsed_evidence"] = evidence
    return out


def _should_judge(row: dict[str, Any]) -> bool:
    return (
        row.get("annotated_label") == "yes"
        and row.get("pred_label") == "yes"
        and bool(row.get("parsed_contribution"))
        and bool(row.get("parsed_evidence"))
    )


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def process(
    input_path: Path,
    save_path: Path,
    provider: str = DEFAULT_PROVIDER,
    judge_model: str | None = None,
    batch_size: int = 16,
    max_workers: int = 16,
    max_rows: int | None = None,
    env_file: Path | None = None,
    enable_thinking: bool = False,
) -> None:
    load_optional_dotenv(env_file=env_file, default_env_file=DEFAULT_PROJECT_ENV_FILE)

    if provider not in PROVIDER_CONFIGS:
        raise ValueError(
            f"Unknown provider: {provider}. Choices: {list(PROVIDER_CONFIGS)}"
        )
    if enable_thinking and provider != "bailian":
        log.warning(
            "--enable_thinking is only supported for provider=bailian; "
            "ignoring for provider=%s",
            provider,
        )
        enable_thinking = False
    cfg = PROVIDER_CONFIGS[provider]
    if judge_model is None:
        judge_model = cfg["default_model"]

    api_key = os.environ.get(cfg["api_key_env"], "")
    if not api_key:
        raise RuntimeError(f"{cfg['api_key_env']} is not set.")
    client = _build_client(provider, api_key)

    entity_api_key = os.environ.get(ENTITY_EXTRACTOR_API_KEY_ENV, "")
    if not entity_api_key:
        raise RuntimeError(
            f"{ENTITY_EXTRACTOR_API_KEY_ENV} is not set "
            f"(needed for entity-fidelity extractor: {ENTITY_EXTRACTOR_MODEL})."
        )
    entity_client = build_entity_extractor_client(entity_api_key)

    template = _load_template()

    log.info("Loading input rows from %s ...", input_path)
    input_rows = _load_input_rows(input_path)
    log.info("Loaded %d input rows", len(input_rows))

    scan_limit = (
        min(max_rows, len(input_rows)) if max_rows is not None else len(input_rows)
    )
    done_hashes = _load_done_hashes(save_path)

    # First pass: enrich labels for all rows (cheap, no LLM).
    enriched: list[dict[str, Any]] = []
    pending: list[int] = []
    already_done = 0
    for idx in range(scan_limit):
        row = input_rows[idx]
        h = _pair_hash(row["query"], row["document"])
        if h in done_hashes:
            already_done += 1
            enriched.append({})  # placeholder; skipped rows not re-emitted
            continue
        enriched_row = _enrich_label_fields(row)
        enriched.append(enriched_row)
        if _should_judge(enriched_row):
            pending.append(idx)

    to_pass_through = [
        idx for idx in range(scan_limit) if enriched[idx] and idx not in set(pending)
    ]

    log.info("=" * 60)
    log.info("Pred Quality Evaluation")
    log.info("=" * 60)
    log.info("Input:               %s", input_path)
    log.info("Output:              %s", save_path)
    log.info("Provider:            %s", provider)
    log.info("Judge model:         %s", judge_model)
    log.info("Enable thinking:     %s", enable_thinking)
    log.info("Scan limit:          %d", scan_limit)
    log.info("Already done:        %d", already_done)
    log.info("Needs judging:       %d", len(pending))
    log.info("Pass-through (no judge): %d", len(to_pass_through))
    log.info("-" * 60)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Write pass-through rows in order (no LLM needed).
    if to_pass_through:
        pass_rows: list[dict[str, Any]] = []
        for idx in to_pass_through:
            row = dict(enriched[idx])
            if row.get("annotated_label") == "yes" and row.get("pred_label") == "yes":
                # yes-yes but parse failed
                row["eval_status"] = "skipped_no_parse"
            else:
                row["eval_status"] = "skipped_not_yesyes"
            row["eval_scores"] = None
            row["eval_reason"] = None
            row["eval_thinking"] = None
            row["eval_raw_content"] = None
            row["entity_fidelity"] = None
            row["judge_model"] = judge_model
            row["provider"] = provider
            row["enable_thinking"] = enable_thinking
            row["entity_extractor_model"] = ENTITY_EXTRACTOR_MODEL
            pass_rows.append(row)
        _append_rows(pass_rows, save_path)
        log.info("Wrote %d pass-through rows", len(pass_rows))

    if not pending:
        log.info("Nothing to judge.")
        return

    # 2) Judge yes-yes rows.
    written = 0
    failed = 0
    t_start = time.monotonic()
    pbar = tqdm(total=len(pending), desc="Judging", unit="row", dynamic_ncols=True)

    def _evaluate_one(
        prompt: str, evidence: str, document: str
    ) -> tuple[str | None, str | None, dict[str, Any]]:
        fidelity = compute_entity_fidelity(
            evidence, document, entity_client, ENTITY_EXTRACTOR_MODEL
        )
        content, reasoning = _call_judge(
            client, provider, judge_model, prompt, enable_thinking=enable_thinking
        )
        return content, reasoning, fidelity

    for batch_start in range(0, len(pending), batch_size):
        batch_indices = pending[batch_start : batch_start + batch_size]

        prompts: dict[int, str] = {}
        for idx in batch_indices:
            row = enriched[idx]
            prompts[idx] = template.render(
                query=row["query"],
                document=row["document"],
                contribution=row["parsed_contribution"],
                evidence=row["parsed_evidence"],
            )

        results: dict[int, tuple[str | None, str | None, dict[str, Any]]] = {}
        worker_count = min(max_workers, len(batch_indices))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            future_to_idx = {
                pool.submit(
                    _evaluate_one,
                    prompts[idx],
                    enriched[idx]["parsed_evidence"],
                    enriched[idx]["document"],
                ): idx
                for idx in batch_indices
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:  # noqa: BLE001
                    log.warning("Unexpected error for row %d: %s", idx, exc)
                    results[idx] = (
                        None,
                        None,
                        {"score": 1.0, "extracted": [], "missing": []},
                    )

        batch_output: list[dict[str, Any]] = []
        for idx in batch_indices:
            content, reasoning, fidelity = results.get(
                idx, (None, None, {"score": 1.0, "extracted": [], "missing": []})
            )
            out_row = dict(enriched[idx])
            out_row["judge_model"] = judge_model
            out_row["provider"] = provider
            out_row["enable_thinking"] = enable_thinking
            out_row["entity_extractor_model"] = ENTITY_EXTRACTOR_MODEL
            out_row["entity_fidelity"] = fidelity
            out_row["eval_raw_content"] = content

            if content is None:
                out_row["eval_status"] = "failed"
                out_row["eval_scores"] = None
                out_row["eval_reason"] = None
                out_row["eval_thinking"] = reasoning
                failed += 1
            else:
                scores, reason = _parse_judge_output(content)
                if scores is None:
                    out_row["eval_status"] = "failed"
                    out_row["eval_scores"] = None
                    out_row["eval_reason"] = reason or content[:2000]
                    out_row["eval_thinking"] = reasoning
                    failed += 1
                else:
                    out_row["eval_status"] = "scored"
                    out_row["eval_scores"] = scores
                    out_row["eval_reason"] = reason
                    out_row["eval_thinking"] = reasoning
                    written += 1

            batch_output.append(out_row)

        _append_rows(batch_output, save_path)
        pbar.update(len(batch_indices))
        pbar.set_postfix_str(f"ok={written} fail={failed}")

    pbar.close()

    elapsed = time.monotonic() - t_start
    log.info("-" * 60)
    log.info("FINISHED in %s", _fmt_duration(elapsed))
    log.info("  scored=%d, failed=%d", written, failed)
    log.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen reranker predictions with Kimi k2.5 as judge.",
    )
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--save_path", type=Path, default=DEFAULT_SAVE_PATH)
    parser.add_argument(
        "--provider",
        type=str,
        default=DEFAULT_PROVIDER,
        choices=list(PROVIDER_CONFIGS),
        help="LLM provider for the judge call.",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=None,
        help="Override model id; defaults to the provider's default model.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_workers", type=int, default=64)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--env_file", type=Path, default=None)
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help=("Bailian only: enable DeepSeek thinking mode (r1 behavior, streamed)."),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(verbose=args.verbose)

    if not args.input_path.exists():
        log.error("input_path not found: %s", args.input_path)
        sys.exit(1)

    process(
        input_path=args.input_path,
        save_path=args.save_path,
        provider=args.provider,
        judge_model=args.judge_model,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        max_rows=args.max_rows,
        env_file=args.env_file,
        enable_thinking=args.enable_thinking,
    )


if __name__ == "__main__":
    main()
