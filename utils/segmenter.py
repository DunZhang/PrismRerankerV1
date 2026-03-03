"""Rule-based document segmenter for reranker training.

Splits Markdown or plain-text documents into paragraph-level segments
and wraps each with indexed ``<segment_i>`` tags, providing anchors
for the model to reference during inference.

Usage::

    from utils.segmenter import segment_document

    tagged = segment_document(raw_doc)
    # => "<segment_1>...</segment_1>\n<segment_2>...</segment_2>\n..."
"""

import functools
import os
import re
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault(
    "TIKTOKEN_CACHE_DIR",
    str(Path(__file__).resolve().parent / ".tiktoken_cache"),
)

import tiktoken  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MIN_TOKENS: int = 60
"""Segments shorter than this (in tokens) are merged with neighbors."""

DEFAULT_MAX_TOKENS: int = 300
"""Paragraphs longer than this (in tokens) are split further."""

DEFAULT_WINDOW_TOKENS: int = 300
"""Token-window fallback for last-resort splitting."""

_MD_HEADER_RE = re.compile(r"(?=^#{2,6}\s+)", re.MULTILINE)
"""Lookahead split point at ``##`` .. ``######`` headers."""

_MD_DETECT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^#{1,6}\s+", re.MULTILINE),
    re.compile(r"^```", re.MULTILINE),
    re.compile(r"^[\-\*]\s+", re.MULTILINE),
    re.compile(r"^\d+\.\s+", re.MULTILINE),
]
_MD_DETECT_THRESHOLD: int = 2

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_ENC = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    """Return the number of tokens in *text* using cl100k_base."""
    return len(_ENC.encode(text))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _is_markdown(text: str) -> bool:
    """Detect whether *text* looks like Markdown."""
    hits = sum(1 for pat in _MD_DETECT_PATTERNS if pat.search(text))
    return hits >= _MD_DETECT_THRESHOLD


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines and drop empty fragments."""
    return [p for part in text.split("\n\n") if (p := part.strip())]


def _split_by_window(text: str, window: int) -> list[str]:
    """Split *text* into chunks of at most *window* tokens.

    Encodes the text, takes the first *window* tokens, decodes back to
    find the character boundary, then breaks at the last whitespace.
    Falls back to a hard cut at the token boundary for CJK text.
    """
    chunks: list[str] = []
    while _token_len(text) > window:
        tokens = _ENC.encode(text)
        prefix = _ENC.decode(tokens[:window])
        cut = prefix.rfind(" ")
        if cut <= 0:
            cut = len(prefix)
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks


def _split_long(
    text: str,
    max_tokens: int,
    window_tokens: int,
    is_md: bool,
) -> list[str]:
    """Break an oversized paragraph into smaller pieces.

    Strategy (applied in order until the chunk is small enough):
    1. Markdown headers (``##`` / ``###`` …) — only for Markdown docs.
    2. Single newlines (``\\n``).
    3. Fixed token-window fallback.
    """
    if _token_len(text) <= max_tokens:
        return [text]

    # --- step 1: Markdown header split ---
    if is_md:
        parts = [s.strip() for s in _MD_HEADER_RE.split(text) if s.strip()]
        if len(parts) > 1:
            result: list[str] = []
            for part in parts:
                result.extend(
                    _split_long(part, max_tokens, window_tokens, False)
                )
            return result

    # --- step 2: single-newline split with greedy accumulation ---
    lines = text.split("\n")
    if len(lines) > 1:
        chunks: list[str] = []
        buf = lines[0]
        for line in lines[1:]:
            candidate = buf + "\n" + line
            if _token_len(candidate) > max_tokens and buf:
                chunks.append(buf.strip())
                buf = line
            else:
                buf = candidate
        if buf.strip():
            chunks.append(buf.strip())
        if len(chunks) > 1:
            result = []
            for chunk in chunks:
                if _token_len(chunk) > max_tokens:
                    result.extend(
                        _split_by_window(chunk, window_tokens)
                    )
                else:
                    result.append(chunk)
            return result

    # --- step 3: fixed-window fallback ---
    return _split_by_window(text, window_tokens)


def _merge_short(segments: list[str], min_tokens: int) -> list[str]:
    """Greedily merge consecutive short segments.

    Walks forward; accumulates segments into a buffer until the buffer
    reaches *min_tokens*, then flushes.  Any trailing residue is folded
    into the last flushed segment to avoid a tiny tail.
    """
    if not segments:
        return []

    result: list[str] = []
    buf = ""
    for seg in segments:
        buf = f"{buf}\n\n{seg}" if buf else seg
        if _token_len(buf) >= min_tokens:
            result.append(buf)
            buf = ""
    if buf:
        if result:
            result[-1] = f"{result[-1]}\n\n{buf}"
        else:
            result.append(buf)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split_segments(
    text: str,
    *,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    window_tokens: int = DEFAULT_WINDOW_TOKENS,
) -> list[str]:
    """Split a document into paragraph-level segments.

    Args:
        text: The raw document string (Markdown or plain text).
        min_tokens: Merge segments shorter than this with neighbors.
        max_tokens: Split paragraphs longer than this.
        window_tokens: Token window for last-resort splitting.

    Returns:
        Ordered list of segment strings (no XML tags).
    """
    text = text.strip()
    if not text:
        return []

    is_md = _is_markdown(text)
    paragraphs = _split_paragraphs(text)

    # Expand oversized paragraphs.
    expanded: list[str] = []
    for para in paragraphs:
        expanded.extend(
            _split_long(para, max_tokens, window_tokens, is_md)
        )

    return _merge_short(expanded, min_tokens)


def segment_document(
    text: str,
    *,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    window_tokens: int = DEFAULT_WINDOW_TOKENS,
) -> str:
    """Split a document and wrap each segment with indexed tags.

    Args:
        text: The raw document string (Markdown or plain text).
        min_tokens: Merge segments shorter than this with neighbors.
        max_tokens: Split paragraphs longer than this.
        window_tokens: Token window for last-resort splitting.

    Returns:
        A string where each segment is wrapped as
        ``<segment_1>…</segment_1>``, joined by newlines.
    """
    segments = split_segments(
        text,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        window_tokens=window_tokens,
    )
    return "\n\n".join(
        f"<segment_{i}>\n{seg}\n</segment_{i}>"
        for i, seg in enumerate(segments, 1)
    )


def batch_split_segments(
    texts: list[str],
    *,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    window_tokens: int = DEFAULT_WINDOW_TOKENS,
    num_workers: int | None = None,
) -> list[list[str]]:
    """Split multiple documents into segments in parallel.

    Args:
        texts: List of raw document strings.
        min_tokens: Merge segments shorter than this with neighbors.
        max_tokens: Split paragraphs longer than this.
        window_tokens: Token window for last-resort splitting.
        num_workers: Number of worker processes. ``None`` uses CPU count.

    Returns:
        List of segment lists, one per input document, in order.
    """
    fn = functools.partial(
        split_segments,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        window_tokens=window_tokens,
    )
    with Pool(num_workers) as pool:
        return pool.map(fn, texts)


def batch_segment_documents(
    texts: list[str],
    *,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    window_tokens: int = DEFAULT_WINDOW_TOKENS,
    num_workers: int | None = None,
) -> list[str]:
    """Split multiple documents and wrap segments with indexed tags.

    Args:
        texts: List of raw document strings.
        min_tokens: Merge segments shorter than this with neighbors.
        max_tokens: Split paragraphs longer than this.
        window_tokens: Token window for last-resort splitting.
        num_workers: Number of worker processes. ``None`` uses CPU count.

    Returns:
        List of tagged strings, one per input document, in order.
    """
    fn = functools.partial(
        segment_document,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        window_tokens=window_tokens,
    )
    with Pool(num_workers) as pool:
        return pool.map(fn, texts)


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_md = """\
# 布洛芬使用须知

## 常见副作用

布洛芬的常见副作用包括胃肠不适、头晕和皮疹，通常在停药后可自行缓解。
如出现症状加重，应及时咨询医生。

## 严重不良反应

罕见情况下，布洛芬可能导致严重过敏反应或肝肾功能异常，需立即就医。

## 用药提示

本品为非处方药，请按说明书或在药师指导下使用。

注意事项。
"""

    sample_plain = (
        "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) "
        "used for treating pain, fever, and inflammation. "
        "Common side effects include stomach upset, dizziness, "
        "and rash, which usually resolve after stopping the drug.\n\n"
        "In rare cases, ibuprofen may cause severe allergic "
        "reactions or liver/kidney dysfunction, requiring "
        "immediate medical attention.\n\n"
        "This is an over-the-counter medicine. Please follow the "
        "instructions or consult a pharmacist before use."
    )

    print("=== Markdown ===")
    print(segment_document(sample_md))
    print()
    print("=== Plain text ===")
    print(segment_document(sample_plain))
    print()

    # Batch example
    print("=== Batch (2 workers) ===")
    results = batch_segment_documents(
        [sample_md, sample_plain], num_workers=2
    )
    single = [segment_document(sample_md), segment_document(sample_plain)]
    assert results == single, "Batch result mismatch!"
    print(f"Batch processed {len(results)} docs, results match single.")
