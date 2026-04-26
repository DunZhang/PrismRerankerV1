"""Shared prompt constants and rendering for Qwen3-Reranker family models.

All prompt templates are managed as Jinja2 ``.j2`` files under
``shared/templates/``.  This module loads them once and exposes thin
rendering and message-building helpers so that neither ``train`` nor
``rank_evaluate`` need to know about the template engine.
"""

from __future__ import annotations

from jinja2 import Environment, PackageLoader

# ---------------------------------------------------------------------------
# Jinja2 environment – loads templates from shared/templates/
# ---------------------------------------------------------------------------
_env = Environment(
    loader=PackageLoader("shared", "templates"),
    keep_trailing_newline=True,
    autoescape=False,
)

_raw_template = _env.get_template("reranker_raw.j2")

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
ORIGINAL_SYSTEM_PROMPT: str = (
    "Judge whether the Document meets the requirements based on "
    "the Query and the Instruct provided. Note that the answer "
    'can only be "yes" or "no".'
)

TRAINING_SYSTEM_PROMPT: str = (
    "Judge whether the Document meets the requirements based on "
    "the Query and the Instruct provided. "
)

DEFAULT_EVAL_INSTRUCTION: str = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

TRAINING_INSTRUCTION: str = (
    'Judge if the document is relevant to the query. Reply "yes" or "no".\n'
    'On "yes", also emit:\n'
    "<contribution>One sentence covering every core point the document "
    "contributes to the query, without elaboration.</contribution>\n"
    "<evidence>Self-contained rewrite of the query-relevant content. Rules:\n"
    "- Faithful: rephrase only; add or infer nothing.\n"
    "- Self-contained: evidence alone must fully answer the query.\n"
    "- Concise: drop query-irrelevant background.\n"
    "- Verbatim (no translation): proper nouns, terms, abbreviations, "
    "numbers, dates, code, URLs.\n"
    "- Output language: multilingual doc → query's language; else doc's language."
    "</evidence>"
)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_raw_prompt(
    query: str,
    doc: str,
    instruction: str = TRAINING_INSTRUCTION,
    system_prompt: str = TRAINING_SYSTEM_PROMPT,
) -> str:
    """Render the full raw reranker prompt (no chat template needed)."""
    return _raw_template.render(
        system_prompt=system_prompt,
        instruction=instruction,
        query=query,
        doc=doc,
    )


__all__ = [
    "ORIGINAL_SYSTEM_PROMPT",
    "TRAINING_SYSTEM_PROMPT",
    "DEFAULT_EVAL_INSTRUCTION",
    "TRAINING_INSTRUCTION",
    "render_raw_prompt",
]
