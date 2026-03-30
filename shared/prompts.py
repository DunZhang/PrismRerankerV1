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
    "Given a query and a document, judge whether the document "
    'is relevant to the query. Answer "yes" or "no", '
    "then provide in XML:\n"
    "1. <contribution>: what the document contributes to the query.\n"
    "2. <evidence>: a self-contained rewrite of relevant content."
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
