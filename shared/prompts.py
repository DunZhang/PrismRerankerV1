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
_chat_template = _env.get_template("reranker_chat.j2")

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = (
    "Judge whether the Document meets the requirements based on "
    "the Query and the Instruct provided. Note that the answer "
    'can only be "yes" or "no".'
)

SUFFIX: str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

DEFAULT_EVAL_INSTRUCTION: str = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

TRAINING_INSTRUCTION: str = (
    "Given a query and a document, determine whether "
    "the document directly and specifically answers or addresses "
    'the query. After providing a "yes" or "no", then provide '
    "the following in XML format:\n"
    "1. The aspects of the Document that help answer the query.\n"
    "2. An evidence statement based on the above, "
    "integrated and rewritten to be understandable even without "
    "the Document. Please retain original phrasing as much as "
    "possible."
)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_raw_prompt(
    query: str,
    doc: str,
    instruction: str = TRAINING_INSTRUCTION,
) -> str:
    """Render the full raw reranker prompt (no chat template needed)."""
    return _raw_template.render(
        system_prompt=SYSTEM_PROMPT,
        instruction=instruction,
        query=query,
        doc=doc,
    )


def render_chat_user_content(
    query: str,
    doc: str,
    instruction: str = DEFAULT_EVAL_INSTRUCTION,
) -> str:
    """Render the user-role content for chat-template-based models."""
    return _chat_template.render(
        instruction=instruction,
        query=query,
        doc=doc,
    )


def build_chat_messages(
    query: str,
    doc: str,
    instruction: str = DEFAULT_EVAL_INSTRUCTION,
) -> list[dict[str, str]]:
    """Build the chat-template messages for one reranker prompt."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": render_chat_user_content(
                query,
                doc,
                instruction=instruction,
            ),
        },
    ]


__all__ = [
    "SYSTEM_PROMPT",
    "SUFFIX",
    "DEFAULT_EVAL_INSTRUCTION",
    "TRAINING_INSTRUCTION",
    "render_raw_prompt",
    "render_chat_user_content",
    "build_chat_messages",
]
