from __future__ import annotations

DEFAULT_LORA_TARGET_MODULES: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Token IDs are shared by Qwen3-Reranker-0.6B and 4B.
YES_TOKEN_ID: int = 9693
NO_TOKEN_ID: int = 2152

# This is the exact raw prompt expected by the model.
PROMPT_TEMPLATE: str = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on "
    "the Query and the Instruct provided. Note that the answer "
    'can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
    "<Instruct>: Given a query and a document, determine whether "
    "the document directly and specifically answers or addresses "
    'the query. After providing a "yes" or "no", then provide '
    "the following in XML format:\n"
    "1. The aspects of the Document that help answer the query.\n"
    "2. The specific segments of the Document (by ID) that are "
    "helpful for the query.\n"
    "3. An evidence statement based on the above segments, "
    "integrated and rewritten to be understandable even without "
    "the Document. Please retain original phrasing as much as "
    "possible.\n"
    "<Query>: {query}\n"
    "<Document>: {doc}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n"
    "</think>\n\n"
)


def build_prompt(query: str, doc: str) -> str:
    """Build the raw reranker prompt without any chat template."""
    return PROMPT_TEMPLATE.format(query=query, doc=doc)
