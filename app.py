"""Minimal Streamlit UI for the Prism reranker.

Usage:
    uv run streamlit run app.py
"""

from __future__ import annotations

import html
from threading import Thread
from typing import Iterator

import streamlit as st
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

MODEL_PATH = "/mnt/g/prism_released_models/Prism-Qwen3.5-Reranker-4B/"

SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on "
    "the Query and the Instruct provided. "
)

INSTRUCTION = (
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
    "- Output language: multilingual doc -> query's language; else doc's language."
    "</evidence>"
)

PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n"
    "<Instruct>: {instruction}\n"
    "<Query>: {query}\n"
    "<Document>: {doc}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def build_prompt(query: str, doc: str) -> str:
    return PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT, instruction=INSTRUCTION, query=query, doc=doc
    )


@st.cache_resource(show_spinner=False)
def load_model() -> tuple[AutoTokenizer, AutoModelForCausalLM, int, int]:
    """Load tokenizer + model once and cache across reruns."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("no", add_special_tokens=False)[0]
    return tokenizer, model, yes_id, no_id


@torch.no_grad()
def compute_score(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    yes_id: int,
    no_id: int,
) -> float:
    """Score = softmax over {yes, no} at the first generated-token position."""
    logits = model(input_ids=input_ids).logits[0, -1].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    yes_p = logprobs[yes_id].exp()
    no_p = logprobs[no_id].exp()
    return (yes_p / (yes_p + no_p)).item()


def stream_generate(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int = 4096,
) -> Iterator[str]:
    """Yield decoded text chunks as the model generates."""
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    thread = Thread(target=model.generate, kwargs=kwargs)
    thread.start()
    try:
        for chunk in streamer:
            yield chunk
    finally:
        thread.join()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Prism Reranker", page_icon="✦", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #fbfaf7;
    --surface: #ffffff;
    --border: #e8e6e0;
    --text: #1c1c1c;
    --muted: #6b6b6b;
    --accent: #2d2d2d;
    --good: #2f7a4d;
    --warn: #b07a1a;
    --bad:  #b34646;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, system-ui, sans-serif;
    color: var(--text);
}

.stApp { background: var(--bg); }

/* hide default streamlit chrome */
header, footer, #MainMenu { visibility: hidden; }
.block-container { padding-top: 3rem; padding-bottom: 3rem; max-width: 1920px; }

.title-wrap { text-align: center; margin-bottom: 2.5rem; }
.title {
    font-family: 'Inter', sans-serif;
    font-size: 2.6rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--text);
    margin: 0;
}
.subtitle {
    color: var(--muted);
    font-size: 1.05rem;
    font-weight: 400;
    margin-top: 0.5rem;
}

/* card */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.5rem 1.75rem;
}

/* labels */
.field-label {
    color: var(--muted);
    font-size: 0.95rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
}

/* text areas */
.stTextArea textarea {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1.05rem !important;
    line-height: 1.6 !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
    box-shadow: none !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(0,0,0,0.05) !important;
}

/* button */
div.stButton { margin-top: 0.5rem; }
div.stButton > button {
    width: 100%;
    background: var(--accent);
    color: #fff;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    letter-spacing: 0.01em;
    font-size: 1.1rem;
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: 0.85rem 1rem;
    transition: background 0.15s ease, transform 0.05s ease;
    box-shadow: none;
}
div.stButton > button:hover { background: #000; border-color: #000; }
div.stButton > button:active { transform: translateY(1px); }

/* score */
.score-card {
    text-align: center;
    margin-top: 1.75rem;
}
.score-label {
    color: var(--muted);
    font-size: 0.95rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.score-num {
    font-family: 'Inter', sans-serif;
    font-size: 4.2rem;
    font-weight: 600;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-top: 0.6rem;
    font-variant-numeric: tabular-nums;
}
.score-bar {
    height: 4px;
    border-radius: 2px;
    overflow: hidden;
    background: #efece5;
    margin-top: 1.1rem;
}
.score-bar-fill {
    height: 100%;
    transition: width 0.6s ease;
}
.verdict {
    font-size: 1.05rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    margin-top: 1rem;
}

/* stream */
.stream-head {
    color: var(--muted);
    font-size: 0.95rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 2.2rem 0 0.7rem 0;
}
.stream-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.5rem 1.7rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text);
    font-size: 1.02rem;
    line-height: 1.75;
    min-height: 220px;
    white-space: pre-wrap;
    word-break: break-word;
}
.xml-tag { color: #8a6dc1; font-weight: 500; }

.cursor {
    display: inline-block;
    width: 7px; height: 1em;
    background: var(--text);
    margin-left: 2px;
    vertical-align: text-bottom;
    animation: blink 1s step-end infinite;
    opacity: 0.7;
}
@keyframes blink { 50% { opacity: 0; } }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="title-wrap">'
    '<div class="title">Prism Reranker</div>'
    '<div class="subtitle">Score, contribution, and evidence — in one pass.</div>'
    "</div>",
    unsafe_allow_html=True,
)

with st.spinner("Loading model..."):
    tokenizer, model, yes_id, no_id = load_model()

st.markdown('<div class="field-label">Query</div>', unsafe_allow_html=True)
query = st.text_area(
    "query",
    height=140,
    label_visibility="collapsed",
    placeholder="What is the boiling point of water at sea level?",
)

st.markdown('<div class="field-label">Document</div>', unsafe_allow_html=True)
doc = st.text_area(
    "document",
    height=460,
    label_visibility="collapsed",
    placeholder="Paste your document here. Long documents are fine — the model is trained on inputs up to 10K tokens.",
)

st.write("")
submit = st.button("Submit")


def colorize_stream(text: str) -> str:
    """Escape text and recolor reranker-specific tags."""
    safe = html.escape(text)
    safe = safe.replace(
        "&lt;contribution&gt;",
        '<span class="xml-tag">&lt;contribution&gt;</span>',
    )
    safe = safe.replace(
        "&lt;/contribution&gt;",
        '<span class="xml-tag">&lt;/contribution&gt;</span>',
    )
    safe = safe.replace(
        "&lt;evidence&gt;",
        '<span class="xml-tag">&lt;evidence&gt;</span>',
    )
    safe = safe.replace(
        "&lt;/evidence&gt;",
        '<span class="xml-tag">&lt;/evidence&gt;</span>',
    )
    return safe


def score_color(score: float) -> str:
    if score >= 0.7:
        return "#2f7a4d"
    if score >= 0.4:
        return "#b07a1a"
    return "#b34646"


def verdict(score: float) -> str:
    if score >= 0.7:
        return "Highly relevant"
    if score >= 0.4:
        return "Partial match"
    return "Low relevance"


if submit:
    if not query.strip() or not doc.strip():
        st.error("Both query and document are required.")
    else:
        prompt = build_prompt(query, doc)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        score_slot = st.empty()
        with st.spinner("Computing score..."):
            score = compute_score(model, input_ids, yes_id, no_id)

        c = score_color(score)
        score_slot.markdown(
            f"""
<div class="card score-card">
  <div class="score-label">Relevance score</div>
  <div class="score-num" style="color:{c};">{score:.4f}</div>
  <div class="score-bar">
    <div class="score-bar-fill" style="width:{score * 100:.2f}%; background:{c};"></div>
  </div>
  <div class="verdict" style="color:{c};">{verdict(score)}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="stream-head">Contribution &amp; Evidence</div>',
            unsafe_allow_html=True,
        )

        text_slot = st.empty()
        accumulated = ""
        for chunk in stream_generate(tokenizer, model, input_ids):
            accumulated += chunk
            text_slot.markdown(
                f'<div class="stream-box">{colorize_stream(accumulated)}'
                '<span class="cursor"></span></div>',
                unsafe_allow_html=True,
            )

        text_slot.markdown(
            f'<div class="stream-box">{colorize_stream(accumulated)}</div>',
            unsafe_allow_html=True,
        )
