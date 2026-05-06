"""Prism Search — query + Tavily retrieval + Prism GGUF reranking.

Flow: user enters query + access key → validate key → Tavily search →
      Prism GGUF scores each result with streaming → display ranked results.

Usage:
    uv run streamlit run web_application/app.py
"""

from __future__ import annotations

import html
import itertools
import json
import math
import os
import random
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from filelock import FileLock
from openai import OpenAI
from tavily import TavilyClient

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from shared.prompts import render_raw_prompt  # noqa: E402

SERVER_URL = "http://localhost:54580/v1"
MODEL_NAME = "prism-gguf"
GENERATION_TEMPERATURE = 0.4
KEYS_DB_PATH = Path(__file__).parent / "keys_db.json"
KEYS_DB_LOCK = FileLock(str(KEYS_DB_PATH) + ".lock")
ENV_PATH = ROOT / ".env"
MAX_DOC_CHARS = 12_000
CANDIDATE_POOL_SIZE = 10
NUM_DOCS_TO_SCORE = 5
HF_COLLECTION = "https://huggingface.co/collections/infgrad/prism-reranker"
PAPER_URL = "https://arxiv.org/abs/2604.23734"
ARCH_IMAGE = Path(__file__).parent / "ma.png"


# ---------------------------------------------------------------------------
# Access key database (simple JSON: {"key": remaining_uses})
# ---------------------------------------------------------------------------


def load_keys_db() -> dict[str, int]:
    if not KEYS_DB_PATH.exists():
        return {}
    return json.loads(KEYS_DB_PATH.read_text())


def save_keys_db(db: dict[str, int]) -> None:
    KEYS_DB_PATH.write_text(json.dumps(db, indent=2))


def validate_and_consume_key(key: str) -> tuple[bool, str]:
    """Return (ok, message). Decrements remaining uses by 1 on success."""
    with KEYS_DB_LOCK:
        db = load_keys_db()
        if key not in db:
            return False, "Invalid access key."
        remaining = db[key]
        if remaining <= 0:
            return False, "Access key quota exhausted."
        db[key] = remaining - 1
        save_keys_db(db)
    return True, f"{remaining - 1} uses remaining"


# ---------------------------------------------------------------------------
# Tavily search (tries keys in order, skips exhausted ones)
# ---------------------------------------------------------------------------


def _load_tavily_keys() -> list[str]:
    load_dotenv(ENV_PATH)
    keys: list[str] = []
    i = 1
    while True:
        k = os.getenv(f"TAVILY_API_KEY_{i}")
        if k is None:
            break
        keys.append(k)
        i += 1
    return keys


def tavily_search(query: str) -> list[dict[str, Any]]:
    """Search via Tavily, cycling through keys until one succeeds."""
    keys = _load_tavily_keys()
    last_err: Exception | None = None
    for api_key in keys:
        try:
            client = TavilyClient(api_key)
            response = client.search(
                query=query,
                search_depth="basic",
                max_results=20,
                include_raw_content="markdown",
                chunks_per_source=1,
            )
            return response.get("results", [])
        except Exception as e:
            last_err = e
    raise RuntimeError(f"All Tavily keys failed. Last error: {last_err}")


# ---------------------------------------------------------------------------
# Prism GGUF scoring (via llama.cpp OpenAI-compatible server)
# ---------------------------------------------------------------------------


def _extract_score(content: list[dict[str, Any]]) -> float:
    for entry in content:
        yes_lp: float | None = None
        no_lp: float | None = None
        for cand in entry.get("top_logprobs") or []:
            tok = (cand.get("token") or "").strip().lower()
            if tok == "yes" and yes_lp is None:
                yes_lp = cand.get("logprob")
            elif tok == "no" and no_lp is None:
                no_lp = cand.get("logprob")
        if yes_lp is not None or no_lp is not None:
            lp_yes = yes_lp if yes_lp is not None else -100.0
            lp_no = no_lp if no_lp is not None else -100.0
            return 1.0 / (1.0 + math.exp(-(lp_yes - lp_no)))
    return 0.0


def score_and_generate(prompt: str, client: OpenAI) -> tuple[float, Iterator[str]]:
    """Score relevance and optionally stream contribution+evidence text."""
    resp1 = client.completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        max_tokens=1,
        logprobs=30,
        temperature=0.0,
        extra_body={"cache_prompt": True},
    )
    choice1 = resp1.choices[0]
    first_token: str = choice1.text
    raw_lp = choice1.logprobs
    content: list[dict[str, Any]] = []
    if raw_lp is not None:
        content = raw_lp.model_dump().get("content") or []
    score = _extract_score(content)

    if first_token.strip().lower() != "yes":
        return score, iter([first_token])

    stream = client.completions.create(
        model=MODEL_NAME,
        prompt=prompt + first_token,
        max_tokens=20480,
        temperature=GENERATION_TEMPERATURE,
        stream=True,
        extra_body={"cache_prompt": True},
    )

    def _chunks() -> Iterator[str]:
        for chunk in stream:
            text = chunk.choices[0].text
            if text:
                yield text

    return score, itertools.chain([first_token], _chunks())


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------


def score_color(score: float) -> str:
    if score >= 0.7:
        return "#2f7a4d"
    if score >= 0.4:
        return "#b07a1a"
    return "#b34646"


def colorize_tags(text: str) -> str:
    safe = html.escape(text)
    for tag in ("contribution", "evidence"):
        safe = safe.replace(
            f"&lt;{tag}&gt;", f'<span class="xml-tag">&lt;{tag}&gt;</span>'
        )
        safe = safe.replace(
            f"&lt;/{tag}&gt;", f'<span class="xml-tag">&lt;/{tag}&gt;</span>'
        )
    return safe


def render_result_card(
    rank: int,
    title: str,
    url: str,
    score: float,
    text: str,
    streaming: bool = False,
) -> str:
    c = score_color(score)
    safe_title = html.escape(title or url or "—")
    safe_url = html.escape(url or "#")
    cursor = '<span class="cursor"></span>' if streaming else ""
    text_section = (
        f'<div class="card-text">{colorize_tags(text)}{cursor}</div>'
        if text.strip()
        else ""
    )
    return f"""
<div class="result-card">
  <div class="card-header">
    <span class="rank-badge">#{rank}</span>
    <div class="card-title-wrap">
      <a class="card-title" href="{safe_url}" target="_blank">{safe_title}</a>
    </div>
    <span class="score-badge" style="color:{c};">{score:.3f}</span>
  </div>
  <div class="score-bar">
    <div class="score-bar-fill" style="width:{score * 100:.2f}%; background:{c};"></div>
  </div>
  {text_section}
</div>
"""


def render_pending_card(index: int, title: str, url: str) -> str:
    safe_title = html.escape(title or url or f"Document {index + 1}")
    safe_url = html.escape(url or "#")
    return f"""
<div class="result-card result-card-pending">
  <div class="card-header">
    <span class="rank-badge rank-pending">#{index + 1}</span>
    <div class="card-title-wrap">
      <a class="card-title" href="{safe_url}" target="_blank">{safe_title}</a>
    </div>
    <span class="score-badge" style="color:#ccc;">—</span>
  </div>
  <div class="score-bar">
    <div class="score-bar-fill" style="width:0%; background:#eee;"></div>
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# CSS + JS
# ---------------------------------------------------------------------------

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

header, footer, #MainMenu { visibility: hidden; }
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: min(1200px, 96vw);
    padding-left: max(1rem, 2vw) !important;
    padding-right: max(1rem, 2vw) !important;
}

/* title */
.title-wrap { text-align: center; margin-bottom: 1.5rem; }
.title {
    font-size: 2.2rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--text);
    margin: 0;
}
.subtitle {
    color: var(--muted);
    font-size: 1rem;
    font-weight: 400;
    margin-top: 0.4rem;
}
.hf-link {
    margin-top: 1.1rem;
}
.hf-link a {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.6rem 1.4rem;
    font-size: 1rem;
    font-weight: 600;
    color: #fff;
    background: linear-gradient(135deg, #ff9d3d 0%, #ff6b6b 100%);
    border-radius: 999px;
    text-decoration: none;
    letter-spacing: 0.01em;
    box-shadow: 0 4px 14px rgba(255, 107, 107, 0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
}
.hf-link a:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(255, 107, 107, 0.45);
    filter: brightness(1.05);
    text-decoration: none;
}
.hf-link a:active { transform: translateY(0); }
.paper-link-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.6rem 1.4rem;
    font-size: 1rem;
    font-weight: 600;
    color: var(--accent) !important;
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 999px !important;
    text-decoration: none !important;
    letter-spacing: 0.01em;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    background: none;
}
.paper-link-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(0,0,0,0.10) !important;
    border-color: #aaa !important;
    text-decoration: none !important;
}
.paper-link-btn:active { transform: translateY(0) !important; }

/* about section */
.about-section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.5rem;
    margin-bottom: 1.5rem;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem 2rem;
}
@media (max-width: 640px) {
    .about-section { grid-template-columns: 1fr; }
}
.about-en, .about-zh {
    font-size: 1.05rem;
    line-height: 1.7;
    color: var(--muted);
}
.about-en { border-right: 1px solid var(--border); padding-right: 1.5rem; }
@media (max-width: 640px) {
    .about-en { border-right: none; padding-right: 0;
                border-bottom: 1px solid var(--border); padding-bottom: 0.75rem; }
}
.about-en a, .about-zh a {
    color: var(--accent);
    text-decoration: none;
    font-weight: 500;
}
.about-en a:hover, .about-zh a:hover { text-decoration: underline; }
.about-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #aaa;
    margin-bottom: 0.35rem;
}

/* inputs */
.stTextInput input {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
    box-shadow: none !important;
}
.stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(0,0,0,0.05) !important;
}

/* button */
div.stButton > button {
    width: 100%;
    background: var(--accent);
    color: #fff;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 1rem;
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: 0.65rem 1rem;
    transition: background 0.15s ease, transform 0.05s ease;
    box-shadow: none;
}
div.stButton > button:hover { background: #000; border-color: #000; }
div.stButton > button:active { transform: translateY(1px); }

/* result cards */
.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.85rem;
    transition: border-color 0.2s ease;
}
.result-card:hover { border-color: #ccc; }
.result-card-pending { opacity: 0.45; }

.card-header {
    display: flex;
    align-items: center;
    gap: 0.65rem;
}

.rank-badge {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--muted);
    min-width: 1.6rem;
    flex-shrink: 0;
}
.rank-pending { color: #ccc; }

.card-title-wrap {
    flex: 1;
    min-width: 0;
}
.card-title {
    font-size: 0.92rem;
    font-weight: 500;
    color: var(--text);
    text-decoration: none;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    display: block;
}
.card-title:hover { text-decoration: underline; color: var(--accent); }

.score-badge {
    font-size: 1rem;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    flex-shrink: 0;
}

.score-bar {
    height: 3px;
    border-radius: 2px;
    overflow: hidden;
    background: #efece5;
    margin: 0.6rem 0 0 0;
}
.score-bar-fill {
    height: 100%;
    transition: width 0.5s ease;
}

.card-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.86rem;
    line-height: 1.65;
    color: var(--text);
    margin-top: 0.8rem;
    white-space: pre-wrap;
    word-break: break-word;
}
.xml-tag { color: #8a6dc1; font-weight: 500; }

/* model architecture image */
.arch-section {
    text-align: center;
    margin: 0 0 1.5rem 0;
}
.arch-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #aaa;
    margin-bottom: 0.6rem;
}

.cursor {
    display: inline-block;
    width: 6px; height: 1em;
    background: var(--text);
    margin-left: 2px;
    vertical-align: text-bottom;
    animation: blink 1s step-end infinite;
    opacity: 0.7;
}
@keyframes blink { 50% { opacity: 0; } }

.stAlert { border-radius: 8px !important; }
</style>

<script>
/* Mask the access key field without using type="password" (avoids password-manager popup).
   Uses -webkit-text-security for visual masking + a toggle eye button. */
(function() {
    function patchKeyInput() {
        document.querySelectorAll('input[placeholder="Access key"]').forEach(function(el) {
            if (el._patched) return;
            el._patched = true;
            el.setAttribute('autocomplete', 'off');
            el.setAttribute('data-lpignore', 'true');
            el.setAttribute('data-form-type', 'other');
            el.setAttribute('data-1p-ignore', 'true');
            el.style.webkitTextSecurity = 'disc';
            el.style.paddingRight = '2.5rem';

            var btn = document.createElement('button');
            btn.type = 'button';
            btn.title = 'Show / hide key';
            btn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>';
            btn.style.cssText = 'position:absolute;right:0.55rem;top:50%;transform:translateY(-50%);background:none;border:none;cursor:pointer;color:#888;padding:0.15rem;line-height:0;z-index:10;';

            var visible = false;
            btn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                visible = !visible;
                el.style.webkitTextSecurity = visible ? 'none' : 'disc';
                btn.style.color = visible ? '#333' : '#888';
            });

            var parent = el.parentElement;
            if (parent) {
                parent.style.position = 'relative';
                parent.appendChild(btn);
            }
        });
    }
    patchKeyInput();
    new MutationObserver(patchKeyInput).observe(document.body, {childList: true, subtree: true});
})();
</script>
"""

# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Prism Search", page_icon="✦", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    f'<div class="title-wrap">'
    f'<div class="title">Prism Search</div>'
    f'<div class="subtitle">目前为您服务的是 Prism-Qwen3.5-Reranker-4B.Q8_0.gguf 模型</div>'
    f'<div class="hf-link">'
    f'<a href="{HF_COLLECTION}" target="_blank">Open source on HuggingFace ↗</a>'
    f'&nbsp;&nbsp;'
    f'<a href="{PAPER_URL}" target="_blank" class="paper-link-btn">Technical Report ↗</a>'
    f'</div>'
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="about-section">
  <div>
    <div class="about-en">
      Prism Search retrieves live web pages via <strong>Tavily</strong>, then runs each
      result through <a href="{HF_COLLECTION}" target="_blank">Prism Reranker</a> —
      an open-source model that goes beyond a plain relevance score. In a single pass it
      produces: a calibrated <strong>relevance score</strong>, a one-sentence
      <strong>contribution</strong> summarising what the page adds to your query, and a
      self-contained <strong>evidence</strong> passage faithfully extracted from the
      document — ready to feed directly into a downstream LLM without the web noise.
      See the <a href="{PAPER_URL}" target="_blank">technical report</a> for details.
    </div>
  </div>
  <div>
    <div class="about-zh">
      Prism Search 通过 <strong>Tavily</strong> 实时抓取网页原文，再由
      <a href="{HF_COLLECTION}" target="_blank">Prism Reranker</a>
      对每篇文档进行深度分析。与只输出一个数字的传统 reranker 不同，
      Prism 在一次推理中同时返回三项结果：校准后的<strong>相关度评分</strong>、
      一句话<strong>贡献摘要</strong>（说明该页面对查询的核心价值），
      以及去除噪声后的<strong>证据段落</strong>（可直接送入下游 LLM，无需再读原文）。
      详见<a href="{PAPER_URL}" target="_blank">技术报告</a>。
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if ARCH_IMAGE.exists():
    st.markdown(
        '<div class="arch-section"><div class="arch-label">Model Architecture</div></div>',
        unsafe_allow_html=True,
    )
    arch_l, arch_c, arch_r = st.columns([1, 6, 1])
    with arch_c:
        st.image(str(ARCH_IMAGE), use_container_width=True)

with st.form("search_form"):
    query = st.text_input(
        "query",
        label_visibility="collapsed",
        placeholder="Enter your search query... / 输入搜索问题",
    )
    col_key, col_btn = st.columns([3, 1])
    with col_key:
        access_key = st.text_input(
            "key",
            label_visibility="collapsed",
            placeholder="Access key",
            type="password",
        )
    with col_btn:
        submit = st.form_submit_button("Search")

st.write("")

if submit:
    if not query.strip():
        st.error("Please enter a query.")
        st.stop()
    if not access_key.strip():
        st.error("Please enter your access key.")
        st.stop()

    ok, msg = validate_and_consume_key(access_key.strip())
    if not ok:
        st.error(msg)
        st.stop()

    st.caption(f"Key accepted — {msg}")

    status = st.empty()
    status.info("Searching the web via Tavily...")

    try:
        results = tavily_search(query.strip())
    except RuntimeError as e:
        status.error(str(e))
        st.stop()

    if not results:
        status.warning("No results returned by Tavily.")
        st.stop()

    pool = results[:CANDIDATE_POOL_SIZE]
    if len(pool) > NUM_DOCS_TO_SCORE:
        results = random.sample(pool, NUM_DOCS_TO_SCORE)
    else:
        results = pool

    status.info(
        f"Randomly picked {len(results)} documents to rerank. "
        "Starting Prism analysis..."
    )

    # Pre-render all cards as pending so the user sees the full list immediately
    slots = [st.empty() for _ in results]
    for i, doc in enumerate(results):
        title = doc.get("title") or doc.get("url") or f"Document {i + 1}"
        url = doc.get("url") or ""
        slots[i].markdown(render_pending_card(i, title, url), unsafe_allow_html=True)

    # Score each document with streaming
    try:
        prism_client = OpenAI(base_url=SERVER_URL, api_key="not-needed")
    except Exception as e:
        status.error(f"Cannot connect to Prism server at {SERVER_URL}: {e}")
        st.stop()

    result_data: list[dict[str, Any]] = []

    def render_final(slot: Any, rank: int, r: dict[str, Any]) -> None:
        with slot.container():
            st.markdown(
                render_result_card(rank, r["title"], r["url"], r["score"], r["text"]),
                unsafe_allow_html=True,
            )
            if r["raw_content"].strip():
                with st.expander("Show source"):
                    st.code(r["raw_content"], language=None, wrap_lines=True)

    for i, doc in enumerate(results):
        title = doc.get("title") or doc.get("url") or f"Document {i + 1}"
        url = doc.get("url") or ""
        raw_content = (doc.get("raw_content") or doc.get("content") or "")[
            :MAX_DOC_CHARS
        ]

        status.info(f"Analyzing {i + 1} / {len(results)} — {title[:70]}...")

        prompt = render_raw_prompt(query.strip(), raw_content)

        try:
            score, text_stream = score_and_generate(prompt, prism_client)
        except Exception as e:
            status.error(f"Prism server error on document {i + 1}: {e}")
            st.stop()

        accumulated = ""
        for chunk in text_stream:
            accumulated += chunk
            slots[i].markdown(
                render_result_card(
                    i + 1, title, url, score, accumulated, streaming=True
                ),
                unsafe_allow_html=True,
            )

        result_entry = {
            "title": title,
            "url": url,
            "score": score,
            "text": accumulated,
            "raw_content": raw_content,
        }
        render_final(slots[i], i + 1, result_entry)
        result_data.append(result_entry)

    # Re-render in descending score order
    sorted_results = sorted(result_data, key=lambda x: x["score"], reverse=True)
    for rank, (slot, r) in enumerate(zip(slots, sorted_results), 1):
        render_final(slot, rank, r)

    status.success(f"Done — ranked {len(sorted_results)} documents.")
