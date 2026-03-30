"""Streamlit WebUI for Qwen3-Reranker inference.

Usage:
    uv run streamlit run webui.py
"""

from __future__ import annotations

import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared.prompts import (
    TRAINING_INSTRUCTION,
    TRAINING_SYSTEM_PROMPT,
    render_raw_prompt,
)

# ---------------------------------------------------------------------------
# Config — edit these before running
# ---------------------------------------------------------------------------
MODEL_PATH: str = "/mnt/g/train_output/v2_test/samples-4500-merged"
MAX_NEW_TOKENS: int = 512

YES_TOKEN_ID: int = 9693
NO_TOKEN_ID: int = 2152


# ---------------------------------------------------------------------------
# Model loading (cached, only runs once across all reruns)
# ---------------------------------------------------------------------------
@st.cache_resource
def _load_model() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    print(f"Loading model from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    print("Model loaded.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    query: str,
    doc: str,
    max_new_tokens: int,
) -> tuple[float, str]:
    """Run model inference, returning (score, generated_text)."""
    prompt = render_raw_prompt(
        query,
        doc,
        instruction=TRAINING_INSTRUCTION,
        system_prompt=TRAINING_SYSTEM_PROMPT,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    # --- Score: P(yes) via logits at the last prompt position ---
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    yes_no = logits[:, [YES_TOKEN_ID, NO_TOKEN_ID]]
    probs = torch.softmax(yes_no, dim=1)
    score = probs[0, 0].item()

    # --- Generate full text ---
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated_text = tokenizer.decode(
        gen_ids[0][input_len:], skip_special_tokens=True
    )

    return score, generated_text


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Reranker Inference", layout="wide")
    st.title("Qwen3-Reranker Inference")

    model, tokenizer = _load_model()

    # --- Input ---
    query = st.text_area("Query", height=80)
    doc = st.text_area("Document", height=200)

    if st.button("Run", type="primary"):
        if not query.strip() or not doc.strip():
            st.warning("Please enter both query and document.")
            return

        with st.spinner("Running inference..."):
            score, generated_text = run_inference(
                model, tokenizer, query, doc, MAX_NEW_TOKENS
            )

        # --- Output ---
        st.subheader("Score")
        st.metric(label="P(yes)", value=f"{score:.6f}")

        st.subheader("Generated Text")
        st.text_area("Output", value=generated_text, height=400, disabled=True)


if __name__ == "__main__":
    main()
