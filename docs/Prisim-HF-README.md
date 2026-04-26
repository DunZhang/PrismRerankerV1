---
license: mit
language:
- en
- zh
- multilingual
pipeline_tag: text-ranking
library_name: transformers
tags:
- reranker
- retrieval
- rag
- agentic-search
- qwen3.5
---

# Prism-Reranker

**Beyond Relevance Scoring — Jointly Producing Contributions and Evidence for Agentic Retrieval.**

A reranker family that, unlike standard rerankers that emit only a relevance score, returns three things in a single forward pass: a calibrated score, a one-sentence *contribution*, and a self-contained *evidence* passage extracted from the document.

![Model Architecture](./model_architecture.png)

## Released models

Five checkpoints are released on the Hugging Face Hub. Four are fine-tuned from the **Qwen3.5** backbone; one (`-4B-exp`) is an experimental extension built on top of **Qwen3-Reranker-4B**, demonstrating that the same recipe transfers to an existing LLM-based reranker without losing ranking quality.

| Model | Backbone | Parameters | Hugging Face |
|---|---|---|---|
| Prism-Qwen3.5-Reranker-0.8B | Qwen3.5 | 0.8B | [infgrad/Prism-Qwen3.5-Reranker-0.8B](https://huggingface.co/infgrad/Prism-Qwen3.5-Reranker-0.8B) |
| Prism-Qwen3.5-Reranker-2B   | Qwen3.5 | 2B   | [infgrad/Prism-Qwen3.5-Reranker-2B](https://huggingface.co/infgrad/Prism-Qwen3.5-Reranker-2B) |
| Prism-Qwen3.5-Reranker-4B   | Qwen3.5 | 4B   | [infgrad/Prism-Qwen3.5-Reranker-4B](https://huggingface.co/infgrad/Prism-Qwen3.5-Reranker-4B) |
| Prism-Qwen3.5-Reranker-9B   | Qwen3.5 | 9B   | [infgrad/Prism-Qwen3.5-Reranker-9B](https://huggingface.co/infgrad/Prism-Qwen3.5-Reranker-9B) |
| Prism-Qwen3-Reranker-4B-exp | Qwen3-Reranker-4B | 4B | [infgrad/Prism-Qwen3-Reranker-4B-exp](https://huggingface.co/infgrad/Prism-Qwen3-Reranker-4B-exp) |



## Why this model?

In agentic / RAG pipelines, a relevance score is rarely the end goal. After deciding a document is relevant, the agent still has to read it, denoise it, and decide what to do next. Prism-Reranker folds that work into the reranker itself:

- **Relevance score** — `s(q, d) = σ(ℓ_yes − ℓ_no) ∈ (0, 1)`. Calibrated, ranking-ready.
- **`<contribution>`** — one sentence stating *every* core point the document contributes to the query. Useful for the agent to plan its next step without re-reading the doc.
- **`<evidence>`** — a self-contained, faithfully-rephrased rewrite of the query-relevant content. Drops irrelevant background, preserves verbatim proper nouns / numbers / dates / code / URLs. You can feed `<evidence>` directly to a downstream LLM and skip the raw document — saving context tokens and removing web-noise.

If the document is not relevant, the model outputs `no` and stops. No contribution/evidence is generated.

## Highlights

- **Backbones**: Qwen3.5 series for the four main sizes, no architectural changes; one extension variant on top of Qwen3-Reranker-4B.
- **Context length**: training data capped at **10K tokens** per example, covering most real-world documents.
- **Multilingual**: Chinese / English primary; other languages supported but with less coverage.
- **Keyword-query robust**: agents often emit keyword-style queries instead of well-formed questions. ~30% of training queries were rewritten by an LLM into keyword form, so the model handles both natural and keyword queries.
- **Real-world data distribution**: in addition to open reranker datasets (MS MARCO, T2Ranking, MIRACL, …), training includes synthetic queries paired with real Tavily / Exa web-search results, matching what an actual agent sees at inference time.
- **Length × score balanced**: training data was rebalanced so that document length is not a relevance shortcut.
- **Training recipe**: distillation (point-wise MSE on a strong commercial reranker's scores) + SFT on `yes/no` + `<contribution>` + `<evidence>`, supervised by a 5-LLM-as-judge ensemble.

## Quickstart

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "infgrad/Prism-Qwen3.5-Reranker-4B"  # or any sibling repo above

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


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="sdpa",
).eval()

yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
no_id = tokenizer.encode("no", add_special_tokens=False)[0]


@torch.no_grad()
def rerank(query: str, doc: str, max_new_tokens: int = 512):
    prompt = build_prompt(query, doc)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    # Relevance score = softmax over {yes, no} at the first generated token.
    first_logprobs = torch.log_softmax(out.scores[0][0].float(), dim=-1)
    yes_p = first_logprobs[yes_id].exp()
    no_p = first_logprobs[no_id].exp()
    score = (yes_p / (yes_p + no_p)).item()

    # Decoded text holds yes/no plus <contribution>...</contribution><evidence>...</evidence>
    gen_ids = out.sequences[0, input_ids.shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {"score": score, "text": text}


example = rerank(
    query="What is the boiling point of water at sea level?",
    doc=(
        "Water boils at 100 C (212 F) at standard atmospheric pressure (1 atm), "
        "which corresponds to sea-level conditions."
    ),
)
print(example)
```

Expected shape of the output:

```text
{
  "score": 0.98,
  "text": "yes\n<contribution>...</contribution>\n<evidence>...</evidence>"
}
```

For irrelevant pairs the score is close to 0 and `text` is just `"no"`.


## Notes on usage

- The first generated token is always `yes` or `no` — the score is well-defined even if you stop generation immediately (cheap mode: `max_new_tokens=1`). Generate further only when you also want contribution/evidence.
- Inputs longer than 10K tokens may degrade — truncate the document side first.
- Greedy decoding is fine for ranking. For diverse evidence rephrasings, use `temperature=0.3-0.5`.



## Contact

Dun Zhang — `dunnzhang0@gmail.com` (independent researcher).
