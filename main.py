QUERY = "What is the boiling point of water at sea level?"
DOCUMENTS = [
    "Water boils at 100 C (212 F) at standard atmospheric pressure (1 atm), "
    "which corresponds to sea-level conditions.",
    "Mount Everest is the highest mountain on Earth, with a peak elevation "
    "of 8,848 meters above sea level.",
]

import torch
from sentence_transformers import CrossEncoder

MODEL_PATH = "/mnt/g/prism_released_models/Prism-Qwen3.5-Reranker-0.8B"  # or any sibling repo above

ce = CrossEncoder(MODEL_PATH, model_kwargs={"torch_dtype": torch.bfloat16})

# 1) Score (q, d) pairs. The default activation is Sigmoid, so scores are in (0, 1)
# and equal to s(q, d) = sigmoid(logit_yes - logit_no) — identical to path A above.
pairs = [(QUERY, doc) for doc in DOCUMENTS]
scores = ce.predict(pairs)
print(scores)
# array([0.98, 0.01], dtype=float32)

# 2) Rank documents directly.
ranked = ce.rank(QUERY, DOCUMENTS, return_documents=True)
for r in ranked:
    print(f"{r['score']:.3f}\t{r['corpus_id']}\t{r['text'][:80]}")

#######################################################################
#######################################################################
#######################################################################
#######################################################################
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    "- Output language: multilingual doc → query's language; else doc's language."
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


for doc in DOCUMENTS:
    print(rerank(QUERY, doc))