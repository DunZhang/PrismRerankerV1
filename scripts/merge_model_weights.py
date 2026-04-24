"""模型权重融合（Weight Interpolation）

θ_final = α * θ_SFT + (1 - α) * θ_base
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_weights(
    sft_model_path: str,
    base_model_path: str,
    output_path: str,
    alpha: float = 0.7,
) -> None:
    print(f"Loading SFT model from {sft_model_path}")
    sft_model = AutoModelForCausalLM.from_pretrained(
        sft_model_path, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    print(f"Loading base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    sft_sd = sft_model.state_dict()
    base_sd = base_model.state_dict()

    assert sft_sd.keys() == base_sd.keys(), "Model architectures do not match"

    merged_sd: dict[str, torch.Tensor] = {}
    for key in sft_sd:
        sft_t = sft_sd[key]
        base_t = base_sd[key]
        if not sft_t.is_floating_point():
            merged_sd[key] = sft_t
            continue
        if sft_t.shape == base_t.shape:
            merged_sd[key] = alpha * sft_t + (1 - alpha) * base_t
        else:
            min_slices = tuple(
                slice(0, min(s, b)) for s, b in zip(sft_t.shape, base_t.shape)
            )
            merged = sft_t.clone()
            merged[min_slices] = (
                alpha * sft_t[min_slices] + (1 - alpha) * base_t[min_slices]
            )
            merged_sd[key] = merged
            print(
                f"  Shape mismatch on {key}: "
                f"SFT={list(sft_t.shape)} Base={list(base_t.shape)}, "
                f"merged common region, kept SFT remainder"
            )

    print(f"Merged {len(merged_sd)} parameters with alpha={alpha}")

    sft_model.load_state_dict(merged_sd)

    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model to {output_path}")
    sft_model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    tokenizer.save_pretrained(output_path)
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear weight interpolation")
    parser.add_argument("--sft_model", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.7)
    args = parser.parse_args()

    merge_weights(args.sft_model, args.base_model, args.output, args.alpha)
