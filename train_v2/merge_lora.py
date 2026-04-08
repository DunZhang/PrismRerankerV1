"""合并 LoRA adapter 到基础模型并保存为完整模型。"""

import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_and_save(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
) -> None:
    """加载基础模型和 LoRA adapter，合并后保存。"""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model from: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path, safe_serialization=True)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)

    print("Done!")


if __name__ == "__main__":
    merge_and_save(
        base_model_path="/mnt/data/public_models/Qwen3-Reranker-0.6B",
        adapter_path="/mnt/data/train_output/sft_test_v1/samples-12000/",
        # output_path="/mnt/data/train_output/test1_baseline/samples-400_megred_lora",
        output_path="/root/samples-12000-merged",
    )
