"""合并 LoRA adapter 到基础模型并保存为完整模型。"""

import shutil

import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM

SKIP_PATTERNS = {"LICENSE", "README.md", "model.safetensors"}


def _should_skip(filename: str) -> bool:
    """判断是否跳过该文件（模型权重、LICENSE、README）。"""
    if filename in SKIP_PATTERNS:
        return True
    if filename.startswith("model.safetensors"):
        return True
    return False


def merge_and_save(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
) -> None:
    """加载基础模型和 LoRA adapter，合并后保存。"""
    base_dir = Path(base_model_path)
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

    print("Copying auxiliary files from base model...")
    for src_file in base_dir.iterdir():
        if src_file.is_file() and not _should_skip(src_file.name):
            dst_file = output_dir / src_file.name
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)
                print(f"  Copied: {src_file.name}")

    print("Done!")


if __name__ == "__main__":
    merge_and_save(
        base_model_path="/mnt/data/public_models/Qwen3.5-4B",
        adapter_path=f"/mnt/data/train_output/Prism-Qwen3.5-Reranker-4B/samples-31606-epoch-1",
        # output_path="/mnt/data/train_output/test1_baseline/samples-400_megred_lora",
        output_path=f"/root/prism_released_models/Prism-Qwen3.5-Reranker-4B",
    )
