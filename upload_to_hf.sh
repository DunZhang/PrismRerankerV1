#!/usr/bin/env bash
# Upload CrossEncoder integration files to all 5 Prism reranker HF repos.
# Run after `hf auth login`.
set -euo pipefail

BASE=/mnt/g/prism_released_models

REPOS=(
  "infgrad/Prism-Qwen3.5-Reranker-0.8B:Prism-Qwen3.5-Reranker-0.8B"
  "infgrad/Prism-Qwen3.5-Reranker-2B:Prism-Qwen3.5-Reranker-2B"
  "infgrad/Prism-Qwen3.5-Reranker-4B:Prism-Qwen3.5-Reranker-4B"
  "infgrad/Prism-Qwen3.5-Reranker-9B:Prism-Qwen3.5-Reranker-9B"
  "infgrad/Prism-Qwen3-Reranker-4B-exp:Prism-Qwen3-Reranker-4B-exp"
)

INCLUDES=(
  --include "README.md"
  --include "chat_template.jinja"
  --include "modules.json"
  --include "sentence_bert_config.json"
  --include "config_sentence_transformers.json"
  --include "1_LogitScore/config.json"
)

for entry in "${REPOS[@]}"; do
  repo_id="${entry%%:*}"
  dir_name="${entry#*:}"
  local_path="$BASE/$dir_name"

  echo ""
  echo "=================================================================="
  echo "[$(date +%H:%M:%S)] Uploading -> $repo_id"
  echo "      from $local_path"
  echo "=================================================================="

  hf upload "$repo_id" "$local_path" \
    "${INCLUDES[@]}" \
    --commit-message "Add CrossEncoder integration"

  echo "[$(date +%H:%M:%S)] Done: $repo_id"
done

echo ""
echo "All 5 repos uploaded."
