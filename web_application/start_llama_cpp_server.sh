#!/usr/bin/env bash
# Start an OpenAI-compatible GGUF server via the native llama.cpp llama-server.
#
# First-time setup (build llama-server with CUDA, run once):
#   git clone --depth 1 https://github.com/ggml-org/llama.cpp.git vendor/llama.cpp
#   cmake -B vendor/llama.cpp/build -S vendor/llama.cpp \
#       -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
#   cmake --build vendor/llama.cpp/build --config Release -j --target llama-server
#
# Usage:
#   bash web_application/start_llama_cpp_server.sh [model.gguf] [port] [gpu_layers] [ctx]

set -e

MODEL="${1:-/mnt/g/prism_released_models/Prism-Qwen3.5-Reranker-4B/Prism-Qwen3.5-Reranker-4B.Q8_0.gguf}"
PORT="${2:-54580}"
N_GPU_LAYERS="${3:-99}"
N_CTX="${4:-20480}"

cd "$(dirname "$0")/.."

LLAMA_SERVER_BIN="vendor/llama.cpp/build/bin/llama-server"

exec "$LLAMA_SERVER_BIN" \
    -m             "$MODEL" \
    --host         "127.0.0.1" \
    --port         "$PORT" \
    -ngl           "$N_GPU_LAYERS" \
    -c             "$N_CTX" \
    --alias        "prism-gguf" \
    --cache-reuse  256 \
    -fa            on
