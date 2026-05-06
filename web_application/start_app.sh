#!/usr/bin/env bash
# Start the Prism Search Streamlit app and Cloudflare Tunnel.
#
# Usage:
#   bash web_application/start_app.sh [port]

PORT="${1:-8501}"

cd "$(dirname "$0")/.."

# Start Cloudflare Tunnel in background (force HTTP/2 to avoid QUIC instability on WSL2)
cloudflared tunnel  run prism-app &
TUNNEL_PID=$!

# On exit, kill the tunnel too
trap "kill $TUNNEL_PID 2>/dev/null" EXIT

exec uv run streamlit run web_application/app.py \
    --server.port "$PORT" \
    --server.headless true
