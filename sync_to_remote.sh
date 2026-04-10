#!/bin/bash
# 实时同步项目到远程服务器
# 用法: bash sync_to_remote.sh

set -euo pipefail

# ============ 配置 ============
REMOTE_HOST="guangdong-b-is.cloud.infini-ai.com"
# REMOTE_PORT="42541"
REMOTE_PORT="43242"
REMOTE_USER="root"
REMOTE_PATH="/mnt/data/codes/PrismRerankerV1/"
LOCAL_PATH="/mnt/d/Codes/PrismRerankerV1/"
KEY_SRC="/mnt/d/keys/ed25519_np/id_ed25519"
KEY_TMP="/tmp/id_ed25519_sync"
INTERVAL=0.5  # 轮询间隔（秒）
# ==============================

# 复制私钥并修正权限（WSL2 挂载盘权限为 0777，SSH 拒绝使用）
cp "$KEY_SRC" "$KEY_TMP"
chmod 600 "$KEY_TMP"

SSH_CMD="ssh -i $KEY_TMP -p $REMOTE_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10"

RSYNC_EXCLUDES=(
    --exclude='.venv/'
    --exclude='__pycache__/'
    --exclude='.git/'
    --exclude='.ruff_cache/'
    --exclude='.mypy_cache/'
    --exclude='.claude/'
    --exclude='*.pyc'
    --exclude='*.egg-info/'
    --exclude='.pixi/'
    --exclude='*.jsonl'
    --exclude='*.json'
)

do_sync() {
    rsync -aiz \
        "${RSYNC_EXCLUDES[@]}" \
        -e "$SSH_CMD" \
        "$LOCAL_PATH" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
}

# 首次全量同步
echo "[sync] 首次全量同步..."
do_sync
echo "[sync] 首次同步完成，开始监听变更（每 ${INTERVAL}s 轮询）"
echo "[sync] 按 Ctrl+C 停止"

# 持续轮询
while true; do
    sleep "$INTERVAL"
    if output=$(do_sync 2>/dev/null); then
        [ -n "$output" ] && echo "[sync] $(date '+%H:%M:%S')" && echo "$output"
    else
        echo "[sync] $(date '+%H:%M:%S') 同步失败，将在下次重试"
    fi
done
