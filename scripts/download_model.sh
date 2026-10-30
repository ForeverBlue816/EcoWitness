#!/bin/bash
# Download the base model into $SCRATCH on a LOGIN node; jobs then run offline.
#   bash scripts/download_model.sh                       # Qwen/Qwen3.5-9B -> $SCRATCH/ecowitness/models/Qwen3.5-9B
#   MODEL_ID=Qwen/Qwen3.5-4B bash scripts/download_model.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cineca_env.sh
MODEL_ID=${MODEL_ID:-Qwen/Qwen3.5-9B}
MODEL_DIR=${MODEL_DIR_OVERRIDE:-$MODEL_ROOT/$(basename "$MODEL_ID")}
mkdir -p "$MODEL_DIR" "$TINY_DIR" "$HF_HOME"
python - <<EOF
from huggingface_hub import snapshot_download
print(snapshot_download('$MODEL_ID', local_dir='$MODEL_DIR'))          # ~19 GB for the 9B model, public, no token needed
print(snapshot_download('tiny-random/qwen3.5', local_dir='$TINY_DIR'))  # 8 MB debug model for the smoke test
EOF
du -sh "$MODEL_DIR" "$TINY_DIR"
