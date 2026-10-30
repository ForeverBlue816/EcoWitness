#!/bin/bash
# End-to-end dry run with the 8 MB tiny-random Qwen3.5 on the LOGIN node CPU (about two minutes): data loading,
# collator, LoRA, 3 optimisation steps, evaluation on a few items. Run after setup_env.sh and download_model.sh.
#   bash scripts/smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cineca_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=$RUNS_DIR/smoke
python -m ecowitness.train --model "$TINY_DIR" --data_dir "$DATA_DIR" --workspace "$WORKSPACE" --splits val --langs en --max_items 6 \
  --epochs 1 --batch_size 2 --grad_accum 1 --bf16 false --gradient_checkpointing false --num_workers 0 --output_dir "$OUT" \
  --lora_r 4 --lora_alpha 8 --logging_steps 1 --save_steps 1000
python -m ecowitness.evaluate --model "$TINY_DIR" --adapter "$OUT/adapter" --data_dir "$DATA_DIR" --workspace "$WORKSPACE" --split val \
  --langs en --modes base revise chain --max_items 3 --batch_size 2 --max_new_tokens 8 --device cpu --dtype fp32 --out "$OUT/eval"
echo "SMOKE TEST OK"
