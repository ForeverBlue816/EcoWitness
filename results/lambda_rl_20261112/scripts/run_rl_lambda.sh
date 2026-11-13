#!/bin/bash
# RL training (configs/rl.yaml) on 2 x H100, then the test-set evaluation of the adapter. No zero-shot evaluation.
# The commands of repo/scripts/train.slurm and eval.slurm, without Slurm / modules.
#   tmux new -d -s ew_rl "bash /lambda/nfs/ICLR/ecowitness/run_rl_lambda.sh"          (add RESUME=1 to continue a stopped run)
# Differences to the 4 x A100 run, both because of the machine:
#   --prompts_per_step 16   2 GPUs x 16 prompts x grad_accum 2 = 64 prompts per update, 2,500 steps, 1,250 updates:
#                           the same step and update structure as 4 GPUs x 8 x 2 (user 2026-11-12: larger batch for speed)
#   --gen_token_budget 64000, --train_token_budget 16000   both doubled with the prompts per GPU, so a step samples in as few
#                           chunks, and trains as many candidates per prompt, as before (H100 has 80 GB)
#   --max_hours 0    no 24-hour job limit here, so the run is not cut at 23.3 h
set -euo pipefail
source /lambda/nfs/ICLR/ecowitness/lambda_env.sh
cd "$CODE_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
RUN=${RUN:-rl}; CONFIG=${CONFIG:-configs/rl.yaml}; OUT=$RUNS_DIR/$RUN; LOG=$EW_ROOT/logs
mkdir -p "$OUT" "$LOG"
python - <<PY
import json
r = json.loads(open('$DATA_DIR/train.en.jsonl').readline())
assert 'answer_reward' in r, 'train.en.jsonl has no answer_reward: this data was built before the RL commit (b00ba9d)'
PY
echo "code commit: $(git rev-parse --short HEAD)  config=$CONFIG  out=$OUT"; nvidia-smi -L
if [ ! -f "$OUT/adapter/adapter_model.safetensors" ] || [ "${RESUME:-0}" = "1" ]; then
  $EW_TORCHRUN --nproc_per_node=2 --master_port=29417 -m ecowitness.rl --config "$CONFIG" --model "$MODEL_DIR" --data_dir "$DATA_DIR" \
    --workspace "$WORKSPACE" --output_dir "$OUT" --prompts_per_step 16 --grad_accum 2 --gen_token_budget 64000 --train_token_budget 16000 --max_hours 0 $([ "${RESUME:-0}" = "1" ] && echo "--resume true") ${TRAIN_EXTRA:-} 2>&1 | tee -a "$LOG/train_$RUN.out"
fi
echo "TRAIN DONE $OUT/adapter"
CUDA_VISIBLE_DEVICES=0 python -m ecowitness.evaluate --model "$MODEL_DIR" --adapter "$OUT/adapter" --data_dir "$DATA_DIR" --workspace "$WORKSPACE" \
  --split test --langs en --modes base silent revise chain --out "$OUT/eval_test" --batch_size "${EVAL_BS:-8}" ${EVAL_EXTRA:-} 2>&1 | tee -a "$LOG/eval_$RUN.out"
echo "EVAL DONE $OUT/eval_test/metrics.json"
