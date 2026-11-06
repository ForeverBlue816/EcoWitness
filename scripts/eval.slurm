#!/bin/bash
#SBATCH -A iscrb_wearusfm
#SBATCH -p boost_usr_prod
#SBATCH --time 06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=ecowitness_eval
#SBATCH --output=logs/eval_%j.out
# Usage:
#   RUN=base sbatch scripts/eval.slurm                   # evaluate $RUNS_DIR/base/adapter on test (en + zh)
#   RUN=zeroshot ADAPTER=none sbatch scripts/eval.slurm  # zero-shot base model
set -euo pipefail
cd "${CODE_DIR:-$SLURM_SUBMIT_DIR}"
source scripts/cineca_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN=${RUN:-base}
ADAPTER=${ADAPTER:-$RUNS_DIR/$RUN/adapter}
SPLIT=${SPLIT:-test}
LANGS=${LANGS:-en}
MODES=${MODES:-base silent revise chain}
mkdir -p logs "$RUNS_DIR/$RUN"
echo "code commit: $(git rev-parse --short HEAD 2>/dev/null)"
ADAPTER_ARG=""
if [ "$ADAPTER" != "none" ]; then ADAPTER_ARG="--adapter $ADAPTER"; fi
python -m ecowitness.evaluate --model "$MODEL_DIR" $ADAPTER_ARG --data_dir "$DATA_DIR" --workspace "$WORKSPACE" \
  --split "$SPLIT" --langs $LANGS --modes $MODES --out "$RUNS_DIR/$RUN/eval_${SPLIT}" --batch_size "${EVAL_BS:-8}" ${EVAL_EXTRA:-}
echo "EVAL DONE $RUNS_DIR/$RUN/eval_${SPLIT}/metrics.json"
