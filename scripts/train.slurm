#!/bin/bash
#SBATCH -A iscrb_wearusfm
#SBATCH -p boost_usr_prod
#SBATCH --time 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --job-name=ecowitness_train
#SBATCH --output=logs/train_%j.out
# Usage (from $HOME/ecowitness, normally via scripts/run_all.sh):
#   CONFIG=configs/base.yaml RUN=base sbatch scripts/train.slurm
set -euo pipefail
cd "${CODE_DIR:-$SLURM_SUBMIT_DIR}"
source scripts/cineca_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
RUN=${RUN:-base}
CONFIG=${CONFIG:-configs/base.yaml}
OUT=$RUNS_DIR/$RUN
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
mkdir -p "$OUT" logs
echo "model=$MODEL_DIR data=$DATA_DIR images=$WORKSPACE config=$CONFIG out=$OUT gpus=$NGPU"
echo "code commit: $(git rev-parse --short HEAD 2>/dev/null) $(git status --porcelain 2>/dev/null | grep -q . && echo "(uncommitted changes)")"
nvidia-smi -L
$EW_TORCHRUN --nproc_per_node="$NGPU" --master_port=$((29000 + SLURM_JOB_ID % 1000)) \
  -m ecowitness.train --config "$CONFIG" --model "$MODEL_DIR" --data_dir "$DATA_DIR" --workspace "$WORKSPACE" --output_dir "$OUT" ${TRAIN_EXTRA:-}
echo "TRAIN DONE $OUT/adapter"
