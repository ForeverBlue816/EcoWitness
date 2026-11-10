#!/bin/bash
# One-click pipeline on Leonardo (run from $HOME/ecowitness): checks -> train -> eval, chained with SLURM dependencies.
#   bash scripts/run_all.sh                    # main run (configs/rl.yaml, verifier-guided RL) + zero-shot baseline evaluation
#   RUN=equivariance CONFIG=configs/equivariance.yaml bash scripts/run_all.sh    # supervised counterfactual-equivariance baseline
#   RUN=base CONFIG=configs/base.yaml bash scripts/run_all.sh     # previous method: SFT + isolation loss
#   bash scripts/run_all.sh zeroshot           # only the zero-shot evaluation
#   bash scripts/run_all.sh ablations          # every config in configs/ (each train + eval)
#   TRAIN_EXTRA="--max_items 2000 --epochs 1" RUN=trial bash scripts/run_all.sh    # small trial run
set -euo pipefail
cd "$(dirname "$0")/.."
EW_VARS_ONLY=1 source scripts/cineca_env.sh
export CODE_DIR=$PWD
mkdir -p logs "$RUNS_DIR"

[ -f "$VENV/bin/activate" ] || { echo "venv missing: bash scripts/setup_env.sh"; exit 1; }
[ -f "$MODEL_DIR/config.json" ] || { echo "model missing at $MODEL_DIR: bash scripts/download_model.sh"; exit 1; }
[ -d "$WORKSPACE/satellite_rgb" ] || { echo "satellite_rgb/ missing under $WORKSPACE (unzip ecowitness_data.zip there)"; exit 1; }
[ -f "$DATA_DIR/train.en.jsonl" ] || { echo "data missing at $DATA_DIR (unzip ecowitness_data.zip into $SCRATCH_ROOT)"; exit 1; }

submit_pair () {   # $1=run name  $2=config
  local run=$1 cfg=$2 tid eid
  tid=$(RUN=$run CONFIG=$cfg sbatch -A "$EW_ACCOUNT" -p "$EW_PARTITION" --parsable scripts/train.slurm)
  echo "train  $run  job $tid  ($cfg)"
  eid=$(RUN=$run sbatch -A "$EW_ACCOUNT" -p "$EW_PARTITION" --parsable --dependency=afterok:$tid scripts/eval.slurm)
  echo "eval   $run  job $eid  (after $tid)"
}

case "${1:-main}" in
  zeroshot)
    RUN=zeroshot ADAPTER=none sbatch -A "$EW_ACCOUNT" -p "$EW_PARTITION" scripts/eval.slurm ;;
  ablations)
    for cfg in configs/*.yaml; do submit_pair "$(basename "$cfg" .yaml)" "$cfg"; done ;;
  main|*)
    submit_pair "${RUN:-rl}" "${CONFIG:-configs/rl.yaml}"
    RUN=zeroshot ADAPTER=none sbatch -A "$EW_ACCOUNT" -p "$EW_PARTITION" scripts/eval.slurm ;;
esac
squeue -u "$USER"
echo "results: $RUNS_DIR/<run>/eval_test/metrics.json   (SCRATCH is purged after 40 days: copy adapters/metrics to \$HOME or \$FAST)"
