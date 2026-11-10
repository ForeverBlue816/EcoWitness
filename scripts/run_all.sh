#!/bin/bash
# One-click pipeline on Leonardo (run from $HOME/ecowitness): checks -> train -> eval, chained with SLURM dependencies.
#   bash scripts/run_all.sh                    # main run (configs/rl.yaml, verifier-guided RL) + zero-shot baseline evaluation
#   RUN=equivariance CONFIG=configs/equivariance.yaml bash scripts/run_all.sh    # supervised counterfactual-equivariance baseline
#   RUN=base CONFIG=configs/base.yaml bash scripts/run_all.sh     # previous method: SFT + isolation loss
#   bash scripts/run_all.sh zeroshot           # only the zero-shot evaluation
#   bash scripts/run_all.sh ablations          # every config in configs/ (each train + eval)
#   bash scripts/run_all.sh resume             # continue the RL run from its newest checkpoint (after a time-budget stop), then evaluate
#   bash scripts/run_all.sh trial              # 30-minute trial on the debug QOS: 1 node, 4 GPUs, 50 RL steps, no evaluation
#   CONFIG=configs/equivariance.yaml bash scripts/run_all.sh trial     # the same for a supervised config (2,000 groups)
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
  # RL runs stop by themselves after max_hours (configs/rl*.yaml, 23.3 h), write a checkpoint and exit normally
  local limit=24:00:00
  tid=$(RUN=$run CONFIG=$cfg TRAIN_EXTRA="${TRAIN_EXTRA:-}" sbatch -A "$EW_ACCOUNT" -p "$EW_PARTITION" --parsable --time="${EW_TRAIN_TIME:-$limit}" scripts/train.slurm)
  echo "train  $run  job $tid  ($cfg)"
  eid=$(RUN=$run sbatch -A "$EW_ACCOUNT" -p "$EW_PARTITION" --parsable --dependency=afterok:$tid scripts/eval.slurm)
  echo "eval   $run  job $eid  (after $tid)"
}

case "${1:-main}" in
  trial)
    # Speed / memory check, not a result: the debug QOS starts within minutes and stops the job after 30 minutes.
    # The log of the steps that ran is what matters; a time-out is fine.
    cfg=${CONFIG:-configs/rl.yaml}
    case "$(basename "$cfg")" in
      rl*) extra="--episodes 1600 --logging_steps 5 --save_steps 100000 --eval_steps 25 --val_items 48" ;;   # 50 steps on 4 GPUs
      *)   extra="--max_items 2000 --epochs 1 --logging_steps 5 --save_steps 100000" ;;
    esac
    tid=$(RUN=${RUN:-trial} CONFIG=$cfg TRAIN_EXTRA="${TRAIN_EXTRA:-$extra}" sbatch -A "$EW_ACCOUNT" -p "$EW_PARTITION" --parsable \
          --qos="${EW_TRIAL_QOS:-boost_qos_dbg}" --time=00:30:00 --nodes=1 --gres=gpu:4 --job-name=ecowitness_trial scripts/train.slurm)
    echo "trial  job $tid  ($cfg, 1 node x 4 GPUs, 30 min)   log: logs/train_$tid.out"
    squeue -u "$USER" --start
    exit 0 ;;
  resume)
    TRAIN_EXTRA="--resume true ${TRAIN_EXTRA:-}" submit_pair "${RUN:-rl}" "${CONFIG:-configs/rl.yaml}" ;;
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
