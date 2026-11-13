#!/bin/bash
# Test-set evaluation of an adapter, split over the two GPUs by record type (the command of repo/scripts/eval.slurm, twice):
#   GPU 0: base + silent      GPU 1: revise + chain
# then the predictions are concatenated and scored once (ecowitness.score), which gives the same metrics.json as one run over
# all four modes (the silent behaviour metrics need the base predictions; both are in part A and in the merged file).
#   bash /lambda/nfs/ICLR/ecowitness/run_eval_lambda.sh part a|b      (one tmux session each)
#   bash /lambda/nfs/ICLR/ecowitness/run_eval_lambda.sh merge
set -euo pipefail
source /lambda/nfs/ICLR/ecowitness/lambda_env.sh
cd "$CODE_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN=${RUN:-rl}; OUT=$RUNS_DIR/$RUN; P=$OUT/eval_test_parts
case "$1" in
  part)
    case "$2" in a) GPU=0; MODES="base silent" ;; b) GPU=1; MODES="revise chain" ;; esac
    CUDA_VISIBLE_DEVICES=$GPU python -m ecowitness.evaluate --model "$MODEL_DIR" --adapter "$OUT/adapter" --data_dir "$DATA_DIR" --workspace "$WORKSPACE" \
      --split test --langs en --modes $MODES --out "$P/$2" --batch_size "${EVAL_BS:-8}" ${EVAL_EXTRA:-} 2>&1 | tee "$EW_ROOT/logs/eval_${RUN}_$2.out"
    echo "EVAL PART $2 DONE" ;;
  merge)
    mkdir -p "$OUT/eval_test"
    cat "$P/a/predictions.jsonl" "$P/b/predictions.jsonl" > "$OUT/eval_test/predictions.jsonl"
    python -m ecowitness.score --data "$DATA_DIR/test.en.jsonl" --predictions "$OUT/eval_test/predictions.jsonl" --label "$RUN (adapter $OUT/adapter, test, en, base silent revise chain)" ;;
esac
