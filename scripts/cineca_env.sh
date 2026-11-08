#!/bin/bash
# Shared Leonardo environment. Source it, do not run it:   source scripts/cineca_env.sh
# Layout (same convention as PhysioWave-T/docs/cineca_setup.md):
#   $HOME/ecowitness          code  (backed up, 50 GB quota)
#   $HOME/ecowitness_env      venv on top of the cineca-ai module (torch >= 2.6 from the module, else a wheel in the venv)
#   $SCRATCH/ecowitness/...   data, satellite images, model weights, runs  (SCRATCH is purged 40 days after creation;
#                             everything here is rebuildable, but copy adapters/metrics you want to keep to $HOME or $FAST)
# EW_VARS_ONLY=1 source scripts/cineca_env.sh   -> only export the paths, no module load / venv activation.
export EW_ACCOUNT=${EW_ACCOUNT:-iscrb_wearusfm}
export EW_PARTITION=${EW_PARTITION:-boost_usr_prod}
export EW_MODULE=${EW_MODULE:-cineca-ai/4.3.0}

export CODE_DIR=${CODE_DIR:-$HOME/ecowitness}
export VENV=${VENV:-$HOME/ecowitness_env}
export SCRATCH_ROOT=${SCRATCH_ROOT:-${SCRATCH:-$HOME}/ecowitness}
export WORKSPACE=${WORKSPACE:-$SCRATCH_ROOT}                 # directory that holds satellite_rgb/
export DATA_DIR=${DATA_DIR:-$SCRATCH_ROOT/data}              # {train,val,test}.{en,zh}.jsonl
export MODEL_ROOT=${MODEL_ROOT:-$SCRATCH_ROOT/models}
export MODEL_DIR=${MODEL_DIR:-$MODEL_ROOT/Qwen3.5-9B}
export TINY_DIR=${TINY_DIR:-$MODEL_ROOT/tiny-random-qwen3.5}
export RUNS_DIR=${RUNS_DIR:-$SCRATCH_ROOT/runs}

export HF_HOME=${HF_HOME:-$SCRATCH_ROOT/hf_cache}
# Keep every compiler / package cache off $HOME (50 GB quota): triton and fla write compiled kernels and autotune
# results to ~/.triton and ~/.cache by default, which fills the quota ("OSError: [Errno 122] Disk quota exceeded").
export EW_CACHE=${EW_CACHE:-$SCRATCH_ROOT/cache}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$EW_CACHE/xdg}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$EW_CACHE/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$EW_CACHE/torchinductor}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$EW_CACHE/pip}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$EW_CACHE/nv}
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$PIP_CACHE_DIR" "$CUDA_CACHE_PATH" 2>/dev/null || true
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [ "${EW_VARS_ONLY:-0}" != "1" ]; then
  module purge
  module load profile/deeplrn
  module load "$EW_MODULE" || echo "WARNING: module $EW_MODULE failed to load (run: module av cineca-ai)"
  if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
    # the module's PYTHONPATH would otherwise shadow the venv's transformers / peft / accelerate
    VENV_SITE=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    export PYTHONPATH="$VENV_SITE${PYTHONPATH:+:$PYTHONPATH}"
  else
    echo "NOTE: venv $VENV not found yet; run scripts/setup_env.sh"
  fi
fi
# Always launch with the venv's interpreter so workers inherit its site-packages (never the torchrun console script).
export EW_TORCHRUN="python -m torch.distributed.run"
