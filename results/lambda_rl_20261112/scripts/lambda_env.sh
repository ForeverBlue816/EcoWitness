# EcoWitness on the Lambda machine (2 x H100, no Slurm, no modules). Source it:  source /lambda/nfs/ICLR/ecowitness/lambda_env.sh
# Same variables as repo/scripts/cineca_env.sh, without the module loads.
export EW_ROOT=/lambda/nfs/ICLR/ecowitness
export CODE_DIR=$EW_ROOT/repo
export VENV=/home/ubuntu/vla_local/envs/ecowitness
export SCRATCH_ROOT=$EW_ROOT
export WORKSPACE=$EW_ROOT/dataset_v10                 # holds satellite_rgb/ (root of EcoWitness_dataset_v10.zip)
export DATA_DIR=$EW_ROOT/data_rl/data                 # {train,val,test}.en.jsonl with answer_reward (data.zip, uploaded 2026-11-12 12:38 UTC)
export MODEL_ROOT=$EW_ROOT/models
export MODEL_DIR=$MODEL_ROOT/Qwen3.5-9B
export TINY_DIR=$MODEL_ROOT/tiny-random-qwen3.5
export RUNS_DIR=$EW_ROOT/runs
export HF_HOME=$EW_ROOT/hf_cache
export EW_CACHE=/home/ubuntu/vla_local/ecowitness_cache   # compiler caches on the local disk
export XDG_CACHE_HOME=$EW_CACHE/xdg TRITON_CACHE_DIR=$EW_CACHE/triton TORCHINDUCTOR_CACHE_DIR=$EW_CACHE/torchinductor
export CUDA_CACHE_PATH=$EW_CACHE/nv PYTORCH_KERNEL_CACHE_PATH=$EW_CACHE/torch_kernels
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$PYTORCH_KERNEL_CACHE_PATH" "$XDG_CACHE_HOME/torch/kernels"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export EW_TORCHRUN="python -m torch.distributed.run"
