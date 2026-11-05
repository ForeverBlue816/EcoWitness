#!/bin/bash
# One-time setup on a Leonardo LOGIN node (compute nodes have no internet):
#   cd $HOME/ecowitness && bash scripts/setup_env.sh
# Order matters: load the cineca-ai module FIRST, then create the venv on top of it (--system-site-packages), then
# install only what the module lacks, with --no-deps for anything that depends on torch so pip never pulls its own
# torch wheel over the tuned Leonardo build.
set -euo pipefail
cd "$(dirname "$0")/.."
EW_VARS_ONLY=1 source scripts/cineca_env.sh

module purge
module av cineca-ai 2>&1 | grep -i cineca-ai || true
module load profile/deeplrn
module load "$EW_MODULE"
python - <<'EOF'
import sys, torch
print('module python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda, '|', torch.__file__)
assert sys.version_info >= (3, 10), 'python >= 3.10 required'
EOF

if [ ! -d "$VENV" ]; then
  python -m venv "$VENV" --system-site-packages
fi
source "$VENV/bin/activate"
# The cineca-ai module exports PYTHONPATH=<module site-packages>, which outranks the venv. Put the venv first so packages
# installed here (transformers 5.x, ...) win over the module's copies; torch is not installed here, so it stays the module's.
VENV_SITE=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export PYTHONPATH="$VENV_SITE${PYTHONPATH:+:$PYTHONPATH}"
echo "venv site-packages first on PYTHONPATH: $VENV_SITE"
# Do NOT upgrade pip, and never call the bin/pip script: a pip copied into the venv would clash with the module's pip
# through PYTHONPATH ("cannot import name 'cli'"). A pip left over from an earlier run is removed.
rm -rf "$VENV_SITE"/pip "$VENV_SITE"/pip-*.dist-info "$VENV/bin/pip" "$VENV/bin/pip3" "$VENV"/bin/pip3.*
PIP="python -m pip"
$PIP --version

# 1) pure-python dependencies (no torch dependency anywhere in this list)
$PIP install --no-cache-dir "tokenizers>=0.22" "safetensors>=0.5" "huggingface_hub>=0.34" regex filelock packaging pyyaml tqdm requests psutil pillow openai typing_extensions
# 2) torch-dependent packages WITHOUT their dependency resolution, so the module's torch stays in place
$PIP install --no-cache-dir --no-deps "transformers>=5.3" "peft>=0.18" "accelerate>=1.5"
# 3) torchvision is needed by the fast Qwen image processor; only install if the module lacks it, matching torch's version
python -c "import torchvision" 2>/dev/null || echo "NOTE: torchvision missing -> the code falls back to the slow PIL image processor (fine for 256x256 images)"

python - <<'EOF'
import torch, transformers, peft, accelerate
print('torch', torch.__version__, torch.__file__)
assert 'ecowitness_env' not in torch.__file__, 'torch was installed inside the venv! remove it: python -m pip uninstall torch'
print('transformers', transformers.__version__, transformers.__file__)
assert 'ecowitness_env' in transformers.__file__, 'the module\'s transformers shadows the venv copy (PYTHONPATH order)'
print('peft', peft.__version__, '| accelerate', accelerate.__version__)
from transformers import AutoConfig
assert 'qwen3_5' in transformers.models.auto.configuration_auto.CONFIG_MAPPING_NAMES, 'this transformers has no qwen3_5; need >= 5.3'
print('qwen3_5 supported')
EOF

# optional fused kernels for the Gated DeltaNet layers (need a CUDA toolchain; harmless if they fail)
$PIP install --no-cache-dir --no-deps flash-linear-attention causal-conv1d 2>/dev/null || echo "NOTE: fused linear-attention kernels not installed; training uses the slower torch fallback"
$PIP install --no-cache-dir nvitop || true
echo "Environment ready. Activate with:  source $CODE_DIR/scripts/cineca_env.sh"
