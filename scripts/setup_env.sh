#!/bin/bash
# One-time setup on a Leonardo LOGIN node (compute nodes have no internet):
#   cd $HOME/ecowitness && bash scripts/setup_env.sh
# Order: load the cineca-ai module FIRST (Python, CUDA libraries, numpy ...), create the venv on top of it
# (--system-site-packages), then install what the module lacks.
# Qwen3.5 needs transformers >= 5.3, which needs torch >= 2.6. If the module's torch is older (cineca-ai/4.3.0 ships
# torch 2.2), a PyTorch CUDA wheel is installed into the venv and shadows the module's torch. Override with
#   EW_TORCH_SPEC="torch==2.7.1 torchvision==0.22.1" EW_TORCH_INDEX=https://download.pytorch.org/whl/cu126
set -euo pipefail
cd "$(dirname "$0")/.."
EW_VARS_ONLY=1 source scripts/cineca_env.sh
EW_TORCH_SPEC=${EW_TORCH_SPEC:-"torch==2.7.1 torchvision==0.22.1"}
EW_TORCH_INDEX=${EW_TORCH_INDEX:-https://download.pytorch.org/whl/cu126}

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
# installed here (torch if needed, transformers 5.x, newer regex / tokenizers / huggingface_hub) win over the module's.
VENV_SITE=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export PYTHONPATH="$VENV_SITE${PYTHONPATH:+:$PYTHONPATH}"
echo "venv site-packages first on PYTHONPATH: $VENV_SITE"
# Do NOT upgrade pip, and never call the bin/pip script: a pip copied into the venv clashes with the module's pip
# through PYTHONPATH ("cannot import name 'cli'"). A pip left over from an earlier run is removed.
rm -rf "$VENV_SITE"/pip "$VENV_SITE"/pip-*.dist-info "$VENV/bin/pip" "$VENV/bin/pip3" "$VENV"/bin/pip3.*
PIP="python -m pip"
$PIP --version

# 1) torch: keep the module's build when it is new enough, otherwise install a CUDA wheel into the venv
if python -c 'import sys, torch; from packaging.version import Version; sys.exit(0 if Version(torch.__version__).release >= (2, 6) else 1)'; then
  echo "module torch is >= 2.6: keeping it"
else
  echo "module torch is older than 2.6 (transformers 5 needs >= 2.6): installing $EW_TORCH_SPEC from $EW_TORCH_INDEX into the venv (~3 GB)"
  $PIP install --no-cache-dir $EW_TORCH_SPEC --index-url "$EW_TORCH_INDEX"
fi

# 2) transformers WITH its own dependencies (none of them is torch), so regex / tokenizers / huggingface_hub /
#    safetensors are brought to the versions transformers 5 requires instead of the module's older copies
$PIP install --no-cache-dir "transformers>=5.3"
# 3) peft and accelerate depend on torch: --no-deps so they never pull a second torch; their other deps are listed here
$PIP install --no-cache-dir --no-deps "peft>=0.18" "accelerate>=1.5"
$PIP install --no-cache-dir psutil pyyaml pillow openai

python - <<'EOF'
import torch, transformers, peft, accelerate
from packaging.version import Version
print('torch', torch.__version__, '| cuda', torch.version.cuda, '|', torch.__file__)
assert Version(torch.__version__).release >= (2, 6), 'torch < 2.6 is still the one imported (PYTHONPATH order?)'
print('transformers', transformers.__version__, transformers.__file__)
assert 'ecowitness_env' in transformers.__file__, "the module's transformers shadows the venv copy (PYTHONPATH order)"
import transformers.utils.import_utils as iu
assert iu.is_torch_available(), 'transformers disabled PyTorch: check the torch version above'
print('peft', peft.__version__, '| accelerate', accelerate.__version__)
assert 'qwen3_5' in transformers.models.auto.configuration_auto.CONFIG_MAPPING_NAMES, 'this transformers has no qwen3_5; need >= 5.3'
from transformers import AutoModelForImageTextToText  # noqa: F401  (import works with torch enabled)
try:
    import torchvision
    print('torchvision', torchvision.__version__)
except ImportError:
    print('NOTE: torchvision missing -> the code falls back to the slow PIL image processor (fine for 256x256 images)')
print('qwen3_5 supported')
EOF

# Fused kernels for the Gated DeltaNet layers, installed one by one so a failed build does not take the others down.
# flash-linear-attention is pure Python + triton (no compilation); --no-deps so it never touches torch.
$PIP install --no-cache-dir --no-deps flash-linear-attention fla-core einops \
  || echo "NOTE: flash-linear-attention not installed; chunk_gated_delta_rule uses the slow torch fallback"
# causal-conv1d: its prebuilt wheels need glibc >= 2.32; Leonardo (RHEL 8, glibc 2.28) cannot load them
# ("GLIBC_2.32 not found"). Not installed by default: the short convolution's torch fallback is cheap next to the gated
# delta rule. EW_CAUSAL_CONV1D=1 compiles it from source instead (needs nvcc; 10-20 min).
$PIP uninstall -y causal-conv1d >/dev/null 2>&1 || true
if [ "${EW_CAUSAL_CONV1D:-0}" = "1" ]; then
  $PIP install --no-cache-dir --no-deps --no-build-isolation --no-binary causal-conv1d causal-conv1d \
    || echo "NOTE: causal-conv1d failed to build; the torch fallback for the short convolution is used"
fi
# fla imports only on a machine with a GPU driver; on the login node just check that it is installed
python -c "import importlib.metadata as m; print('flash-linear-attention', m.version('flash-linear-attention'))" || true
$PIP install --no-cache-dir nvitop || true
echo "Environment ready. Activate with:  source $CODE_DIR/scripts/cineca_env.sh"
