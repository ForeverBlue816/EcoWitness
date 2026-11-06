"""EcoWitness: evidence-revision training and evaluation for the v9 river-ecology benchmark."""
import glob as _glob
import os as _os
import sys as _sys

# A None entry in sys.modules makes importlib.util.find_spec() return None, so the availability checks in
# transformers / accelerate report the package as absent and never import it.

# DeepSpeed is not used. On Leonardo the cineca-ai module ships a DeepSpeed built for its own torch 2.2; accelerate and
# transformers import it as soon as it is findable, and it then initialises the venv's triton, which fails without a GPU
# driver ("0 active drivers") or clashes with the newer torch. EW_ALLOW_DEEPSPEED=1 keeps it visible.
if _os.environ.get('EW_ALLOW_DEEPSPEED', '0') != '1' and 'deepspeed' not in _sys.modules:
    _sys.modules['deepspeed'] = None

# flash-linear-attention (fla) kernels are triton-only: importing fla on a machine without an NVIDIA GPU (login nodes,
# CPU smoke tests) raises "0 active drivers", and transformers imports it while loading Qwen3.5. Hide it there, so the
# torch fallback is used. EW_FLA=on forces it visible, EW_FLA=off hides it everywhere.
_fla = _os.environ.get('EW_FLA', 'auto').lower()
if 'fla' not in _sys.modules and (_fla == 'off' or (_fla == 'auto' and not _glob.glob('/dev/nvidia[0-9]*'))):
    _sys.modules['fla'] = None
