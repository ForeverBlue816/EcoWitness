"""EcoWitness: evidence-revision training and evaluation for the v9 river-ecology benchmark."""
import os as _os
import sys as _sys

# DeepSpeed is not used. On Leonardo the cineca-ai module ships a DeepSpeed built for its own torch 2.2; accelerate and
# transformers import it as soon as it is findable, and it then initialises the venv's triton, which fails without a GPU
# driver ("0 active drivers") or clashes with the newer torch. A None entry in sys.modules makes find_spec() return None,
# so every is_deepspeed_available() check is False. Set EW_ALLOW_DEEPSPEED=1 to keep it visible.
if _os.environ.get('EW_ALLOW_DEEPSPEED', '0') != '1' and 'deepspeed' not in _sys.modules:
    _sys.modules['deepspeed'] = None
