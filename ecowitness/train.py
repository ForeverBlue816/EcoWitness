"""LoRA fine-tuning of Qwen3.5 (image-text-to-text) on the witness protocol with the isolation loss.

    torchrun --nproc_per_node=4 -m ecowitness.train --config configs/base.yaml --output_dir runs/base

Loss = token cross-entropy on the assistant lines (answer, support, load_bearing, changed)
     + lambda_iso * symmetric KL between the answer-letter distributions of a base question and its keep-uncited revision.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, Trainer, TrainingArguments

from .data import ALL_FIELDS, WitnessCollator, WitnessDataset, load_processor

HERE = Path(__file__).resolve().parents[1]
DEFAULTS = {
    'model': 'Qwen/Qwen3.5-9B', 'workspace': str(HERE.parent), 'data_dir': str(HERE / 'data'), 'langs': ['en', 'zh'], 'splits': ['train'],
    'modes': ['base', 'revise'], 'fields': list(ALL_FIELDS), 'pair': True, 'exclude_families': [], 'edit_classes': None, 'max_items': 0,
    'lambda_iso': 0.5, 'lora_r': 32, 'lora_alpha': 64, 'lora_dropout': 0.05, 'lora_merger': False,
    'epochs': 3, 'lr': 1e-4, 'warmup_ratio': 0.03, 'batch_size': 1, 'grad_accum': 8, 'max_length': 6144, 'bf16': True,
    'gradient_checkpointing': True, 'attn': 'sdpa', 'logging_steps': 20, 'save_steps': 500, 'save_total_limit': 2,
    'num_workers': 4, 'seed': 0, 'output_dir': str(HERE / 'runs/base'), 'report_to': 'none', 'resume': False,
}
TARGET_RE = r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj\w*|out_proj|gate_proj|up_proj|down_proj)'
MERGER_RE = r'.*visual\.merger.*(linear_fc\d|proj)'


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            ap.add_argument(f'--{k}', type=lambda s: s.lower() in ('1', 'true', 'yes'), default=None)
        elif isinstance(v, list) or v is None:
            ap.add_argument(f'--{k}', nargs='*', default=None)
        else:
            ap.add_argument(f'--{k}', type=type(v), default=None)
    a = ap.parse_args()
    cfg = dict(DEFAULTS)
    if a.config:
        cfg.update(yaml.safe_load(Path(a.config).read_text(encoding='utf8')) or {})
    for k in DEFAULTS:
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    return cfg


class WitnessTrainer(Trainer):
    def __init__(self, *args, lambda_iso=0.0, letter_ids=None, **kw):
        super().__init__(*args, **kw)
        # The model accepts loss kwargs, so the Trainer would skip dividing the loss by gradient_accumulation_steps and
        # expect num_items_in_batch to be used instead. compute_loss returns a per-batch mean (CE + iso KL), so let the
        # Trainer do the division; otherwise gradients come out grad_accum times too large.
        self.model_accepts_loss_kwargs = False
        self.lambda_iso = lambda_iso
        self.letter_ids = letter_ids
        self._iso_log = []

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        answer_pos, group, role = inputs.pop('answer_pos'), inputs.pop('pair_group'), inputs.pop('pair_role')
        outputs = model(**inputs)
        loss = outputs.loss
        if self.lambda_iso > 0 and (group >= 0).any():
            logits = outputs.logits
            idx = torch.arange(logits.shape[0], device=logits.device)
            sel = logits[idx, answer_pos - 1][:, self.letter_ids].float()  # logits that predict the answer token
            logp = F.log_softmax(sel, dim=-1)
            kls = []
            for g in group[group >= 0].unique():
                members = (group == g).nonzero().flatten()
                if len(members) != 2:
                    continue
                a, b = logp[members[0]], logp[members[1]]
                kls.append(0.5 * ((a.exp() * (a - b)).sum() + (b.exp() * (b - a)).sum()))
            if kls:
                iso = torch.stack(kls).mean()
                self._iso_log.append(float(iso.detach()))
                loss = loss + self.lambda_iso * iso
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kw):
        if self._iso_log:
            logs['iso_kl'] = sum(self._iso_log) / len(self._iso_log)
            self._iso_log = []
        super().log(logs, *args, **kw)


def main():
    cfg = parse()
    out = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        (out / 'config.json').write_text(json.dumps(cfg, indent=1), encoding='utf8')
    processor = load_processor(cfg['model'])
    model = AutoModelForImageTextToText.from_pretrained(cfg['model'], dtype=torch.bfloat16 if cfg['bf16'] else torch.float32, attn_implementation=cfg['attn'])
    if hasattr(model, 'mtp'):
        model.mtp = None  # the multi-token-prediction head is not trained or used here
    for p in model.parameters():
        p.requires_grad_(False)
    target = TARGET_RE if not cfg['lora_merger'] else f'({TARGET_RE})|({MERGER_RE})'
    lcfg = LoraConfig(r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'], lora_dropout=cfg['lora_dropout'], target_modules=target, bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, lcfg)
    n_lora = sum(1 for n, _ in model.named_modules() if n.endswith('lora_A'))
    if n_lora == 0:
        raise RuntimeError('LoRA matched no modules; check TARGET_RE against model.named_modules()')
    if cfg['gradient_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        model.print_trainable_parameters()
        print(f'LoRA on {n_lora} modules')
    paths = [Path(cfg['data_dir']) / f'{s}.{l}.jsonl' for s in cfg['splits'] for l in cfg['langs']]
    ds = WitnessDataset(paths, modes=cfg['modes'], fields=cfg['fields'], pair=cfg['pair'], exclude_families=cfg['exclude_families'],
                        edit_classes=cfg['edit_classes'], max_items=cfg['max_items'], seed=cfg['seed'])
    collator = WitnessCollator(processor, cfg['workspace'], cfg['max_length'])
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        print(f'{len(ds)} groups / {ds.n_records} records from {[str(p) for p in paths]}')
    args = TrainingArguments(
        output_dir=str(out), num_train_epochs=cfg['epochs'], learning_rate=cfg['lr'], warmup_steps=cfg['warmup_ratio'], lr_scheduler_type='cosine',  # transformers 5: a float warmup_steps < 1 is a ratio
        per_device_train_batch_size=cfg['batch_size'], gradient_accumulation_steps=cfg['grad_accum'], bf16=cfg['bf16'],
        logging_steps=cfg['logging_steps'], save_steps=cfg['save_steps'], save_total_limit=cfg['save_total_limit'], save_strategy='steps',
        dataloader_num_workers=cfg['num_workers'], remove_unused_columns=False, report_to=cfg['report_to'], seed=cfg['seed'],
        ddp_find_unused_parameters=False, gradient_checkpointing=False, label_names=['labels'], weight_decay=0.0, max_grad_norm=1.0,
    )
    trainer = WitnessTrainer(model=model, args=args, train_dataset=ds, data_collator=collator, lambda_iso=cfg['lambda_iso'], letter_ids=collator.letter_ids)
    trainer.train(resume_from_checkpoint=cfg['resume'] or None)
    trainer.save_model(str(out / 'adapter'))
    processor.save_pretrained(str(out / 'adapter'))


if __name__ == '__main__':
    main()
