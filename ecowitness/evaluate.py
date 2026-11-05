"""Run a Qwen3.5 model (zero-shot, or with a LoRA adapter) on the witness protocol and score it.

    python -m ecowitness.evaluate --model $MODEL --adapter runs/base/adapter --split test --langs en zh --out runs/base/eval_test

Writes predictions.jsonl plus metrics.json (records with truth) or check.json (public test pack). Metrics are defined in
score.py; the generation loop (own-history / teacher-forced chains, own-prior revisions) is in runner.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText

from . import runner
from .data import WitnessCollator, load_processor

HERE = Path(__file__).resolve().parents[1]


class Generator:
    def __init__(self, model_path, adapter, workspace, max_new_tokens=64, batch_size=8, dtype=torch.bfloat16, device='cuda'):
        self.processor = load_processor(adapter or model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=dtype, attn_implementation='sdpa')
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
            self.model = self.model.merge_and_unload()
        self.model.to(device).eval()
        self.device = device
        self.coll = WitnessCollator(self.processor, workspace)
        self.max_new_tokens, self.batch_size = max_new_tokens, batch_size

    @torch.no_grad()
    def generate(self, message_lists):
        outs = []
        for i in range(0, len(message_lists), self.batch_size):
            chunk = message_lists[i:i + self.batch_size]
            prompts, images = [], []
            for msgs in chunk:
                p, paths = self.coll.render({'messages': msgs})
                prompts.append(p)
                images.extend(self.coll.load_images(paths))
            self.processor.tokenizer.padding_side = 'left'
            batch = self.processor(text=prompts, images=images or None, padding=True, return_tensors='pt').to(self.device)
            gen = self.model.generate(**batch, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self.processor.tokenizer.pad_token_id)
            new = gen[:, batch['input_ids'].shape[1]:]
            outs.extend(self.processor.batch_decode(new, skip_special_tokens=True))
        return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3.5-9B')
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--workspace', default=str(HERE.parent), help='folder that holds satellite_rgb/')
    ap.add_argument('--data_dir', default=str(HERE / 'data'))
    ap.add_argument('--split', default='test')
    ap.add_argument('--langs', nargs='+', default=['en'])
    ap.add_argument('--modes', nargs='+', default=['base', 'revise', 'chain'])
    ap.add_argument('--families', nargs='*', default=None)
    ap.add_argument('--max_items', type=int, default=0, help='per mode and language (0 = all)')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--max_new_tokens', type=int, default=64)
    ap.add_argument('--teacher_forced', action='store_true', help='chains: feed gold previous answers (needs targets, owner only)')
    ap.add_argument('--own_history', action='store_true', help='revisions: feed the model\'s own base answer as its prior reply (base mode must run too)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dtype', default='bf16')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[a.dtype]
    gen = Generator(a.model, a.adapter, a.workspace, a.max_new_tokens, a.batch_size, dtype, a.device)
    runner.run(gen, a.data_dir, a.split, a.langs, a.modes, a.out, families=a.families, max_items=a.max_items,
               teacher_forced=a.teacher_forced, own_history=a.own_history, meta={'model': a.model, 'adapter': a.adapter})


if __name__ == '__main__':
    main()
