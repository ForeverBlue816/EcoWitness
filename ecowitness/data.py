"""Dataset and collator for witness-protocol SFT with paired (base, keep-uncited revision) groups.

A dataset item is a *group*: one or two records that must land in the same micro-batch so that the isolation loss can
compare their answer distributions. The collator flattens groups, renders the Qwen chat template with thinking
disabled, masks everything but the final assistant turn, and records the position of the answer token.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from . import protocol

LETTERS = [' A', ' B', ' C', ' D', ' none']


def load_processor(path):
    """AutoProcessor; falls back to the slow PIL image processor when torchvision (needed by the fast one) is absent."""
    from transformers import AutoProcessor
    try:
        import torchvision  # noqa: F401
        return AutoProcessor.from_pretrained(path)
    except ImportError:
        return AutoProcessor.from_pretrained(path, use_fast=False)
ALL_FIELDS = ('changed', 'answer', 'support', 'load_bearing')


def load_jsonl(path):
    with Path(path).open(encoding='utf8') as f:
        return [json.loads(l) for l in f if l.strip()]


def strip_target(target, fields):
    return '\n'.join(l for l in target.split('\n') if l.split(':')[0] in fields)


class WitnessDataset(Dataset):
    def __init__(self, paths, modes=('base', 'revise'), fields=ALL_FIELDS, pair=True, families=None, exclude_families=(),
                 edit_classes=None, max_items=0, seed=0):
        rows = []
        for p in paths:
            rows.extend(load_jsonl(p))
        rows = [r for r in rows if r['mode'] in modes and (not families or r['family'] in families) and r['family'] not in set(exclude_families)
                and (edit_classes is None or r['mode'] != 'revise' or r['edit_class'] in edit_classes)]
        for r in rows:
            r['target'] = strip_target(r['target'], fields)
        groups = []
        if pair:
            by_pair = {}
            for r in rows:
                pid = (r['pair_id'], r['lang'])  # en and zh versions of a question are never paired with each other
                by_pair.setdefault(pid, {'base': [], 'keep_uncited': [], 'other': []})
                key = 'base' if r['mode'] == 'base' else ('keep_uncited' if r['edit_class'] == 'keep_uncited' else 'other')
                by_pair[pid][key].append(r)
            for pid, d in by_pair.items():
                base, keeps = d['base'], d['keep_uncited']
                for i, b in enumerate(base):
                    g = [b] + ([keeps[i]] if i < len(keeps) else [])
                    groups.append(g)
                groups.extend([k] for k in keeps[len(base):])
                groups.extend([o] for o in d['other'])
        else:
            groups = [[r] for r in rows]
        rng = random.Random(seed)
        rng.shuffle(groups)
        if max_items:
            groups = groups[:max_items]
        self.groups = groups
        self.n_records = sum(len(g) for g in groups)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        return self.groups[i]


class WitnessCollator:
    def __init__(self, processor, workspace, max_length=4096):
        self.processor = processor
        self.tok = processor.tokenizer
        self.workspace = Path(workspace)
        self.max_length = max_length
        self.letter_ids = [self.tok.encode(s, add_special_tokens=False)[0] for s in LETTERS]
        self.im_end = self.tok.convert_tokens_to_ids('<|im_end|>')

    def render(self, rec):
        chat, paths = protocol.to_chat(rec['messages'])
        prompt = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return prompt, paths

    def load_images(self, paths):
        return [Image.open(self.workspace / p).convert('RGB') for p in paths]

    def __call__(self, groups):
        recs = [r for g in groups for r in g]
        texts, images, target_ids, roles, gids = [], [], [], [], []
        for gi, g in enumerate(groups):
            for r in g:
                prompt, paths = self.render(r)
                tgt = r['target'] + '<|im_end|>'
                texts.append(prompt + tgt + '\n')
                images.extend(self.load_images(paths))
                target_ids.append(self.tok.encode(tgt, add_special_tokens=False))
                roles.append(0 if r['mode'] == 'base' else 1)
                gids.append(gi if len(g) == 2 else -1)
        self.tok.padding_side = 'right'
        batch = self.processor(text=texts, images=images or None, padding=True, return_tensors='pt')
        ids, att = batch['input_ids'], batch['attention_mask']
        labels = torch.full_like(ids, -100)
        answer_pos = torch.zeros(len(recs), dtype=torch.long)
        for i, t in enumerate(target_ids):
            L = int(att[i].sum())
            start = L - 1 - len(t)  # the sequence ends with target + '\n'
            assert ids[i, start:start + len(t)].tolist() == t, 'target tokens are not a suffix of the rendered sequence'
            labels[i, start:start + len(t)] = ids[i, start:start + len(t)]
            k = next(j for j, x in enumerate(t) if x in self.letter_ids)
            answer_pos[i] = start + k
        if ids.shape[1] > self.max_length:
            raise ValueError(f'sequence of {ids.shape[1]} tokens exceeds max_length={self.max_length}')
        batch['labels'] = labels
        batch['answer_pos'] = answer_pos
        batch['pair_group'] = torch.tensor(gids)
        batch['pair_role'] = torch.tensor(roles)
        return dict(batch)
