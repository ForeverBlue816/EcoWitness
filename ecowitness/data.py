"""Dataset and collator for witness-protocol SFT with counterfactual groups.

A dataset item is a *group*: a base question plus up to `group_size - 1` companions that must land in the same
micro-batch so the equivariance loss can compare their answer distributions. Companions, in priority order:
    silent:invariant     the evidence was edited (ecological or generic axis) and the answer must not move
    silent:flip          the evidence was edited and the answer must move to the rule-derived new one
    silent:equivariant   the options were permuted and the answer letter must move with them
    revise:keep_uncited  the announced revision of an unrelated witness (the original isolation pair)
The collator flattens groups, renders the Qwen chat template with thinking disabled, masks everything but the final
assistant turn, and records the answer-token position plus the group bookkeeping the loss needs.
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
LETTER_INDEX = {l.strip(): i for i, l in enumerate(LETTERS)}
KIND = {'base': 0, 'invariant': 1, 'flip': 2, 'equivariant': 3}
COMPANIONS = ('silent:invariant', 'silent:flip', 'silent:equivariant', 'revise:keep_uncited')


def load_processor(path):
    """AutoProcessor; falls back to the slow PIL image processor when torchvision (needed by the fast one) is absent."""
    from transformers import AutoProcessor
    try:
        import torchvision  # noqa: F401
        return AutoProcessor.from_pretrained(path)
    except ImportError:
        return AutoProcessor.from_pretrained(path, use_fast=False)
ALL_FIELDS = ('changed', 'answer', 'support', 'load_bearing')


from .runner import load_jsonl  # noqa: E402  (shared with the torch-free runner and scorer)


def strip_target(target, fields):
    return '\n'.join(l for l in target.split('\n') if l.split(':')[0] in fields)


def companion_key(r):
    if r['mode'] == 'silent':
        return 'silent:' + r['effect']
    if r['mode'] == 'revise' and r.get('edit_class') == 'keep_uncited':
        return 'revise:keep_uncited'
    return None


def record_kind(r):
    if r['mode'] == 'base':
        return KIND['base']
    if r['mode'] == 'silent':
        return KIND[r['effect']]
    return KIND['invariant']   # keep_uncited revision


def approx_tokens(r):
    """Rough token count of a conversation (text / 3.2 + 64 per image), for the group budget."""
    n, images = 0, 0
    for m in r['messages']:
        c = m['content']
        if isinstance(c, str):
            n += len(c)
        else:
            for it in c:
                if it['type'] == 'image':
                    images += 1
                else:
                    n += len(it['text'])
    return int(n / 3.2) + 64 * images + 40


class WitnessDataset(Dataset):
    def __init__(self, paths, modes=('base', 'revise', 'silent'), fields=ALL_FIELDS, pair=True, families=None, exclude_families=(),
                 edit_classes=None, silent_axes=None, companions=COMPANIONS, group_size=3, group_budget=8000, max_items=0, seed=0):
        rows = []
        for p in paths:
            rows.extend(load_jsonl(p))
        rows = [r for r in rows if r['mode'] in modes and (not families or r['family'] in families) and r['family'] not in set(exclude_families)
                and (edit_classes is None or r['mode'] != 'revise' or r['edit_class'] in edit_classes)
                and (silent_axes is None or r['mode'] != 'silent' or r['axis'] in silent_axes)]
        for r in rows:
            r['target'] = strip_target(r['target'], fields)
        groups = []
        if pair:
            by_pair = {}
            for r in rows:
                pid = (r['pair_id'], r['lang'])  # en and zh versions of a question are never grouped together
                d = by_pair.setdefault(pid, {'base': [], 'other': []})
                key = 'base' if r['mode'] == 'base' else companion_key(r)
                d.setdefault(key or 'other', []).append(r)
            for pid, d in by_pair.items():
                base = d['base']
                if base:
                    b = base[0]
                    g, cost = [b], approx_tokens(b)
                    # priority order, one of each kind first, then a second round
                    for _round in range(2):
                        for key in companions:
                            if len(g) >= group_size or not d.get(key):
                                continue
                            c = d[key][0]
                            new_cost = max(cost, approx_tokens(c)) * (len(g) + 1)
                            if new_cost > group_budget:
                                continue
                            g.append(d[key].pop(0))
                            cost = max(cost, approx_tokens(c))
                    groups.append(g)
                    groups.extend([x] for x in base[1:])
                for key, rest in d.items():
                    if key != 'base':
                        groups.extend([x] for x in rest)
        else:
            groups = [[r] for r in rows]
        rng = random.Random(seed)
        rng.shuffle(groups)
        if max_items:   # trial runs: keep half of the budget for grouped items so the equivariance terms are exercised
            multi = [g for g in groups if len(g) > 1][: (max_items + 1) // 2]
            single = [g for g in groups if len(g) == 1][: max_items - len(multi)]
            groups = multi + single
            rng.shuffle(groups)
        self.groups = groups
        self.n_records = sum(len(g) for g in groups)
        self.n_companions = sum(len(g) - 1 for g in groups if len(g) > 1)

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
        texts, images, target_ids, kinds, gids, perms, olds = [], [], [], [], [], [], []
        identity = list(range(len(LETTERS)))
        for gi, g in enumerate(groups):
            for r in g:
                prompt, paths = self.render(r)
                tgt = r['target'] + '<|im_end|>'
                texts.append(prompt + tgt + '\n')
                images.extend(self.load_images(paths))
                target_ids.append(self.tok.encode(tgt, add_special_tokens=False))
                kinds.append(record_kind(r))
                gids.append(gi if len(g) > 1 else -1)
                # companion letter j holds the base option whose old letter maps to j: perm_index[j] = base index
                perm = r.get('perm')
                if perm:
                    inv = {new: old for old, new in perm.items()}
                    perms.append([LETTER_INDEX[inv[l.strip()]] for l in LETTERS])
                else:
                    perms.append(identity)
                olds.append(LETTER_INDEX.get(r.get('base_answer'), -1))
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
        batch['cf_group'] = torch.tensor(gids)
        batch['cf_kind'] = torch.tensor(kinds)
        batch['cf_perm'] = torch.tensor(perms)
        batch['cf_old'] = torch.tensor(olds)
        return dict(batch)
