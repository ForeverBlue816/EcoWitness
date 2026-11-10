"""Verifier-guided reinforcement learning of Qwen3.5 (LoRA) on the witness protocol.

    python -m torch.distributed.run --nproc_per_node=4 -m ecowitness.rl --config configs/rl.yaml --output_dir runs/rl

Why RL fits this benchmark. Every record is a verifiable episode: the rule engine knows the value of every option and
the exact support / load-bearing witness sets of every evidence state, so any reply can be scored exactly, with partial
credit that follows the ecology (reward.py). No reference reasoning is needed, and nothing has to be paired.

One step, per GPU
  1. take `prompts_per_step` episodes (a base question, a silently edited evidence state, or an announced revision);
  2. sample `samples_per_prompt` replies from the current policy (no gradient);
  3. score every reply with the verifier; advantage = reward - mean reward of that prompt's samples;
  4. optionally add the reference reply as one more candidate whose advantage is 1 - mean sampled reward
     (`gold_prob`): a demonstration that fades by itself once the policy's own replies reach the reference reward;
  5. merge identical replies, keep the candidates with the largest |advantage| that fit `train_token_budget`, and
     take one gradient step on  - sum_c advantage_c * mean_t log pi(reply_c[t] | prompt)  / prompts.
Prompts on which all samples earn the same reward have zero advantage and cost no backward pass, so the gradient budget
goes to the questions the policy is unsure about.

`think: true` leaves the model's reasoning block open: the reward is the only supervision of the reasoning (there are
no reference reasoning traces), and a reply that runs out of `max_new_tokens` before answering earns the format
penalty. Reference replies are not injected in that mode; start from an SFT or RL adapter with `init_adapter` instead.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageTextToText

from .data import WitnessCollator, approx_tokens, load_processor
from .reward import PARTIAL, WEIGHTS, reward
from .runner import load_jsonl
from .train import MERGER_RE, TARGET_RE

HERE = Path(__file__).resolve().parents[1]
DEFAULTS = {
    'model': 'Qwen/Qwen3.5-9B', 'workspace': str(HERE.parent), 'data_dir': str(HERE / 'data'), 'langs': ['en'], 'splits': ['train'],
    'modes': ['base', 'silent', 'revise'], 'mode_weights': [0.4, 0.35, 0.25], 'exclude_families': [], 'episodes': 80000,
    'prompts_per_step': 4, 'samples_per_prompt': 4, 'temperature': 1.0, 'top_p': 1.0, 'think': False, 'max_new_tokens': 320, 'max_reply_tokens': 300,
    'gold_prob': 0.5, 'gold_coef': 1.0, 'gen_token_budget': 32000, 'train_token_budget': 8000,
    'partial_credit': PARTIAL, 'w_answer': WEIGHTS['answer'], 'w_load_bearing': WEIGHTS['load_bearing'], 'w_support': WEIGHTS['support'], 'w_changed': WEIGHTS['changed'],
    'lora_r': 32, 'lora_alpha': 64, 'lora_dropout': 0.0, 'lora_merger': False, 'init_adapter': '',
    'lr': 5e-5, 'warmup_ratio': 0.03, 'grad_accum': 2, 'max_grad_norm': 1.0, 'bf16': True, 'gradient_checkpointing': True, 'attn': 'sdpa',
    'logging_steps': 10, 'save_steps': 500, 'save_total_limit': 2, 'seed': 0, 'output_dir': str(HERE / 'runs/rl'), 'skip_steps': 0,
}


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            ap.add_argument(f'--{k}', type=lambda s: s.lower() in ('1', 'true', 'yes'), default=None)
        elif isinstance(v, list):
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
    cfg['mode_weights'] = [float(x) for x in cfg['mode_weights']]
    return cfg


class Episodes(Dataset):
    """Pre-drawn batches of episodes. Modes are mixed by `mode_weights`; inside a mode records are drawn without
    replacement and reshuffled when exhausted. Batches hold prompts of similar length."""

    def __init__(self, paths, modes, mode_weights, episodes, batch, exclude_families=(), seed=0, tokenizer=None, max_reply_tokens=0):
        rows = [r for p in paths for r in load_jsonl(p) if r['mode'] in modes and r['family'] not in set(exclude_families)]
        self.too_long = 0
        if tokenizer is not None and max_reply_tokens:   # a reference reply that cannot fit the sampling limit can never earn full reward
            keep = [r for r in rows if len(r['target']) <= max_reply_tokens or len(tokenizer.encode(r['target'], add_special_tokens=False)) <= max_reply_tokens]
            self.too_long, rows = len(rows) - len(keep), keep
        if rows and 'answer_reward' not in rows[0]:
            raise ValueError('records have no answer_reward: rebuild the data with the current build_revisions.py')
        rng = random.Random(seed)
        pools = {m: [r for r in rows if r['mode'] == m] for m in modes}
        weights = {m: w for m, w in zip(modes, mode_weights) if pools[m]}
        total = sum(weights.values())
        n = episodes or len(rows)
        items = []
        for m, w in weights.items():
            pool, k = pools[m], round(n * w / total)
            rng.shuffle(pool)
            items += [pool[i % len(pool)] for i in range(k)]
        rng.shuffle(items)
        self.batches = []
        for i in range(0, len(items), batch * 64):                      # length buckets
            chunk = sorted(items[i:i + batch * 64], key=approx_tokens)
            group = [chunk[j:j + batch] for j in range(0, len(chunk), batch)]
            rng.shuffle(group)
            self.batches += group
        self.n_episodes = len(items)
        self.by_mode = {m: sum(r['mode'] == m for r in items) for m in modes}

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        return self.batches[i]


class Policy:
    """Sampling and scoring around one LoRA model. `model` may be DDP-wrapped; `core` is the unwrapped PeftModel."""

    def __init__(self, cfg, model, core, processor, device):
        self.cfg, self.model, self.core, self.processor, self.device = cfg, model, core, processor, device
        self.tok = processor.tokenizer
        self.coll = WitnessCollator(processor, cfg['workspace'], think=cfg['think'])
        eos = core.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {self.tok.convert_tokens_to_ids('<|im_end|>')}
        self.pad = self.tok.pad_token_id
        self.weights = {'answer': cfg['w_answer'], 'load_bearing': cfg['w_load_bearing'], 'support': cfg['w_support'], 'changed': cfg['w_changed']}

    def encode(self, prompts, images, side):
        self.tok.padding_side = side
        flat = [im for ims in images for im in ims]
        return self.processor(text=prompts, images=flat or None, padding=True, return_tensors='pt')

    def trim(self, ids):
        """Generated ids up to and including the first end token; padding after it is dropped."""
        out = []
        for t in ids:
            out.append(t)
            if t in self.eos:
                break
        return out

    @torch.no_grad()
    def sample(self, prompts, images, k):
        """k sampled replies per prompt -> list (per prompt) of lists of token-id lists."""
        cfg = self.cfg
        order = [i for i in range(len(prompts)) for _ in range(k)]
        longest = max(approx_tokens_text(p) for p in prompts)
        chunk = max(1, cfg['gen_token_budget'] // max(longest, 1))
        out = [[] for _ in prompts]
        self.core.eval()
        for s in range(0, len(order), chunk):
            idx = order[s:s + chunk]
            batch = self.encode([prompts[i] for i in idx], [images[i] for i in idx], 'left').to(self.device)
            gen = self.core.generate(**batch, max_new_tokens=cfg['max_new_tokens'], do_sample=True, temperature=cfg['temperature'], top_p=cfg['top_p'],
                                     top_k=0, repetition_penalty=1.0, pad_token_id=self.pad, use_cache=True)   # plain sampling: the log-probabilities of step 5 are those of the sampler
            new = gen[:, batch['input_ids'].shape[1]:].tolist()
            for i, row in zip(idx, new):
                out[i].append(self.trim(row))
        self.core.train()
        return out

    def text(self, ids):
        t = self.tok.decode([t for t in ids if t not in self.eos and t != self.pad], skip_special_tokens=False)
        # with reasoning on, a reply that never closes its reasoning block has not answered, whatever the block mentions
        return t if (not self.cfg['think'] or '</think>' in t) else ''

    def loss(self, prompts, images, replies, weights, n_prompts):
        """- sum_c w_c * mean_t log pi(reply_c[t]) / n_prompts, one padded forward pass over the selected candidates."""
        batch = self.encode(prompts, images, 'right')
        ids, att = batch['input_ids'], batch['attention_mask']
        lens = att.sum(1).tolist()
        width = max(l + len(r) for l, r in zip(lens, replies))
        new_ids = torch.full((len(prompts), width), self.pad, dtype=ids.dtype)
        new_att = torch.zeros((len(prompts), width), dtype=att.dtype)
        for j, (l, r) in enumerate(zip(lens, replies)):
            new_ids[j, :l] = ids[j, :l]
            new_ids[j, l:l + len(r)] = torch.tensor(r, dtype=ids.dtype)
            new_att[j, :l + len(r)] = 1
        extra = {}
        for key, v in batch.items():
            if key in ('input_ids', 'attention_mask'):
                continue
            if torch.is_tensor(v) and v.dim() == 2 and v.shape == ids.shape:   # other per-token tensors: extend with zeros
                e = torch.zeros((len(prompts), width), dtype=v.dtype)
                e[:, :v.shape[1]] = v
                v = e
            extra[key] = v
        inputs = {'input_ids': new_ids.to(self.device), 'attention_mask': new_att.to(self.device), **{k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in extra.items()}}
        logits = self.model(**inputs).logits
        total = logits.new_zeros((), dtype=torch.float32)
        for j, (l, r) in enumerate(zip(lens, replies)):
            ce = F.cross_entropy(logits[j, l - 1:l - 1 + len(r)].float(), new_ids[j, l:l + len(r)].to(self.device), reduction='mean')
            total = total + weights[j] * ce
        return total / n_prompts

    def step(self, records, rng):
        """-> (loss tensor, stats dict) for one batch of episodes."""
        cfg = self.cfg
        rendered = [self.coll.render(r) for r in records]
        prompts = [p for p, _ in rendered]
        images = [self.coll.load_images(paths) for _, paths in rendered]
        samples = self.sample(prompts, images, cfg['samples_per_prompt'])
        cands, stats = [], {'reward': [], 'parsed': [], 'correct': [], 'answer': [], 'load_bearing': [], 'support': [], 'informative': 0, 'distinct': [], 'new_tokens': []}
        for i, r in enumerate(records):
            scored = [reward(r, self.text(ids), self.weights, cfg['partial_credit']) for ids in samples[i]]
            base = sum(s['total'] for s in scored) / len(scored)
            merged = {}
            for ids, s in zip(samples[i], scored):
                key = tuple(ids)
                merged[key] = merged.get(key, 0.0) + (s['total'] - base) / len(scored)
                for k in ('parsed', 'correct', 'answer', 'load_bearing', 'support'):
                    stats[k].append(float(s[k]))
                stats['reward'].append(s['total'])
                stats['new_tokens'].append(len(ids))
            stats['distinct'].append(len(merged))
            if not cfg['think'] and cfg['gold_prob'] > 0 and rng.random() < cfg['gold_prob']:
                gold = tuple(self.tok.encode(r['target'] + '<|im_end|>', add_special_tokens=False))
                gain = reward(r, r['target'], self.weights, cfg['partial_credit'])['total'] - base
                if gain > 0:
                    merged[gold] = merged.get(gold, 0.0) + cfg['gold_coef'] * gain
            if any(abs(w) > 1e-6 for w in merged.values()):
                stats['informative'] += 1
            cands += [(abs(w), i, list(ids), w) for ids, w in merged.items() if ids]
        cands.sort(key=lambda c: -c[0])
        chosen, longest = [], 0
        for c in cands:
            length = approx_tokens_text(prompts[c[1]]) + len(c[2])
            if chosen and max(longest, length) * (len(chosen) + 1) > cfg['train_token_budget']:
                continue
            chosen.append(c)
            longest = max(longest, length)
        if not chosen:   # every sampled reply was empty: a zero-weight pass keeps the processes in step
            chosen = [(0.0, 0, self.tok.encode(records[0]['target'] + '<|im_end|>', add_special_tokens=False), 0.0)]
        loss = self.loss([prompts[c[1]] for c in chosen], [images[c[1]] for c in chosen], [c[2] for c in chosen], [c[3] for c in chosen], len(records))
        out = {k: sum(v) / max(len(v), 1) for k, v in stats.items() if isinstance(v, list)}
        out.update({'informative_prompts': stats['informative'] / len(records), 'candidates_trained': len(chosen), 'abs_advantage': sum(c[0] for c in chosen) / max(len(chosen), 1)})
        return loss, out


def approx_tokens_text(prompt):
    return int(len(prompt) / 3.2) + 64 * prompt.count('<|image_pad|>') + 40


def main():
    cfg = parse()
    out = Path(cfg['output_dir'])
    accelerator = Accelerator(gradient_accumulation_steps=cfg['grad_accum'])
    main_proc = accelerator.is_main_process
    if main_proc:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'config.json').write_text(json.dumps(cfg, indent=1), encoding='utf8')
    processor = load_processor(cfg['model'])
    model = AutoModelForImageTextToText.from_pretrained(cfg['model'], dtype=torch.bfloat16 if cfg['bf16'] else torch.float32, attn_implementation=cfg['attn'])
    if hasattr(model, 'mtp'):
        model.mtp = None
    for p in model.parameters():
        p.requires_grad_(False)
    if cfg['init_adapter']:
        model = PeftModel.from_pretrained(model, cfg['init_adapter'], is_trainable=True)
    else:
        target = TARGET_RE if not cfg['lora_merger'] else f'({TARGET_RE})|({MERGER_RE})'
        model = get_peft_model(model, LoraConfig(r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'], lora_dropout=cfg['lora_dropout'], target_modules=target, bias='none', task_type='CAUSAL_LM'))
    if not any(n.endswith('lora_A') for n, _ in model.named_modules()):
        raise RuntimeError('no LoRA modules: check TARGET_RE or init_adapter')
    if cfg['gradient_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    paths = [Path(cfg['data_dir']) / f'{s}.{l}.jsonl' for s in cfg['splits'] for l in cfg['langs']]
    ds = Episodes(paths, cfg['modes'], cfg['mode_weights'], cfg['episodes'], cfg['prompts_per_step'], cfg['exclude_families'], cfg['seed'],
                  tokenizer=processor.tokenizer, max_reply_tokens=cfg['max_reply_tokens'])
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=lambda b: b[0], num_workers=0)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg['lr'], weight_decay=0.0)
    steps = math.ceil(len(ds) / accelerator.num_processes)
    updates = math.ceil(steps / cfg['grad_accum'])
    warm = max(1, int(cfg['warmup_ratio'] * updates))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda u: min(1.0, (u + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(u, updates) / updates))))
    model, optim, loader = accelerator.prepare(model, optim, loader)   # the scheduler is stepped by hand, once per update
    core = accelerator.unwrap_model(model)
    policy = Policy(cfg, model, core, processor, accelerator.device)
    if main_proc:
        core.print_trainable_parameters()
        print(f'{ds.n_episodes} episodes {ds.by_mode} in {len(ds)} batches ({ds.too_long} records with a reply over {cfg["max_reply_tokens"]} tokens left out); {accelerator.num_processes} process(es) x {cfg["prompts_per_step"]} prompts x '
              f'{cfg["samples_per_prompt"]} samples; {steps} steps, {updates} updates; think={cfg["think"]} max_new_tokens={cfg["max_new_tokens"]}', flush=True)
    rng = random.Random(cfg['seed'] * 1000 + accelerator.process_index)
    log, t0 = [], time.time()
    core.train()

    def save(name):
        d = out / name
        core.save_pretrained(str(d))
        processor.save_pretrained(str(d))
        return d

    for step, records in enumerate(loader, 1):
        if step <= cfg['skip_steps']:
            continue
        with accelerator.accumulate(model):
            loss, stats = policy.step(records, rng)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg['max_grad_norm'])
            optim.step()
            if accelerator.sync_gradients:
                sched.step()
            optim.zero_grad()
        log.append({**stats, 'loss': float(loss.detach())})
        if main_proc and step % cfg['logging_steps'] == 0:
            avg = {k: round(sum(x[k] for x in log) / len(log), 4) for k in log[0]}
            line = {'step': step, 'of': steps, 'sec_per_step': round((time.time() - t0) / len(log), 2), 'lr': sched.get_last_lr()[0], **avg}
            print(json.dumps(line), flush=True)
            with (out / 'rl_log.jsonl').open('a', encoding='utf8') as f:
                f.write(json.dumps(line) + '\n')
            log, t0 = [], time.time()
        if main_proc and step % cfg['save_steps'] == 0:
            save(f'checkpoint-{step}')
            old = sorted((p for p in out.glob('checkpoint-*') if p.is_dir()), key=lambda p: int(p.name.split('-')[1]))
            for p in old[:-cfg['save_total_limit']]:
                shutil.rmtree(p, ignore_errors=True)
    accelerator.wait_for_everyone()
    if main_proc:
        save('adapter')
        print('RL DONE', out / 'adapter', flush=True)


if __name__ == '__main__':
    main()
