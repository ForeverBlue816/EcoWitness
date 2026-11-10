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

What a run writes into output_dir
  rl_log.jsonl     one line per `logging_steps`: training metrics averaged over all GPUs (reward, accuracy overall / by
                   mode / by question group, citation F1, informative prompts, seconds per step, peak GPU memory)
  val_log.jsonl    greedy replies on a fixed validation sample before training, every `eval_steps` and at the end
  progress.json    step, elapsed and remaining hours, latest metrics (cat it at any time)
  checkpoint-N/    adapter + optimiser state every `save_steps` and when the time budget ends the run
  adapter/         the final adapter;  curves.png: the training curves (plot_rl.py)
`--resume true` continues from the newest checkpoint in output_dir (or starts fresh when there is none).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageTextToText, StoppingCriteria, StoppingCriteriaList

from .data import WitnessCollator, approx_tokens, load_processor
from .reward import PARTIAL, WEIGHTS, reward
from .runner import load_jsonl
from .train import MERGER_RE, TARGET_RE

HERE = Path(__file__).resolve().parents[1]
DEFAULTS = {
    'model': 'Qwen/Qwen3.5-9B', 'workspace': str(HERE.parent), 'data_dir': str(HERE / 'data'), 'langs': ['en'], 'splits': ['train'],
    'modes': ['base', 'silent', 'revise'], 'mode_weights': [0.4, 0.35, 0.25], 'exclude_families': [], 'episodes': 80000,
    'prompts_per_step': 8, 'samples_per_prompt': 4, 'temperature': 1.0, 'top_p': 1.0, 'think': False, 'max_new_tokens': 320, 'max_reply_tokens': 300,
    'gold_prob': 0.5, 'gold_coef': 1.0, 'gen_token_budget': 32000, 'train_token_budget': 8000,
    'partial_credit': PARTIAL, 'w_answer': WEIGHTS['answer'], 'w_load_bearing': WEIGHTS['load_bearing'], 'w_support': WEIGHTS['support'], 'w_changed': WEIGHTS['changed'],
    'lora_r': 32, 'lora_alpha': 64, 'lora_dropout': 0.0, 'lora_merger': False, 'init_adapter': '',
    'lr': 5e-5, 'warmup_ratio': 0.03, 'grad_accum': 2, 'max_grad_norm': 1.0, 'bf16': True, 'gradient_checkpointing': True, 'attn': 'sdpa',
    'logging_steps': 10, 'save_steps': 500, 'save_total_limit': 2, 'seed': 0, 'output_dir': str(HERE / 'runs/rl'), 'skip_steps': 0,
    'reply_slack': 1.3, 'max_hours': 0.0, 'resume': False, 'eval_steps': 250, 'val_items': 240, 'val_split': 'val',
}


MODES = ('base', 'silent', 'revise')
GROUPS = ('A', 'B', 'C', 'D', 'E', 'K')
SAMPLE_KEYS = ('reward', 'parsed', 'correct', 'answer', 'load_bearing', 'support', 'new_tokens')
TRAIN_KEYS = (*SAMPLE_KEYS, *(f'correct_{m}' for m in MODES), *(f'correct_{g}' for g in GROUPS), 'distinct', 'informative_prompts',
              'candidates_trained', 'abs_advantage', 'sample_sec', 'train_sec', 'loss')
VAL_KEYS = ('reward', 'parsed', 'correct', 'answer', 'load_bearing', 'support', *(f'correct_{m}' for m in MODES), *(f'correct_{g}' for g in GROUPS))


class Tally:
    """Sums and counts per metric, so that metrics can be added up across GPUs and steps before dividing."""

    def __init__(self, keys):
        self.keys = keys
        self.v = {k: [0.0, 0.0] for k in keys}

    def add(self, key, value, n=1.0):
        self.v[key][0] += float(value)
        self.v[key][1] += n

    def tensor(self, device):
        return torch.tensor([[x for k in self.keys for x in self.v[k]]], dtype=torch.float64, device=device)

    @staticmethod
    def means(keys, total):
        """total: summed tensor of 2 * len(keys) -> {key: sum / count} for the keys that were observed."""
        t = total.tolist()
        return {k: round(t[2 * i] / t[2 * i + 1], 4) for i, k in enumerate(keys) if t[2 * i + 1] > 0}


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


class RowLimit(StoppingCriteria):
    """Stops each row of a batch at its own number of new tokens. The decoding loop runs as long as its longest row, so
    one runaway sample would otherwise cost every row of the chunk `max_new_tokens` decoding steps."""

    def __init__(self, prompt_len, limits):
        self.prompt_len, self.limits = prompt_len, limits

    def __call__(self, input_ids, scores, **kwargs):
        return (input_ids.shape[1] - self.prompt_len) >= self.limits


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
    def sample(self, prompts, images, k, limits=None, greedy=False):
        """k replies per prompt -> list (per prompt) of lists of token-id lists. `limits`: new-token cap per prompt.
        Sampled from the policy, or its single greedy reply when `greedy` (validation)."""
        cfg = self.cfg
        limits = limits or [cfg['max_new_tokens']] * len(prompts)
        order = [i for i in range(len(prompts)) for _ in range(k)]
        longest = max(approx_tokens_text(p) for p in prompts)
        chunk = max(1, cfg['gen_token_budget'] // max(longest, 1))
        out = [[] for _ in prompts]
        self.core.eval()
        for s in range(0, len(order), chunk):
            idx = order[s:s + chunk]
            batch = self.encode([prompts[i] for i in idx], [images[i] for i in idx], 'left').to(self.device)
            width = batch['input_ids'].shape[1]
            stop = StoppingCriteriaList([RowLimit(width, torch.tensor([limits[i] for i in idx], device=self.device))])
            how = {'do_sample': False} if greedy else {'do_sample': True, 'temperature': cfg['temperature'], 'top_p': cfg['top_p'], 'top_k': 0}
            # plain sampling (no top-k / top-p / repetition penalty): the log-probabilities of step 5 are those of the sampler
            gen = self.core.generate(**batch, max_new_tokens=max(limits[i] for i in idx), repetition_penalty=1.0, pad_token_id=self.pad, use_cache=True,
                                     stopping_criteria=stop, **how)
            new = gen[:, width:].tolist()
            for i, row in zip(idx, new):
                out[i].append(self.trim(row[:limits[i]]))
        self.core.train()
        return out

    def text(self, ids):
        t = self.tok.decode([t for t in ids if t not in self.eos and t != self.pad], skip_special_tokens=False)
        # with reasoning on, a reply that never closes its reasoning block has not answered, whatever the block mentions
        return t if (not self.cfg['think'] or '</think>' in t) else ''

    def loss(self, prompts, images, replies, weights, n_prompts):
        """- sum_c w_c * mean_t log pi(reply_c[t]) / n_prompts, one forward pass over the selected candidates.

        Sequences are padded on the left so that every reply ends at the last position. The vocabulary projection
        (248k entries) is then needed only at the last `longest reply` positions (`logits_to_keep`), not at every
        prompt and image position: full-sequence logits ran out of memory in supervised training on the 64 GB A100s."""
        batch = self.encode(prompts, images, 'left')
        ids, att = batch['input_ids'], batch['attention_mask']
        lens = att.sum(1).tolist()
        old_width = ids.shape[1]
        width = max(l + len(r) for l, r in zip(lens, replies))
        longest = max(len(r) for r in replies)
        new_ids = torch.full((len(prompts), width), self.pad, dtype=ids.dtype)
        new_att = torch.zeros((len(prompts), width), dtype=att.dtype)
        shifts = []
        for j, (l, r) in enumerate(zip(lens, replies)):
            start = width - len(r) - l                                  # left padding of this row
            new_ids[j, start:start + l] = ids[j, old_width - l:]
            new_ids[j, width - len(r):] = torch.tensor(r, dtype=ids.dtype)
            new_att[j, start:] = 1
            shifts.append(start - (old_width - l))
        extra = {}
        for key, v in batch.items():
            if key in ('input_ids', 'attention_mask'):
                continue
            if torch.is_tensor(v) and v.dim() == 2 and v.shape == ids.shape:   # other per-token tensors: move with the tokens
                e = torch.zeros((len(prompts), width), dtype=v.dtype)
                for j, (l, sh) in enumerate(zip(lens, shifts)):
                    e[j, old_width - l + sh:old_width + sh] = v[j, old_width - l:]
                v = e
            extra[key] = v
        inputs = {'input_ids': new_ids.to(self.device), 'attention_mask': new_att.to(self.device), **{k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in extra.items()}}
        keep = longest + 1                                              # positions width-longest-1 .. width-1
        logits = self.model(**inputs, logits_to_keep=keep).logits       # (candidates, keep, vocabulary)
        if logits.shape[1] != keep:
            raise RuntimeError(f'the model ignored logits_to_keep: got {tuple(logits.shape)}')
        total = logits.new_zeros((), dtype=torch.float32)
        for j, r in enumerate(replies):
            # column c holds position width - keep + c, which predicts the token at width - keep + c + 1
            ce = F.cross_entropy(logits[j, keep - 1 - len(r):keep - 1].float(), new_ids[j, width - len(r):].to(self.device), reduction='mean')
            total = total + weights[j] * ce
        return total / n_prompts

    def limits(self, records):
        """New-token cap per record: `reply_slack` times the reference reply plus a little. Reasoning has no reference length."""
        cfg = self.cfg
        if cfg['think']:
            return None
        return [min(cfg['max_new_tokens'], int(cfg['reply_slack'] * len(self.tok.encode(r['target'], add_special_tokens=False))) + 16) for r in records]

    def prepare(self, records):
        rendered = [self.coll.render(r) for r in records]
        return [p for p, _ in rendered], [self.coll.load_images(paths) for _, paths in rendered]

    def step(self, records, rng):
        """-> (loss tensor, Tally) for one batch of episodes."""
        cfg = self.cfg
        tally = Tally(TRAIN_KEYS)
        prompts, images = self.prepare(records)
        t0 = time.time()
        samples = self.sample(prompts, images, cfg['samples_per_prompt'], self.limits(records))
        tally.add('sample_sec', time.time() - t0)
        cands = []
        for i, r in enumerate(records):
            scored = [reward(r, self.text(ids), self.weights, cfg['partial_credit']) for ids in samples[i]]
            base = sum(x['total'] for x in scored) / len(scored)
            merged = {}
            for ids, x in zip(samples[i], scored):
                key = tuple(ids)
                merged[key] = merged.get(key, 0.0) + (x['total'] - base) / len(scored)
                for k in ('parsed', 'correct', 'answer', 'load_bearing', 'support'):
                    tally.add(k, x[k])
                tally.add('reward', x['total'])
                tally.add('new_tokens', len(ids))
                tally.add(f'correct_{r["mode"]}', x['correct'])
                tally.add(f'correct_{r["group"]}', x['correct'])
            tally.add('distinct', len(merged))
            if not cfg['think'] and cfg['gold_prob'] > 0 and rng.random() < cfg['gold_prob']:
                gold = tuple(self.tok.encode(r['target'] + '<|im_end|>', add_special_tokens=False))
                gain = reward(r, r['target'], self.weights, cfg['partial_credit'])['total'] - base
                if gain > 0:
                    merged[gold] = merged.get(gold, 0.0) + cfg['gold_coef'] * gain
            tally.add('informative_prompts', any(abs(w) > 1e-6 for w in merged.values()))
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
        tally.add('candidates_trained', len(chosen))
        for c in chosen:
            tally.add('abs_advantage', c[0])
        return loss, tally

    @torch.no_grad()
    def validate(self, records, batch=16):
        """Greedy replies on validation records -> Tally (VAL_KEYS). These are the numbers a learning curve should show:
        the training metrics are measured on replies sampled at temperature 1."""
        tally = Tally(VAL_KEYS)
        for i in range(0, len(records), batch):
            recs = records[i:i + batch]
            prompts, images = self.prepare(recs)
            outs = self.sample(prompts, images, 1, self.limits(recs), greedy=True)
            for r, o in zip(recs, outs):
                x = reward(r, self.text(o[0]), self.weights, self.cfg['partial_credit'])
                for k in ('parsed', 'correct', 'answer', 'load_bearing', 'support'):
                    tally.add(k, x[k])
                tally.add('reward', x['total'])
                tally.add(f'correct_{r["mode"]}', x['correct'])
                tally.add(f'correct_{r["group"]}', x['correct'])
        return tally


def validation_sample(cfg, tokenizer):
    """A fixed, mode-balanced sample of validation records (the same for every run with the same seed)."""
    path = Path(cfg['data_dir']) / f'{cfg["val_split"]}.{cfg["langs"][0]}.jsonl'
    if not path.exists() or cfg['val_items'] <= 0:
        return []
    rows = [r for r in load_jsonl(path) if r['mode'] in MODES and 'answer_reward' in r
            and len(tokenizer.encode(r['target'], add_special_tokens=False)) <= cfg['max_reply_tokens']]
    rng = random.Random(f'val:{cfg["seed"]}')
    out = []
    for m in MODES:
        pool = [r for r in rows if r['mode'] == m]
        rng.shuffle(pool)
        out += pool[:cfg['val_items'] // len(MODES)]
    out.sort(key=approx_tokens)   # similar lengths together, and an even mix across processes after striding
    return out


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
        if not cfg['resume']:   # a fresh run must not append to the logs, or later resume from the checkpoints, of an earlier run here
            stamp = time.strftime('%Y%m%d-%H%M%S')
            for old in [out / n for n in ('rl_log.jsonl', 'val_log.jsonl', 'progress.json')] + sorted(out.glob('checkpoint-*')):
                if old.exists():
                    old.replace(out / f'before-{stamp}-{old.name}')
    processor = load_processor(cfg['model'])
    model = AutoModelForImageTextToText.from_pretrained(cfg['model'], dtype=torch.bfloat16 if cfg['bf16'] else torch.float32, attn_implementation=cfg['attn'])
    if hasattr(model, 'mtp'):
        model.mtp = None
    for p in model.parameters():
        p.requires_grad_(False)
    state, skip, hours_before = None, cfg['skip_steps'], 0.0
    if cfg['resume']:   # newest checkpoint of this output_dir, if any
        cks = sorted((c for c in out.glob('checkpoint-*') if (c / 'state.pt').exists()), key=lambda c: int(c.name.split('-')[1]))
        if cks:
            state = torch.load(cks[-1] / 'state.pt', map_location='cpu', weights_only=False)
            cfg['init_adapter'], skip = str(cks[-1]), state['step']
            log_path = out / 'rl_log.jsonl'
            logged = [json.loads(x) for x in log_path.read_text(encoding='utf8').splitlines() if x.strip()] if log_path.exists() else []
            hours_before = max([x.get('hours', 0.0) for x in logged if x['step'] <= skip], default=0.0)
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
    n_proc, rank, device = accelerator.num_processes, accelerator.process_index, accelerator.device
    steps = math.ceil(len(ds) / n_proc)
    updates = math.ceil(steps / cfg['grad_accum'])
    warm = max(1, int(cfg['warmup_ratio'] * updates))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda u: min(1.0, (u + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(u, updates) / updates))))
    model, optim, loader = accelerator.prepare(model, optim, loader)   # the scheduler is stepped by hand, once per update
    if state:
        optim.load_state_dict(state['optim'])
        sched.load_state_dict(state['sched'])
    core = accelerator.unwrap_model(model)
    policy = Policy(cfg, model, core, processor, device)
    val = validation_sample(cfg, processor.tokenizer)[rank::n_proc] if cfg['eval_steps'] > 0 else []
    cuda = device.type == 'cuda'
    if main_proc:
        core.print_trainable_parameters()
        print(f'{ds.n_episodes} episodes {ds.by_mode} in {len(ds)} batches ({ds.too_long} records with a reply over {cfg["max_reply_tokens"]} tokens left out); '
              f'{n_proc} process(es) x {cfg["prompts_per_step"]} prompts x {cfg["samples_per_prompt"]} samples; {steps} steps, {updates} updates; '
              f'think={cfg["think"]} max_new_tokens={cfg["max_new_tokens"]}; time budget {cfg["max_hours"] or "none"} h; '
              + (f'resuming after step {skip}' if skip else 'fresh start'), flush=True)
    rng = random.Random(cfg['seed'] * 1000 + rank + 7919 * skip)
    started = time.time()
    core.train()

    def save_checkpoint(step):
        d = out / f'checkpoint-{step}'
        core.save_pretrained(str(d))
        processor.save_pretrained(str(d))
        torch.save({'step': step, 'optim': optim.state_dict(), 'sched': sched.state_dict()}, d / 'state.pt')
        old = sorted((p for p in out.glob('checkpoint-*') if p.is_dir()), key=lambda p: int(p.name.split('-')[1]))
        for p in old[:-cfg['save_total_limit']]:
            shutil.rmtree(p, ignore_errors=True)

    def run_validation(step):
        """Every process scores its share of the validation sample; one collective call adds the shares up."""
        if not cfg['eval_steps']:
            return {}
        total = accelerator.gather(policy.validate(val).tensor(device)).sum(0)
        core.train()
        m = Tally.means(VAL_KEYS, total)
        if main_proc:
            line = {'step': step, 'items': int(total[1].item()), 'hours': round((time.time() - started) / 3600 + hours_before, 3), **m}
            with (out / 'val_log.jsonl').open('a', encoding='utf8') as f:
                f.write(json.dumps(line) + '\n')
            bar.write('VAL ' + json.dumps(line))
        return m

    from tqdm import tqdm
    bar = tqdm(total=steps, initial=skip, disable=not main_proc, file=sys.stdout, mininterval=15, smoothing=0.05, dynamic_ncols=False, ncols=150, desc='RL')
    last_val = run_validation(skip) if not skip else {}
    window = torch.zeros(2 * len(TRAIN_KEYS), dtype=torch.float64, device=device)
    peak, t_log, n_log = [0.0, 0.0], time.time(), 0
    step, stopped = skip, False
    for step, records in enumerate(loader, 1):
        if step <= skip:
            continue
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with accelerator.accumulate(model):
            loss, tally = policy.step(records, rng)
            mem_sample = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else 0.0
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg['max_grad_norm'])
            optim.step()
            if accelerator.sync_gradients:
                sched.step()
            optim.zero_grad()
        mem_train = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else 0.0
        tally.add('loss', float(loss.detach()))
        tally.add('train_sec', time.time() - t0 - tally.v['sample_sec'][0])
        # One collective call per step carries the metrics of all GPUs, their peak memory and whether any process is over
        # the time budget, so every process takes the same decisions at the same step (one stopping alone would hang the rest).
        over = float(cfg['max_hours'] > 0 and (time.time() - started) / 3600 + hours_before > cfg['max_hours'])
        extra = torch.tensor([[mem_sample, mem_train, over]], dtype=torch.float64, device=device)
        got = accelerator.gather(torch.cat([tally.tensor(device), extra], 1))
        window += got[:, :-3].sum(0)
        peak = [max(peak[0], got[:, -3].max().item()), max(peak[1], got[:, -2].max().item())]
        n_log += 1
        bar.update(1)
        done = step == steps
        stopped = got[:, -1].max().item() > 0 and not done
        if step % cfg['logging_steps'] == 0 or done or stopped:
            m = Tally.means(TRAIN_KEYS, window)
            sec = (time.time() - t_log) / n_log
            elapsed = (time.time() - started) / 3600 + hours_before
            line = {'step': step, 'of': steps, 'hours': round(elapsed, 3), 'sec_per_step': round(sec, 2), 'lr': sched.get_last_lr()[0], **m,
                    'mem_sample_gb': round(peak[0], 1), 'mem_train_gb': round(peak[1], 1)}
            if main_proc:
                with (out / 'rl_log.jsonl').open('a', encoding='utf8') as f:
                    f.write(json.dumps(line) + '\n')
                bar.write(json.dumps(line))
                bar.set_postfix_str(f'reward {m.get("reward", 0):.3f} acc {m.get("correct", 0):.3f} val_acc {last_val.get("correct", float("nan")):.3f} '
                                    f'{sec:.1f}s/step mem {max(peak):.0f}G')
                progress = {'step': step, 'of': steps, 'percent': round(100 * step / steps, 1), 'hours_elapsed': round(elapsed, 2),
                            'hours_remaining_at_current_speed': round((steps - step) * sec / 3600, 2), 'time_budget_hours': cfg['max_hours'], 'finished': False,
                            'latest_training': line, 'latest_validation': last_val}
                (out / 'progress.json').write_text(json.dumps(progress, indent=1), encoding='utf8')
            window.zero_()
            peak, t_log, n_log = [0.0, 0.0], time.time(), 0
        if cfg['eval_steps'] and (step % cfg['eval_steps'] == 0 or done or stopped):
            last_val = run_validation(step)
        if main_proc and (step % cfg['save_steps'] == 0 or stopped):
            save_checkpoint(step)
        if stopped:
            if main_proc:
                bar.write(f'time budget of {cfg["max_hours"]} h reached at step {step} of {steps}: checkpoint-{step} saved. Continue with  TRAIN_EXTRA="--resume true"')
            break
    bar.close()
    accelerator.wait_for_everyone()
    if main_proc:
        core.save_pretrained(str(out / 'adapter'))
        processor.save_pretrained(str(out / 'adapter'))
        if (out / 'progress.json').exists():
            progress = json.loads((out / 'progress.json').read_text(encoding='utf8'))
            progress.update({'finished': not stopped, 'stopped_by_time_budget': stopped, 'latest_validation': last_val})
            (out / 'progress.json').write_text(json.dumps(progress, indent=1), encoding='utf8')
        try:
            from .plot_rl import plot
            print('curves:', plot(out), flush=True)
        except Exception as e:   # drawing must never cost a finished run
            print(f'curves not drawn ({type(e).__name__}: {e}); run  python -m ecowitness.plot_rl {out}', flush=True)
        print('RL DONE', out / 'adapter', '(stopped by the time budget; resume to finish)' if stopped else '', flush=True)


if __name__ == '__main__':
    main()
