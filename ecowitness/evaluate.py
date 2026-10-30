"""Evaluate a Qwen3.5 model (zero-shot, or with a LoRA adapter) on the witness protocol.

    python -m ecowitness.evaluate --model $MODEL --adapter runs/base/adapter --split test --langs en --out runs/base/eval_test_en

Metrics: base accuracy and citation precision/recall; revision F1, contagion rate (answers changed although an
unrelated witness changed), inertia (answers kept although a load-bearing witness changed); multi-step chain accuracy by
step with the model's own history (or gold history with --teacher_forced).
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText

from . import protocol
from .data import WitnessCollator, load_jsonl, load_processor

HERE = Path(__file__).resolve().parents[1]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(2 * p * r / (p + r), 4) if p + r else 0.0}


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


def set_scores(pred, truth):
    p, t = set(pred), set(truth)
    return len(p & t), len(p - t), len(t - p)


def score_single(records, preds):
    """Base and single-step revision metrics."""
    m = {'n': len(records)}
    acc = collections.defaultdict(lambda: [0, 0])
    cite = {'support': [0, 0, 0], 'load_bearing': [0, 0, 0]}
    lb_exact = [0, 0]
    rev = {'tp': 0, 'fp': 0, 'fn': 0}
    by_class = collections.defaultdict(lambda: {'n': 0, 'acc': 0, 'pred_changed': 0})
    none_acc = [0, 0]
    for r, out in zip(records, preds):
        p = protocol.parse_output(out)
        ok = p['answer'] == r['truth']['answer']
        for key in ('all', 'group:' + r['group'], 'family:' + r['family'], 'tier:' + r['tier']):
            acc[key][0] += ok
            acc[key][1] += 1
        for k in cite:
            tp, fp, fn = set_scores(p[k], r['truth'][k])
            cite[k][0] += tp
            cite[k][1] += fp
            cite[k][2] += fn
        lb_exact[0] += set(p['load_bearing']) == set(r['truth']['load_bearing'])
        lb_exact[1] += 1
        if r['truth']['answer'] == 'none':
            none_acc[0] += ok
            none_acc[1] += 1
        if r['mode'] == 'revise':
            prev = protocol.parse_output(r['messages'][-2]['content'])['answer']
            pred_changed = p['changed'] if p['changed'] is not None else (p['answer'] != prev)
            truth_changed = r['truth']['changed']
            rev['tp'] += pred_changed and truth_changed
            rev['fp'] += pred_changed and not truth_changed
            rev['fn'] += (not pred_changed) and truth_changed
            c = by_class[r['edit_class']]
            c['n'] += 1
            c['acc'] += ok
            c['pred_changed'] += pred_changed
    m['accuracy'] = {k: round(v[0] / v[1], 4) for k, v in sorted(acc.items())}
    m['citation'] = {k: prf(*v) for k, v in cite.items()}
    m['load_bearing_exact'] = round(lb_exact[0] / lb_exact[1], 4) if lb_exact[1] else None
    m['none_accuracy'] = round(none_acc[0] / none_acc[1], 4) if none_acc[1] else None
    if any(r['mode'] == 'revise' for r in records):
        m['revision_f1'] = prf(rev['tp'], rev['fp'], rev['fn'])
        m['by_edit_class'] = {k: {'n': v['n'], 'accuracy': round(v['acc'] / v['n'], 4), 'pred_changed_rate': round(v['pred_changed'] / v['n'], 4)} for k, v in sorted(by_class.items())}
        ku, kc, fl = by_class.get('keep_uncited'), by_class.get('keep_cited'), by_class.get('flip')
        m['contagion_rate'] = round(ku['pred_changed'] / ku['n'], 4) if ku and ku['n'] else None
        m['redundancy_overreaction_rate'] = round(kc['pred_changed'] / kc['n'], 4) if kc and kc['n'] else None
        m['inertia_rate'] = round(1 - fl['pred_changed'] / fl['n'], 4) if fl and fl['n'] else None
    return m


def run_chains(gen, chains, teacher_forced):
    """Returns per-chain list of (step outputs)."""
    histories = [list(c['messages']) for c in chains]
    base_out = gen.generate(histories)
    step_out = [[] for _ in chains]
    max_len = max(len(c['steps']) for c in chains)
    for k in range(max_len):
        idx = [i for i, c in enumerate(chains) if len(c['steps']) > k]
        for i in idx:
            prev_text = (chains[i]['target'] if k == 0 else chains[i]['steps'][k - 1]['target']) if teacher_forced else (base_out[i] if k == 0 else step_out[i][-1])
            histories[i] = histories[i] + [{'role': 'assistant', 'content': prev_text.strip()}, chains[i]['steps'][k]['message']]
        outs = gen.generate([histories[i] for i in idx])
        for i, o in zip(idx, outs):
            step_out[i].append(o)
    return base_out, step_out


def score_chains(chains, base_out, step_out):
    by_step = collections.defaultdict(lambda: [0, 0])
    changed_ok = collections.defaultdict(lambda: [0, 0])
    final = [0, 0]
    for c, b, outs in zip(chains, base_out, step_out):
        by_step[0][0] += protocol.parse_output(b)['answer'] == c['truth']['answer']
        by_step[0][1] += 1
        prev = protocol.parse_output(b)['answer']
        for k, (s, o) in enumerate(zip(c['steps'], outs), 1):
            p = protocol.parse_output(o)
            by_step[k][0] += p['answer'] == s['truth']['answer']
            by_step[k][1] += 1
            pc = p['changed'] if p['changed'] is not None else (p['answer'] != prev)
            changed_ok[k][0] += pc == s['truth']['changed']
            changed_ok[k][1] += 1
            prev = p['answer']
        final[0] += protocol.parse_output(outs[-1])['answer'] == c['steps'][-1]['truth']['answer'] if outs else 0
        final[1] += 1
    return {'n': len(chains), 'accuracy_by_step': {k: round(v[0] / v[1], 4) for k, v in sorted(by_step.items())},
            'changed_accuracy_by_step': {k: round(v[0] / v[1], 4) for k, v in sorted(changed_ok.items())}, 'final_step_accuracy': round(final[0] / final[1], 4) if final[1] else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3.5-9B')
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--workspace', default=str(HERE.parent))
    ap.add_argument('--data_dir', default=str(HERE / 'data'))
    ap.add_argument('--split', default='test')
    ap.add_argument('--langs', nargs='+', default=['en'])
    ap.add_argument('--modes', nargs='+', default=['base', 'revise', 'chain'])
    ap.add_argument('--families', nargs='*', default=None)
    ap.add_argument('--max_items', type=int, default=0, help='per mode and language (0 = all)')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--max_new_tokens', type=int, default=64)
    ap.add_argument('--teacher_forced', action='store_true', help='chains: feed gold previous answers instead of the model\'s own')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dtype', default='bf16')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[a.dtype]
    gen = Generator(a.model, a.adapter, a.workspace, a.max_new_tokens, a.batch_size, dtype, a.device)
    metrics = {'model': a.model, 'adapter': a.adapter, 'split': a.split, 'teacher_forced': a.teacher_forced}
    t0 = time.time()
    with (out / 'predictions.jsonl').open('w', encoding='utf8') as pf:
        for lang in a.langs:
            rows = load_jsonl(Path(a.data_dir) / f'{a.split}.{lang}.jsonl')
            if a.families:
                rows = [r for r in rows if r['family'] in a.families]
            for mode in a.modes:
                recs = [r for r in rows if r['mode'] == mode]
                if a.max_items:
                    recs = recs[:a.max_items]
                if not recs:
                    continue
                if mode == 'chain':
                    base_out, step_out = run_chains(gen, recs, a.teacher_forced)
                    metrics[f'{lang}/chain'] = score_chains(recs, base_out, step_out)
                    for r, b, s in zip(recs, base_out, step_out):
                        pf.write(json.dumps({'id': r['id'], 'base_output': b, 'step_outputs': s}, ensure_ascii=False) + '\n')
                else:
                    preds = gen.generate([r['messages'] for r in recs])
                    metrics[f'{lang}/{mode}'] = score_single(recs, preds)
                    for r, p in zip(recs, preds):
                        pf.write(json.dumps({'id': r['id'], 'output': p, 'parsed': protocol.parse_output(p), 'truth': r['truth']}, ensure_ascii=False) + '\n')
                print(lang, mode, json.dumps(metrics[f'{lang}/{mode}'], ensure_ascii=False)[:600], flush=True)
    metrics['seconds'] = round(time.time() - t0, 1)
    (out / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding='utf8')
    print(json.dumps(metrics, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
