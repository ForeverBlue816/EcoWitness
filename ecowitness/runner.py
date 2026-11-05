"""Generation loop shared by the local runner (evaluate.py) and the API runner (evaluate_api.py). No torch here, so the
API runner and the scorer work on a laptop without GPU packages.

A generator is any object with `generate(list_of_message_lists) -> list_of_strings`.

predictions.jsonl, one line per record:
    base / revise   {"id", "lang", "mode", "output"}                 (+ "prior": own base output under --own_history)
    chain           {"id", "lang", "mode", "base_output", "step_outputs": [...]}
Scoring needs the records with their truth (score.py); the public test pack has no truth, so there the loop only writes
predictions and a format/coverage check.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import protocol

MODE_ORDER = {'base': 0, 'revise': 1, 'chain': 2}


def load_jsonl(path):
    with Path(path).open(encoding='utf8') as f:
        return [json.loads(l) for l in f if l.strip()]


def has_truth(records):
    return bool(records) and all('truth' in r for r in records)


def with_own_prior(record, prior_output):
    """Copy of a revise record whose prior assistant turn is the model's own base reply instead of the gold one."""
    msgs = list(record['messages'])
    msgs[-2] = {'role': 'assistant', 'content': prior_output.strip()}
    return msgs


def run_chains(gen, chains, teacher_forced=False):
    """Multi-step revisions. Own history: every step sees the model's previous replies. Teacher forced (needs targets,
    so owner-only): every step sees the gold previous replies. Returns (base outputs, per-chain step outputs)."""
    if teacher_forced and not all('target' in c and all('target' in s for s in c['steps']) for c in chains):
        raise ValueError('teacher-forced chains need the records with targets (not available in the public test pack)')
    histories = [list(c['messages']) for c in chains]
    base_out = gen.generate(histories)
    step_out = [[] for _ in chains]
    for k in range(max(len(c['steps']) for c in chains)):
        idx = [i for i, c in enumerate(chains) if len(c['steps']) > k]
        for i in idx:
            c = chains[i]
            if teacher_forced:
                prev = c['target'] if k == 0 else c['steps'][k - 1]['target']
            else:
                prev = base_out[i] if k == 0 else step_out[i][-1]
            histories[i] = histories[i] + [{'role': 'assistant', 'content': prev.strip()}, c['steps'][k]['message']]
        outs = gen.generate([histories[i] for i in idx])
        for i, o in zip(idx, outs):
            step_out[i].append(o)
    return base_out, step_out


def run(gen, data_dir, split, langs, modes, out, families=None, max_items=0, teacher_forced=False, own_history=False, meta=None, log=print):
    """Generate for every requested (lang, mode), write predictions.jsonl, and score when the records carry truth."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_records, pred_lines = [], []
    base_outputs = {}  # (qa_id, lang) -> own base output, for --own_history
    for lang in langs:
        path = Path(data_dir) / f'{split}.{lang}.jsonl'
        if not path.exists():
            path = Path(data_dir) / f'{split}.public.{lang}.jsonl'
        rows = load_jsonl(path)
        if families:
            rows = [r for r in rows if r['family'] in families]
        for mode in sorted(modes, key=MODE_ORDER.get):
            recs = [r for r in rows if r['mode'] == mode]
            if max_items:
                recs = recs[:max_items]
            if not recs:
                continue
            if mode == 'chain':
                base_out, step_out = run_chains(gen, recs, teacher_forced)
                for r, b, s in zip(recs, base_out, step_out):
                    pred_lines.append({'id': r['id'], 'lang': lang, 'mode': mode, 'history': 'gold' if teacher_forced else 'own',
                                       'base_output': b, 'step_outputs': s})
            else:
                msgs, priors = [], []
                for r in recs:
                    prior = base_outputs.get((r['qa_id'], lang)) if (own_history and mode == 'revise') else None
                    msgs.append(with_own_prior(r, prior) if prior is not None else r['messages'])
                    priors.append(prior)
                outs = gen.generate(msgs)
                for r, o, prior in zip(recs, outs, priors):
                    line = {'id': r['id'], 'lang': lang, 'mode': mode, 'output': o}
                    if prior is not None:
                        line['prior'] = prior
                    pred_lines.append(line)
                    if mode == 'base':
                        base_outputs[(r['qa_id'], lang)] = o
            all_records.extend(recs)
            log(f'{lang}/{mode}: {len(recs)} records done, {time.time() - t0:.0f}s', flush=True)
    with (out / 'predictions.jsonl').open('w', encoding='utf8') as f:
        for line in pred_lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    from . import score
    report = score.score_all(all_records, pred_lines, meta={**(meta or {}), 'split': split, 'teacher_forced': teacher_forced,
                                                             'own_history': own_history, 'seconds': round(time.time() - t0, 1)})
    (out / ('metrics.json' if has_truth(all_records) else 'check.json')).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf8')
    log(json.dumps(report.get('scorecard', report), ensure_ascii=False, indent=1))
    return report
