"""Run the witness protocol through an OpenAI-compatible chat API (GPT, Gemini, Qwen-Max, a local vLLM server) as a
prompt-only baseline, then score with the same metrics as evaluate.py.

    export OPENAI_API_KEY=...; export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    python -m ecowitness.evaluate_api --model qwen3-vl-plus --split test --langs en --modes base revise --out runs/api_qwen
Images are sent as base64 PNG. Results are cached per record id so an interrupted run can resume.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path

from . import protocol
from .data import load_jsonl
from .evaluate import run_chains, score_chains, score_single

HERE = Path(__file__).resolve().parents[1]


class ApiGenerator:
    def __init__(self, model, workspace, cache_path, temperature=0.0, max_tokens=256, extra_body=None, sleep=0.0):
        from openai import OpenAI
        self.client = OpenAI()
        self.model, self.workspace = model, Path(workspace)
        self.temperature, self.max_tokens, self.extra_body, self.sleep = temperature, max_tokens, extra_body or {}, sleep
        self.cache_path = cache_path
        self.cache = {}
        if cache_path.exists():
            for l in cache_path.read_text(encoding='utf8').splitlines():
                if l.strip():
                    d = json.loads(l)
                    self.cache[d['key']] = d['output']

    def to_api(self, msgs):
        out = []
        for m in msgs:
            if isinstance(m['content'], str):
                out.append({'role': m['role'], 'content': m['content']})
                continue
            parts = []
            for it in m['content']:
                if it['type'] == 'image':
                    b = base64.b64encode((self.workspace / it['path']).read_bytes()).decode()
                    parts.append({'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b}'}})
                else:
                    parts.append({'type': 'text', 'text': it['text']})
            out.append({'role': m['role'], 'content': parts})
        return out

    def generate(self, message_lists):
        outs = []
        for msgs in message_lists:
            key = json.dumps(msgs, ensure_ascii=False, sort_keys=True)[:4000] + str(hash(json.dumps(msgs, ensure_ascii=False, sort_keys=True)))
            if key in self.cache:
                outs.append(self.cache[key])
                continue
            for attempt in range(5):
                try:
                    resp = self.client.chat.completions.create(model=self.model, messages=self.to_api(msgs), temperature=self.temperature,
                                                               max_tokens=self.max_tokens, extra_body=self.extra_body)
                    text = resp.choices[0].message.content or ''
                    break
                except Exception as ex:  # noqa: BLE001
                    print('retry', attempt, ex)
                    time.sleep(2 * (attempt + 1))
            else:
                text = ''
            self.cache[key] = text
            with self.cache_path.open('a', encoding='utf8') as f:
                f.write(json.dumps({'key': key, 'output': text}, ensure_ascii=False) + '\n')
            outs.append(text)
            if self.sleep:
                time.sleep(self.sleep)
        return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--workspace', default=str(HERE.parent))
    ap.add_argument('--data_dir', default=str(HERE / 'data'))
    ap.add_argument('--split', default='test')
    ap.add_argument('--langs', nargs='+', default=['en'])
    ap.add_argument('--modes', nargs='+', default=['base', 'revise'])
    ap.add_argument('--families', nargs='*', default=None)
    ap.add_argument('--max_items', type=int, default=0)
    ap.add_argument('--teacher_forced', action='store_true')
    ap.add_argument('--no_thinking', action='store_true', help='send chat_template_kwargs.enable_thinking=false (vLLM / DashScope)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    extra = {'chat_template_kwargs': {'enable_thinking': False}} if a.no_thinking else None
    gen = ApiGenerator(a.model, a.workspace, out / 'cache.jsonl', extra_body=extra)
    metrics = {'model': a.model, 'split': a.split, 'api_base': os.environ.get('OPENAI_BASE_URL')}
    with (out / 'predictions.jsonl').open('w', encoding='utf8') as pf:
        for lang in a.langs:
            rows = load_jsonl(Path(a.data_dir) / f'{a.split}.{lang}.jsonl')
            if a.families:
                rows = [r for r in rows if r['family'] in a.families]
            for mode in a.modes:
                recs = [r for r in rows if r['mode'] == mode][: a.max_items or None]
                if not recs:
                    continue
                if mode == 'chain':
                    b, s = run_chains(gen, recs, a.teacher_forced)
                    metrics[f'{lang}/chain'] = score_chains(recs, b, s)
                    for r, bo, so in zip(recs, b, s):
                        pf.write(json.dumps({'id': r['id'], 'base_output': bo, 'step_outputs': so}, ensure_ascii=False) + '\n')
                else:
                    preds = gen.generate([r['messages'] for r in recs])
                    metrics[f'{lang}/{mode}'] = score_single(recs, preds)
                    for r, p in zip(recs, preds):
                        pf.write(json.dumps({'id': r['id'], 'output': p, 'parsed': protocol.parse_output(p), 'truth': r['truth']}, ensure_ascii=False) + '\n')
                print(lang, mode, json.dumps(metrics[f'{lang}/{mode}'], ensure_ascii=False)[:600], flush=True)
    (out / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
