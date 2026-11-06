"""Run the witness protocol through an OpenAI-compatible chat API (GPT, Gemini, Qwen-Max, a local vLLM server) as a
prompt-only baseline, then score with the same metrics as evaluate.py. Needs only `openai` (no torch).

    export OPENAI_API_KEY=...; export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    python -m ecowitness.evaluate_api --model qwen3-vl-plus --split test --langs en zh --out runs/api_qwen --no_thinking
Images are sent as base64 PNG. Replies are cached per conversation so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
from pathlib import Path

from . import runner

HERE = Path(__file__).resolve().parents[1]


class ApiGenerator:
    def __init__(self, model, workspace, cache_path, temperature=0.0, max_tokens=256, extra_body=None, sleep=0.0, workers=1):
        from openai import OpenAI
        self.client = OpenAI()
        self.model, self.workspace = model, Path(workspace)
        self.temperature, self.max_tokens, self.extra_body, self.sleep, self.workers = temperature, max_tokens, extra_body or {}, sleep, workers
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

    def one(self, msgs):
        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(model=self.model, messages=self.to_api(msgs), temperature=self.temperature,
                                                           max_tokens=self.max_tokens, extra_body=self.extra_body)
                return resp.choices[0].message.content or ''
            except Exception as ex:  # noqa: BLE001
                print('retry', attempt, ex, flush=True)
                time.sleep(2 * (attempt + 1))
        return ''

    def generate(self, message_lists):
        keys = [hashlib.sha256((self.model + json.dumps(m, ensure_ascii=False, sort_keys=True)).encode()).hexdigest() for m in message_lists]
        todo = [(k, m) for k, m in zip(keys, message_lists) if k not in self.cache]
        if todo:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.workers) as ex, self.cache_path.open('a', encoding='utf8') as f:
                for (k, _), text in zip(todo, ex.map(lambda km: self.one(km[1]), todo)):
                    self.cache[k] = text
                    f.write(json.dumps({'key': k, 'output': text}, ensure_ascii=False) + '\n')
                    if self.sleep:
                        time.sleep(self.sleep)
        return [self.cache[k] for k in keys]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--workspace', default=str(HERE.parent), help='folder that holds satellite_rgb/')
    ap.add_argument('--data_dir', default=str(HERE / 'data'))
    ap.add_argument('--split', default='test')
    ap.add_argument('--langs', nargs='+', default=['en'])
    ap.add_argument('--modes', nargs='+', default=['base', 'silent', 'revise', 'chain'])
    ap.add_argument('--families', nargs='*', default=None)
    ap.add_argument('--max_items', type=int, default=0)
    ap.add_argument('--teacher_forced', action='store_true', help='chains: gold previous answers (needs targets, owner only)')
    ap.add_argument('--own_history', action='store_true', help='revisions: the model\'s own base answer as its prior reply')
    ap.add_argument('--no_thinking', action='store_true', help='send chat_template_kwargs.enable_thinking=false (vLLM / DashScope)')
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--max_tokens', type=int, default=256, help='raise for thinking models whose reasoning counts against the limit')
    ap.add_argument('--workers', type=int, default=4, help='parallel requests')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    extra = {'chat_template_kwargs': {'enable_thinking': False}} if a.no_thinking else None
    gen = ApiGenerator(a.model, a.workspace, out / 'cache.jsonl', temperature=a.temperature, max_tokens=a.max_tokens, extra_body=extra, workers=a.workers)
    runner.run(gen, a.data_dir, a.split, a.langs, a.modes, out, families=a.families, max_items=a.max_items, teacher_forced=a.teacher_forced,
               own_history=a.own_history, meta={'model': a.model, 'api_base': os.environ.get('OPENAI_BASE_URL'), 'temperature': a.temperature})


if __name__ == '__main__':
    main()
