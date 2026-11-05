"""Assemble the public test pack for external evaluators.

    python scripts/build_test_pack.py            # -> ../ecowitness_testpack_v9/ and ../ecowitness_testpack_v9.tar.gz

Public test records keep only what is needed to run a model: the conversation, the revision notices, ids and
family/tier labels. Removed: target, truth, edit_class (flip / keep = the label), the full edit dicts (only type and the
witness ids named in the notice stay), pair_id. The dev set (= val split) is shipped complete, with answers, so evaluators
can check their pipeline end to end with score.py. Images: only the PNGs referenced by test and dev.

Known, documented limitation: a single-step revision record shows the gold base answer as the model's previous turn
(that is the task). The pack is for cooperative evaluation; base items must be answered from the base records.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]          # ecowitness/
ROOT = HERE.parent                                  # folder that holds satellite_rgb/
sys.path.insert(0, str(HERE))
from ecowitness.runner import load_jsonl  # noqa: E402

NAME = 'ecowitness_testpack_v9'
LANGS = ('en',)   # the released benchmark is English only
PACK = ROOT / NAME
PUBLIC_KEYS = ('id', 'qa_id', 'sample_id', 'family', 'group', 'tier', 'lang', 'mode', 'messages')
EDIT_KEYS = ('type', 'wid', 'old_wid')
CODE = ['__init__.py', 'protocol.py', 'runner.py', 'score.py', 'evaluate.py', 'evaluate_api.py', 'data.py']


def public_record(r):
    p = {k: r[k] for k in PUBLIC_KEYS}
    if r['mode'] == 'revise':
        p['edit'] = {k: r['edit'][k] for k in EDIT_KEYS if k in r['edit']}
    if r['mode'] == 'chain':
        p['steps'] = [{'message': s['message'], 'edit': {k: s['edit'][k] for k in EDIT_KEYS if k in s['edit']}} for s in r['steps']]
    return p


def images_of(records):
    paths = set()
    for r in records:
        msgs = list(r['messages']) + [s['message'] for s in r.get('steps', [])]
        for m in msgs:
            if isinstance(m['content'], list):
                paths.update(it['path'] for it in m['content'] if it.get('type') == 'image')
    return sorted(paths)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if PACK.exists():
        shutil.rmtree(PACK)
    (PACK / 'ecowitness').mkdir(parents=True)
    manifest = {'name': NAME, 'source': 'ecowitness/data (built from qualitative_benchmark_20261105_v9)', 'files': {}, 'counts': {}}
    images = set()
    for lang in LANGS:
        test = load_jsonl(HERE / 'data' / f'test.{lang}.jsonl')
        pub = [public_record(r) for r in test]
        for r in pub:
            blob = json.dumps(r, ensure_ascii=False)
            assert '"truth"' not in blob and '"target"' not in blob and 'edit_class' not in blob, r['id']
        out = PACK / f'test.public.{lang}.jsonl'
        out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in pub), encoding='utf8')
        manifest['counts'][f'test.{lang}'] = {m: sum(r['mode'] == m for r in pub) for m in ('base', 'revise', 'chain')}
        manifest['counts'][f'test.{lang}']['chain_steps'] = sum(len(r['steps']) for r in pub if r['mode'] == 'chain')
        images.update(images_of(test))
        dev = load_jsonl(HERE / 'data' / f'val.{lang}.jsonl')
        (PACK / f'dev.{lang}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in dev), encoding='utf8')
        manifest['counts'][f'dev.{lang}'] = {m: sum(r['mode'] == m for r in dev) for m in ('base', 'revise', 'chain')}
        images.update(images_of(dev))
    (PACK / 'satellite_rgb').mkdir()
    for p in sorted(images):
        shutil.copy2(ROOT / p, PACK / p)
    manifest['counts']['images'] = len(images)
    for f in CODE:
        shutil.copy2(HERE / 'ecowitness' / f, PACK / 'ecowitness' / f)
    shutil.copy2(HERE / 'docs' / 'testpack_README.md', PACK / 'README.md')
    (PACK / 'requirements_api.txt').write_text('openai>=1.40\nnumpy\n', encoding='utf8')
    shutil.copy2(HERE / 'requirements.txt', PACK / 'requirements_local.txt')
    for p in sorted(PACK.rglob('*')):
        if p.is_file() and p.parent != PACK / 'satellite_rgb':
            manifest['files'][p.relative_to(PACK).as_posix()] = {'sha256': sha256(p), 'bytes': p.stat().st_size}
    manifest['files']['satellite_rgb/'] = {'files': len(images), 'bytes': sum(p.stat().st_size for p in (PACK / 'satellite_rgb').iterdir())}
    (PACK / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + '\n', encoding='utf8')
    tar = ROOT / f'{NAME}.tar.gz'
    with tarfile.open(tar, 'w:gz') as tf:
        tf.add(PACK, arcname=NAME)
    (ROOT / f'{NAME}.tar.gz.sha256').write_text(f'{sha256(tar)}  {NAME}.tar.gz\n', encoding='ascii')
    print(json.dumps(manifest['counts'], ensure_ascii=False, indent=1))
    print('pack:', PACK, '\ntar :', tar, round(tar.stat().st_size / 1e6, 1), 'MB')


if __name__ == '__main__':
    main()
