"""Build the EcoWitness dataset from the benchmark (v10 by default, EW_BENCH selects another): witness-protocol base
questions, single-step counterfactual revisions (flip / keep-cited / keep-uncited), multi-step revision chains, and
silent counterfactuals (the edited evidence asked as a fresh question: no notice, no previous reply) along the
ecological axes (taxonomy, trait, abundance, status) and the generic ones (witness, image, record, presentation).

Run from the `ecowitness` directory (needs the benchmark release, its tools and the v5 source reports next to it):
    python -m ecowitness.build_revisions --splits train val test --langs en --out data

Record modes: base, revise, chain (unchanged from v1 of the data; same seeds, byte-identical), silent (new).
A silent record carries `axis`, `effect` (invariant / flip / equivariant), `edit_class`, `base_answer` and, for a
permutation, `perm` (old letter -> new letter). Val/test also get removal probes (one load-bearing and one other
witness removed) for the attribution-faithfulness metric.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import random
from pathlib import Path

from . import protocol, rules

HERE = Path(__file__).resolve().parents[1]
ROOT = rules.ROOT
V8 = rules.V8
V5 = ROOT / 'qualitative_vqa_20261023_v5'
A_EDITS = 1          # image families: one swap per question (the image is the only witness)
EDITS_PER_ITEM = 4
SILENT_PER_ITEM = 2  # silent counterfactuals per question: one invariant + one flip, ecological axes first
PERM_RATE = 0.5      # share of train questions that also get an option-permuted copy (val/test: every question)
PROBE_REMOVALS = 2   # val/test: silent removals of one load-bearing and one other witness (attribution faithfulness)


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf8').splitlines() if x.strip()]


def load_pools(evidence, records_by_split):
    images, labels, by_family, by_split = {}, {}, collections.defaultdict(list), collections.defaultdict(list)
    for split, rows in records_by_split.items():
        for r in rows['en']:
            im = r['inputs'].get('image')
            if im and r['sample_id'] not in images:
                images[r['sample_id']] = {'path': im['path'], 'date': im['acquisition_date']}
                by_split[split].append(r['sample_id'])
    for e in evidence.values():
        if e['task'][0] == 'A' and e['tier'] in ('easy', 'medium') and e['sample_id'] in images:
            labels[(e['sample_id'], e['task'])] = e['label']['category']
            by_family[(e['task'], e['split'])].append(e['sample_id'])
    dna_path = V8 / 'reports/dna_evidence.json'          # v9 and v10 ship their own DNA evidence
    if not dna_path.exists():
        dna_path = V5 / 'reports/dna_evidence.json'      # v8 used the v5 evidence
    dna = json.loads(dna_path.read_text(encoding='utf8'))
    return rules.Pools(dna, images, labels, dict(by_family), dict(by_split))


def image_label(fam, private):
    if fam[0] == 'A':
        return private['label']['category']
    if fam == 'C1-SceneAndCommunity':
        return private['visual_label']['category']
    return None


def classify(edit, base, new):
    if new['answer'] != base['answer']:
        return 'flip'
    return 'keep_cited' if rules.touches(edit, base['support']) else 'keep_uncited'


def public_edit(edit):
    e = {k: v for k, v in edit.items() if k not in ('card',)}
    if 'card' in edit:
        e['name'] = edit['card']['taxon_label']
    return e


def finish_edit(edit, new_state_by_lang):
    """Fill the rendered card strings used by the revision notice."""
    if edit['type'] in ('add', 'reassign'):
        for lang, st in new_state_by_lang.items():
            c = st.card_by_wid(edit['wid'])
            text = protocol.render_card(c, lang)
            if edit['type'] == 'reassign':
                text = text.split('] ', 1)[1]  # the notice already names the witness
            edit['rendered' if lang == 'en' else 'rendered_zh'] = text
        if 'rendered_zh' not in edit:
            edit['rendered_zh'] = edit['rendered']
        if 'rendered' not in edit:
            edit['rendered'] = edit['rendered_zh']


def select_edits(cands, fam, rng, n_max):
    """cands: list of (edit, analysis, cls). Balanced pick: flips, one keep_cited, keep_uncited."""
    by = collections.defaultdict(list)
    for c in cands:
        by[c[2]].append(c)
    for v in by.values():
        rng.shuffle(v)
    if fam[0] == 'A':  # the image is the only witness: a same-category swap is a "keep although the cited witness changed"
        pool = by['flip'] if (by['flip'] and (not by['keep_cited'] or rng.random() < 0.6)) else by['keep_cited']
        return pool[:A_EDITS]
    out = by['flip'][:2] + by['keep_cited'][:1] + by['keep_uncited'][:1]
    spare = by['flip'][2:] + by['keep_uncited'][1:] + by['keep_cited'][1:]
    out += spare[: max(0, n_max - len(out))]
    return out[:n_max]


def select_silent(cands, rng, n):
    """cands: list of (edit, states, analysis, cls). Alternate invariant / flip picks, ecological axes before generic
    ones, random within an axis group."""
    def ordered(xs):
        eco = [x for x in xs if x[0]['axis'] in rules.ECO_AXES]
        gen = [x for x in xs if x[0]['axis'] not in rules.ECO_AXES]
        rng.shuffle(eco)
        rng.shuffle(gen)
        return eco + gen
    inv = ordered([x for x in cands if x[3] != 'flip'])
    fl = ordered([x for x in cands if x[3] == 'flip'])
    out = []
    for i in range(n):
        first, second = (inv, fl) if i % 2 == 0 else (fl, inv)
        src = first if first else second
        if src:
            out.append(src.pop(0))
    return out


def try_edit(edit, states, langs, pools):
    """Apply one edit in every language; -> (edit, new states, analyses) or None when it is ill-posed."""
    e = copy.deepcopy(edit)
    try:
        ns = {lang: rules.apply_edit(states[lang], copy.deepcopy(e) if lang != 'en' else e, pools) for lang in langs}
    except (StopIteration, KeyError, ValueError):
        return None
    ana = {lang: rules.analyse(ns[lang]) for lang in langs}
    if any(a is None for a in ana.values()):
        return None
    if len({a['answer'] for a in ana.values()}) != 1:
        raise AssertionError(f'language disagreement after edit {e}')
    return e, ns, ana


def silent_records(r, recs, p, lab, states, base, langs, pools, split, args, stats, meta):
    """Silent counterfactuals for one question: edited evidence asked afresh (no notice), removal probes (val/test)
    and an option permutation."""
    fam = r['task']
    rng = random.Random(f'{args.seed}:{r["qa_id"]}:silent')
    cands = []
    edits = rules.ecological_edits(states['en'], pools, rng)
    # generic edits; a sampling-date change is announced in the context text, not in the evidence, so it cannot be silent
    edits += [e for e in rules.candidate_edits(states['en'], pools, rng, split) if not (e['type'] == 'set_field' and e.get('field') == 'sampling_date')]
    for edit in edits:
        t = try_edit(edit, states, langs, pools)
        if t is None:
            stats[split]['ill_posed_silent'][fam] += 1
            continue
        e, ns, ana = t
        cands.append((e, ns, ana, classify(e, base['en'], ana['en'])))
    chosen = select_silent(cands, rng, args.silent_per_item)
    if split != 'train' and args.probe_removals:
        lb = set(base['en']['load_bearing'])
        rem = rules.removable(states['en'])
        picks = rng.sample([w for w in rem if w in lb], min(1, sum(w in lb for w in rem))) + rng.sample([w for w in rem if w not in lb], min(args.probe_removals - 1, sum(w not in lb for w in rem)))
        for wid in picks:
            t = try_edit({'type': 'remove', 'wid': wid, 'axis': 'witness', 'probe': True}, states, langs, pools)
            if t is not None:
                e, ns, ana = t
                chosen.append((e, ns, ana, classify(e, base['en'], ana['en'])))
    out = {lang: [] for lang in langs}
    k = 0
    for e, ns, ana, cls in chosen:
        k += 1
        effect = 'flip' if cls == 'flip' else 'invariant'
        for lang in langs:
            truth = {kk: ana[lang][kk] for kk in ('answer', 'support', 'load_bearing')}
            msgs = [{'role': 'system', 'content': protocol.SYSTEM[lang]}, protocol.user_message(ns[lang], recs[lang], lang)]
            out[lang].append({'id': f'{r["qa_id"]}|{lang}|s{k}', 'mode': 'silent', 'lang': lang, **meta, 'messages': msgs,
                              'target': protocol.render_target(truth, False), 'truth': truth, 'pair_id': r['qa_id'], 'edit_class': cls,
                              'edit': public_edit(e), 'axis': e['axis'], 'effect': effect, 'base_answer': base[lang]['answer'], 'probe': bool(e.get('probe'))})
        stats[split]['silent_axis'][e['axis']] += 1
        stats[split]['silent_effect'][effect] += 1
        if e.get('relation'):
            stats[split]['silent_relation'][e['relation']] += 1
    # option permutation: same evidence, moved options -> the answer letter moves with them
    if rng.random() < (args.perm_rate if split == 'train' else 1.0):
        k += 1
        perm = rules.random_perm(list(r['options']), rng)
        for lang in langs:
            prec = rules.permute_options(recs[lang], perm)
            st = rules.build_state(prec, p, lang, lab)
            ana = rules.analyse(st)
            assert ana is not None and ana['answer'] == perm[base[lang]['answer']], r['qa_id']
            truth = {kk: ana[kk] for kk in ('answer', 'support', 'load_bearing')}
            msgs = [{'role': 'system', 'content': protocol.SYSTEM[lang]}, protocol.user_message(st, prec, lang)]
            out[lang].append({'id': f'{r["qa_id"]}|{lang}|s{k}', 'mode': 'silent', 'lang': lang, **meta, 'messages': msgs,
                              'target': protocol.render_target(truth, False), 'truth': truth, 'pair_id': r['qa_id'], 'edit_class': 'permute',
                              'edit': {'type': 'permute'}, 'axis': 'presentation', 'effect': 'equivariant', 'base_answer': base[lang]['answer'],
                              'perm': {**perm, 'none': 'none'}, 'probe': False})
        stats[split]['silent_axis']['presentation'] += 1
        stats[split]['silent_effect']['equivariant'] += 1
    return out


def build_split(split, langs, records, evidence, pools, args, stats):
    rng_master = random.Random(f'{args.seed}:{split}')
    en_rows = records['en']
    zh_map = {r['qa_id']: r for r in records.get('zh', [])}
    out = {lang: [] for lang in langs}
    holdout = set(args.holdout_families) if split == 'train' else set()
    for r in en_rows:
        fam = r['task']
        if fam in holdout:
            stats[split]['held_out'][fam] += 1
            continue
        p = evidence[r['qa_id']]
        rng = random.Random(f'{args.seed}:{r["qa_id"]}')
        lab = image_label(fam, p)
        recs = {'en': r}
        if 'zh' in langs:
            recs['zh'] = zh_map[r['qa_id']]
        states = {lang: rules.build_state(recs[lang], p, lang, lab) for lang in langs}
        base = {lang: rules.analyse(states[lang]) for lang in langs}
        if any(b is None for b in base.values()):
            stats[split]['ill_posed_base'][fam] += 1
            continue
        assert all(b['answer'] == recs[lang]['answer'] for lang, b in base.items()), r['qa_id']
        meta = {'qa_id': r['qa_id'], 'sample_id': r['sample_id'], 'family': fam, 'group': fam[0], 'tier': p['tier'], 'split': split}
        pair_id = r['qa_id']
        # ---- base question
        for lang in langs:
            msgs = [{'role': 'system', 'content': protocol.SYSTEM[lang]}, protocol.user_message(states[lang], recs[lang], lang)]
            truth = {k: base[lang][k] for k in ('answer', 'support', 'load_bearing')}
            out[lang].append({'id': f'{r["qa_id"]}|{lang}|base', 'mode': 'base', 'lang': lang, **meta, 'messages': msgs,
                              'target': protocol.render_target(truth, False), 'truth': truth, 'pair_id': pair_id, 'edit_class': None, 'edit': None})
        stats[split]['base'][fam] += 1
        # ---- single-step revisions
        cands = []
        for edit in rules.candidate_edits(states['en'], pools, rng, split):
            e = copy.deepcopy(edit)
            try:
                new_states = {lang: rules.apply_edit(states[lang], copy.deepcopy(e) if lang != 'en' else e, pools) for lang in langs}
            except (StopIteration, KeyError, ValueError):
                continue
            ana = {lang: rules.analyse(new_states[lang]) for lang in langs}
            if any(a is None for a in ana.values()):
                stats[split]['ill_posed_edit'][fam] += 1
                continue
            if len({a['answer'] for a in ana.values()}) != 1:
                raise AssertionError(f'language disagreement after edit {e} on {r["qa_id"]}')
            # the applied edit dict for 'en' carries wid/old/etc.; zh copies were filled identically
            finish_edit(e, new_states)
            cands.append((e, ana, classify(e, base['en'], ana['en'])))
        chosen = select_edits(cands, fam, rng, args.edits_per_item)
        for k, (e, ana, cls) in enumerate(chosen, 1):
            for lang in langs:
                truth = {'changed': ana[lang]['answer'] != base[lang]['answer'], **{kk: ana[lang][kk] for kk in ('answer', 'support', 'load_bearing')}}
                msgs = [{'role': 'system', 'content': protocol.SYSTEM[lang]}, protocol.user_message(states[lang], recs[lang], lang),
                        {'role': 'assistant', 'content': protocol.render_target(base[lang], False)}, protocol.revision_message(e, lang)]
                out[lang].append({'id': f'{r["qa_id"]}|{lang}|e{k}', 'mode': 'revise', 'lang': lang, **meta, 'messages': msgs,
                                  'target': protocol.render_target(truth, True), 'truth': truth, 'pair_id': pair_id, 'edit_class': cls, 'edit': public_edit(e)})
            stats[split]['revise'][cls] += 1
            stats[split]['revise_by_family'][fam] += 1
            stats[split]['edit_types'][e['type']] += 1
            stats[split]['revised_answer']['none' if ana['en']['answer'] == 'none' else 'option'] += 1
        # ---- multi-step chains
        n_chains = args.train_chains if split == 'train' else args.chains_per_item
        for c in range(n_chains):
            length = rng.randint(2, 5)
            cur = {lang: states[lang] for lang in langs}
            prev = {lang: base[lang] for lang in langs}
            steps = {lang: [] for lang in langs}
            for step in range(length):
                step_cands = []
                for edit in rules.candidate_edits(cur['en'], pools, rng, split):
                    e = copy.deepcopy(edit)
                    try:
                        ns = {lang: rules.apply_edit(cur[lang], copy.deepcopy(e) if lang != 'en' else e, pools) for lang in langs}
                    except (StopIteration, KeyError, ValueError):
                        continue
                    ana = {lang: rules.analyse(ns[lang]) for lang in langs}
                    if any(a is None for a in ana.values()):
                        continue
                    finish_edit(e, ns)
                    step_cands.append((e, ns, ana, classify(e, prev['en'], ana['en'])))
                if not step_cands:
                    break
                flips = [x for x in step_cands if x[3] == 'flip']
                keeps = [x for x in step_cands if x[3] != 'flip']
                pick = rng.choice(flips if flips and (not keeps or rng.random() < 0.5) else keeps)
                e, ns, ana, cls = pick
                for lang in langs:
                    truth = {'changed': ana[lang]['answer'] != prev[lang]['answer'], **{kk: ana[lang][kk] for kk in ('answer', 'support', 'load_bearing')}}
                    steps[lang].append({'message': protocol.revision_message(e, lang), 'target': protocol.render_target(truth, True), 'truth': truth,
                                        'edit_class': cls, 'edit': public_edit(e)})
                cur, prev = ns, ana
            if len(steps['en']) >= 2:
                for lang in langs:
                    msgs = [{'role': 'system', 'content': protocol.SYSTEM[lang]}, protocol.user_message(states[lang], recs[lang], lang)]
                    out[lang].append({'id': f'{r["qa_id"]}|{lang}|chain{c + 1}', 'mode': 'chain', 'lang': lang, **meta, 'messages': msgs,
                                      'target': protocol.render_target(base[lang], False), 'truth': {k: base[lang][k] for k in ('answer', 'support', 'load_bearing')},
                                      'pair_id': pair_id, 'edit_class': None, 'edit': None, 'steps': steps[lang]})
                stats[split]['chains'][len(steps['en'])] += 1
        # ---- silent counterfactuals (own RNG stream: base / revise / chain records above stay byte-identical)
        if args.silent_per_item or args.probe_removals:
            srows = silent_records(r, recs, p, lab, states, base, langs, pools, split, args, stats, meta)
            for lang in langs:
                out[lang].extend(srows[lang])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    ap.add_argument('--langs', nargs='+', default=['en'])
    ap.add_argument('--out', default=str(HERE / 'data'))
    ap.add_argument('--edits-per-item', type=int, default=EDITS_PER_ITEM)
    ap.add_argument('--chains-per-item', type=int, default=1, help='chains per val/test question')
    ap.add_argument('--train-chains', type=int, default=0, help='chains per train question (0 = single-step training only)')
    ap.add_argument('--silent-per-item', type=int, default=SILENT_PER_ITEM, help='silent counterfactuals per question (0 = none)')
    ap.add_argument('--perm-rate', type=float, default=PERM_RATE, help='share of train questions with an option-permuted copy (val/test: all)')
    ap.add_argument('--probe-removals', type=int, default=PROBE_REMOVALS, help='val/test: silent witness removals per question for attribution faithfulness')
    ap.add_argument('--holdout-families', nargs='*', default=['C1-SceneAndCommunity'], help='families removed from train (compositional generalisation)')
    ap.add_argument('--seed', default='ecowitness-v1')
    ap.add_argument('--limit', type=int, default=0, help='debug: only the first N questions per split')
    args = ap.parse_args()

    evidence = {e['qa_id']: e for e in jsonl(V8 / 'reports/answer_evidence.jsonl')}
    records = {}
    for split in ('train', 'val', 'test'):
        records[split] = {'en': jsonl(V8 / f'{split}.jsonl'), 'zh': jsonl(V8 / f'{split}.zh.jsonl')}
    pools = load_pools(evidence, records)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {s: collections.defaultdict(collections.Counter) for s in args.splits}
    for split in args.splits:
        recs = records[split]
        if args.limit:
            recs = {k: v[: args.limit] for k, v in recs.items()}
        out = build_split(split, args.langs, recs, evidence, pools, args, stats)
        for lang, rows in out.items():
            p = out_dir / f'{split}.{lang}.jsonl'
            with p.open('w', encoding='utf8') as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
            stats[split]['rows'][lang] = len(rows)
        print(split, json.dumps({k: dict(v) if isinstance(v, dict) else v for k, v in stats[split].items()}, ensure_ascii=False))
    (out_dir / 'stats.json').write_text(json.dumps({s: {k: (dict(v) if isinstance(v, dict) else v) for k, v in st.items()} for s, st in stats.items()}, ensure_ascii=False, indent=1), encoding='utf8')
    (out_dir / 'build_args.json').write_text(json.dumps(vars(args), ensure_ascii=False, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
