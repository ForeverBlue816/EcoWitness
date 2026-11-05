"""Scoring for the witness protocol: accuracy, abstention, citation faithfulness, revision behaviour, multi-step drift,
cross-lingual consistency, and bootstrap confidence intervals.

    python -m ecowitness.score --data data/test.en.jsonl data/test.zh.jsonl --predictions runs/x/predictions.jsonl --out runs/x/metrics.json

Records without truth (the public test pack) get a format / coverage check only (check.json).

Metric families
  accuracy          option accuracy overall and by group / family / difficulty tier, 95 % bootstrap CI
  abstention        precision / recall / F1 of answering "none" (no option fits the evidence); false_none_rate
  citation          support and load_bearing precision / recall / F1; load_bearing_exact; sufficiency (every truly
                    load-bearing witness is inside the cited support); invalid_citation_rate (cites an ID that is not in
                    the evidence); faithful_accuracy (answer right AND load-bearing set exactly right)
  revision          post-revision accuracy by edit class (flip / keep_cited / keep_uncited) and by edit type;
                    revision_f1 of the "changed" line; contagion_rate (changed although an unrelated witness changed);
                    redundancy_overreaction_rate (changed although only a redundant cited witness changed);
                    inertia_rate (kept although a load-bearing witness changed); stale_citation_rate (cites a withdrawn
                    witness); self_contradiction_rate ("changed" line disagrees with the answer actually given)
  chain             accuracy by step with the model's own history, final-step accuracy, whole-chain accuracy, drift,
                    recovery / slip rates between consecutive steps
  cross_lingual     agreement of the en and zh answers to the same question, both-correct rate
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

from . import protocol
from .runner import has_truth, load_jsonl

CI_RESAMPLES = 1000
WITHDRAWING = {'remove': 'wid', 'swap_image': 'old_wid'}


# --------------------------------------------------------------------------------------------------- helpers
def text_of(message):
    c = message['content']
    return c if isinstance(c, str) else ' '.join(it.get('text', '') for it in c)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(2 * p * r / (p + r), 4) if p + r else 0.0}


def rate(num, den):
    return round(num / den, 4) if den else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def bootstrap_ci(bits, seed=0):
    """95 % percentile bootstrap of a mean over 0/1 values."""
    bits = [b for b in bits if b is not None]
    if len(bits) < 2:
        return None
    try:
        import numpy as np
        arr = np.asarray(bits, dtype=np.float32)
        rng = np.random.default_rng(seed)
        means = arr[rng.integers(0, len(arr), size=(CI_RESAMPLES, len(arr)))].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
    except ImportError:
        rng = random.Random(seed)
        n = len(bits)
        means = sorted(sum(rng.choices(bits, k=n)) / n for _ in range(CI_RESAMPLES))
        lo, hi = means[int(0.025 * CI_RESAMPLES)], means[int(0.975 * CI_RESAMPLES) - 1]
    return [round(float(lo), 4), round(float(hi), 4)]


def live_ids(record, step=None):
    """Witness IDs that exist in the conversation at a given point: every ID mentioned in a user turn, minus the ones
    withdrawn by the revisions announced so far (step=None: the record's own edit; step=k: chain steps 0..k)."""
    msgs = list(record['messages'])
    edits = []
    if record['mode'] == 'revise':
        edits = [record.get('edit') or {}]
    elif record['mode'] == 'chain' and step is not None:
        msgs += [s['message'] for s in record['steps'][:step + 1]]
        edits = [s.get('edit') or {} for s in record['steps'][:step + 1]]
    ids = set()
    for m in msgs:
        if m['role'] == 'user':
            ids.update(protocol.ID_RE.findall(text_of(m)))
    withdrawn = set()
    for e in edits:  # in order: a witness withdrawn earlier can be re-introduced by a later addition under the same id
        if e.get('type') in WITHDRAWING and e.get(WITHDRAWING[e['type']]):
            withdrawn.add(e[WITHDRAWING[e['type']]])
        elif e.get('type') == 'add' and e.get('wid'):
            withdrawn.discard(e['wid'])
    return ids - withdrawn, withdrawn


def prior_answer(record):
    return protocol.parse_output(record['messages'][-2]['content'])['answer']


# --------------------------------------------------------------------------------------------------- per-record rows
def single_row(r, pred):
    p = protocol.parse_output(pred['output'])
    live, withdrawn = live_ids(r)
    cited = set(p['support']) | set(p['load_bearing'])
    row = {'lang': r['lang'], 'mode': r['mode'], 'group': r['group'], 'family': r['family'], 'tier': r['tier'], 'qa_id': r['qa_id'],
           'pred_answer': p['answer'], 'parsed': p['answer'] is not None,
           'full_format': p['answer'] is not None and 'support' in pred['output'].lower() and 'load' in pred['output'].lower()
                          and (r['mode'] != 'revise' or p['changed'] is not None),
           'invalid_cite': bool(cited - live), 'stale_cite': bool(cited & withdrawn) if withdrawn else None,
           'n_support': len(set(p['support'])), 'n_lb': len(set(p['load_bearing']))}
    if r['mode'] == 'revise':
        own_prior = pred.get('prior')
        prev = protocol.parse_output(own_prior)['answer'] if own_prior is not None else prior_answer(r)
        row['pred_changed'] = p['changed'] if p['changed'] is not None else (p['answer'] != prev)
        row['changed_line'] = p['changed']
        row['self_contra'] = (p['changed'] != (p['answer'] != prev)) if (p['changed'] is not None and p['answer'] is not None) else None
        row['edit_class'] = r.get('edit_class')
        row['edit_type'] = (r.get('edit') or {}).get('type')
        row['own_prior'] = own_prior is not None
        row['prev'] = prev
    if 'truth' not in r:
        return row
    t = r['truth']
    row['correct'] = p['answer'] == t['answer']
    row['truth_none'] = t['answer'] == 'none'
    row['pred_none'] = p['answer'] == 'none'
    ps, ts = set(p['support']), set(t['support'])
    pl, tl = set(p['load_bearing']), set(t['load_bearing'])
    row['sup'] = (len(ps & ts), len(ps - ts), len(ts - ps))
    row['lb'] = (len(pl & tl), len(pl - tl), len(tl - pl))
    row['lb_exact'] = pl == tl
    row['sufficient'] = tl <= ps
    row['faithful'] = row['correct'] and row['lb_exact']
    if r['mode'] == 'revise':
        row['truth_changed'] = (t['answer'] != row['prev']) if row['own_prior'] else t['changed']
    return row


def chain_rows(r, pred):
    outs = [pred['base_output']] + list(pred['step_outputs'])
    gold_history = pred.get('history') == 'gold'   # teacher forced: the model saw the gold previous replies
    rows = []
    prev = None
    prev_correct = None
    for k, o in enumerate(outs):
        if k > len(r['steps']):
            break
        p = protocol.parse_output(o)
        live, withdrawn = live_ids(r, step=k - 1) if k else live_ids(r)
        cited = set(p['support']) | set(p['load_bearing'])
        row = {'lang': r['lang'], 'step': k, 'length': len(r['steps']), 'parsed': p['answer'] is not None,
               'invalid_cite': bool(cited - live), 'stale_cite': bool(cited & withdrawn) if withdrawn else None}
        if k:
            row['pred_changed'] = p['changed'] if p['changed'] is not None else (p['answer'] != prev)
        if 'truth' in r:
            t = r['truth'] if k == 0 else r['steps'][k - 1]['truth']
            row['correct'] = p['answer'] == t['answer']
            row['prev_correct'] = prev_correct
            if k:
                # "changed" is judged against what the model was actually shown as its previous reply
                truth_changed = t['changed'] if gold_history else (t['answer'] != prev)
                row['changed_ok'] = row['pred_changed'] == truth_changed
                row['edit_class'] = r['steps'][k - 1].get('edit_class')
            prev_correct = row['correct']
        rows.append(row)
        prev = p['answer']
    return rows


# --------------------------------------------------------------------------------------------------- aggregation
def agg_single(rows, with_ci=True):
    n = len(rows)
    m = {'n': n, 'parse_rate': rate(sum(x['parsed'] for x in rows), n), 'full_format_rate': rate(sum(x['full_format'] for x in rows), n),
         'invalid_citation_rate': rate(sum(x['invalid_cite'] for x in rows), n),
         'mean_support_size': mean([x['n_support'] for x in rows]), 'mean_load_bearing_size': mean([x['n_lb'] for x in rows])}
    if not rows or 'correct' not in rows[0]:
        return m
    acc = collections.defaultdict(list)
    for x in rows:
        for key in ('all', 'group:' + x['group'], 'family:' + x['family'], 'tier:' + x['tier']):
            acc[key].append(x['correct'])
    m['accuracy'] = {k: rate(sum(v), len(v)) for k, v in sorted(acc.items())}
    if with_ci:
        m['accuracy_ci95'] = bootstrap_ci(acc['all'])
    tn = [x for x in rows if x['truth_none']]
    pn = [x for x in rows if x['pred_none']]
    tp = sum(x['pred_none'] for x in tn)
    m['abstention'] = {'n_truth_none': len(tn), **prf(tp, len(pn) - tp, len(tn) - tp),
                       'false_none_rate': rate(sum(x['pred_none'] for x in rows if not x['truth_none']), n - len(tn))}
    m['citation'] = {name: prf(*(sum(x[k][i] for x in rows) for i in range(3))) for name, k in (('support', 'sup'), ('load_bearing', 'lb'))}
    m['load_bearing_exact'] = rate(sum(x['lb_exact'] for x in rows), n)
    m['sufficiency'] = rate(sum(x['sufficient'] for x in rows), n)
    m['faithful_accuracy'] = rate(sum(x['faithful'] for x in rows), n)
    right = [x for x in rows if x['correct']]
    m['load_bearing_exact_given_correct'] = rate(sum(x['lb_exact'] for x in right), len(right))
    if rows[0]['mode'] == 'revise':
        own = any(x['own_prior'] for x in rows)
        m['history'] = 'own' if own else 'gold'
        tp = sum(x['pred_changed'] and x['truth_changed'] for x in rows)
        fp = sum(x['pred_changed'] and not x['truth_changed'] for x in rows)
        fn = sum((not x['pred_changed']) and x['truth_changed'] for x in rows)
        m['revision'] = {'changed_f1': prf(tp, fp, fn), 'change_detection_accuracy': rate(n - fp - fn, n),
                         'self_contradiction_rate': mean([x['self_contra'] for x in rows]),
                         'stale_citation_rate': mean([x['stale_cite'] for x in rows]),
                         'accuracy_truth_none': rate(sum(x['correct'] for x in tn), len(tn))}
        by = collections.defaultdict(list)
        for x in rows:
            by['class:' + str(x['edit_class'])].append(x)
            by['type:' + str(x['edit_type'])].append(x)
        m['by_edit'] = {k: {'n': len(v), 'accuracy': rate(sum(x['correct'] for x in v), len(v)), 'pred_changed_rate': rate(sum(x['pred_changed'] for x in v), len(v)),
                            'changed_line_accuracy': rate(sum(x['pred_changed'] == x['truth_changed'] for x in v), len(v)),
                            'load_bearing_f1': prf(*(sum(x['lb'][i] for x in v) for i in range(3)))['f1']} for k, v in sorted(by.items())}
        cls = {k[6:]: v for k, v in by.items() if k.startswith('class:')}
        if not own:
            ku, kc, fl = cls.get('keep_uncited', []), cls.get('keep_cited', []), cls.get('flip', [])
            m['revision'].update({
                'contagion_rate': rate(sum(x['pred_changed'] for x in ku), len(ku)),
                'contagion_ci95': bootstrap_ci([x['pred_changed'] for x in ku]) if with_ci else None,
                'redundancy_overreaction_rate': rate(sum(x['pred_changed'] for x in kc), len(kc)),
                'inertia_rate': rate(sum(not x['pred_changed'] for x in fl), len(fl)),
                'inertia_ci95': bootstrap_ci([not x['pred_changed'] for x in fl]) if with_ci else None,
                'flip_accuracy': rate(sum(x['correct'] for x in fl), len(fl)),
                'keep_accuracy': rate(sum(x['correct'] for x in ku + kc), len(ku) + len(kc))})
    return m


def agg_chain(rows, with_ci=True):
    n_chains = sum(1 for x in rows if x['step'] == 0)
    m = {'n': n_chains, 'parse_rate': rate(sum(x['parsed'] for x in rows), len(rows)),
         'invalid_citation_rate': rate(sum(x['invalid_cite'] for x in rows), len(rows)),
         'stale_citation_rate': mean([x['stale_cite'] for x in rows])}
    if not rows or 'correct' not in rows[0]:
        return m
    by_step = collections.defaultdict(list)
    ch_ok = collections.defaultdict(list)
    by_len = collections.defaultdict(list)
    for x in rows:
        by_step[x['step']].append(x['correct'])
        if x['step']:
            ch_ok[x['step']].append(x['changed_ok'])
        if x['step'] == x['length']:
            by_len[x['length']].append(x['correct'])
    m['accuracy_by_step'] = {k: rate(sum(v), len(v)) for k, v in sorted(by_step.items())}
    m['changed_line_accuracy_by_step'] = {k: rate(sum(v), len(v)) for k, v in sorted(ch_ok.items())}
    finals = [x['correct'] for x in rows if x['step'] == x['length']]
    m['final_step_accuracy'] = rate(sum(finals), len(finals))
    if with_ci:
        m['final_step_accuracy_ci95'] = bootstrap_ci(finals)
    m['final_step_accuracy_by_length'] = {k: rate(sum(v), len(v)) for k, v in sorted(by_len.items())}
    # whole chain: every step including the base answer correct
    per_chain = collections.defaultdict(list)
    for x in rows:
        per_chain[x['chain_key']].append(x['correct'])
    m['whole_chain_accuracy'] = rate(sum(all(v) for v in per_chain.values()), len(per_chain))
    first, last = m['accuracy_by_step'].get(0), m['final_step_accuracy']
    m['drift'] = round(first - last, 4) if first is not None and last is not None else None
    later = [x for x in rows if x['step'] and x['prev_correct'] is not None]
    m['recovery_rate'] = rate(sum(x['correct'] for x in later if not x['prev_correct']), sum(not x['prev_correct'] for x in later))
    m['slip_rate'] = rate(sum(not x['correct'] for x in later if x['prev_correct']), sum(x['prev_correct'] for x in later))
    by_cls = collections.defaultdict(list)
    for x in later:
        by_cls[str(x.get('edit_class'))].append(x)
    m['step_accuracy_by_edit_class'] = {k: {'n': len(v), 'accuracy': rate(sum(x['correct'] for x in v), len(v)),
                                            'pred_changed_rate': rate(sum(x['pred_changed'] for x in v), len(v))} for k, v in sorted(by_cls.items())}
    return m


# --------------------------------------------------------------------------------------------------- ecology view
COMPETENCY = {'A': 'habitat_from_imagery', 'B': 'community_trait_reasoning', 'C': 'scene_community_fusion', 'D': 'diversity_and_conservation',
              'E': 'river_network_and_connectivity', 'K': 'species_knowledge'}
CONSERVATION_FAMILIES = {'D1-ThreatenedRecords', 'D2-DetectionPlausibility', 'K1-NonNativeSet', 'K3-MarineStraySet', 'K6-SeaRouteSet', 'B6-StatusCount'}
COUNT_FAMILIES = {'B1-DominantGuild', 'B6-StatusCount', 'D4-RareSpeciesCount', 'D5-SharedSpecies', 'D6-FamilyRichness', 'D8-RareDetectionCount'}
ORDINAL_FAMILIES = {'A3-ChannelWidth', 'A4-WaterFraction', 'A5-Sinuosity', 'A6-RiparianVegetation', 'A7-BuiltUpBank'}
READ_FAMILIES = {'D7-ReadDominance', 'D8-RareDetectionCount', 'D9-DominanceClass'}
NUM_RE = re.compile(r'\d+(?:\.\d+)?')


def options_of(record):
    """letter -> option text, read back from the first user turn."""
    text = text_of(record['messages'][1])
    head = text.rfind('\nOptions:')
    if head < 0:
        head = text.rfind('\n选项：')
    out = {}
    for line in text[head:].splitlines():
        m = re.match(r'([ABCD])\. (.*)', line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def option_number(text):
    nums = [float(x) for x in NUM_RE.findall(text)]
    return sum(nums) / len(nums) if nums else None


def witness_kind(wid):
    return 'image' if wid.startswith('IMG') else 'record' if wid.startswith('R') else 'match_row' if wid.startswith('ROW') else 'species_card'


def ecology_rows(r, pred, row):
    """Add ecological readings to a base/revise row: count error, ordinal distance, modality grounding, edited-witness kind."""
    fam = r['family']
    row['competency'] = COMPETENCY.get(r['group'], r['group'])
    row['conservation'] = fam in CONSERVATION_FAMILIES
    row['reads'] = fam in READ_FAMILIES
    p = protocol.parse_output(pred['output'])
    truth = r.get('truth', {}).get('answer')
    if fam in COUNT_FAMILIES or fam in ORDINAL_FAMILIES:
        opts = options_of(r)
        tv = option_number(opts.get(truth, '')) if truth in opts else None
        pv = option_number(opts.get(p['answer'], '')) if p['answer'] in opts else None
        if tv is not None and pv is not None:
            if fam in COUNT_FAMILIES:
                row['count_error'] = pv - tv
            else:
                order = sorted(opts, key=lambda L: option_number(opts[L]))
                row['ordinal_distance'] = abs(order.index(p['answer']) - order.index(truth))
    cited = {witness_kind(w) for w in set(p['support']) | set(p['load_bearing'])}
    if r['group'] == 'A' or fam == 'C3-ImageTiming':
        row['cites_image'] = 'image' in cited
    if fam == 'C1-SceneAndCommunity':
        row['cites_both_modalities'] = {'image', 'species_card'} <= cited
    if fam == 'C2-MigrationBarrier':
        row['cites_both_modalities'] = {'record', 'species_card'} <= cited
    if r['mode'] == 'revise':
        e = r.get('edit') or {}
        row['edited_kind'] = 'image' if e.get('type') == 'swap_image' else witness_kind(e.get('wid') or 'SP')
    return row


def agg_ecology(rows):
    scored = [x for x in rows if 'correct' in x]
    if not scored:
        return None
    m = {}
    by = collections.defaultdict(list)
    for x in scored:
        by[x['competency']].append(x)
    m['competency_profile'] = {k: {'n': len(v), 'accuracy': rate(sum(x['correct'] for x in v), len(v)), 'faithful_accuracy': rate(sum(x['faithful'] for x in v), len(v)),
                                   'load_bearing_f1': prf(*(sum(x['lb'][i] for x in v) for i in range(3)))['f1']} for k, v in sorted(by.items())}
    cons = [x for x in scored if x['conservation']]
    m['conservation_items'] = {'n': len(cons), 'accuracy': rate(sum(x['correct'] for x in cons), len(cons)),
                               'note': 'threatened / protected / non-native / marine-stray / sea-route / native-status questions'}
    reads = [x for x in scored if x['reads']]
    if reads:
        m['read_abundance_items'] = {'n': len(reads), 'accuracy': rate(sum(x['correct'] for x in reads), len(reads))}
    cnt = [x['count_error'] for x in scored if x.get('count_error') is not None]
    if cnt:
        m['species_counts'] = {'n': len(cnt), 'mean_absolute_error': round(sum(abs(e) for e in cnt) / len(cnt), 4), 'bias': round(sum(cnt) / len(cnt), 4),
                               'over_count_rate': rate(sum(e > 0 for e in cnt), len(cnt)), 'under_count_rate': rate(sum(e < 0 for e in cnt), len(cnt)),
                               'note': 'count questions only (B1, B6, D4, D5, D6, D8); error in species / families, none answers excluded'}
    ordv = [x['ordinal_distance'] for x in scored if x.get('ordinal_distance') is not None]
    if ordv:
        m['habitat_bins'] = {'n': len(ordv), 'exact_rate': rate(sum(d == 0 for d in ordv), len(ordv)), 'adjacent_bin_error_rate': rate(sum(d == 1 for d in ordv), len(ordv)),
                             'far_error_rate': rate(sum(d >= 2 for d in ordv), len(ordv)), 'mean_bin_distance': round(sum(ordv) / len(ordv), 4),
                             'note': 'ordered image classes (channel width, water fraction, sinuosity, riparian vegetation, built-up bank)'}
    ci = [x['cites_image'] for x in scored if 'cites_image' in x]
    cb = [x['cites_both_modalities'] for x in scored if 'cites_both_modalities' in x]
    m['modality_grounding'] = {'image_items_citing_image': rate(sum(ci), len(ci)), 'fusion_items_citing_both_sources': rate(sum(cb), len(cb))}
    rev = [x for x in scored if x['mode'] == 'revise' and 'edited_kind' in x]
    if rev:
        byk = collections.defaultdict(list)
        for x in rev:
            byk[x['edited_kind']].append(x)
        m['revision_by_edited_witness'] = {}
        for k, v in sorted(byk.items()):
            ku = [x for x in v if x.get('edit_class') == 'keep_uncited']
            fl = [x for x in v if x.get('edit_class') == 'flip']
            m['revision_by_edited_witness'][k] = {'n': len(v), 'accuracy': rate(sum(x['correct'] for x in v), len(v)),
                                                  'contagion_rate': rate(sum(x['pred_changed'] for x in ku), len(ku)), 'inertia_rate': rate(sum(not x['pred_changed'] for x in fl), len(fl))}
    return m


def cross_lingual(single):
    base = collections.defaultdict(dict)
    for x in single:
        if x['mode'] == 'base' and 'correct' in x:
            base[x['qa_id']][x['lang']] = x
    pairs = [v for v in base.values() if 'en' in v and 'zh' in v]
    if not pairs:
        return None
    agree = [v['en']['pred_answer'] is not None and v['en']['pred_answer'] == v['zh']['pred_answer'] for v in pairs]
    return {'n_pairs': len(pairs), 'answer_agreement_rate': rate(sum(agree), len(pairs)),
            'both_correct_rate': rate(sum(v['en']['correct'] and v['zh']['correct'] for v in pairs), len(pairs)),
            'accuracy_en': rate(sum(v['en']['correct'] for v in pairs), len(pairs)), 'accuracy_zh': rate(sum(v['zh']['correct'] for v in pairs), len(pairs))}


# --------------------------------------------------------------------------------------------------- entry points
def score_all(records, predictions, meta=None):
    by_id = {p['id']: p for p in predictions}
    truth = has_truth(records)
    cov = collections.defaultdict(lambda: [0, 0])
    single, chain = [], []
    for r in records:
        c = cov[f"{r['lang']}/{r['mode']}"]
        c[1] += 1
        p = by_id.get(r['id'])
        if p is None:
            continue
        c[0] += 1
        if r['mode'] == 'chain':
            rows = chain_rows(r, p)
            for x in rows:
                x['chain_key'] = r['id']
            chain.extend(rows)
            if len(p['step_outputs']) != len(r['steps']):
                cov['chain_step_count_mismatch'][0] += 1
        else:
            single.append(ecology_rows(r, p, single_row(r, p)))
    report = {'meta': meta or {}, 'has_truth': truth,
              'coverage': {k: {'predicted': v[0], 'total': v[1]} for k, v in sorted(cov.items()) if v[1]},
              'unknown_prediction_ids': len(set(by_id) - {r['id'] for r in records})}
    if cov['chain_step_count_mismatch'][0]:
        report['chain_step_count_mismatch'] = cov['chain_step_count_mismatch'][0]
    langs = sorted({x['lang'] for x in single} | {x['lang'] for x in chain})
    scopes = ([('all', None)] if len(langs) > 1 else []) + [(l, l) for l in langs]
    for name, lang in scopes:
        s = [x for x in single if lang is None or x['lang'] == lang]
        c = [x for x in chain if lang is None or x['lang'] == lang]
        block = {}
        for mode in ('base', 'revise'):
            rows = [x for x in s if x['mode'] == mode]
            if rows:
                block[mode] = agg_single(rows, with_ci=truth)
        if c:
            block['chain'] = agg_chain(c, with_ci=truth)
        if truth and s:
            block['ecology'] = agg_ecology(s)
        report[name] = block
    if truth:
        xl = cross_lingual(single)
        if xl:
            report['cross_lingual'] = xl
        report['scorecard'] = scorecard(report, scopes[0][0])
    return report


def scorecard(report, scope):
    b, r, c = report[scope].get('base', {}), report[scope].get('revise', {}), report[scope].get('chain', {})
    g = lambda d, *ks: _dig(d, ks)
    card = {
        'base_accuracy': g(b, 'accuracy', 'all'), 'base_accuracy_ci95': b.get('accuracy_ci95'),
        'faithful_accuracy': b.get('faithful_accuracy'), 'load_bearing_f1': g(b, 'citation', 'load_bearing', 'f1'),
        'support_f1': g(b, 'citation', 'support', 'f1'), 'sufficiency': b.get('sufficiency'),
        'invalid_citation_rate': b.get('invalid_citation_rate'), 'none_f1_after_revision': g(r, 'abstention', 'f1'),
        'post_revision_accuracy': g(r, 'accuracy', 'all'), 'changed_f1': g(r, 'revision', 'changed_f1', 'f1'),
        'contagion_rate': g(r, 'revision', 'contagion_rate'), 'inertia_rate': g(r, 'revision', 'inertia_rate'),
        'redundancy_overreaction_rate': g(r, 'revision', 'redundancy_overreaction_rate'), 'stale_citation_rate': g(r, 'revision', 'stale_citation_rate'),
        'chain_final_accuracy': c.get('final_step_accuracy'), 'chain_drift': c.get('drift'), 'chain_recovery_rate': c.get('recovery_rate'),
        'cross_lingual_agreement': _dig(report, ('cross_lingual', 'answer_agreement_rate')),
        'conservation_accuracy': _dig(report[scope], ('ecology', 'conservation_items', 'accuracy')),
        'species_count_mae': _dig(report[scope], ('ecology', 'species_counts', 'mean_absolute_error')),
        'habitat_far_error_rate': _dig(report[scope], ('ecology', 'habitat_bins', 'far_error_rate')),
        'parse_rate': b.get('parse_rate')}
    return {k: v for k, v in card.items() if v is not None}


def _dig(d, keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', nargs='+', required=True, help='record files (with truth for scoring, public files for a check)')
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--out', default=None, help='metrics json (default: next to the predictions)')
    ap.add_argument('--label', default=None)
    a = ap.parse_args()
    records = [r for p in a.data for r in load_jsonl(p)]
    preds = load_jsonl(a.predictions)
    report = score_all(records, preds, meta={'label': a.label, 'predictions': a.predictions, 'data': a.data})
    out = Path(a.out) if a.out else Path(a.predictions).with_name('metrics.json' if report['has_truth'] else 'check.json')
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf8')
    print(json.dumps(report.get('scorecard') or report, ensure_ascii=False, indent=1))
    print('written', out)


if __name__ == '__main__':
    main()
