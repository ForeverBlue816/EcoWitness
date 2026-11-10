"""Verifier reward for one model reply. No torch here.

The benchmark's rule engine knows, for every evidence state, the value of every option and the exact support and
load-bearing witness sets. That makes each record a verifiable episode: any reply can be scored without a judge.

    reward = weighted mean of
        answer        graded: 1 for the correct option; partial credit by ecological closeness for the others
                      (a species count one off, an adjacent habitat class, an overlapping set of records, one of two
                      verdicts right), scaled by PARTIAL; 0 for every option when the right reply is "none"
                      (see rules.answer_rewards)
        load_bearing  F1 of the cited load-bearing witnesses against the rule-derived set
        support       F1 of the cited support witnesses
        changed       revision replies only: the "changed:" line is right
    A reply without a parsable answer gets FORMAT_PENALTY.
"""
from __future__ import annotations

from . import protocol

WEIGHTS = {'answer': 0.6, 'load_bearing': 0.25, 'support': 0.15, 'changed': 0.15}
FORMAT_PENALTY = -0.2
PARTIAL = 0.5   # a near miss earns at most half of the answer credit, so exactness always pays


def f1(pred, truth):
    pred, truth = set(pred), set(truth)
    if not pred and not truth:
        return 1.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 2 * p * r / (p + r)


def reward(record, output, weights=WEIGHTS, partial=PARTIAL):
    """-> dict(total, parsed, correct, answer, load_bearing, support, changed)."""
    p = protocol.parse_output(output)
    t = record['truth']
    if p['answer'] is None:
        return {'total': FORMAT_PENALTY, 'parsed': False, 'correct': False, 'answer': 0.0, 'load_bearing': 0.0, 'support': 0.0, 'changed': None}
    graded = record.get('answer_reward')
    credit = float(graded[p['answer']]) if graded else float(p['answer'] == t['answer'])
    parts = {'answer': credit if credit >= 1.0 else partial * credit,
             'load_bearing': f1(p['load_bearing'], t['load_bearing']), 'support': f1(p['support'], t['support'])}
    if 'changed' in t:
        parts['changed'] = float(p['changed'] is not None and p['changed'] == t['changed'])
    total = sum(weights[k] * v for k, v in parts.items()) / max(sum(weights[k] for k in parts), 1e-9)
    return {'total': total, 'parsed': True, 'correct': p['answer'] == t['answer'], 'changed': parts.get('changed'),
            **{k: parts[k] for k in ('answer', 'load_bearing', 'support')}}
