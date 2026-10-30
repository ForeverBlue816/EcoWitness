"""Witness protocol: evidence IDs, prompt rendering (en/zh), target rendering and output parsing.

Every piece of evidence becomes an addressable *witness* with an ID:
  IMG, IMG2, ...   satellite image(s)            R1, R2, ...   fields of a site record
  SP-01, SP-02     matched-fish cards             A-01 / B-01   cards of site A / site B (two-site questions)
  ROW-01           sequence match rows (B5)
The model answers with three fixed lines (a fourth, "changed:", in revision mode).
"""
from __future__ import annotations

import json
import re

ID_RE = re.compile(r'\b(?:IMG\d*|R\d+|SP-\d+|A-\d+|B-\d+|ROW-\d+)\b')
ANSWER_RE = re.compile(r'answer\s*[:：]\s*\**\s*\(?([ABCD]|none|无)\)?', re.I)
CHANGED_RE = re.compile(r'changed\s*[:：]\s*\**\s*(yes|no|是|否)', re.I)
SUPPORT_RE = re.compile(r'support\s*[:：]\s*\[([^\]]*)\]', re.I)
LB_RE = re.compile(r'load[_ ]?bearing\s*[:：]\s*\[([^\]]*)\]', re.I)

SYSTEM = {
    'en': (
        'You are an ecological evidence auditor. You receive a question about a river site, a satellite image when one is '
        'supplied, and numbered evidence items (witnesses). Answer the multiple-choice question and cite the witnesses by ID.\n'
        'Reply with exactly these lines and nothing else:\n'
        'answer: <A, B, C, D, or none>\n'
        'support: [IDs of every witness your answer relies on]\n'
        'load_bearing: [IDs whose removal alone would change the answer]\n'
        'Use "none" only when no option is consistent with the evidence. When a revision of the evidence is announced, '
        're-evaluate the same question and put a first line "changed: yes" or "changed: no" before the three lines above.'
    ),
    'zh': (
        '你是一名生态证据审核员。你会收到一个关于河流样点的问题、可能有一幅卫星影像，以及带编号的证据条目（证人）。'
        '请回答选择题，并按编号引用证人。\n'
        '只输出下面几行，不要输出其他内容：\n'
        'answer: <A、B、C、D 或 none>\n'
        'support: [你的答案所依赖的全部证人编号]\n'
        'load_bearing: [单独去掉就会改变答案的证人编号]\n'
        '只有在没有任何选项与证据相符时才回答 none。当收到证据修订通知时，请对同一问题重新判断，并在上面三行之前先输出一行 "changed: yes" 或 "changed: no"。'
    ),
}
HEAD = {'en': ('Background', 'Evidence (cite by ID)', 'Question', 'Options'), 'zh': ('背景', '证据（按编号引用）', '问题', '选项')}


def fmt_val(v):
    if v is None:
        return 'null'
    if isinstance(v, float):
        return f'{v:.4f}'.rstrip('0').rstrip('.') if abs(v) < 1 else f'{v:g}'
    return str(v)


def render_card(c, lang, highlighted=False):
    parts = [c['taxon_label']]
    for k, v in (c.get('attributes') or {}).items():
        parts.append(f'{k}={fmt_val(v)}')
    for k, v in (c.get('status') or {}).items():
        parts.append(f'{k}={fmt_val(v)}')
    if c.get('list_no') is not None:
        parts.append(f'list_no={c["list_no"]}')
    if highlighted:
        parts.append('highlighted' if lang == 'en' else '高亮物种')
    return f'[{c["wid"]}] ' + ' | '.join(parts)


def render_row(r, lang):
    return (f'[{r["wid"]}] reference_id={r["reference_id"]} ({r["taxon_label"]}) | query_coverage={r["query_coverage"]:.4f} | '
            f'method={r["method"]} | orientation={r["orientation"]} | reference_length_nt={r["reference_length_nt"]}')


def render_evidence(state, lang):
    """Lines of the evidence block for a State (see rules.State)."""
    lines = []
    if state.image:
        im = state.image
        lines.append(f'[{im["wid"]}] ' + (f'Satellite RGB image acquired on {im["date"]}.' if lang == 'en' else f'卫星 RGB 影像，拍摄于 {im["date"]}。'))
    for f, wid in state.record_wids.items():
        lines.append(f'[{wid}] {f} = {fmt_val(state.record.get(f))}')
    if state.sites:
        for tag in ('A', 'B'):
            site = state.sites[tag]
            lines.append((f'Site {tag} (sampled {site["sampling_date"]}):' if lang == 'en' else f'样点 {tag}（采样日期 {site["sampling_date"]}）：'))
            lines.extend(render_card(c, lang) for c in state.cards if c.get('site') == tag)
    else:
        lines.extend(render_card(c, lang, highlighted=c['wid'] in state.highlight) for c in state.cards)
    if state.rows:
        lines.append('Match rows:' if lang == 'en' else '匹配行：')
        lines.extend(render_row(r, lang) for r in state.rows)
    return lines


def user_message(state, record, lang):
    """First user turn: image item(s) + text with background, evidence, question, options."""
    h = HEAD[lang]
    notes = [record['inputs'].get(k) for k in ('attribute_note', 'knowledge_note') if record['inputs'].get(k)]
    if state.record_note:
        notes.append(state.record_note)
    bg = record['context'] + ('' if not notes else ('\n' + '\n'.join(notes)))
    text = '\n\n'.join([f'{h[0]}:\n{bg}', f'{h[1]}:\n' + '\n'.join(render_evidence(state, lang)), f'{h[2]}:\n{record["question"]}',
                        f'{h[3]}:\n' + '\n'.join(f'{k}. {v}' for k, v in record['options'].items())])
    content = []
    if state.image:
        content.append({'type': 'image', 'path': state.image['path']})
    content.append({'type': 'text', 'text': text})
    return {'role': 'user', 'content': content}


def revision_message(edit, lang):
    """User turn announcing one evidence revision. `edit` is the dict produced by rules.apply_edit."""
    t = edit['type']
    content = []
    if lang == 'en':
        if t == 'remove':
            text = f'Revision: [{edit["wid"]}] has been withdrawn ({edit.get("reason", "the record failed quality control")}). Re-evaluate the question above.'
        elif t == 'add':
            text = f'Revision: a new item has been added: {edit["rendered"]}. Re-evaluate the question above.'
        elif t == 'reassign':
            text = f'Revision: [{edit["wid"]}] has been reassigned; the record is now {edit["rendered"]} (previously {edit["old_name"]}). Re-evaluate the question above.'
        elif t == 'set_field':
            text = f'Revision: [{edit["wid"]}] {edit["field"]} is corrected to {fmt_val(edit["new"])} (previously {fmt_val(edit["old"])}). Re-evaluate the question above.'
        elif t == 'swap_image':
            text = f'Revision: [{edit["old_wid"]}] is withdrawn. The image below, [{edit["wid"]}], acquired on {edit["date"]}, replaces it. Re-evaluate the question above.'
        else:
            raise ValueError(t)
    else:
        if t == 'remove':
            text = f'修订：[{edit["wid"]}] 已被撤销（{edit.get("reason_zh", "该记录未通过质控")}）。请对上面的问题重新判断。'
        elif t == 'add':
            text = f'修订：新增一条证据：{edit["rendered_zh"]}。请对上面的问题重新判断。'
        elif t == 'reassign':
            text = f'修订：[{edit["wid"]}] 的物种归属已更正，该记录现在是 {edit["rendered_zh"]}（原为 {edit["old_name"]}）。请对上面的问题重新判断。'
        elif t == 'set_field':
            text = f'修订：[{edit["wid"]}] {edit["field"]} 更正为 {fmt_val(edit["new"])}（原为 {fmt_val(edit["old"])}）。请对上面的问题重新判断。'
        elif t == 'swap_image':
            text = f'修订：[{edit["old_wid"]}] 已撤销，由下面这幅拍摄于 {edit["date"]} 的影像 [{edit["wid"]}] 替代。请对上面的问题重新判断。'
        else:
            raise ValueError(t)
    if t == 'swap_image':
        content.append({'type': 'image', 'path': edit['path']})
    content.append({'type': 'text', 'text': text})
    return {'role': 'user', 'content': content}


def ids_sorted(ids):
    def key(w):
        m = re.match(r'([A-Z]+)-?(\d*)', w)
        return (m.group(1), int(m.group(2) or 0))
    return sorted(set(ids), key=key)


def render_target(truth, revise):
    lines = []
    if revise:
        lines.append('changed: ' + ('yes' if truth['changed'] else 'no'))
    lines.append('answer: ' + truth['answer'])
    lines.append('support: [' + ', '.join(ids_sorted(truth['support'])) + ']')
    lines.append('load_bearing: [' + ', '.join(ids_sorted(truth['load_bearing'])) + ']')
    return '\n'.join(lines)


def parse_output(text):
    """Parse a model reply. Missing fields come back as None / empty lists."""
    text = text.split('</think>')[-1]
    m = ANSWER_RE.search(text)
    answer = None
    if m:
        a = m.group(1).upper()
        answer = 'none' if a in ('NONE', '无') else a
    c = CHANGED_RE.search(text)
    changed = None if not c else c.group(1).lower() in ('yes', '是')
    sup = SUPPORT_RE.search(text)
    lb = LB_RE.search(text)
    return {'answer': answer, 'changed': changed,
            'support': ID_RE.findall(sup.group(1)) if sup else [],
            'load_bearing': ID_RE.findall(lb.group(1)) if lb else []}


def to_chat(messages, with_images=True):
    """Convert stored messages (content items with 'path') into the list the Qwen chat template expects,
    plus the ordered image paths."""
    out, paths = [], []
    for m in messages:
        if isinstance(m['content'], str):
            out.append({'role': m['role'], 'content': m['content']})
            continue
        items = []
        for it in m['content']:
            if it['type'] == 'image':
                paths.append(it['path'])
                if with_images:
                    items.append({'type': 'image'})
            else:
                items.append({'type': 'text', 'text': it['text']})
        out.append({'role': m['role'], 'content': items})
    return out, paths


if __name__ == '__main__':
    print(json.dumps(parse_output('changed: yes\nanswer: C\nsupport: [SP-03, IMG]\nload_bearing: [SP-03]'), ensure_ascii=False))
