"""Re-evaluable conclusion rules for every v8 family, counterfactual edits, and leave-one-out load-bearing analysis.

A State holds the evidence of one question (cards, match rows, record fields, image). `evaluate(state)` returns the
gold *value* (count, set of list numbers, tuple, ...) plus the support witnesses; `letter_of` maps a value onto the
fixed option letters of the question ('none' when no option matches). Load-bearing witnesses are found by leave-one-out:
a witness is load-bearing when removing it alone changes the value.
"""
from __future__ import annotations

import collections
import copy
import importlib
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Source benchmark: v9 by default (v8 rules on the rebuilt DNA evidence); EW_BENCH=qualitative_benchmark_20261026_v8 for v8.
BENCH = ROOT / os.environ.get('EW_BENCH', 'qualitative_benchmark_20261105_v9')
V8 = BENCH  # older name, kept for callers
sys.path.insert(0, str(BENCH / 'tools'))
bv = importlib.import_module(sorted(p.stem for p in (BENCH / 'tools').glob('build_v*.py'))[-1])
import diversity_families as div  # noqa: E402
import environment_families as envf  # noqa: E402
import knowledge_families as kn  # noqa: E402

ILL = object()  # ill-posed after an edit (no single correct option, question no longer well defined)
INT = re.compile(r'\d+')
A_OPTS = {'A1-NearshoreCover': bv.A1_OPT, 'A2-LandscapePattern': bv.A2_OPT, 'A3-ChannelWidth': bv.A3_OPT, 'A4-WaterFraction': bv.A4_OPT,
          'A5-Sinuosity': envf.A5_OPT, 'A6-RiparianVegetation': envf.A6_OPT, 'A7-BuiltUpBank': envf.A7_OPT}
TEXT_OPTS = {'C2-MigrationBarrier': bv.C2_OPT, 'C3-ImageTiming': bv.C3_OPT, 'E1-ConnectivityStatus': envf.E1_OPT, 'E2-ReachType': envf.E2_OPT,
             'D3-SiteComparison': div.D3_OPT, 'D9-DominanceClass': div.D9_OPT, **A_OPTS}
COUNT_FAMILIES = {'B1-DominantGuild', 'B6-StatusCount', 'D4-RareSpeciesCount', 'D5-SharedSpecies', 'D6-FamilyRichness', 'D8-RareDetectionCount'}
SET_FAMILIES = {'D1-ThreatenedRecords', 'D2-DetectionPlausibility', 'D7-ReadDominance', 'K1-NonNativeSet', 'K3-MarineStraySet', 'K4-FourCarpsSet',
                'K5-LargeBodiedSet', 'K6-SeaRouteSet'}
K_GOLD = {'K1-NonNativeSet': lambda n: n in kn.ALIEN, 'K3-MarineStraySet': kn.is_marine, 'K4-FourCarpsSet': lambda n: n in kn.FOUR_CARPS,
          'K5-LargeBodiedSet': lambda n: (kn.max_length(n) or 0) >= 100, 'K6-SeaRouteSet': lambda n: n in kn.SEA_ROUTE}
K_TRAP = {'K1-NonNativeSet': kn.is_native, 'K3-MarineStraySet': kn.is_obligate_freshwater,
          'K4-FourCarpsSet': lambda n: kn.is_native(n) and n not in kn.FOUR_CARPS and kn.traits().get(n, {}).get('family') == 'Cyprinidae',
          'K5-LargeBodiedSet': lambda n: 0 < (kn.max_length(n) or 0) <= 40, 'K6-SeaRouteSet': kn.is_obligate_freshwater}
LANG_I = {'en': 0, 'zh': 1}


@dataclass
class State:
    family: str
    params: dict
    cards: list = field(default_factory=list)      # dicts with wid, taxon_label, attributes, status, list_no, site
    rows: list = field(default_factory=list)       # B5 match rows with wid
    record: dict = field(default_factory=dict)     # field -> value
    record_wids: dict = field(default_factory=dict)  # field -> wid
    record_note: str | None = None
    image: dict | None = None                      # wid, path, date, category (label for the visual family), sample_id
    sites: dict | None = None                      # {'A': {'sampling_date':..}, 'B': {...}} for two-site families
    highlight: set = field(default_factory=set)    # wids of highlighted cards (B4)
    option_values: dict = field(default_factory=dict)  # letter -> parsed value
    next_sp: int = 1

    def clone(self):
        return copy.deepcopy(self)

    def card_by_wid(self, wid):
        return next(c for c in self.cards if c['wid'] == wid)


# ----------------------------------------------------------------------------- building a state from a v8 record

def _inv(d, i):
    return {v[i].rstrip('.。'): k for k, v in d.items()}


def _greedy(text, mapping, i, separators):
    """Parse a concatenation of known phrases (FUNC texts) separated by any of `separators` into keys."""
    phrases = sorted(((v[i], k) for k, v in mapping.items()), key=lambda x: -len(x[0]))
    out, pos = [], 0
    while pos < len(text):
        for sep in separators:
            if text.startswith(sep, pos):
                pos += len(sep)
                break
        else:
            for ph, k in phrases:
                if text.startswith(ph, pos):
                    out.append(k)
                    pos += len(ph)
                    break
            else:
                raise ValueError(f'cannot parse {text!r} at {pos}')
    return out


def parse_options(record, fam, params, lang):
    i = LANG_I[lang]
    out = {}
    for L, text in record['options'].items():
        t = text.strip()
        if fam in COUNT_FAMILIES or (fam == 'B5-GatedTrace' and params['b5_variant'] == 'species_count'):
            out[L] = int(INT.search(t).group())
        elif fam == 'B3-UniqueCarrier':
            out[L] = int(INT.search(t).group())
        elif fam in SET_FAMILIES:
            out[L] = frozenset(int(x) for x in INT.findall(t))
        elif fam == 'B5-GatedTrace':
            out[L] = frozenset(_greedy(t, bv.FUNC[params['field']], i, (', ', '、')))
        elif fam == 'B4-PairedProfile':
            f0, f1 = params['fields']
            pairs = []
            for part in re.split(r'; |；', t.rstrip('.。')):
                # f0 phrase first, then the connector, then the f1 phrase
                k0 = next(k for k, v in sorted(bv.FUNC[f0].items(), key=lambda x: -len(x[1][i])) if part.startswith(v[i]))
                rest = part[len(bv.FUNC[f0][k0][i]):]
                conn = ' with ' if i == 0 else '，对应'
                assert rest.startswith(conn), (part, rest)
                rest = rest[len(conn):]
                k1 = next(k for k, v in bv.FUNC[f1].items() if rest == v[i])
                pairs.append((k0, k1))
            out[L] = tuple(sorted(pairs))
        elif fam == 'C1-SceneAndCommunity':
            vopt = A_OPTS[params['visual_family']]
            vis = next(c for c, texts in vopt.items() if t.startswith(texts[i].rstrip('.。')))
            rest = t[len(vopt[vis][i].rstrip('.。')):]
            out[L] = (vis, int(INT.search(rest).group()))
        elif fam in TEXT_OPTS:
            inv = _inv(TEXT_OPTS[fam], i)
            out[L] = inv[t.rstrip('.。')]
        else:
            raise KeyError(fam)
    return out


def letter_of(state, value):
    if value is ILL:
        return None
    for L, v in state.option_values.items():
        if v == value:
            return L
    return 'none'


def build_state(record, private, lang, image_label=None):
    """record: a v8 row (en or zh); private: its reports/answer_evidence.jsonl row; image_label: category of the image
    for the visual family (A*, C1) or None."""
    fam = record['task']
    inp = record['inputs']
    params = dict(private)
    if fam == 'B5-GatedTrace':  # train augmentation overwrites private['variant'] with a number; read the variant off the options
        params['b5_variant'] = 'species_count' if all(re.fullmatch(r'\d+ (species|种)', t.strip()) for t in record['options'].values()) else 'value_set'
    st = State(family=fam, params=params)
    st.option_values = parse_options(record, fam, params, lang)
    if inp.get('image'):
        st.image = {'wid': 'IMG', 'path': inp['image']['path'], 'date': inp['image']['acquisition_date'], 'category': image_label, 'sample_id': record['sample_id']}
    if 'site_A' in inp:
        st.sites = {t: {'sampling_date': inp[f'site_{t}']['sampling_date']} for t in ('A', 'B')}
        for t in ('A', 'B'):
            for c in inp[f'site_{t}']['fish']:
                c = copy.deepcopy(c)
                c['wid'] = f'{t}-{c["list_no"]:02d}'
                c['site'] = t
                st.cards.append(c)
    elif inp.get('matched_fish'):
        for n, c in enumerate(inp['matched_fish'], 1):
            c = copy.deepcopy(c)
            c['wid'] = f'SP-{n:02d}'
            st.cards.append(c)
        st.next_sp = len(st.cards) + 1
    if inp.get('highlighted_reference_ids'):
        st.highlight = {c['wid'] for c in st.cards if c['reference_id'] in inp['highlighted_reference_ids']}
    if inp.get('match_rows'):
        name = {c['reference_id']: c['taxon_label'] for c in st.cards}
        for n, r in enumerate(inp['match_rows'], 1):
            r = copy.deepcopy(r)
            r['wid'] = f'ROW-{n:02d}'
            r['taxon_label'] = name.get(r['reference_id'], '?')
            st.rows.append(r)
    rec = inp.get('site_record') or inp.get('site_connectivity_record')
    if rec:
        n = 1
        for k, v in rec.items():
            if k == 'note':
                st.record_note = ('Note: ' if lang == 'en' else '说明：') + v
                continue
            st.record[k] = v
            st.record_wids[k] = f'R{n}'
            n += 1
    if fam == 'C3-ImageTiming':
        samp = re.search(r'\d{4}-\d{2}-\d{2}', record['context']).group()
        st.record['sampling_date'] = samp
        st.record_wids['sampling_date'] = 'R1'
    return st


# ----------------------------------------------------------------------------- evaluators

def attr(c, f):
    v = (c.get('attributes') or {}).get(f)
    return None if v in (None, 'not_resolved') else v


def status(c, k):
    return (c.get('status') or {}).get(k)


def threatened(c):
    return status(c, 'iucn_category') in div.THREATENED_IUCN or str(status(c, 'national_protection_status') or '').startswith('Class')


def option_cards(st):
    nos = set().union(*st.option_values.values()) if st.option_values else set()
    return [c for c in st.cards if c.get('list_no') in nos]


def c3_category(acq, samp):
    gap = abs((acq - samp).days)
    if gap <= 7:
        return 'near'
    if gap <= 45:
        return 'mid_same' if bv.season(acq) == bv.season(samp) else 'mid_diff'
    return 'far'


def e2_class(order, area, slope):
    if order is None or area is None or slope is None:
        return None
    if order <= 2 and slope >= 10 and area < 500:
        return 'headwater'
    if order >= 6 and slope < 2 and area >= 10000:
        return 'large_lowland'
    if order <= 3 and slope < 1.5 and area < 1000:
        return 'lowland_small'
    if 3 <= order <= 5 and 2 <= slope < 10 and 300 <= area < 20000:
        return 'hill'
    return None


def evaluate(st):
    """-> (value, support_wids). value may be ILL."""
    fam, p = st.family, st.params
    if fam[0] == 'A':
        if st.image is None or st.image.get('category') is None:
            return ILL, []
        return st.image['category'], [st.image['wid']]
    if fam == 'B1-DominantGuild':
        f, tgt = p.get('field', 'trophic_guild'), p['target']
        sup = [c['wid'] for c in st.cards if attr(c, f) == tgt]
        return len(sup), sup
    if fam == 'B3-UniqueCarrier':
        f = p['field']
        cnt = collections.Counter(attr(c, f) for c in st.cards if attr(c, f))
        by_no = {c.get('list_no'): c for c in st.cards}
        uniq = {no for no, c in by_no.items() if attr(c, f) and cnt[attr(c, f)] == 1}
        hits = [L for L, n in st.option_values.items() if n in uniq]
        if len(hits) > 1:
            return ILL, []
        sup = []
        for n in st.option_values.values():
            c = by_no.get(n)
            if not c:
                continue
            sup.append(c['wid'])
            sup.extend(o['wid'] for o in st.cards if o is not c and attr(c, f) and attr(o, f) == attr(c, f))
        return (st.option_values[hits[0]] if hits else None), sorted(set(sup))
    if fam == 'B4-PairedProfile':
        f0, f1 = p['fields']
        hl = [c for c in st.cards if c['wid'] in st.highlight]
        if len(hl) != 2 or any(attr(c, f0) is None or attr(c, f1) is None for c in hl):
            return None, [c['wid'] for c in hl]
        return tuple(sorted((attr(c, f0), attr(c, f1)) for c in hl)), [c['wid'] for c in hl]
    if fam == 'B5-GatedTrace':
        f, gate = p['field'], p['gate']
        info = {c['reference_id']: c for c in st.cards}
        kept = [r for r in st.rows if r['query_coverage'] >= gate and r['reference_id'] in info]
        sup = [r['wid'] for r in kept] + sorted({info[r['reference_id']]['wid'] for r in kept})
        if p['b5_variant'] == 'value_set':
            vals = frozenset(attr(info[r['reference_id']], f) for r in kept) - {None}
            return (vals if vals else None), sup
        return len({r['reference_id'] for r in kept}), sup
    if fam == 'B6-StatusCount':
        sup = [c['wid'] for c in st.cards if status(c, 'native_flag') == 'yes']
        return len(sup), sup
    if fam == 'C1-SceneAndCommunity':
        if st.image is None or st.image.get('category') is None:
            return ILL, []
        tgt = p['biology_target']
        sup = [c['wid'] for c in st.cards if attr(c, 'trophic_guild') == tgt]
        return (st.image['category'], len(sup)), [st.image['wid']] + sup
    if fam == 'C2-MigrationBarrier':
        dia = [c['wid'] for c in st.cards if attr(c, 'migration_type') == 'diadromous']
        dd = st.record.get('distance_to_downstream_dam_km')
        sup = dia or [c['wid'] for c in st.cards if attr(c, 'migration_type')]
        return (bool(dia), dd is not None and dd < 100), sup + [st.record_wids['distance_to_downstream_dam_km']]
    if fam == 'C3-ImageTiming':
        acq, samp = date.fromisoformat(st.image['date']), date.fromisoformat(st.record['sampling_date'])
        return c3_category(acq, samp), [st.image['wid'], st.record_wids['sampling_date']]
    if fam == 'D1-ThreatenedRecords':
        gold = [c for c in st.cards if threatened(c)]
        return frozenset(c['list_no'] for c in gold), sorted({c['wid'] for c in gold + option_cards(st)})
    if fam == 'D2-DetectionPlausibility':
        gold = [c for c in st.cards if threatened(c) and (status(c, 'national_detection_frequency') or 0) > div.FLAG_OCCUPANCY]
        return frozenset(c['list_no'] for c in gold), sorted({c['wid'] for c in gold + option_cards(st)})
    if fam == 'D3-SiteComparison':
        n = {t: sum(c['site'] == t for c in st.cards) for t in 'AB'}
        a = {t: sum(c['site'] == t and status(c, 'alien_flag') == 'yes' for c in st.cards) for t in 'AB'}
        if n['A'] == n['B'] or a['A'] == a['B']:
            return ILL, []
        return ('A' if n['A'] > n['B'] else 'B', 'A' if a['A'] > a['B'] else 'B'), [c['wid'] for c in st.cards]
    if fam == 'D4-RareSpeciesCount':
        thr = p['threshold_pct'] / 100
        occ = [(c, status(c, 'national_detection_frequency')) for c in st.cards]
        if any(o is None for _, o in occ):
            return ILL, []
        sup = [c['wid'] for c, o in occ if o < thr]
        return len(sup), sup
    if fam == 'D5-SharedSpecies':
        names = {t: {c['taxon_label'] for c in st.cards if c['site'] == t} for t in 'AB'}
        shared = names['A'] & names['B']
        return len(shared), [c['wid'] for c in st.cards if c['taxon_label'] in shared]
    if fam == 'D6-FamilyRichness':
        fams = [status(c, 'family') for c in st.cards]
        if not fams or any(f is None for f in fams):
            return ILL, []
        return len(set(fams)), [c['wid'] for c in st.cards]
    if fam in ('D7-ReadDominance', 'D8-RareDetectionCount', 'D9-DominanceClass'):
        shares = [(c, status(c, 'read_share')) for c in st.cards]
        if not shares or any(s is None for _, s in shares):
            return ILL, []
        if fam == 'D8-RareDetectionCount':
            thr = round(p['threshold_pct'] / 100, 4)
            sup = [c['wid'] for c, s in shares if s < thr]
            return len(sup), sup
        top = max(s for _, s in shares)
        tops = [c for c, s in shares if s == top]
        if fam == 'D7-ReadDominance':
            if len(tops) != 1:
                return ILL, []
            return frozenset({tops[0]['list_no']}), sorted({c['wid'] for c in tops + option_cards(st)})
        return div.dominance_class(top), [c['wid'] for c in st.cards]
    if fam == 'E1-ConnectivityStatus':
        csi, up = st.record.get('river_connectivity_status_index_pct'), st.record.get('distance_to_nearest_upstream_large_dam_km')
        if csi is None:
            return ILL, []
        return (csi >= envf.CSI_THRESHOLD, up is not None and up < envf.DAM_KM), [st.record_wids['river_connectivity_status_index_pct'], st.record_wids['distance_to_nearest_upstream_large_dam_km']]
    if fam == 'E2-ReachType':
        cls = e2_class(st.record.get('strahler_stream_order'), st.record.get('upstream_catchment_area_km2'), st.record.get('mean_valley_slope_deg'))
        return (ILL if cls is None else cls), [st.record_wids[k] for k in ('strahler_stream_order', 'upstream_catchment_area_km2', 'mean_valley_slope_deg')]
    if fam in K_GOLD:
        gold = [c for c in st.cards if K_GOLD[fam](c['taxon_label'])]
        return frozenset(c['list_no'] for c in gold), sorted({c['wid'] for c in gold + option_cards(st)})
    raise KeyError(fam)


FIXED_LB = {'C3-ImageTiming': lambda st: [st.image['wid'], st.record_wids['sampling_date']],
            'E1-ConnectivityStatus': lambda st: [st.record_wids['river_connectivity_status_index_pct'], st.record_wids['distance_to_nearest_upstream_large_dam_km']],
            'E2-ReachType': lambda st: [st.record_wids[k] for k in ('strahler_stream_order', 'upstream_catchment_area_km2', 'mean_valley_slope_deg')],
            'C2-MigrationBarrier': lambda st: [st.record_wids['distance_to_downstream_dam_km']]}


def removable(st):
    if st.family == 'B5-GatedTrace':
        return [r['wid'] for r in st.rows]
    if st.family[0] == 'A' or st.family in ('C3-ImageTiming', 'E1-ConnectivityStatus', 'E2-ReachType'):
        return []
    return [c['wid'] for c in st.cards]


def without(st, wid):
    s = copy.copy(st)  # shallow: cards/rows are never mutated by evaluate(), so sharing them is safe
    s.cards = [c for c in st.cards if c['wid'] != wid]
    s.rows = [r for r in st.rows if r['wid'] != wid]
    return s


def analyse(st):
    """-> dict(value, answer, support, load_bearing) or None when the state is ill-posed."""
    value, support = evaluate(st)
    if value is ILL:
        return None
    answer = letter_of(st, value)
    lb = []
    for wid in removable(st):
        v2, _ = evaluate(without(st, wid))
        if v2 is ILL or v2 != value:
            lb.append(wid)
    if st.family[0] == 'A' or st.family == 'C1-SceneAndCommunity':
        lb.append(st.image['wid'])
    if st.family in FIXED_LB:
        lb.extend(FIXED_LB[st.family](st))
    return {'value': value, 'answer': answer, 'support': sorted(set(support)), 'load_bearing': sorted(set(lb))}


# ----------------------------------------------------------------------------- counterfactual edits

def make_card(taxon, template, site=None):
    """A card for `taxon` (dna_evidence matched_taxa entry) with the same key layout as `template`."""
    c = {'reference_id': min(taxon['reference_ids']), 'taxon_label': taxon['scientific_name']}
    if 'attributes' in template:
        c['attributes'] = {f: (bv.trait(taxon, f) or 'not_resolved') for f in template['attributes']}
    if 'status' in template:
        c['status'] = {}
        for k in template['status']:
            if k == 'national_detection_frequency':
                c['status'][k] = div.occupancy(taxon['scientific_name'])
            elif k == 'family':
                c['status'][k] = div.family_of(taxon['scientific_name'])
            elif k == 'read_share':
                c['status'][k] = div.share_of(taxon)
            else:
                c['status'][k] = taxon['status'].get(k)
    if site:
        c['site'] = site
    return c


def apply_edit(st, edit, pools):
    """Return the new state for an edit dict (type: remove|add|reassign|set_field|swap_image) and complete the dict."""
    s = st.clone()
    t = edit['type']
    if t == 'remove':
        s = without(s, edit['wid'])
    elif t == 'add':
        taxon = pools.taxa[edit['name']]
        site = edit.get('site')
        template = next((c for c in s.cards if c.get('site') == site), s.cards[0]) if site else s.cards[0]
        c = make_card(taxon, template, site)
        if site:
            n = max(x['list_no'] for x in s.cards if x['site'] == site) + 1
            c['list_no'] = n
            c['wid'] = f'{site}-{n:02d}'
        else:
            if 'list_no' in template:
                c['list_no'] = s.next_sp  # keeps list_no equal to the SP number even after earlier removals
            c['wid'] = f'SP-{s.next_sp:02d}'
            s.next_sp += 1
        s.cards.append(c)
        edit['wid'], edit['card'] = c['wid'], c
    elif t == 'reassign':
        old = s.card_by_wid(edit['wid'])
        new = make_card(pools.taxa[edit['name']], old, old.get('site'))
        new['wid'] = old['wid']
        if 'list_no' in old:
            new['list_no'] = old['list_no']
        edit['old_name'] = old['taxon_label']
        s.cards[s.cards.index(old)] = new
        edit['card'] = new
    elif t == 'set_field':
        if edit.get('row'):
            r = next(r for r in s.rows if r['wid'] == edit['wid'])
            edit['old'] = r[edit['field']]
            r[edit['field']] = edit['new']
        else:
            edit['old'] = s.record.get(edit['field'])
            s.record[edit['field']] = edit['new']
            edit['wid'] = s.record_wids[edit['field']]
    elif t == 'set_card':   # one attribute / status field of a species card (trait, read share, conservation status, family)
        c = s.card_by_wid(edit['wid'])
        slot = c.setdefault(edit['slot'], {})
        edit['old'] = slot.get(edit['key'])
        slot[edit['key']] = edit['new']
        edit['name'] = c['taxon_label']
    elif t == 'swap_image':
        im = pools.images[edit['sample_id']]
        n = int(re.sub(r'\D', '', s.image['wid']) or 1) + 1
        edit['old_wid'] = s.image['wid']
        edit['wid'] = f'IMG{n}'
        edit['date'], edit['path'] = im['date'], im['path']
        cat = None
        vf = st.params.get('visual_family') or st.family
        if vf[0] == 'A':
            cat = pools.labels.get((edit['sample_id'], vf))
        s.image = {'wid': edit['wid'], 'path': im['path'], 'date': im['date'], 'category': cat, 'sample_id': edit['sample_id']}
    else:
        raise ValueError(t)
    return s


def touches(edit, wids):
    """Does the edit touch any witness in `wids`?"""
    ids = {edit.get('wid'), edit.get('old_wid')}
    return bool(ids & set(wids))


# ----------------------------------------------------------------------------- edit axes
# Every edit lies on one axis. The ecological axes (taxonomy, trait, abundance, status) change the biology of a card and
# leave its witness id in place; the generic axes change which witnesses exist (witness), the image, a site-record field
# (record) or only the presentation of the options (presentation: a permutation, under which the answer letter moves).
ECO_AXES = ('taxonomy', 'trait', 'abundance', 'status')
GENERIC_AXES = ('witness', 'image', 'record', 'presentation')
STATUS_VALUE_KEYS = ('iucn_category', 'national_protection_status', 'native_flag', 'alien_flag', 'invasive_flag', 'invasion_region')
CARD_PROFILE_SKIP = {'read_share', 'family'}   # per-sample / taxonomic fields, not part of a species' ecological profile


def axis_of(edit):
    t = edit['type']
    if t in ('remove', 'add'):
        return 'witness'
    if t == 'swap_image':
        return 'image'
    if t == 'set_field':
        return 'record'
    if t == 'reassign':
        return 'taxonomy'
    if t == 'permute':
        return 'presentation'
    if t == 'set_card':
        if edit['key'] == 'read_share':
            return 'abundance'
        return 'trait' if edit['slot'] == 'attributes' else 'status'
    raise ValueError(t)


def card_profile(c):
    """The ecological profile of a card: its attributes and status fields except the per-sample / taxonomic ones."""
    return {('attributes', k): v for k, v in (c.get('attributes') or {}).items()} | \
           {('status', k): v for k, v in (c.get('status') or {}).items() if k not in CARD_PROFILE_SKIP}


def share_candidates(st, card):
    """Read-share values for `card` around the thresholds the family's rule uses, plus a halving and a doubling. The
    shares of a sample sum to at most 1, so a value never exceeds what the other species leave."""
    fam, p = st.family, st.params
    share = status(card, 'read_share')
    room = round(1.0 - sum((status(c, 'read_share') or 0) for c in st.cards if c is not card), 4)
    thr = []
    if fam == 'D8-RareDetectionCount':
        thr = [round(p['threshold_pct'] / 100, 4)]
    elif fam == 'D9-DominanceClass':
        thr = list(div.D9_BOUNDS)
    elif fam == 'D7-ReadDominance':
        thr = [max(status(c, 'read_share') or 0 for c in st.cards)]
    vals = {round(share / 2, 4), round(min(share * 2, room), 4)}
    for t in thr:
        vals.update({round(t + 0.01, 4), round(max(t - 0.01, 0.0002), 4)})
    return sorted(v for v in vals if 0.0001 <= v <= room and abs(v - share) > 0.0005)


def ecological_edits(st, pools, rng, per_card=3):
    """Edits along the ecological axes for up to `per_card` cards: congener / profile-matched reassignments (taxonomy),
    a changed functional trait (trait), a changed read share (abundance), a changed conservation or origin field
    (status). Which of them flip the answer is decided by the rules afterwards, not here."""
    fam = st.family
    E = []
    if not st.cards or fam[0] == 'A' or fam in ('C3-ImageTiming', 'B5-GatedTrace'):
        return E
    present = {c['taxon_label'] for c in st.cards}
    pool = pools.pool_for(fam, st.params, present)
    pset = pools.pool_set(fam, st.params)
    for c in rng.sample(st.cards, min(per_card, len(st.cards))):
        # --- taxonomy: same family with the same ecological profile (a congener that should not matter), same family with
        #     a different profile, or another family with the same profile
        if pool and fam != 'B4-PairedProfile':
            prof = card_profile(c)
            fam_name = div.family_of(c['taxon_label'])
            congeners = [n for n in pools.names_in_family(fam_name) if n in pset and n not in present] if fam_name else []
            rng.shuffle(congeners)
            same, diff = [], []
            for n in congeners:
                (same if card_profile(make_card(pools.taxa[n], c)) == prof else diff).append(n)
                if len(same) >= 2 and len(diff) >= 1:
                    break
            cross = []
            for n in rng.sample(pool, min(60, len(pool))):
                if div.family_of(n) != fam_name and card_profile(make_card(pools.taxa[n], c)) == prof:
                    cross.append(n)
                    if len(cross) >= 1:
                        break
            for rel, names in (('congener_same_profile', same[:2]), ('congener_other_profile', diff[:1]), ('other_family_same_profile', cross)):
                E += [{'type': 'reassign', 'wid': c['wid'], 'name': n, 'relation': rel} for n in names]
        # --- trait: one functional attribute set to another value of its vocabulary
        for k, v in (c.get('attributes') or {}).items():
            if k in bv.FUNC:
                others = [x for x in bv.FUNC[k] if x != v]
                E.append({'type': 'set_card', 'wid': c['wid'], 'slot': 'attributes', 'key': k, 'new': rng.choice(others)})
        # --- abundance: read share moved around the thresholds of the rule
        if status(c, 'read_share') is not None:
            cands = share_candidates(st, c)
            E += [{'type': 'set_card', 'wid': c['wid'], 'slot': 'status', 'key': 'read_share', 'new': v} for v in rng.sample(cands, min(3, len(cands)))]
        # --- status: conservation category, protection class, origin flags, detection frequency, family
        for k, v in (c.get('status') or {}).items():
            if k in STATUS_VALUE_KEYS:
                others = [x for x in pools.status_values.get(k, []) if x != v]
                if others:
                    E.append({'type': 'set_card', 'wid': c['wid'], 'slot': 'status', 'key': k, 'new': rng.choice(others)})
            elif k == 'national_detection_frequency' and v is not None:
                thr = [div.FLAG_OCCUPANCY] + ([st.params['threshold_pct'] / 100] if 'threshold_pct' in st.params else [])
                cands = sorted({round(v / 2, 4), round(min(v * 2, 0.9), 4)} | {round(t + 0.01, 4) for t in thr} | {round(max(t - 0.01, 0.001), 4) for t in thr})
                E += [{'type': 'set_card', 'wid': c['wid'], 'slot': 'status', 'key': k, 'new': x} for x in rng.sample(cands, min(2, len(cands))) if abs(x - v) > 0.0005]
            elif k == 'family' and v is not None:
                others = [x for x in pools.status_values.get('family', []) if x != v]
                if others:
                    E.append({'type': 'set_card', 'wid': c['wid'], 'slot': 'status', 'key': k, 'new': rng.choice(others)})
    for e in E:
        e['axis'] = axis_of(e)
    return E


def random_perm(letters, rng):
    """A non-identity map old letter -> new letter."""
    while True:
        new = list(letters)
        rng.shuffle(new)
        if new != list(letters):
            return dict(zip(letters, new))


def permute_options(record, perm):
    """A copy of the record whose option contents are moved by `perm` (old letter -> new letter); the letters A-D stay
    in place, so the correct letter becomes perm[old answer]. The evidence is untouched."""
    rec = copy.deepcopy(record)
    rec['options'] = {perm[L]: text for L, text in record['options'].items()}
    rec['options'] = {L: rec['options'][L] for L in record['options']}   # keep the A, B, C, D order
    rec['answer'] = perm[record['answer']]
    return rec


E1_CSI = [20.0, 60.0, 90.0, 93.0, 97.0, 99.0]
E1_DAM = [12.0, 35.0, 48.0, 62.0, 120.0, None]
C2_DD = [25.0, 70.0, 95.0, 140.0, 400.0, None]
E2_VALUES = {'strahler_stream_order': [1, 2, 3, 4, 5, 6, 8], 'upstream_catchment_area_km2': [120.0, 450.0, 900.0, 3000.0, 15000.0, 60000.0],
             'mean_valley_slope_deg': [0.3, 1.2, 3.0, 6.0, 12.0, 20.0]}
C3_SHIFTS = [-100, -60, -30, -12, -5, 5, 12, 30, 60, 100]
COVERAGES = [0.93, 0.95, 0.96, 0.97, 0.98, 1.0]


def candidate_edits(st, pools, rng, split):
    """Enumerate single-step edits worth considering for this state (dicts, not yet applied)."""
    fam = st.family
    E = []
    # --- image swaps
    if fam[0] == 'A' or fam == 'C1-SceneAndCommunity':
        vf = st.params.get('visual_family') or fam
        same = [sid for sid in pools.by_family.get((vf, split), []) if sid != st.image['sample_id'] and pools.labels[(sid, vf)] == st.image['category']]
        diff = [sid for sid in pools.by_family.get((vf, split), []) if pools.labels[(sid, vf)] != st.image['category']]
        E += [{'type': 'swap_image', 'sample_id': sid} for sid in rng.sample(same, min(2, len(same))) + rng.sample(diff, min(3, len(diff)))]
    if fam == 'C3-ImageTiming':
        cands = [sid for sid in pools.by_split.get(split, []) if sid != st.image['sample_id']]
        E += [{'type': 'swap_image', 'sample_id': sid} for sid in rng.sample(cands, min(3, len(cands)))]
        for d in rng.sample(C3_SHIFTS, 5):
            new = (date.fromisoformat(st.record['sampling_date']) + timedelta(days=d)).isoformat()
            E.append({'type': 'set_field', 'field': 'sampling_date', 'new': new})
    # --- record fields
    if fam == 'E1-ConnectivityStatus':
        E += [{'type': 'set_field', 'field': 'river_connectivity_status_index_pct', 'new': v} for v in rng.sample(E1_CSI, 3)]
        E += [{'type': 'set_field', 'field': 'distance_to_nearest_upstream_large_dam_km', 'new': v} for v in rng.sample(E1_DAM, 3)]
        E += [{'type': 'set_field', 'field': 'dam_count_upstream', 'new': float(rng.choice([0, 1, 3, 7, 12]))}]
    if fam == 'E2-ReachType':
        for f, vals in E2_VALUES.items():
            E += [{'type': 'set_field', 'field': f, 'new': v} for v in rng.sample(vals, 3)]
        for f in ('site_elevation_m', 'long_term_mean_discharge_m3_s'):
            if f in st.record and st.record[f] is not None:
                E.append({'type': 'set_field', 'field': f, 'new': round(st.record[f] * rng.choice([0.5, 1.8]), 3)})
    if fam == 'C2-MigrationBarrier':
        E += [{'type': 'set_field', 'field': 'distance_to_downstream_dam_km', 'new': v} for v in rng.sample(C2_DD, 3)]
        for f in ('distance_to_upstream_dam_km', 'dam_count_upstream', 'river_connectivity_index_0_100'):
            if st.record.get(f) is not None:
                E.append({'type': 'set_field', 'field': f, 'new': round(st.record[f] * rng.choice([0.4, 2.0]), 2)})
    # --- match rows (B5)
    if fam == 'B5-GatedTrace':
        E += [{'type': 'remove', 'wid': r['wid'], 'reason': 'the read failed quality control', 'reason_zh': '该序列读段未通过质控'} for r in rng.sample(st.rows, min(8, len(st.rows)))]
        for r in rng.sample(st.rows, min(4, len(st.rows))):
            E.append({'type': 'set_field', 'row': True, 'wid': r['wid'], 'field': 'query_coverage', 'new': rng.choice([v for v in COVERAGES if abs(v - r['query_coverage']) > 0.004])})
        for e in E:
            e['axis'] = axis_of(e)
        return E
    # --- cards
    if st.cards and fam not in ('C3-ImageTiming',) and fam[0] != 'A':
        cards = st.cards
        E += [{'type': 'remove', 'wid': c['wid']} for c in rng.sample(cards, min(10, len(cards)))]
        present = {c['taxon_label'] for c in cards}
        pool = pools.pool_for(fam, st.params, present)
        for name in rng.sample(pool, min(6, len(pool))):
            e = {'type': 'add', 'name': name}
            if st.sites:
                e['site'] = rng.choice('AB')
            E.append(e)
        if fam != 'B4-PairedProfile':
            for c in rng.sample(cards, min(4, len(cards))):
                if pool:
                    E.append({'type': 'reassign', 'wid': c['wid'], 'name': rng.choice(pool), 'relation': 'random'})
    for e in E:
        e['axis'] = axis_of(e)
    return E


class Pools:
    """Species, image and label pools used to synthesise additions, reassignments and image swaps."""

    def __init__(self, dna_evidence, images, labels, by_family, by_split):
        self.taxa = {}
        for d in dna_evidence.values():
            for t in d.get('matched_taxa', []):
                name = t['scientific_name']
                # keep one entry per species, preferring one that carries a read share (needed for D7-D9 additions)
                if name not in self.taxa or (self.taxa[name].get('read_share') is None and t.get('read_share') is not None):
                    self.taxa[name] = t
        self.images, self.labels, self.by_family, self.by_split = images, labels, by_family, by_split
        self._cache = {}
        self._fam = collections.defaultdict(list)
        for name in sorted(self.taxa):
            f = div.family_of(name)
            if f:
                self._fam[f].append(name)
        # observed values of the status fields, for status edits
        vals = collections.defaultdict(set)
        for t in self.taxa.values():
            for k, v in (t.get('status') or {}).items():
                if k in STATUS_VALUE_KEYS and v is not None:
                    vals[k].add(v)
            f = div.family_of(t['scientific_name'])
            if f:
                vals['family'].add(f)
        self.status_values = {k: sorted(v) for k, v in vals.items()}

    def names_in_family(self, family):
        return self._fam.get(family, [])

    def pool_set(self, fam, params):
        self.pool_for(fam, params, set())
        key = (fam, params.get('target'), params.get('field'), params.get('threshold_pct'))
        return set(self._cache[key])

    def pool_for(self, fam, params, present):
        key = (fam, params.get('target'), params.get('field'), params.get('threshold_pct'))
        if key not in self._cache:
            names = sorted(self.taxa)
            if fam in K_GOLD:
                names = [n for n in names if n not in kn.UNCERTAIN_ORIGIN and (K_GOLD[fam](n) or K_TRAP[fam](n))]
            elif fam in ('B1-DominantGuild', 'C1-SceneAndCommunity'):
                f = params.get('field', 'trophic_guild')
                names = [n for n in names if bv.trait(self.taxa[n], f)]
            elif fam in ('B3-UniqueCarrier', 'C2-MigrationBarrier'):
                f = params.get('field', 'migration_type')
                names = [n for n in names if bv.trait(self.taxa[n], f)]
            elif fam == 'B4-PairedProfile':
                names = [n for n in names if all(bv.trait(self.taxa[n], f) for f in params['fields'])]
            elif fam in ('D2-DetectionPlausibility', 'D4-RareSpeciesCount'):
                names = [n for n in names if div.occupancy(n) is not None]
            elif fam == 'D6-FamilyRichness':
                names = [n for n in names if div.family_of(n)]
            elif fam in ('D7-ReadDominance', 'D8-RareDetectionCount', 'D9-DominanceClass'):
                names = [n for n in names if div.share_of(self.taxa[n]) is not None]
            self._cache[key] = names
        return [n for n in self._cache[key] if n not in present]
