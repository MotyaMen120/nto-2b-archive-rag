# Case Q scorer (metric of section 5 of the case description).
# Run: python score.py <predictions.csv|.jsonl> [--gold train.csv] [--items out.jsonl] [--boot 2000]
# Predictions: submission CSV (qid,answer,evidence with space-separated post ids) or JSONL lines
#   {"qid": str, "answer": str, "evidence": [post id, ...]}; a missing qid counts as skip.
#   --subset scores only the qids present in the predictions (e.g. a held-out part of train).
# Gold: train.csv / examples.csv (answer, evidence columns) or gold JSONL; default: train.csv next to this file.
#   CSV gold has no story ids, so items with the same question text form one group.
# Choices where the card is silent (stated here once):
#   skip -> a = -0.25, cited evidence ignored (e = 0); s by the usual formula (owner 2026-11-16).
#   items with empty gold evidence (gold no_info): e = 1 for a correct answer (nothing to cite), else 0; s by the usual formula,
#     so wrong = -0.35 and skip = -0.175 as on every other item (owner 2026-11-17, review R11).
#   group = (story, question text); items without story are their own group; each item weight = level weight / group size
#     (with mixed levels in a group each member keeps its own level weight, divided by group size).
#   an answer not valid for the item's format counts as a wrong answer (a = -0.5) and is reported as invalid.
#   evidence is de-duplicated (first occurrence kept) and cut to 5; gold evidence is restricted to posts dated <= asked_at.
import sys, os, csv, json, re, collections as C, random

GOLD = [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.csv')]
W = {'light': 1, 'medium': 2, 'heavy': 4}
SPECIAL = {'promised', 'no_info', 'skip'}
STATUS = {'done', 'promised', 'postponed', 'cancelled'}


def _read_csv(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def load_gold(paths):
    items = []
    for p in paths:
        if p.endswith('.csv'):
            for r in _read_csv(p):
                items.append({'id': r['qid'], 'question': r['question'], 'asked_at': r['asked_at'], 'level': r['level'],
                              'answer_format': r['answer_format'], 'answer': r['answer'], 'story': 'text:' + r['question'],
                              'evidence': [{'idx': int(x)} for x in (r.get('evidence') or '').split()]})
        else:
            items += [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
    groups = C.Counter((it.get('story') or it['id'], it['question']) for it in items)
    for it in items:
        it['_w'] = W[it['level']] / groups[(it.get('story') or it['id'], it['question'])]
        it['_G'] = [e['idx'] for e in it.get('evidence', []) if not e.get('after_asked')]
    return items


def norm(ans, fmt):
    """Return (normalized answer, valid?)."""
    a = str(ans).strip().lower()
    if a in SPECIAL: return a, True
    if fmt == 'number':
        try: return float(a.replace(',', '.')), True
        except ValueError: return a, False
    if fmt == 'date': return a, bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}', a))
    if fmt == 'status': return a, a in STATUS
    if fmt == 'yesno': return a, a in ('yes', 'no')
    if fmt == 'choice': return a, a in ('a', 'b', 'c', 'd')
    return a, False


def correct(p, g, fmt):
    if isinstance(p, float) and isinstance(g, float):
        return abs(p - g) <= 0.01 * abs(g) if g != 0 else p == 0
    return p == g


def ap5(ev, G):
    G = set(G); hit = 0; s = 0.0
    for k, x in enumerate(ev[:5], 1):
        if x in G: hit += 1; s += hit / k
    return s / min(len(G), 5)


def score_item(it, pred):
    fmt = it['answer_format']; g, _ = norm(it['answer'], fmt)
    ans = (pred or {}).get('answer', 'skip'); p, valid = norm(ans, fmt)
    ev = []
    for x in (pred or {}).get('evidence', []) or []:
        try: x = int(x)
        except (TypeError, ValueError): continue
        if x not in ev: ev.append(x)
    ev = ev[:5]; G = it['_G']
    if valid and p == 'skip':
        a = -0.25; e = 0.0; s = 0.7 * a + 0.3 * e
    else:
        ok = valid and correct(p, g, fmt)
        if not ok: a = -0.5
        elif g == 'no_info': a = 1.0
        else: a = 1.0 if set(ev) & set(G) else 0.5
        e = ap5(ev, G) if G else (1.0 if ok else 0.0)
        s = 0.7 * a + 0.3 * e
    return dict(qid=it['id'], type=it.get('type', '-'), level=it['level'], w=it['_w'], gold=it['answer'], answer=ans,
                valid=valid, a=a, e=e, s=s, grounded=bool(set(ev) & set(G)), story=it.get('story'))


def total(rows):
    sw = sum(r['w'] for r in rows)
    return 100 * max(0.0, sum(r['w'] * r['s'] for r in rows) / sw) if sw else float('nan')


def summarize(rows, boot=0, seed=0):
    sw = sum(r['w'] for r in rows)
    out = {'S': total(rows), 'S_raw': 100 * sum(r['w'] * r['s'] for r in rows) / sw, 'n': len(rows), 'invalid': sum(not r['valid'] for r in rows),
           'skip': sum(r['answer'] == 'skip' for r in rows)}
    out['level'] = {L: total([r for r in rows if r['level'] == L]) for L in W}
    by = C.defaultdict(list)
    for r in rows: by[r['type']].append(r)
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    out['type'] = {t: dict(n=len(v), a=mean([r['a'] for r in v]), e=mean([r['e'] for r in v if r['e'] is not None]),
                           correct=sum(r['a'] > 0 for r in v)) for t, v in sorted(by.items())}
    out['a_level'] = {L: mean([r['a'] for r in rows if r['level'] == L]) for L in W}
    out['e_level'] = {L: mean([r['e'] for r in rows if r['level'] == L and r['e'] is not None]) for L in W}
    if boot:
        # cluster bootstrap: resample stories (single-post items are their own cluster)
        cl = C.defaultdict(list)
        for r in rows: cl[r['story'] or r['qid']].append(r)
        keys = list(cl); rnd = random.Random(seed); xs = []
        for _ in range(boot):
            xs.append(total([r for k in rnd.choices(keys, k=len(keys)) for r in cl[k]]))
        xs.sort(); out['S_ci95'] = (xs[int(0.025 * boot)], xs[int(0.975 * boot) - 1])
    return out


def run(pred_path, gold=GOLD, boot=0, subset=False):
    items = load_gold(gold)
    preds = {}
    if pred_path.endswith('.csv'):
        for r in _read_csv(pred_path):
            preds[str(r['qid'])] = {'answer': r.get('answer', ''), 'evidence': (r.get('evidence') or '').split()}
    else:
        for l in open(pred_path, encoding='utf-8'):
            if l.strip(): d = json.loads(l); preds[str(d['qid'])] = d
    if subset: items = [it for it in items if it['id'] in preds]
    rows = [score_item(it, preds.get(it['id'])) for it in items]
    return rows, summarize(rows, boot)


def show(sm):
    f = lambda x: '  -  ' if x is None else f'{x:5.2f}'
    ci = f"  CI95 [{sm['S_ci95'][0]:.1f}, {sm['S_ci95'][1]:.1f}]" if 'S_ci95' in sm else ''
    print(f"S = {sm['S']:.2f}{ci}  (raw {sm['S_raw']:.2f})   n={sm['n']} invalid={sm['invalid']} skip={sm['skip']}")
    print('level  S      mean a  mean e')
    for L in W: print(f"{L:6s} {sm['level'][L]:6.2f} {f(sm['a_level'][L])}   {f(sm['e_level'][L])}")
    print('type  n  correct  mean a  mean e')
    for t, v in sm['type'].items(): print(f"{t:4s} {v['n']:2d}  {v['correct']:2d}       {f(v['a'])}   {f(v['e'])}")


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if x not in ('--quiet', '--subset')]; opt = {a[k][2:]: a[k + 1] for k in range(len(a) - 1) if a[k].startswith('--')}
    gold = opt['gold'].split(',') if 'gold' in opt else GOLD
    rows, sm = run(a[0], gold, int(opt.get('boot', 0)), '--subset' in sys.argv)
    if 'items' in opt:
        with open(opt['items'], 'w', encoding='utf-8') as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    if 'json' in opt: json.dump(sm, open(opt['json'], 'w'), indent=1, ensure_ascii=False)
    show(sm)
