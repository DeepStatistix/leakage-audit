"""Agreement between the two independent codings of the sealed subsample (Methods 2.7.2).

Coder 1 is the author's blind coding (data/reliability/author_independent_codes.csv); coder 2 is the automated
coding (data/reliability/automated_codes_sealed.csv). Agreement is computed on the original codes and again after the
codebook-conformance corrections (data/reliability/author_codes_after_codebook_corrections.csv,
data/reliability/coder1_corrections_log.csv). Reports percentage agreement and Cohen's kappa for full-text
eligibility (all sealed papers), items A1-A10 and flags F1-F3 (papers both coders found eligible), and the
leakage-risk level (unweighted and linear-weighted kappa, and Level 3 vs not).

Run:  python scripts/analysis/reliability.py
Output: results/reliability/kappa_summary.csv (identical to data/reliability/kappa_summary.csv)
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
REL = os.path.join(ROOT, 'data', 'reliability')
OUT = os.path.join(ROOT, 'results', 'reliability')
ITEMS = [f'A{i}' for i in range(1, 11)]
FLAGS = {'F1': 'F1_metric', 'F2': 'F2_skill', 'F3': 'F3_claim_gap'}


def norm(v):
    return '' if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip().upper()


def level(codes):
    v = [codes[a] for a in ITEMS[:6] if codes[a] != 'NA']
    if any(c not in ('C', 'N', 'NR') for c in v):
        return np.nan
    if 'N' in v:
        return 3
    nr = v.count('NR')
    return 2 if nr >= 3 else 1 if nr else 0


def kappa(x, y, weights=None):
    """Cohen's kappa; weights='linear' for ordinal levels."""
    x, y = np.asarray(x), np.asarray(y)
    cats = sorted(set(x) | set(y))
    if len(cats) < 2:
        return np.nan
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    o = np.zeros((k, k))
    for a, b in zip(x, y):
        o[idx[a], idx[b]] += 1
    o /= o.sum()
    e = np.outer(o.sum(1), o.sum(0))
    if weights == 'linear':
        w = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    else:
        w = 1 - np.eye(k)
    return 1 - (w * o).sum() / (w * e).sum() if (w * e).sum() > 0 else np.nan


def load(path, corrected=None):
    h = pd.read_csv(path, keep_default_na=False)
    h['paper_id'] = h.paper_id.astype(str).str.strip().str.upper()
    h = h.set_index('paper_id')
    if corrected is not None:
        c = pd.read_csv(corrected, keep_default_na=False).set_index('paper_id')
        for col in ITEMS + list(FLAGS.values()):
            h[col] = c.loc[h.index, col]
    h['elig'] = h.ft_eligible.map(norm)
    for a in ITEMS:
        h[a] = h[a].map(norm)
        h[f'{a}_ev'] = h[f'{a}_evidence']
    for f, col in FLAGS.items():
        h[f] = h[col].map(norm)
    h['excl'] = h.ft_exclusion_reason
    h['notes'] = ''
    return h


def compare(h, a):
    ids = sorted(set(h.index) & set(a.index))
    missing = sorted(set(a.index) - set(h.index)), sorted(set(h.index) - set(a.index))
    rows, dis = [], []

    def add(name, x, y, n_note='', weights=None):
        x, y = list(x), list(y)
        rows.append({'measure': name, 'n': len(x), 'agree_pct': round(100 * np.mean([p == q for p, q in zip(x, y)]), 1),
                     'kappa': round(kappa(x, y, weights), 3) if len(x) else np.nan, 'note': n_note})

    add('FT eligibility', h.loc[ids, 'elig'], a.loc[ids, 'elig'])
    for i in ids:
        if h.at[i, 'elig'] != a.at[i, 'elig']:
            dis.append({'paper_id': i, 'item': 'FT_eligible', 'human': h.at[i, 'elig'], 'ai': a.at[i, 'elig'],
                        'human_evidence': f"{h.at[i, 'excl']} {h.at[i, 'notes']}".strip(),
                        'ai_evidence': f"{a.at[i, 'excl']} {a.at[i, 'notes']}".strip()})

    both = [i for i in ids if h.at[i, 'elig'] == 'Y' and a.at[i, 'elig'] == 'Y']
    for it in ITEMS:
        add(it, h.loc[both, it], a.loc[both, it])
    add('A7 (N merged into NR)', h.loc[both, 'A7'].replace('N', 'NR'), a.loc[both, 'A7'].replace('N', 'NR'),
        'codebook: no naive benchmark = NR')
    pooled_h = np.concatenate([h.loc[both, it].values for it in ITEMS[:6]])
    pooled_a = np.concatenate([a.loc[both, it].values for it in ITEMS[:6]])
    add('A1-A6 pooled', pooled_h, pooled_a, 'items that determine the level')
    for f in FLAGS:
        add(f, h.loc[both, f], a.loc[both, f])

    lh = pd.Series({i: level(h.loc[i]) for i in both})
    la = pd.Series({i: level(a.loc[i]) for i in both})
    ok = lh.notna() & la.notna()
    add('Level 0-3', lh[ok].astype(int), la[ok].astype(int))
    add('Level 0-3 (linear weighted)', lh[ok].astype(int), la[ok].astype(int), weights='linear')
    add('Level 3 vs <3', (lh[ok] == 3).astype(int), (la[ok] == 3).astype(int))

    for i in both:
        for it in ITEMS + list(FLAGS):
            if h.at[i, it] != a.at[i, it]:
                dis.append({'paper_id': i, 'item': it, 'human': h.at[i, it], 'ai': a.at[i, it],
                            'human_evidence': h.at[i, f'{it}_ev'] if it in ITEMS else h.at[i, 'notes'],
                            'ai_evidence': a.at[i, f'{it}_ev'] if it in ITEMS else a.at[i, 'notes']})
        if ok.get(i, False) and lh[i] != la[i]:
            dis.append({'paper_id': i, 'item': 'Level', 'human': int(lh[i]), 'ai': int(la[i]),
                        'human_evidence': '', 'ai_evidence': ''})

    return pd.DataFrame(rows), pd.DataFrame(dis), ids, both, missing


def main():
    os.makedirs(OUT, exist_ok=True)
    a = load(os.path.join(REL, 'automated_codes_sealed.csv'))
    h0 = load(os.path.join(REL, 'author_independent_codes.csv'))
    h1 = load(os.path.join(REL, 'author_independent_codes.csv'),
              os.path.join(REL, 'author_codes_after_codebook_corrections.csv'))
    orig = compare(h0, a)[0]
    corr = compare(h1, a)[0]
    summ = orig.rename(columns={'agree_pct': 'agree_pct_original', 'kappa': 'kappa_original'}).merge(
        corr[['measure', 'agree_pct', 'kappa']].rename(columns={'agree_pct': 'agree_pct_corrected',
                                                               'kappa': 'kappa_corrected'}), on='measure')
    summ.to_csv(os.path.join(OUT, 'kappa_summary.csv'), index=False)
    print(summ.to_string(index=False))


if __name__ == '__main__':
    main()
