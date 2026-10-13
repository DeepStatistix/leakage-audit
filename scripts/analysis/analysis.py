"""Audit analysis (protocol section 9): sample description, RQ1, RQ2, RQ4 (exploratory), flags, reliability.

Input:  data/audit/audit_data.csv, data/reliability/kappa_summary.csv, data/reliability/adjudications.csv,
        data/verification/revision_log.csv
Output: results/audit/ (tables as CSV, figures as PNG/PDF, results.md with every number)

Leakage types are derived from the N codes (A1 split, A2 model selection on test, A3 preprocessing,
A4 decomposition/smoothing, A5 target/horizon/input, A6 tuning on test), so they are reproducible.
Run:  python scripts/analysis/analysis.py
"""
import os

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'results', 'audit')
ITEMS = [f'A{i}' for i in range(1, 11)]
ITEM_NAMES = {'A1': 'Data split', 'A2': 'Validation / model selection', 'A3': 'Scaling / imputation',
              'A4': 'Decomposition / smoothing', 'A5': 'Inputs and horizon', 'A6': 'Hyperparameter tuning',
              'A7': 'Naive baseline', 'A8': 'Statistical comparison', 'A9': 'Reproducibility',
              'A10': 'Metric consistency'}
TYPES = {'A1': 'Split (random / shuffled)', 'A2': 'Model selection on test', 'A3': 'Preprocessing (scaling, imputation)',
         'A4': 'Decomposition / smoothing', 'A5': 'Target / horizon / inputs', 'A6': 'Tuning on test'}
DOMAIN_GROUPS = {
    'Energy': ['Energy', 'Power grid'],
    'Finance / economics': ['Finance/economics', 'Business/demand', 'Business/finance', 'Business/supply chain', 'Tourism'],
    'Hydrology / climate / ocean': ['Hydrology/climate', 'Ocean', 'Agriculture/hydrology', 'Climate/health', 'Space weather'],
    'Environment / air quality': ['Environment', 'Air quality/environment'],
    'Agriculture / food': ['Agriculture/commodity', 'Agriculture/food'],
    'Transport': ['Transport', 'Transport/logistics'],
    'Health': ['Health/epidemiology'],
    'Generic benchmarks': ['Generic/benchmarks'],
}
# figure palette (validated with the dataviz validator): ordinal blue ramp for levels; diverging for codes
LEVEL_COLORS = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b']
CODE_COLORS = {'C': '#2a78d6', 'NR': '#a8a69c', 'NA': '#dddcd6', 'N': '#e34948'}
INK, INK2, SURFACE = '#0b0b0b', '#52514e', '#fcfcfb'


def wilson(k, n):
    lo, hi = proportion_confint(k, n, method='wilson')
    return f'{100 * k / n:.1f}% (95% CI {100 * lo:.1f}-{100 * hi:.1f})'


def load():
    f = pd.read_csv(os.path.join(ROOT, 'data', 'audit', 'audit_data.csv'), keep_default_na=False)
    f['in_sample'] = f.in_sample.astype(str) == 'True'
    f['source'] = np.where(f.coding_route.str.startswith('automated'), 'AI first pass', 'independent double coding')
    grp = {d: g for g, ds in DOMAIN_GROUPS.items() for d in ds}
    f['domain_group'] = f.domain.map(grp).fillna('Other')
    f['level'] = f.level.astype(str).str.replace('.0', '', regex=False)
    f['level3'] = (f.level == '3').astype(int)
    f['uses_decomp'] = (f.A4 != 'NA').astype(int)
    f['quartile'] = f.sjr_quartile_2025
    return f


def fig_levels(e):
    cats = [('0', ''), ('1', ''), ('2', ''), ('3', '3b'), ('3', '3a')]
    labels = ['0 Verifiably causal', '1 Probably causal', '2 Indeterminate', '3b Leakage-prone (low severity)',
              '3a Leakage-prone (high severity)']
    fig, ax = plt.subplots(figsize=(7.2, 2.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    pools = [('S2', 'S2 decomposition stratum'), ('S1', 'S1 general pool')]
    for y, (p, name) in enumerate(pools):
        d = e[e.pool == p]
        left = 0
        for (lv, sev), col in zip(cats, LEVEL_COLORS):
            n = ((d.level == lv) & (d.severity == sev)).sum()
            w = 100 * n / len(d)
            ax.barh(y, w, left=left, color=col, height=0.55, edgecolor=SURFACE, linewidth=2)
            if w >= 6:
                ax.text(left + w / 2, y, f'{w:.0f}%', ha='center', va='center', fontsize=8,
                        color='white' if col in LEVEL_COLORS[2:] else INK)
            left += w
        ax.text(-1, y, f'{name}\n(n = {len(d)})', ha='right', va='center', fontsize=8.5, color=INK)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel('Share of eligible papers (%)', fontsize=8.5, color=INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#c3c2b7')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in LEVEL_COLORS]
    ax.legend(handles, labels, ncol=3, fontsize=7.5, frameon=False, loc='upper center', bbox_to_anchor=(0.42, 1.42))
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'fig_levels.{ext}'), dpi=300, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def fig_items(e):
    fig, ax = plt.subplots(figsize=(7.2, 4.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    order = ['C', 'NR', 'NA', 'N']
    names = {'C': 'C causal / reported', 'NR': 'NR not reported', 'NA': 'NA not applicable', 'N': 'N non-causal / absent'}
    for y, it in enumerate(reversed(ITEMS)):
        left = 0
        for code in order:
            n = (e[it] == code).sum()
            w = 100 * n / len(e)
            ax.barh(y, w, left=left, color=CODE_COLORS[code], height=0.6, edgecolor=SURFACE, linewidth=2)
            if w >= 7:
                ax.text(left + w / 2, y, f'{n}', ha='center', va='center', fontsize=7.5,
                        color='white' if code in ('C', 'N') else INK)
            left += w
    ax.set_yticks(range(len(ITEMS)))
    ax.set_yticklabels([f'{it} {ITEM_NAMES[it]}' for it in reversed(ITEMS)], fontsize=8, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f'Share of eligible papers (%), n = {len(e)}; numbers in bars are paper counts', fontsize=8, color=INK2)
    ax.tick_params(axis='x', colors=INK2, labelsize=8)
    ax.tick_params(axis='y', length=0)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#c3c2b7')
    handles = [plt.Rectangle((0, 0), 1, 1, color=CODE_COLORS[c]) for c in order]
    ax.legend(handles, [names[c] for c in order], ncol=4, fontsize=7.5, frameon=False, loc='upper center',
              bbox_to_anchor=(0.4, 1.09))
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'fig_items.{ext}'), dpi=300, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    f = load()
    s = f[f.in_sample]
    e = s[s.ft_eligible == 'Y'].copy()
    R = ['# Audit results', '', 'Generated by `scripts/analysis/analysis.py` from `data/audit/audit_data.csv`.', '']

    # 1. flow and sample description
    R += ['## 1. Sample', '']
    for p in ('S1', 'S2'):
        sp = s[s.pool == p]
        exc = sp[sp.ft_eligible == 'N'].ft_exclusion_reason.value_counts()
        R.append(f'- {p}: {len(sp)} full texts assessed, {len(sp) - (sp.ft_eligible == "Y").sum()} excluded '
                 f'({"; ".join(f"{k} {v}" for k, v in exc.items())}), {(sp.ft_eligible == "Y").sum()} eligible.')
    desc = []
    for col in ('domain_group', 'model_family'):
        t = pd.crosstab(e[col], e.pool, margins=True, margins_name='Total').sort_values('Total', ascending=False)
        t.to_csv(os.path.join(OUT, f'table_{col}.csv'))
        desc.append((col, t))
    yr = pd.crosstab(e.year, e.pool, margins=True, margins_name='Total')
    yr.to_csv(os.path.join(OUT, 'table_year.csv'))
    R += ['', f'- Publication years {int(e.year.min())}-{int(e.year.max())}; median {int(e.year.median())}.',
          f'- Domains (grouped): ' + '; '.join(f'{k} {v}' for k, v in desc[0][1]['Total'].drop('Total').items()) + '.',
          f'- Model families: ' + '; '.join(f'{k} {v}' for k, v in desc[1][1]['Total'].drop('Total').items()) + '.',
          ('- Journal quartile (SJR best quartile; SJR table nearest the publication year, currently 2025 for all papers): ' + '; '.join(
              f'{k} {v}' for k, v in e.quartile.value_counts().items()) + '.' if (e.quartile != '').all()
           else '- Journal quartile: SJR tables not yet available; RQ4 is run without it.'), '']

    # 2. RQ1 and RQ2: levels
    R += ['## 2. Leakage-risk levels (RQ1, RQ2)', '']
    lv = []
    for p in ('S1', 'S2'):
        d = e[e.pool == p]
        n = len(d)
        row = {'pool': p, 'n': n}
        for L in ('0', '1', '2', '3'):
            row[f'level{L}'] = (d.level == L).sum()
        row['level3a'] = (d.severity == '3a').sum()
        row['level3b'] = (d.severity == '3b').sum()
        lv.append(row)
        R += [f'- **{p}** (n = {n}): Level 0 {wilson(row["level0"], n)}; Level 1 {wilson(row["level1"], n)}; '
              f'Level 2 {wilson(row["level2"], n)}; Level 3 {wilson(row["level3"], n)} '
              f'[3a {row["level3a"]}, 3b {row["level3b"]}].']
    pd.DataFrame(lv).to_csv(os.path.join(OUT, 'table_levels.csv'), index=False)
    n01 = e.level.isin(['0', '1']).sum()
    R += [f'- Both pools: Level 0 {wilson((e.level == "0").sum(), len(e))}; Level 0-1 (at most probably causal) '
          f'{wilson(n01, len(e))}; Level 3 {wilson(e.level3.sum(), len(e))}.', '']
    fig_levels(e)

    # items
    it = pd.DataFrame({c: [(e[i] == c).sum() for i in ITEMS] for c in ('C', 'N', 'NR', 'NA')}, index=ITEMS)
    it.insert(0, 'item', [ITEM_NAMES[i] for i in ITEMS])
    for p in ('S1', 'S2'):
        for c in ('C', 'N', 'NR', 'NA'):
            it[f'{p}_{c}'] = [((e.pool == p) & (e[i] == c)).sum() for i in ITEMS]
    it.to_csv(os.path.join(OUT, 'table_items.csv'))
    R += ['### Item-level distribution (both pools)', '', '| Item | C | N | NR | NA |', '|---|---|---|---|---|']
    R += [f'| {i} {ITEM_NAMES[i]} | {r.C} | {r.N} | {r.NR} | {r.NA} |' for i, r in it.iterrows()]
    fig_items(e)

    # taxonomy from N codes
    tx = pd.DataFrame([{'type': TYPES[i], 'S1': ((e.pool == 'S1') & (e[i] == 'N')).sum(),
                        'S2': ((e.pool == 'S2') & (e[i] == 'N')).sum()} for i in TYPES])
    tx['total'] = tx.S1 + tx.S2
    tx['pct_of_level3'] = (100 * tx.total / e.level3.sum()).round(1)
    tx.to_csv(os.path.join(OUT, 'table_leakage_types.csv'), index=False)
    multi = (e[list(TYPES)].eq('N').sum(axis=1) >= 2).sum()
    R += ['', '### Leakage types (papers with N on the item)', '', '| Type | S1 | S2 | Total | % of Level-3 papers |',
          '|---|---|---|---|---|'] + [f'| {r.type} | {r.S1} | {r.S2} | {r.total} | {r.pct_of_level3} |' for r in tx.itertuples()]
    R += ['', f'- Papers with two or more leakage types: {multi} of {e.level3.sum()} Level-3 papers.']
    nr6 = e[ITEMS[:6]].eq('NR').sum(axis=1)
    R += [f'- Median number of NR codes among A1-A6 per paper: {nr6.median():.0f} (IQR {nr6.quantile(.25):.0f}-'
          f'{nr6.quantile(.75):.0f}); papers with every applicable A1-A6 item reported (no NR): {(nr6 == 0).sum()}.', '']

    # 3. RQ4 exploratory
    R += ['## 3. Exploratory associations (RQ4)', '']
    t = pd.crosstab(e.pool, e.level3)
    odds, pf = stats.fisher_exact(t.values)
    R.append(f'- Level 3, S2 vs S1: {t.loc["S2", 1]}/{t.loc["S2"].sum()} vs {t.loc["S1", 1]}/{t.loc["S1"].sum()}; '
             f'Fisher exact p = {pf:.3f} (OR {odds:.2f}, S2 vs S1).')
    t2 = pd.crosstab(e.uses_decomp, e.level3)
    o2, p2 = stats.fisher_exact(t2.values)
    R.append(f'- Level 3 with vs without decomposition/smoothing (A4 applicable): {t2.loc[1, 1]}/{t2.loc[1].sum()} vs '
             f'{t2.loc[0, 1]}/{t2.loc[0].sum()}; Fisher p = {p2:.3f} (OR {o2:.2f}).')
    if (e.quartile != '').all():
        hq = e.quartile.isin(['Q1', 'Q2']).astype(int)
        t3 = pd.crosstab(hq, e.level3)
        o3, p3 = stats.fisher_exact(t3.values)
        R.append(f'- Level 3 in Q1-Q2 journals vs Q3-Q4/unranked/not indexed: {t3.loc[1, 1]}/{t3.loc[1].sum()} vs '
                 f'{t3.loc[0, 1]}/{t3.loc[0].sum()}; Fisher p = {p3:.3f} (OR {o3:.2f}). Level 0-1 in Q1-Q2: '
                 f'{e[hq == 1].level.isin(["0", "1"]).sum()}/{hq.sum()}; elsewhere {e[hq == 0].level.isin(["0", "1"]).sum()}/{(hq == 0).sum()}.')
    # logistic regression: Level 3 ~ decomposition + domain (Energy / Finance / Hydrology / other) + year
    X = pd.DataFrame({'decomposition': e.uses_decomp,
                      'year_c': e.year - e.year.median()})
    dom = e.domain_group.where(e.domain_group.isin(['Energy', 'Finance / economics', 'Hydrology / climate / ocean']), 'Other')
    X = X.join(pd.get_dummies(dom, prefix='dom', drop_first=False).drop(columns='dom_Other').astype(int))
    has_q = (e.quartile != '').all()
    if has_q:  # Q1/Q2 vs Q3/Q4/not indexed
        X['quartile_Q1Q2'] = e.quartile.isin(['Q1', 'Q2']).astype(int)
    X = sm.add_constant(X)
    try:
        m = sm.Logit(e.level3, X).fit(disp=False)
        ci = m.conf_int()
        lr = pd.DataFrame({'OR': np.exp(m.params), 'CI_low': np.exp(ci[0]), 'CI_high': np.exp(ci[1]), 'p': m.pvalues})
        lr.round(3).to_csv(os.path.join(OUT, 'table_logistic_level3.csv'))
        R += ['', '| Term | OR | 95% CI | p |', '|---|---|---|---|'] + [
            f'| {k} | {r.OR:.2f} | {r.CI_low:.2f}-{r.CI_high:.2f} | {r.p:.3f} |' for k, r in lr.drop('const').iterrows()]
        R.append(f'\nLogistic regression of Level 3 (n = {len(e)}; reference domain = other fields; year centred at '
                 f'{int(e.year.median())}); pseudo-R2 {m.prsquared:.3f}. Exploratory.')
    except Exception as ex:  # noqa: BLE001
        R.append(f'- Logistic regression failed: {ex}')
    # reported accuracy (R2 / NSE) by level
    acc = e[e.best_metric_name.str.upper().isin(['R2', 'NSE'])].copy()
    acc['v'] = pd.to_numeric(acc.best_metric_value, errors='coerce')
    acc = acc.dropna(subset=['v'])
    groups = [g.v.values for _, g in acc.groupby('level') if len(g) >= 2]
    if len(groups) >= 2:
        kw = stats.kruskal(*groups)
        med = acc.groupby('level').v.agg(['size', 'median']).round(3)
        R += ['', f'- Headline R2/NSE was available for only {len(acc)} papers; medians by level: ' +
              '; '.join(f'L{k} {r["median"]} (n={int(r["size"])})' for k, r in med.iterrows()) +
              f'; Kruskal-Wallis p = {kw.pvalue:.3f}. Too few for inference; reported descriptively.']
    else:
        R.append(f'- Headline R2/NSE available for only {len(acc)} papers; not tested.')

    # 4. flags
    R += ['', '## 4. Forensic flags', '']
    for fl, nm in (('F1_metric', 'F1 metric inconsistency'), ('F2_skill', 'F2 implausible skill'),
                   ('F3_claim_gap', 'F3 claim-evidence gap')):
        R.append(f'- {nm}: S1 {((e.pool == "S1") & (e[fl] == "Y")).sum()}/{(e.pool == "S1").sum()}, '
                 f'S2 {((e.pool == "S2") & (e[fl] == "Y")).sum()}/{(e.pool == "S2").sum()}; total {wilson((e[fl] == "Y").sum(), len(e))}.')

    # 5. reliability and verification
    R += ['', '## 5. Reliability and verification', '']
    k = pd.read_csv(os.path.join(ROOT, 'data', 'reliability', 'kappa_summary.csv'), keep_default_na=False)
    R += ['| Measure | n | % agreement | kappa (original) | kappa (after codebook corrections) |', '|---|---|---|---|---|']
    R += [f'| {r.measure} | {r.n} | {r.agree_pct_original} | {r.kappa_original} | {r.kappa_corrected} |' for r in k.itertuples()]
    adj = pd.read_csv(os.path.join(ROOT, 'data', 'reliability', 'adjudications.csv'), keep_default_na=False)
    R += ['', f'- Adjudicated items: {len(adj)}; final code = author {(adj.status == "agreed with coder 1").sum()}, '
              f'AI {(adj.status == "agreed with AI").sum()}, neither {(adj.status == "neither - changed").sum()}.']
    rl = pd.read_csv(os.path.join(ROOT, 'data', 'verification', 'revision_log.csv'), keep_default_na=False)
    rev = e[e.source.str.startswith('AI')]
    R.append(f'- Non-sealed papers: {len(rev)} eligible. AI codes changed in the full-text re-pass: {len(rl)} codes in '
             f'{rl.paper_id.nunique()} papers ({100 * len(rl) / (13 * len(rev)):.1f}% of the item and flag codes); '
             f'codes changed by the author during verification: 0.')
    with open(os.path.join(OUT, 'results.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(R) + '\n')
    print('\n'.join(R))


if __name__ == '__main__':
    main()
