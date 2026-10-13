"""PRISMA 2020 flow diagram for the audit, in the style of the PRISMA 2020 template (Page et al., 2021).

Counts are given as S1 / S2 (general pool / decomposition stratum); the caption explains this. All counts are
computed from the search log constants, data/search/screening_<pool>.csv and data/audit/audit_data.csv.
Run:  python scripts/analysis/prisma_diagram.py
Output: results/audit/fig_prisma.png (600 dpi), .pdf, .svg and prisma_counts.csv
"""
import os

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'results', 'audit')
# from data/search/search_log.md (OpenAlex, run 2026-10-07)
SEARCH = {'S1': {'downloaded': 23778, 'unique': 21329, 'oa': 14019},
          'S2': {'downloaded': 2230, 'unique': 2222, 'oa': 1224}}
FONT = 'Times New Roman'
FS = 8.5           # body text size (pt)
LW = 0.8           # line width
PHASE_FILL = '#d9d9d9'


def counts():
    f = pd.read_csv(os.path.join(ROOT, 'data', 'audit', 'audit_data.csv'), keep_default_na=False)
    f['in_sample'] = f.in_sample.astype(str) == 'True'
    f['source'] = f.coding_route.map(lambda r: 'AI' if r.startswith('automated') else 'indep')
    c = {}
    for p in ('S1', 'S2'):
        s = pd.read_csv(os.path.join(ROOT, 'data', 'search', f'screening_{p}.csv'), keep_default_na=False)
        s = s.rename(columns={'final_decision': 'decision', 'final_reason': 'reason'})
        ins = f[(f.pool == p) & f.in_sample]
        stop = ins.oa_order.max()
        within = s[s.oa_order <= stop]
        inc = within[within.decision.isin(['Include', 'Unsure'])]
        unav = within[within.full_text_retrieved == 'N']
        c[p] = {**SEARCH[p], 'dups': SEARCH[p]['downloaded'] - SEARCH[p]['unique'],
                'not_oa': SEARCH[p]['unique'] - SEARCH[p]['oa'], 'screened': len(s), 'stop': int(stop),
                'not_required': SEARCH[p]['oa'] - int(stop),
                'beyond_stop': len(s) - len(within), 'ta_excluded': int((within.decision == 'Exclude').sum()),
                'ta_reasons': within[within.decision == 'Exclude'].reason.value_counts().to_dict(),
                'sought': len(inc), 'not_retrieved': len(unav), 'assessed': len(ins),
                'ft_excluded': int((ins.ft_eligible == 'N').sum()),
                'ft_reasons': ins[ins.ft_eligible == 'N'].ft_exclusion_reason.value_counts().to_dict(),
                'included': int((ins.ft_eligible == 'Y').sum()),
                'double_coded': int(((ins.ft_eligible == 'Y') & ins.source.str.startswith('indep')).sum()),
                'double_coded_all': int(ins.source.str.startswith('indep').sum())}
    return c


def n(c, key):
    return f'{c["S1"][key]:,} / {c["S2"][key]:,}'


def reasons(c, key):
    d1, d2 = c['S1'][key], c['S2'][key]
    order = sorted(set(d1) | set(d2), key=lambda k: -(d1.get(k, 0) + d2.get(k, 0)))
    nice = {'Not forecasting': 'Not a forecasting study', 'Classification/anomaly task': 'Classification or anomaly task',
            'Not time series': 'Not a time series', 'Review/survey': 'Review or survey',
            'Full text unavailable': 'Full text not available', 'Theory/simulation only': 'Theory or simulation only',
            'No out-of-sample accuracy': 'No out-of-sample evaluation', 'Retracted': 'Retracted',
            'Not English': 'Not in English'}
    return [f'{nice.get(k, k)} (n = {d1.get(k, 0)} / {d2.get(k, 0)})' for k in order]


def box(ax, x, y, w, h, lines, align='center', title=None, pad_top=0.09):
    """Square PRISMA box; y is the top edge. Optional title line in bold."""
    ax.add_patch(Rectangle((x, y - h), w, h, fill=True, fc='white', ec='black', lw=LW))
    pad = 0.09
    if align == 'center':
        txt = '\n'.join(lines)
        ax.text(x + w / 2, y - h / 2, txt, ha='center', va='center', fontsize=FS, family=FONT, linespacing=1.35)
    else:
        yy = y - pad_top
        if title:
            ax.text(x + pad, yy, title, ha='left', va='top', fontsize=FS, family=FONT)
            yy -= 0.18
        ax.text(x + pad + 0.06, yy, '\n'.join(lines), ha='left', va='top', fontsize=FS, family=FONT,
                linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color='black', lw=LW, mutation_scale=9, shrinkA=0, shrinkB=0))


def phase(ax, y_top, y_bot, label, x=0.05, w=0.30):
    ax.add_patch(Rectangle((x, y_bot), w, y_top - y_bot, fc=PHASE_FILL, ec='black', lw=LW))
    ax.text(x + w / 2, (y_top + y_bot) / 2, label, rotation=90, ha='center', va='center', fontsize=FS + 0.5,
            family=FONT, fontweight='bold')


def main():
    c = counts()
    W, H = 7.0, 8.9                              # inches; full page width
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')

    L, LW_ = 0.55, 3.05                          # left column x and width
    R, RW = 3.95, 2.95                           # right column x and width
    lc = L + LW_ / 2                             # left column centre

    # header bar (PRISMA template)
    ax.add_patch(Rectangle((L, H - 0.42), R + RW - L, 0.30, fc=PHASE_FILL, ec='black', lw=LW))
    ax.text((L + R + RW) / 2, H - 0.27, 'Identification of studies via OpenAlex', ha='center', va='center',
            fontsize=FS + 0.5, family=FONT, fontweight='bold')

    # rows: (top y, height)
    y1, h1 = H - 0.62, 0.78
    y2, h2 = y1 - h1 - 0.35, 0.62
    y3, h3 = y2 - h2 - 0.35, 0.55
    y4, h4 = y3 - h3 - 0.35, 0.55
    y5, h5 = y4 - h4 - 0.35, 0.55
    y6, h6 = y5 - h5 - 0.35, 0.80
    hr3 = 1.40                                   # height of the title/abstract exclusion box
    hr5 = 1.13                                   # height of the full-text exclusion box

    # Identification
    box(ax, L, y1, LW_, h1, ['Records identified from OpenAlex', '(journal articles, 2018–2025, English)',
                             f'(n = {n(c, "downloaded")})'])
    box(ax, R, y1, RW, h1, [f'Duplicate records (n = {n(c, "dups")})', f'Not open access (n = {n(c, "not_oa")})'],
        align='left', title='Records removed before screening:', pad_top=0.13)
    arrow(ax, L + LW_, y1 - h1 / 2, R, y1 - h1 / 2)

    box(ax, L, y2, LW_, h2, ['Open-access records, placed in', f'random order (n = {n(c, "oa")})'])
    box(ax, R, y2, RW, h2, ['Records not required after the', f'quota was reached (n = {n(c, "not_required")})'])
    arrow(ax, L + LW_, y2 - h2 / 2, R, y2 - h2 / 2)
    arrow(ax, lc, y1 - h1, lc, y2)

    # Screening
    box(ax, L, y3, LW_, h3, ['Records screened in order', f'(n = {n(c, "stop")})'])
    box(ax, R, y3, RW, hr3, reasons(c, 'ta_reasons'), align='left',
        title=f'Records excluded (n = {n(c, "ta_excluded")}):')
    arrow(ax, L + LW_, y3 - h3 / 2, R, y3 - h3 / 2)
    arrow(ax, lc, y2 - h2, lc, y3)

    # push the following rows below the tall exclusion box
    y4 = min(y4, y3 - hr3 - 0.30)
    box(ax, L, y4, LW_, h4, ['Reports sought for retrieval', f'(n = {n(c, "sought")})'])
    box(ax, R, y4, RW, h4, ['Reports not retrieved', f'(n = {n(c, "not_retrieved")})'])
    arrow(ax, L + LW_, y4 - h4 / 2, R, y4 - h4 / 2)
    arrow(ax, lc, y3 - h3, lc, y4)

    y5 = y4 - h4 - 0.35
    box(ax, L, y5, LW_, h5, ['Reports assessed for eligibility', f'(n = {n(c, "assessed")})'])
    box(ax, R, y5, RW, hr5, reasons(c, 'ft_reasons'), align='left',
        title=f'Reports excluded (n = {n(c, "ft_excluded")}):')
    arrow(ax, L + LW_, y5 - h5 / 2, R, y5 - h5 / 2)
    arrow(ax, lc, y4 - h4, lc, y5)

    # Included
    y6 = min(y6, y5 - hr5 - 0.30)
    total = c['S1']['included'] + c['S2']['included']
    box(ax, L, y6, LW_, h6, ['Studies included in the audit', f'(n = {n(c, "included")}; total {total})',
                             f'Independently double-coded (n = {n(c, "double_coded")})'])
    arrow(ax, lc, y5 - h5, lc, y6)

    phase(ax, y1 + 0.0, y2 - h2, 'Identification')
    phase(ax, y3, y5 - h5 - 0.02 if y5 - h5 > y5 - hr5 else y5 - hr5, 'Screening')
    phase(ax, y6, y6 - h6, 'Included')

    bottom = y6 - h6 - 0.15
    ax.set_ylim(bottom, H)
    fig.set_size_inches(W, H - bottom)
    for ext, kw in (('png', {'dpi': 600}), ('pdf', {}), ('svg', {})):
        fig.savefig(os.path.join(OUT, f'fig_prisma.{ext}'), facecolor='white', **kw)
    pd.DataFrame({p: {k: v for k, v in c[p].items() if not isinstance(v, dict)} for p in c}).to_csv(
        os.path.join(OUT, 'prisma_counts.csv'))
    print('saved; not required:', c['S1']['not_required'], c['S2']['not_required'])


if __name__ == '__main__':
    main()
