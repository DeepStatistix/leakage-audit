"""Figures for Experiment 1 (decomposition leakage), from results/exp1_decomposition/*.csv.

Fig 1  test R2 per dataset, one row per decomposition (DWT, CEEMDAN, TQWT, VMD) and one column per horizon
       (ridge learner): leaky vs stepwise-causal decomposition, a no-decomposition model and persistence.
Fig 2  observed vs leaky vs causal forecasts for SPEI-12 Durban, CEEMDAN-ridge and TQWT-ridge (h = 1, seed 0).
Fig 3  distribution of RMSE inflation (leaky over honest) by decomposition, honest design and horizon.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RES = os.path.join(ROOT, 'results', 'exp1_decomposition')

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
CAUSAL, LEAKY = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False})

LABEL = {'spei6_srinagar': 'SPEI-6 Srinagar', 'spei12_srinagar': 'SPEI-12 Srinagar',
         'spei6_durban': 'SPEI-6 Durban', 'spei12_durban': 'SPEI-12 Durban', 'sunspots': 'Sunspots',
         'wheat_price': 'Wheat price', 'etth1_ot': 'ETTh1 (OT)'}


SHORT = {'spei6_srinagar': 'SPEI-6\nSRI', 'spei12_srinagar': 'SPEI-12\nSRI', 'spei6_durban': 'SPEI-6\nDUR',
         'spei12_durban': 'SPEI-12\nDUR', 'sunspots': 'Sun-\nspots', 'wheat_price': 'Wheat\nprice',
         'etth1_ot': 'ETTh1\nOT'}
DECOMPS = [('dwt', 'DWT'), ('ceemdan', 'CEEMDAN'), ('tqwt', 'TQWT'), ('vmd', 'VMD')]


def fig_r2(met, learner='ridge'):
    """Grid: one row per decomposition, one column per horizon; 4 bars per dataset."""
    med = met.groupby(['dataset', 'horizon', 'pipeline', 'learner'])['R2_NSE'].median()
    ds = list(LABEL)
    fig, axes = plt.subplots(len(DECOMPS), 2, figsize=(7.2, 8.6), dpi=200, sharex=True, sharey=True)
    w = 0.2
    for r, (m, mname) in enumerate(DECOMPS):
        bars = [('persistence', '-', 'Persistence', '#9a9890'), ('plain', learner, 'No decomposition', '#6f6e69'),
                (f'{m}_stepwise', learner, 'Decomposition, causal (stepwise)', CAUSAL),
                (f'{m}_leaky', learner, 'Decomposition, leaky (full series)', LEAKY)]
        for c, h in enumerate((1, 3)):
            ax = axes[r, c]
            for k, (pipe, lrn, lab, col) in enumerate(bars):
                v = [med[(d, h, pipe, lrn)] for d in ds]
                ax.bar(np.arange(len(ds)) + (k - 1.5) * w, np.clip(v, 0, None), width=w - 0.03, color=col,
                       edgecolor='white', linewidth=0.5, label=lab)
            ax.set_ylim(0, 1.02)
            ax.yaxis.grid(True, color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            ax.set_title(f'{mname}, {h}-step-ahead', loc='left', fontsize=8, color=INK)
            if c == 0:
                ax.set_ylabel('Test R²', fontsize=8)
    for ax in axes[-1]:
        ax.set_xticks(np.arange(len(ds)), [SHORT[d] for d in ds], fontsize=6.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, fontsize=6.8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.004, 'Ridge learner; CEEMDAN median over 3 seeds; negative R² shown as 0. SRI = Srinagar, '
                          'DUR = Durban.', fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.015, 1, 0.975))
    fig.savefig(os.path.join(RES, 'fig1_r2_by_pipeline.png'))
    plt.close(fig)


def fig_series(pred, method='ceemdan', mname='CEEMDAN', dataset='spei12_durban', learner='ridge'):
    g = pred.query('dataset==@dataset and learner==@learner and horizon==1 and seed==0')
    obs = g[g.pipeline == f'{method}_leaky'].set_index('date')['y']
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.4), dpi=200, sharex=True, sharey=True)
    for ax, (pipe, col, lab) in zip(axes, [(f'{method}_leaky', LEAKY, f'{mname}-ridge, leaky (full-series decomposition)'),
                                           (f'{method}_stepwise', CAUSAL,
                                            f'{mname}-ridge, causal (stepwise decomposition)')]):
        p = g[g.pipeline == pipe].set_index('date')['y_hat']
        e = obs - p
        r2 = 1 - np.sum(e ** 2) / np.sum((obs - obs.mean()) ** 2)
        ax.plot(obs.index, obs.values, color=INK, lw=1.4, label='Observed')
        ax.plot(p.index, p.values, color=col, lw=1.4, label='Forecast')
        ax.set_title(f'{lab}   R² = {r2:.3f}', loc='left', fontsize=8.5, color=INK)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_ylabel(LABEL[dataset].split()[0])
    axes[0].legend(frameon=False, fontsize=7.5, loc='lower left', ncol=2)
    fig.tight_layout()
    suffix = '' if method == 'ceemdan' else f'_{method}'
    fig.savefig(os.path.join(RES, f'fig2_{dataset}_leaky_vs_causal{suffix}.png'))
    plt.close(fig)


def fig_inflation(cmp_):
    groups = [(m, d) for m, _ in DECOMPS for d in ('causal', 'stepwise')]
    names = [mn + ('\nnaive' if d == 'causal' else '\nstepwise') for m, mn in DECOMPS for d in ('causal', 'stepwise')]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), dpi=200, sharey=True)
    rng = np.random.default_rng(0)
    for ax, h in zip(axes, (1, 3)):
        for i, (m, d) in enumerate(groups):
            v = cmp_.query('horizon==@h and decomposition==@m and honest_design==@d')['inflation_pct'].values
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=8, color=LEAKY, alpha=0.55, linewidths=0)
            med = np.median(v)
            ax.plot([i - 0.28, i + 0.28], [med, med], color=INK, lw=2)
            ax.text(i + 0.31, med, f'{med:.0f}%', va='center', fontsize=6.8, color=INK)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xticks(range(len(groups)), names, fontsize=7)
        ax.set_xlim(-0.6, len(groups) - 0.2)
        ax.set_title(f'{h}-step-ahead', loc='left', fontsize=8.5, color=INK)
        ax.set_ylabel('RMSE inflation (%)')
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    fig.text(0.01, 0.005, 'Each point: one dataset × learner (ridge, MLP) × seed. Bar: median. Honest design: naive '
                          'causal or stepwise.\nInflation = (RMSE honest − RMSE leaky) / RMSE honest.',
             fontsize=6.3, color=MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(os.path.join(RES, 'fig3_inflation_distribution.png'))
    plt.close(fig)


if __name__ == '__main__':
    met = pd.read_csv(os.path.join(RES, 'metrics.csv'))
    pred = pd.read_csv(os.path.join(RES, 'predictions.csv'), parse_dates=['date'])
    cmp_ = pd.read_csv(os.path.join(RES, 'inflation_dm.csv'))
    fig_r2(met)
    fig_series(pred)
    fig_series(pred, 'tqwt', 'TQWT')
    fig_inflation(cmp_)
    print('figures saved to', RES)
