"""Figures for Experiment 2 (scaling, smoothing, test-set tuning, shuffled split).

Fig 4  RMSE inflation (leaky over honest) by mechanism and horizon; each point one dataset x learner x seed.
Fig 5  test-set tuning: expected inflation vs number of settings compared on the test set (median and IQR).
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RES = os.path.join(ROOT, 'results', 'exp2_preprocessing')

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
LEAKY, RIDGE_C, MLP_C = '#eb6834', '#2a78d6', '#1baf7a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False})

GROUPS = [('smoothing', 'ma', 'Centred\nmoving avg.'), ('smoothing', 'sg', 'Savitzky–\nGolay'),
          ('tuning', 'test_tuning', 'Tuning on\ntest set'), ('shuffle', 'random_split', 'Random\nsplit'),
          ('scaling', 'minmax', 'Min–max on\nfull series')]


def fig_inflation(cmp_):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), dpi=200, sharey=True)
    rng = np.random.default_rng(0)
    for ax, h in zip(axes, (1, 3)):
        for i, (mech, meth, _) in enumerate(GROUPS):
            v = cmp_.query('horizon==@h and mechanism==@mech and method==@meth')['inflation_pct'].values
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=8, color=LEAKY, alpha=0.55, linewidths=0)
            med = np.median(v)
            ax.plot([i - 0.28, i + 0.28], [med, med], color=INK, lw=2)
            ax.text(i + 0.3, med, f'{med:.0f}%', va='center', fontsize=6.8, color=INK)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xticks(range(len(GROUPS)), [g[2] for g in GROUPS], fontsize=6.6)
        ax.set_xlim(-0.6, len(GROUPS) - 0.2)
        ax.set_title(f'{h}-step-ahead', loc='left', fontsize=8.5, color=INK)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('RMSE inflation (%)')
    fig.text(0.01, 0.005, 'Each point: one dataset × learner (ridge, MLP) × seed. Bar: median. '
                          'Inflation = (RMSE honest − RMSE leaky) / RMSE honest; negative = leaky version worse.',
             fontsize=6.2, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(os.path.join(RES, 'fig4_inflation_by_mechanism.png'))
    plt.close(fig)


def fig_grid(curve):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=200, sharey=True)
    for ax, h in zip(axes, (1, 3)):
        for learner, col, lab in (('ridge', RIDGE_C, 'Ridge'), ('mlp', MLP_C, 'MLP')):
            g = curve.query('horizon==@h and learner==@learner').groupby('grid_size')['inflation_pct']
            k = g.median().index.values
            ax.fill_between(k, g.quantile(0.25).values, g.quantile(0.75).values, color=col, alpha=0.15, lw=0)
            ax.plot(k, g.median().values, color=col, lw=2, marker='o', ms=4, label=lab)
            ax.text(k[-1] * 1.08, g.median().values[-1], lab, color=INK, fontsize=7, va='center')
        ax.set_xscale('log', base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32], ['1', '2', '4', '8', '16', '32'])
        ax.set_xlim(0.8, 60)
        ax.set_xlabel('Settings compared on the test set')
        ax.set_title(f'{h}-step-ahead', loc='left', fontsize=8.5, color=INK)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Expected RMSE inflation (%)')
    axes[0].legend(frameon=False, fontsize=7, loc='upper left')
    fig.text(0.01, 0.005, 'Line: median over datasets × seeds; band: interquartile range. Honest = setting chosen '
                          'on a validation block before the test period.', fontsize=6.2, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(os.path.join(RES, 'fig5_tuning_inflation_vs_grid_size.png'))
    plt.close(fig)


if __name__ == '__main__':
    fig_inflation(pd.read_csv(os.path.join(RES, 'inflation_dm.csv')))
    fig_grid(pd.read_csv(os.path.join(RES, 'tuning_grid_curve.csv')))
    print('figures saved to', RES)
