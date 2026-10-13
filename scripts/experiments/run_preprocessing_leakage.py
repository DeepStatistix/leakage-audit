"""Experiment 2: leakage through scaling, smoothing, test-set tuning and shuffled splits.

Same datasets, test targets (last 120 points) and horizons (1, 3) as Experiment 1; learners ridge and MLP;
direct forecasts from the last L lags; expanding window, models refit every REFIT origins. In every pair the
leaky and honest versions differ only in the information the step may use.

  scaling    leaky: min-max scaling with the min/max of the FULL series   honest: min/max of data up to the origin
             (MLP and ridge are fitted on the scaled values, with no other scaling)
  smoothing  inputs are lags of a smoothed series, target and evaluation are the raw series
             ma  leaky: centred 5-point moving average      honest: trailing 5-point moving average
             sg  leaky: Savitzky-Golay (7, 2) on the full series   honest: Savitzky-Golay end-point fit on the
                 last 7 observations only
             reference: lags of the raw series
  tuning     a grid of settings (ridge: 5 penalties x 4 lag lengths; MLP: 4 widths x 2 penalties x 4 lag lengths)
             leaky: setting with the lowest TEST error   honest: setting with the lowest error on the 60-point
             validation block before the test period. Also records how the gap grows with grid size.
  shuffle    a random 60 of the 120 test targets is held out; leaky: one model fitted on all other rows, including
             rows after the held-out targets (random split)   honest: rolling-origin forecasts of the same 60 targets

Run:  C:\\Python314\\python.exe -u scripts/experiments/run_preprocessing_leakage.py
Outputs: results/exp2_preprocessing/
"""
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal import savgol_coeffs, savgol_filter
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets as D  # noqa: E402
import run_decomposition_leakage as R  # noqa: E402

OUT = os.path.join(D.ROOT, 'results', 'exp2_preprocessing')
REFIT = 12
N_VAL = 60
MECHANISMS = ['tuning', 'smoothing', 'shuffle', 'scaling']  # slowest first
GRID = {'ridge': [{'alpha': a, 'lags': L} for a in (1e-3, 1e-1, 1.0, 10.0, 100.0) for L in (3, 6, 12, 24)],
        'mlp': [{'hidden': hd, 'alpha': a, 'lags': L} for hd in (8, 16, 32, 64) for a in (1e-4, 1e-2)
                for L in (3, 6, 12, 24)]}


def log(msg):
    with open(os.path.join(OUT, 'progress.log'), 'a', encoding='utf-8') as f:
        f.write(time.strftime('%H:%M:%S ') + msg + '\n')


def maker(kind, seed, cfg=None, inner_scaling=True):
    cfg = cfg or {}
    if kind == 'ridge':
        def make():
            m = Ridge(alpha=cfg.get('alpha', 1.0))
            return R.make_pipeline(R.StandardScaler(), m) if inner_scaling else m
    else:
        def make():
            m = MLPRegressor(hidden_layer_sizes=(cfg.get('hidden', 32),), alpha=cfg.get('alpha', 1e-3),
                             max_iter=2000, random_state=seed)
            return R.make_pipeline(R.StandardScaler(), m) if inner_scaling else m
    if not inner_scaling:
        return make
    return lambda: R.TransformedTargetRegressor(regressor=make(), transformer=R.StandardScaler())


def lagmat(x, lags):
    X = np.full((len(x), lags), np.nan)
    X[lags - 1:] = np.lib.stride_tricks.sliding_window_view(x, lags)
    return X


def roll(x, y, targets, h, lags, make, scale=None):
    """Direct h-step forecasts of y[targets] from lags of input series x; refit every REFIT origins.
    scale: None, 'global' (min-max of the full y) or 'train' (min-max of y up to the refit origin)."""
    X = lagmat(x, lags)
    preds = []
    for k, t in enumerate(targets - h):
        if k % REFIT == 0:
            rows = np.arange(lags - 1, t - h + 1)
            if scale == 'global':
                lo, hi = y.min(), y.max()
            elif scale == 'train':
                lo, hi = y[:t + 1].min(), y[:t + 1].max()
            else:
                lo, hi = 0.0, 1.0
            m = make().fit((X[rows] - lo) / (hi - lo), (y[rows + h] - lo) / (hi - lo))
        preds.append(float(m.predict((X[t][None, :] - lo) / (hi - lo))[0]) * (hi - lo) + lo)
    return np.array(preds)


def causal_savgol(y, window=7, poly=2):
    c = savgol_coeffs(window, poly, pos=window - 1, use='dot')
    out = y.copy()
    for i in range(window - 1, len(y)):
        out[i] = c @ y[i - window + 1:i + 1]
    return out


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run_task(args):
    name, mech, seed = args
    warnings.filterwarnings('ignore')
    s, lags, n_test = D.load(name)
    y = s.values.astype(float)
    n = len(y)
    targets = np.arange(n - n_test, n)
    kinds = ['ridge', 'mlp'] if seed == R.SEEDS[0] else ['mlp']
    preds, grid_rows = [], []
    t0 = time.time()

    def add(h, kind, method, arm, tg, p):
        preds.extend({'dataset': name, 'horizon': h, 'seed': seed, 'mechanism': mech, 'method': method, 'arm': arm,
                      'learner': kind, 'date': s.index[i], 'y': y[i], 'y_hat': v} for i, v in zip(tg, p))

    for h in R.HORIZONS:
        for kind in kinds:
            if mech == 'scaling':
                make = maker(kind, seed, inner_scaling=False)
                add(h, kind, 'minmax', 'leaky', targets, roll(y, y, targets, h, lags, make, 'global'))
                add(h, kind, 'minmax', 'honest', targets, roll(y, y, targets, h, lags, make, 'train'))
            elif mech == 'smoothing':
                make = maker(kind, seed)
                inputs = {('ma', 'leaky'): pd.Series(y).rolling(5, center=True, min_periods=1).mean().values,
                          ('ma', 'honest'): pd.Series(y).rolling(5, min_periods=1).mean().values,
                          ('sg', 'leaky'): savgol_filter(y, 7, 2, mode='interp'),
                          ('sg', 'honest'): causal_savgol(y),
                          ('raw', 'reference'): y}
                for (method, arm), x in inputs.items():
                    add(h, kind, method, arm, targets, roll(x, y, targets, h, lags, make))
            elif mech == 'shuffle':
                rng = np.random.default_rng(1000 + seed)
                held = np.sort(rng.choice(targets, size=n_test // 2, replace=False))
                make = maker(kind, seed)
                X = lagmat(y, lags)
                rows = np.arange(lags - 1, n - h)  # every row with a known target
                train = np.setdiff1d(rows, held - h)
                m = make().fit(X[train], y[train + h])
                add(h, kind, 'random_split', 'leaky', held, m.predict(X[held - h]))
                p = roll(y, y, targets, h, lags, make)
                add(h, kind, 'random_split', 'honest', held, p[np.isin(targets, held)])
            else:  # tuning
                val_targets = np.arange(n - n_test - N_VAL, n - n_test)
                res = []
                for j, cfg in enumerate(GRID[kind]):
                    make = maker(kind, seed, cfg)
                    pv = roll(y, y, val_targets, h, cfg['lags'], make)
                    pt = roll(y, y, targets, h, cfg['lags'], make)
                    res.append((rmse(pv, y[val_targets]), rmse(pt, y[targets]), pt))
                    grid_rows.append({'dataset': name, 'horizon': h, 'seed': seed, 'learner': kind, 'config': j,
                                      'val_rmse': res[-1][0], 'test_rmse': res[-1][1]})
                best_test = int(np.argmin([r[1] for r in res]))
                best_val = int(np.argmin([r[0] for r in res]))
                add(h, kind, 'test_tuning', 'leaky', targets, res[best_test][2])
                add(h, kind, 'test_tuning', 'honest', targets, res[best_val][2])
    log(f'done {name:16s} {mech:10s} seed {seed}  {time.time() - t0:6.1f}s')
    return preds, grid_rows


def grid_curve(grid, sizes=(1, 2, 4, 8, 16, 32), draws=300):
    """Expected inflation when the best of k settings is picked on the test set instead of the validation set."""
    rng = np.random.default_rng(0)
    rows = []
    for key, g in grid.groupby(['dataset', 'horizon', 'seed', 'learner']):
        v, t = g['val_rmse'].values, g['test_rmse'].values
        for k in sizes:
            if k > len(g):
                continue
            infl = []
            for _ in range(draws):
                sub = rng.choice(len(g), size=k, replace=False)
                honest = t[sub[np.argmin(v[sub])]]
                infl.append(100 * (honest - t[sub].min()) / honest)
            rows.append(dict(zip(['dataset', 'horizon', 'seed', 'learner'], key), grid_size=k,
                             inflation_pct=float(np.mean(infl))))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, 'progress.log'), 'w').close()
    for d in D.REGISTRY:
        D.load(d)
    tasks = [(d, m, s) for m in MECHANISMS for s in R.SEEDS for d in D.REGISTRY]
    log(f'start: {len(tasks)} tasks')
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(run_task, tasks))
    pred = pd.DataFrame([r for p, _ in results for r in p])
    grid = pd.DataFrame([r for _, g in results for r in g])
    pred.to_csv(os.path.join(OUT, 'predictions.csv'), index=False)
    grid.to_csv(os.path.join(OUT, 'tuning_grid.csv'), index=False)
    curve = grid_curve(grid)
    curve.to_csv(os.path.join(OUT, 'tuning_grid_curve.csv'), index=False)

    keys = ['dataset', 'horizon', 'seed', 'learner', 'mechanism', 'method']
    met = pred.groupby(keys + ['arm']).apply(R.metrics, include_groups=False).reset_index()
    met.to_csv(os.path.join(OUT, 'metrics.csv'), index=False)

    rows = []
    for k, g in pred.sort_values('date').groupby(keys):
        arms = {a: gg for a, gg in g.groupby('arm')}
        if 'leaky' not in arms:
            continue
        el = (arms['leaky']['y'] - arms['leaky']['y_hat']).values
        eh = (arms['honest']['y'] - arms['honest']['y_hat']).values
        rl, rh = np.sqrt(np.mean(el ** 2)), np.sqrt(np.mean(eh ** 2))
        stat, p = R.dm_test(el, eh, k[1]) if np.any(el != eh) else (0.0, 1.0)
        rows.append(dict(zip(keys, k), RMSE_leaky=rl, RMSE_honest=rh, inflation_pct=100 * (rh - rl) / rh,
                         DM=stat, p=p))
    cmp_ = pd.DataFrame(rows)
    cmp_.to_csv(os.path.join(OUT, 'inflation_dm.csv'), index=False)

    pd.set_option('display.width', 200)
    g = cmp_.groupby(['mechanism', 'method', 'horizon'])
    print(pd.DataFrame({'n': g.size(), 'median_%': g.inflation_pct.median().round(1),
                        'min_%': g.inflation_pct.min().round(1), 'max_%': g.inflation_pct.max().round(1),
                        'leaky_sig': g.apply(lambda x: ((x.DM < 0) & (x.p < .05)).sum(), include_groups=False)}))
    print('\nTuning inflation by grid size (median over cases)\n',
          curve.groupby(['learner', 'grid_size']).inflation_pct.median().round(2).unstack(0))
    log('finished')


if __name__ == '__main__':
    main()
