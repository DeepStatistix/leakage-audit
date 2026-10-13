"""Experiment 1: how much does decomposition leakage inflate forecast accuracy?

For every dataset the same `n_test` final observations are forecast at horizons h = 1 and h = 3
(direct strategy: the forecast of y[t+h] uses data up to origin t; expanding window). Pipelines:

  persistence        y_hat = y[t]
  plain              lag-regression on the raw series (no decomposition)
  <m>_leaky          decompose the FULL series once, then fit/forecast each component from its lags and sum
                     (the procedure found in many published decomposition hybrids)
  <m>_causal         same, but at each origin the series is decomposed using y[0..t] only
  <m>_stepwise       best-practice causal design (stepwise decomposition; Fang et al. 2019, Zuo et al. 2020):
                     every sample i, training or test, is built from a decomposition of the sliding window
                     y[i-W+1..i] only; the last `lags` values of every component feed one model that predicts
                     y[i+h]. Training and test inputs therefore share the same boundary behaviour.
  atrous             causal-by-construction a trous Haar wavelet (applied once; no leakage)

with m in {dwt, ceemdan, tqwt, vmd}, two learners (ridge, mlp) and several seeds (CEEMDAN noise, MLP initialisation;
ridge with a deterministic decomposition does not depend on the seed and is run for the first seed only).
Leaky and causal arms differ only in what the decomposition may see, so their gap is the inflation caused
by decomposition leakage alone.

Run:  C:\\Python314\\python.exe scripts/experiments/run_decomposition_leakage.py [--methods tqwt,vmd]
      With --methods, only those decompositions are run and merged into the existing predictions.csv.
Outputs go to results/exp1_decomposition/.
"""
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets as D  # noqa: E402
import decompositions as DC  # noqa: E402

OUT = os.path.join(D.ROOT, 'results', 'exp1_decomposition')
REFIT = {'ridge': 1, 'mlp': 6}  # refit every k forecast origins
HORIZONS = (1, 3)
SEEDS = (0, 1, 2)
WINDOW = 120  # sliding window for the stepwise design
FAMILIES = ['ceemdan_causal', 'ceemdan_stepwise', 'ceemdan_leaky', 'vmd_causal', 'vmd_stepwise', 'vmd_leaky',
            'tqwt_causal', 'tqwt_stepwise', 'tqwt_leaky', 'dwt_stepwise', 'dwt_causal', 'dwt_leaky',
            'atrous', 'plain']  # slowest first for load balancing


def learner(kind, seed):
    if kind == 'ridge':
        base = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    else:
        base = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=2000,
                                                            random_state=seed))
    return TransformedTargetRegressor(regressor=base, transformer=StandardScaler())


def lag_xy(c, lags, end, h):
    """Training pairs from c[0..end]: inputs c[i-lags+1..i], target c[i+h], for i+h <= end."""
    win = np.lib.stride_tricks.sliding_window_view(c[:end + 1], lags + h)
    return win[:, :lags], win[:, -1]


def decompose(method, x, seed):
    if method == 'ceemdan':
        return DC.ceemdan(x, seed=seed)
    return DC.METHODS[method](x)


def forecast_components(get_comps, origins, lags, kind, h, seed):
    """Per-component direct h-step forecasts, summed. get_comps(t) returns components usable at origin t."""
    preds, models = [], None
    for i, t in enumerate(origins):
        comps = get_comps(t)
        if models is None or i % REFIT[kind] == 0 or len(models) != len(comps):
            models = [learner(kind, seed).fit(*lag_xy(c, lags, t, h)) for c in comps]
        preds.append(sum(float(m.predict(c[t - lags + 1:t + 1][None, :])[0]) for m, c in zip(models, comps)))
    return np.array(preds)


def stepwise_features(method, y, lags, seed):
    """Row i = last `lags` values of every component of the decomposition of y[i-WINDOW+1..i] (NaN before)."""
    n_comp = DC.N_COMP[method]  # e.g. CEEMDAN: up to 6 IMFs + residue; DWT: 4 details + approximation
    F = np.full((len(y), n_comp * lags), np.nan)
    for i in range(WINDOW - 1, len(y)):
        comps = decompose(method, y[i - WINDOW + 1:i + 1], seed)
        padded = np.zeros((n_comp, lags))
        padded[:len(comps) - 1] = comps[:-1, -lags:]
        padded[-1] = comps[-1, -lags:]  # residue / approximation always in the last slot
        F[i] = padded.ravel()
    return F


def forecast_stepwise(F, y, origins, kind, h, seed):
    preds, model = [], None
    first = WINDOW - 1
    for i, t in enumerate(origins):
        if model is None or i % REFIT[kind] == 0:
            rows = np.arange(first, t - h + 1)
            model = learner(kind, seed).fit(F[rows], y[rows + h])
        preds.append(float(model.predict(F[t][None, :])[0]))
    return np.array(preds)


def run_task(args):
    name, family, seed = args
    warnings.filterwarnings('ignore')
    s, lags, n_test = D.load(name)
    y = s.values
    n = len(y)
    targets = np.arange(n - n_test, n)
    seeded = family.startswith('ceemdan')
    kinds = list(REFIT) if (seed == SEEDS[0] or seeded) else ['mlp']
    out = {}
    t0 = time.time()
    if family == 'plain':
        for h in HORIZONS:
            if seed == SEEDS[0]:
                out[('persistence', '-', h)] = y[targets - h]
            for kind in kinds:
                out[('plain', kind, h)] = forecast_components(lambda t: [y], targets - h, lags, kind, h, seed)
    elif family == 'atrous':
        comps = DC.atrous_haar(y)
        for h in HORIZONS:
            for kind in kinds:
                out[('atrous', kind, h)] = forecast_components(lambda t: comps, targets - h, lags, kind, h, seed)
    else:
        method, arm = family.split('_')
        if arm == 'stepwise':
            F = stepwise_features(method, y, lags, seed)
            for h in HORIZONS:
                for kind in kinds:
                    out[(family, kind, h)] = forecast_stepwise(F, y, targets - h, kind, h, seed)
        else:
            if arm == 'leaky':
                full = decompose(method, y, seed)
                get = lambda t: full  # noqa: E731
            else:
                needed = sorted({t for h in HORIZONS for t in targets - h})
                cache = {t: decompose(method, y[:t + 1], seed) for t in needed}
                get = cache.__getitem__
            for h in HORIZONS:
                for kind in kinds:
                    out[(family, kind, h)] = forecast_components(get, targets - h, lags, kind, h, seed)
    rows = []
    for (pipe, kind, h), p in out.items():
        for tgt, yh in zip(targets, p):
            rows.append({'dataset': name, 'horizon': h, 'seed': seed, 'date': s.index[tgt], 'pipeline': pipe,
                         'learner': kind, 'y': y[tgt], 'y_hat': yh})
    print(f'  done {name:16s} {family:17s} seed {seed}  {time.time() - t0:6.1f}s', flush=True)
    return rows


def metrics(g):
    e = g['y'] - g['y_hat']
    sst = np.sum((g['y'] - g['y'].mean()) ** 2)
    return pd.Series({'RMSE': np.sqrt(np.mean(e ** 2)), 'MAE': np.mean(np.abs(e)),
                      'R2_NSE': 1 - np.sum(e ** 2) / sst, 'r': np.corrcoef(g['y'], g['y_hat'])[0, 1]})


def dm_test(e1, e2, h=1):
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold correction (squared-error loss).
    Negative statistic: forecast 1 has lower loss."""
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    gamma = [np.mean((d[k:] - d.mean()) * (d[:n - k] - d.mean())) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    stat = d.mean() / np.sqrt(var) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return stat, 2 * stats.t.sf(abs(stat), df=n - 1)


def comparisons(pred):
    """Leaky vs each honest design, and each honest design vs the no-decomposition model, per seed."""
    err = {k: (g['y'] - g['y_hat']).values
           for k, g in pred.sort_values('date').groupby(['dataset', 'horizon', 'seed', 'learner', 'pipeline'])}
    rows = []
    for (d, h, seed, kind, pipe), e in err.items():
        if not pipe.endswith('_leaky'):
            continue
        m = pipe.split('_')[0]
        base = err.get((d, h, seed, kind, 'plain'), err.get((d, h, SEEDS[0], kind, 'plain')))
        for honest in (f'{m}_causal', f'{m}_stepwise'):
            eh = err.get((d, h, seed, kind, honest), err.get((d, h, SEEDS[0], kind, honest)))
            rl, rc = np.sqrt(np.mean(e ** 2)), np.sqrt(np.mean(eh ** 2))
            s1, p1 = dm_test(e, eh, h)
            s2, p2 = dm_test(eh, base, h)
            rows.append({'dataset': d, 'horizon': h, 'seed': seed, 'learner': kind, 'decomposition': m,
                         'honest_design': honest.split('_')[1], 'RMSE_leaky': rl, 'RMSE_honest': rc,
                         'inflation_pct': 100 * (rc - rl) / rc,
                         'DM_leaky_vs_honest': s1, 'p_leaky_vs_honest': p1,
                         'DM_honest_vs_plain': s2, 'p_honest_vs_plain': p2})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    for d in D.REGISTRY:  # download/cache once before forking
        D.load(d)
    families = FAMILIES
    if '--methods' in sys.argv:
        methods = sys.argv[sys.argv.index('--methods') + 1].split(',')
        families = [f for f in FAMILIES if f.split('_')[0] in methods]
    tasks = [(d, f, s) for f in families for s in SEEDS for d in D.REGISTRY]
    with ProcessPoolExecutor(max_workers=8) as ex:
        rows = [r for res in ex.map(run_task, tasks) for r in res]
    pred = pd.DataFrame(rows)
    path = os.path.join(OUT, 'predictions.csv')
    if families != FAMILIES and os.path.exists(path):  # merge with earlier runs, replacing re-run pipelines
        old = pd.read_csv(path, parse_dates=['date'])
        pred = pd.concat([old[~old.pipeline.isin(pred.pipeline.unique())], pred], ignore_index=True)
    pred.to_csv(path, index=False)

    met = (pred.groupby(['dataset', 'horizon', 'seed', 'pipeline', 'learner'])
           .apply(metrics, include_groups=False).reset_index())
    pers = met[met.pipeline == 'persistence'].set_index(['dataset', 'horizon'])['RMSE']
    met['skill_vs_persistence'] = 1 - met['RMSE'] / pd.Series(list(zip(met.dataset, met.horizon))).map(pers).values
    met.to_csv(os.path.join(OUT, 'metrics.csv'), index=False)

    cmp_ = comparisons(pred)
    cmp_.to_csv(os.path.join(OUT, 'inflation_dm.csv'), index=False)

    pd.set_option('display.width', 220)
    print('\nMedian test R2 across seeds\n',
          met.groupby(['horizon', 'pipeline', 'learner'])['R2_NSE'].median().unstack([1]).round(3).to_string())
    print('\nInflation (%) of leaky over each honest design: median [min, max]\n',
          cmp_.groupby(['horizon', 'decomposition', 'honest_design'])['inflation_pct']
          .agg(['median', 'min', 'max']).round(1).to_string())


if __name__ == '__main__':
    main()
