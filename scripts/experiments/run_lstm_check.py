"""Experiment 1b: does decomposition leakage persist with an LSTM learner?

Same datasets, test targets, horizons (1, 3), seeds and decompositions as run_decomposition_leakage.py.
Pipelines (all LSTM):

  plain                LSTM on lags of the raw series
  <m>_leaky            one LSTM per component of the FULL-series decomposition, forecasts summed
                       (the typical published decomposition-LSTM hybrid)
  <m>_leakymulti       one LSTM whose input is the 12 x K sequence of full-series components
  <m>_stepwise         identical architecture to leakymulti, but every sample's components come from a
                       decomposition of the sliding window y[i-119..i] only (best-practice causal)

leakymulti vs stepwise differ ONLY in what the decomposition may see, so their gap isolates leakage with
the architecture held fixed.

Run:  C:\\Python314\\python.exe -u scripts/experiments/run_lstm_check.py [--methods tqwt,vmd]
      With --methods, only those decompositions are run and merged into the existing predictions_lstm.csv.
Outputs: results/exp1_decomposition/lstm/   Progress: results/exp1_decomposition/lstm/progress.log
"""
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets as D  # noqa: E402
import run_decomposition_leakage as R  # noqa: E402

OUT = os.path.join(R.OUT, 'lstm')
CACHE = os.path.join(D.CACHE, 'decomp_cache')
REFIT = 12
FAMILIES = ['ceemdan_stepwise', 'ceemdan_leaky', 'vmd_leaky', 'tqwt_leaky', 'dwt_leaky', 'vmd_stepwise',
            'ceemdan_leakymulti', 'vmd_leakymulti', 'tqwt_leakymulti', 'tqwt_stepwise', 'dwt_stepwise',
            'dwt_leakymulti', 'plain']


def log(msg):
    with open(os.path.join(OUT, 'progress.log'), 'a', encoding='utf-8') as f:
        f.write(time.strftime('%H:%M:%S ') + msg + '\n')


def fit_predict_lstm(X, y, X_new, seed, hidden=32, lr=5e-3, batch=64, max_epochs=300, patience=20):
    """X: (n, T, C) sequences. Chronological last 15% of rows is the early-stopping validation block."""
    import torch
    from torch import nn
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    mu, sd = X.reshape(-1, X.shape[2]).mean(0), X.reshape(-1, X.shape[2]).std(0) + 1e-8
    ym, ys = y.mean(), y.std() + 1e-8
    Xs = torch.tensor((X - mu) / sd, dtype=torch.float32)
    ysd = torch.tensor((y - ym) / ys, dtype=torch.float32)
    n_val = max(10, int(0.15 * len(y)))
    Xtr, ytr, Xva, yva = Xs[:-n_val], ysd[:-n_val], Xs[-n_val:], ysd[-n_val:]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(X.shape[2], hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            return self.head(self.lstm(x)[0][:, -1]).squeeze(-1)

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    best, best_state, wait = np.inf, None, 0
    for _ in range(max_epochs):
        net.train()
        for idx in torch.randperm(len(ytr), generator=g).split(batch):
            opt.zero_grad()
            loss_fn(net(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_fn(net(Xva), yva).item()
        if v < best - 1e-5:
            best, wait = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        p = net(torch.tensor((X_new - mu) / sd, dtype=torch.float32)).numpy()
    return p * ys + ym


def seq_from_series(c, lags, end, h):
    win = np.lib.stride_tricks.sliding_window_view(c[:end + 1], lags + h)
    return win[:, :lags, None], win[:, -1]


def forecast_per_component(comps, origins, lags, h, seed):
    """Fit one LSTM per component every REFIT origins; forecast each origin from the latest models."""
    preds = np.zeros(len(origins))
    for start in range(0, len(origins), REFIT):
        block = origins[start:start + REFIT]
        t0 = block[0]
        for k, c in enumerate(comps):
            X, y = seq_from_series(c, lags, t0, h)
            X_new = np.stack([c[t - lags + 1:t + 1] for t in block])[:, :, None]
            preds[start:start + len(block)] += fit_predict_lstm(X, y, X_new, seed + 101 * k)
    return preds


def forecast_multichannel(S, y, origins, h, seed, first):
    """S[i] = (lags, K) input sequence available at time i (NaN rows before `first`); target y[i+h]."""
    preds = np.zeros(len(origins))
    for start in range(0, len(origins), REFIT):
        block = origins[start:start + REFIT]
        rows = np.arange(first, block[0] - h + 1)
        preds[start:start + len(block)] = fit_predict_lstm(S[rows], y[rows + h], S[block], seed)
    return preds


def cached(name, fn):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name + '.npy')
    if os.path.exists(path):
        return np.load(path)
    arr = fn()
    np.save(path, arr)
    return arr


def run_task(args):
    name, family, seed = args
    warnings.filterwarnings('ignore')
    s, lags, n_test = D.load(name)
    y = s.values
    n = len(y)
    targets = np.arange(n - n_test, n)
    t0 = time.time()
    out = {}
    if family == 'plain':
        for h in R.HORIZONS:
            out[('plain', h)] = forecast_per_component([y], targets - h, lags, h, seed)
    else:
        method, arm = family.split('_')
        dseed = seed if method == 'ceemdan' else 0
        if arm == 'stepwise':
            F = cached(f'{name}_{method}_stepwise_s{dseed}',
                       lambda: R.stepwise_features(method, y, lags, dseed))
            n_comp = F.shape[1] // lags
            S = F.reshape(n, n_comp, lags).transpose(0, 2, 1)
            first = R.WINDOW - 1
        else:
            full = cached(f'{name}_{method}_full_s{dseed}', lambda: R.decompose(method, y, dseed))
        for h in R.HORIZONS:
            if arm == 'leaky':
                out[(family, h)] = forecast_per_component(full, targets - h, lags, h, seed)
            elif arm == 'leakymulti':
                S_full = np.full((n, lags, len(full)), np.nan)
                for i in range(lags - 1, n):
                    S_full[i] = full[:, i - lags + 1:i + 1].T
                out[(family, h)] = forecast_multichannel(S_full, y, targets - h, h, seed, lags - 1)
            else:
                out[(family, h)] = forecast_multichannel(S, y, targets - h, h, seed, first)
    rows = [{'dataset': name, 'horizon': h, 'seed': seed, 'date': s.index[tg], 'pipeline': pipe,
             'learner': 'lstm', 'y': y[tg], 'y_hat': yh}
            for (pipe, h), p in out.items() for tg, yh in zip(targets, p)]
    log(f'done {name:16s} {family:19s} seed {seed}  {time.time() - t0:6.1f}s')
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, 'progress.log'), 'w').close()
    for d in D.REGISTRY:
        D.load(d)
    families = FAMILIES
    if '--methods' in sys.argv:
        methods = sys.argv[sys.argv.index('--methods') + 1].split(',')
        families = [f for f in FAMILIES if f.split('_')[0] in methods]
    tasks = [(d, f, s) for f in families for s in R.SEEDS for d in D.REGISTRY]
    log(f'start: {len(tasks)} tasks')
    with ProcessPoolExecutor(max_workers=8) as ex:
        rows = [r for res in ex.map(run_task, tasks) for r in res]
    pred = pd.DataFrame(rows)
    path = os.path.join(OUT, 'predictions_lstm.csv')
    if families != FAMILIES and os.path.exists(path):  # merge with earlier runs, replacing re-run pipelines
        old = pd.read_csv(path, parse_dates=['date'])
        pred = pd.concat([old[~old.pipeline.isin(pred.pipeline.unique())], pred], ignore_index=True)
    pred.to_csv(path, index=False)

    met = (pred.groupby(['dataset', 'horizon', 'seed', 'pipeline'])
           .apply(R.metrics, include_groups=False).reset_index())
    base = pd.read_csv(os.path.join(R.OUT, 'metrics.csv'))
    pers = base[base.pipeline == 'persistence'].set_index(['dataset', 'horizon'])['RMSE']
    met['skill_vs_persistence'] = 1 - met['RMSE'] / pd.Series(list(zip(met.dataset, met.horizon))).map(pers).values
    met.to_csv(os.path.join(OUT, 'metrics_lstm.csv'), index=False)

    err = {k: (g['y'] - g['y_hat']).values
           for k, g in pred.sort_values('date').groupby(['dataset', 'horizon', 'seed', 'pipeline'])}
    rows = []
    for (d, h, seed, pipe), e in err.items():
        if pipe == 'plain' or pipe.endswith('_stepwise'):
            continue
        m = pipe.split('_')[0]
        eh, ep = err[(d, h, seed, f'{m}_stepwise')], err[(d, h, seed, 'plain')]
        rl, rc = np.sqrt(np.mean(e ** 2)), np.sqrt(np.mean(eh ** 2))
        s1, p1 = R.dm_test(e, eh, h)
        s2, p2 = R.dm_test(eh, ep, h)
        rows.append({'dataset': d, 'horizon': h, 'seed': seed, 'decomposition': m, 'leaky_design': pipe.split('_')[1],
                     'RMSE_leaky': rl, 'RMSE_stepwise': rc, 'inflation_pct': 100 * (rc - rl) / rc,
                     'DM_leaky_vs_stepwise': s1, 'p_leaky_vs_stepwise': p1,
                     'DM_stepwise_vs_plain': s2, 'p_stepwise_vs_plain': p2})
    cmp_ = pd.DataFrame(rows)
    cmp_.to_csv(os.path.join(OUT, 'inflation_dm_lstm.csv'), index=False)

    pd.set_option('display.width', 200)
    print('\nMedian test R2 (LSTM)\n', met.groupby(['horizon', 'pipeline'])['R2_NSE'].median().unstack(0).round(3))
    print('\nInflation vs stepwise (%)\n', cmp_.groupby(['horizon', 'decomposition', 'leaky_design'])['inflation_pct']
          .agg(['median', 'min', 'max']).round(1))
    log('finished')


if __name__ == '__main__':
    main()
