"""Public datasets for the leakage experiments (downloaded once, cached in data/leakage/).

Series
  spei6_srinagar, spei12_srinagar   SPEI from NASA POWER monthly P and T (Srinagar, 1981-2024)
  spei6_durban,  spei12_durban      same, Mount Edgecombe / Durban, KwaZulu-Natal
  sunspots                          monthly sunspot numbers, 1900-1983
  wheat_price                       FRED PWHEAMTUSDM, global wheat price, monthly, 1992-
  etth1_ot                          ETTh1 oil temperature (common deep-learning benchmark), daily mean

SPEI follows Vicente-Serrano et al. (2010): Thornthwaite PET, k-month water balance, three-parameter
log-logistic fit per calendar month on the full record. Like SPEIbase, the index is treated as given
data; the experiments only vary what the forecasting pipeline does with it.
"""
import io
import os

import numpy as np
import pandas as pd
import requests
from scipy import special, stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CACHE = os.path.join(ROOT, 'data', 'leakage')

STATIONS = {'srinagar': (34.08, 74.80), 'durban': (-29.70, 31.04)}


def _get(url, name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(path, 'w', encoding='utf-8') as f:
            f.write(r.text)
    with open(path, encoding='utf-8') as f:
        return f.read()


def nasa_power(station):
    lat, lon = STATIONS[station]
    url = ('https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=PRECTOTCORR,T2M'
           f'&community=AG&longitude={lon}&latitude={lat}&start=1981&end=2024&format=CSV')
    text = _get(url, f'power_{station}.csv')
    body = text.split('-END HEADER-')[1].strip()
    df = pd.read_csv(io.StringIO(body))
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    long = df.melt(id_vars=['PARAMETER', 'YEAR'], value_vars=months, var_name='mon')
    long['month'] = long['mon'].map({m: i + 1 for i, m in enumerate(months)})
    wide = long.pivot_table(index=['YEAR', 'month'], columns='PARAMETER', values='value').sort_index()
    idx = pd.to_datetime({'year': wide.index.get_level_values(0), 'month': wide.index.get_level_values(1), 'day': 1})
    out = pd.DataFrame({'precip_mm_day': wide['PRECTOTCORR'].values, 't2m': wide['T2M'].values}, index=idx)
    return out.replace(-999, np.nan)


def thornthwaite(t, lat, index):
    """Monthly Thornthwaite PET (mm/month) with daylength and month-length correction."""
    t = np.asarray(t, float)
    clim = pd.Series(t, index=index).groupby(index.month).mean()
    heat = np.sum((np.clip(clim.values, 0, None) / 5.0) ** 1.514)
    a = 6.75e-7 * heat ** 3 - 7.71e-5 * heat ** 2 + 1.792e-2 * heat + 0.49239
    doy = index.dayofyear.values + index.days_in_month.values / 2.0
    decl = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(np.radians(lat)) * np.tan(decl), -1, 1))
    daylen = 24 / np.pi * ws
    pet = 16 * (daylen / 12) * (index.days_in_month.values / 30) * (10 * np.clip(t, 0, None) / heat) ** a
    return np.where(t > 0, pet, 0.0)


def loglogistic_cdf(x, sample):
    """Three-parameter log-logistic fitted by probability-weighted moments (Vicente-Serrano et al., 2010)."""
    s = np.sort(sample)
    n = len(s)
    f = (np.arange(1, n + 1) - 0.35) / n
    w0, w1, w2 = (np.mean((1 - f) ** k * s) for k in range(3))
    beta = (2 * w1 - w0) / (6 * w1 - w0 - 6 * w2)
    g = special.gamma(1 + 1 / beta) * special.gamma(1 - 1 / beta)
    alpha = (w0 - 2 * w1) * beta / g
    gamma = w0 - alpha * g
    z = np.clip(x - gamma, 1e-9, None)
    return 1 / (1 + (alpha / z) ** beta)


def spei(balance, scale, index):
    """Standardise the k-month water balance with a per-calendar-month log-logistic fit."""
    d = pd.Series(balance, index=index).rolling(scale).sum()
    out = pd.Series(np.nan, index=index)
    for m in range(1, 13):
        x = d[(d.index.month == m) & d.notna()]
        with np.errstate(invalid='ignore'):
            p = loglogistic_cdf(x.values, x.values)
        if not np.all(np.isfinite(p)):
            # near-symmetric or left-skewed months break the log-logistic fit; use a normal fit instead
            p = stats.norm.cdf(x.values, x.mean(), x.std(ddof=1))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        out[x.index] = stats.norm.ppf(p)
    return out


def spei_series(station, scale):
    df = nasa_power(station)
    p = df['precip_mm_day'] * df.index.days_in_month
    pet = thornthwaite(df['t2m'].values, STATIONS[station][0], df.index)
    return spei((p - pet).values, scale, df.index).dropna()


def sunspots():
    text = _get('https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv',
                'monthly_sunspots.csv')
    s = pd.read_csv(io.StringIO(text), index_col=0).iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    return s['1900':]


def wheat_price():
    text = _get('https://fred.stlouisfed.org/graph/fredgraph.csv?id=PWHEAMTUSDM', 'fred_wheat.csv')
    s = pd.read_csv(io.StringIO(text), index_col=0).iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    return pd.to_numeric(s, errors='coerce').dropna()


def etth1_ot():
    text = _get('https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv', 'ETTh1.csv')
    df = pd.read_csv(io.StringIO(text), parse_dates=['date'], index_col='date')
    return df['OT'].resample('D').mean().dropna()


# name -> (loader, number of lags, test length)
REGISTRY = {
    'spei6_srinagar': (lambda: spei_series('srinagar', 6), 12, 120),
    'spei12_srinagar': (lambda: spei_series('srinagar', 12), 12, 120),
    'spei6_durban': (lambda: spei_series('durban', 6), 12, 120),
    'spei12_durban': (lambda: spei_series('durban', 12), 12, 120),
    'sunspots': (sunspots, 12, 120),
    'wheat_price': (wheat_price, 12, 120),
    'etth1_ot': (etth1_ot, 14, 120),
}


def load(name):
    loader, lags, n_test = REGISTRY[name]
    s = loader().astype(float)
    return s, lags, n_test


if __name__ == '__main__':
    for name in REGISTRY:
        s, lags, n_test = load(name)
        print(f'{name:16s} n={len(s):5d}  {s.index[0].date()} to {s.index[-1].date()}  '
              f'mean={s.mean():8.3f} sd={s.std():8.3f}  lag-1 r={s.autocorr(1):.3f}')
