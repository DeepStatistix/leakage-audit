# Are time-series forecasts really out-of-sample? Data and code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23346658.svg)](https://doi.org/10.5281/zenodo.23346658)

This repository contains the data and code needed to reproduce the results of

> Gul, A. (2026). Are time-series forecasts really out-of-sample? A systematic audit of information leakage in
> forecasting research. *Manuscript submitted for publication.*

It has two parts: a **systematic audit** of 140 randomly sampled open-access forecasting studies (2018–2025), and
**controlled experiments** that measure how much each kind of leakage inflates reported accuracy.

## Contents

| Folder | What it holds |
|---|---|
| `protocol/` | The audit protocol as frozen before screening (v1.0, 7 October 2026), the final version, and the time-stamped amendment history |
| `data/search/` | Search log, screening rules, the random order of all open-access records for each pool, and the title/abstract screening decisions (abstracts are not redistributed; use the OpenAlex ID or DOI) |
| `data/audit/audit_data.csv` | One row per full text assessed: bibliographic data, SJR quartile, eligibility, codes for items A1–A10 with the supporting evidence (quotation and page), flags F1–F3, and the leakage-risk level |
| `data/reliability/` | The two independent codings of the sealed subsample, the codebook-conformance corrections, the adjudication of every disagreement, and the agreement statistics |
| `data/verification/` | For the remaining studies: the first-pass codes and every change made during the full-text verification |
| `scripts/search/` | OpenAlex search and random ordering |
| `scripts/analysis/` | Audit analysis (prevalence, confidence intervals, logistic model, figures), agreement statistics, PRISMA diagram |
| `scripts/experiments/` | Experiment 1 (decomposition leakage; DWT, CEEMDAN, TQWT, VMD; ridge, MLP and LSTM) and Experiment 2 (preprocessing, smoothing, imputation and tuning leakage) |
| `results/` | All tables and figures of the paper; experiment metrics, inflation estimates with Diebold–Mariano tests, and test-period predictions (`.csv.gz`) |

Full texts of the audited studies are not included, for copyright reasons; every study is identified by DOI and
OpenAlex ID, and every code is supported by a quotation with its page number.

### Codes

Each item is coded **C** (compliant, the paper shows the step is causal), **N** (non-compliant, the paper shows a
leakage-prone step), **NR** (not reported, the step cannot be verified) or **NA** (not applicable). The level rule
(Methods, Eqs. 10–11) maps items A1–A6 to a level: 0 verifiably causal, 1 probably causal, 2 indeterminate, 3
leakage-prone (3a high, 3b low severity). The full codebook is in `protocol/Protocol_final.md`.

`in_sample` marks the studies inside the pre-specified quotas (100 in S1, 40 in S2). Five full texts assessed after
the quota was reached are kept for transparency and excluded from all analyses.

## Reproducing the results

Python 3.12 or later. Results in the paper were produced with Python 3.14.3 on Windows 11 and the versions in
`requirements.txt`.

```
pip install -r requirements.txt

# audit (seconds)
python scripts/analysis/analysis.py          # results/audit/: tables, figures, results.md
python scripts/analysis/reliability.py       # results/reliability/kappa_summary.csv
python scripts/analysis/prisma_diagram.py    # results/audit/fig_prisma.*

# experiments (hours on a CPU; series are downloaded once into data/leakage/)
python scripts/experiments/run_decomposition_leakage.py
python scripts/experiments/run_lstm_check.py
python scripts/experiments/run_preprocessing_leakage.py
python scripts/experiments/plot_decomposition_leakage.py
python scripts/experiments/plot_preprocessing_leakage.py
```

The experiments use seeds 0, 1 and 2, which fix the CEEMDAN noise and the network initialisation. The search
(`scripts/search/search_openalex.py`) is included for completeness; OpenAlex changes over time, so re-running it
will not return exactly the same records. The records actually retrieved are in `data/search/`.

### Public series used in the experiments

- SPEI-6 and SPEI-12 at Srinagar and Durban, computed from NASA POWER monthly precipitation and temperature
- Monthly sunspot numbers, 1900–1983
- Global wheat price, FRED series PWHEAMTUSDM
- ETTh1 oil temperature (Zhou et al., 2021), daily mean

## Licence

Code: MIT (`LICENSE`). Data and results: CC BY 4.0 (`LICENSE-DATA`). Bibliographic records come from OpenAlex
(CC0). Quotations from the audited studies are short extracts used as evidence for scholarly review; copyright
remains with their authors and publishers.

## Citation

Please cite the paper and this repository: Gul, A. (2026). *Are time-series forecasts really out-of-sample? Data and
code* (Version 1.0.0) [Data set and software]. Zenodo. https://doi.org/10.5281/zenodo.23346658
