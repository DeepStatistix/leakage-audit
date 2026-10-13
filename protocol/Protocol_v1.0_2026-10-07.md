# Study Protocol

## Are Time-Series Forecasts Really Out-of-Sample? A Systematic Audit of Information Leakage in Forecasting Research

**Version** 1.0 (draft for registration) · **Date** 7 October 2026
**Investigators** Aqib Gul (lead; coder 1) · [second coder, TBD]
**Registration** Open Science Framework (OSF Registries, "Generalized Systematic Review" template); the protocol is also frozen by a dated commit in the project repository.
**Status at registration:** search not yet run; no papers screened or coded. Pilot experiments on public data (Experiment 1) have been run to calibrate the experimental design. They do not involve the audit sample.

---

## 1. Background and rationale

Time-series forecasting studies are judged by their out-of-sample accuracy. A forecast is out-of-sample only if every step that produces it, including cleaning, imputation, scaling, smoothing, decomposition, feature construction, model selection and fitting, uses information available at the forecast origin. Several common practices quietly break this condition. Examples are decomposing the complete series before splitting it, scaling with full-sample statistics, using centred filters, and choosing models or hyperparameters on the test set. These problems have been documented for wavelet and singular-spectrum hybrids in hydrology (e.g. Du et al., 2017; Quilty & Adamowski, 2018), for decomposition-based prediction more generally, and for machine-learning-based science as a whole (Kaufman et al., 2012; Kapoor & Narayanan, 2023). Their prevalence in the forecasting literature has not been audited systematically.

Leakage is rarely visible from results alone, and published method descriptions are often too incomplete to establish whether a pipeline was causal. This study therefore audits the **verifiability** of out-of-sample claims, the **frequency** of described leakage-prone steps, and, through controlled experiments, the **size of accuracy inflation** these steps produce.

## 2. Research questions

**Main question.** To what extent do published time-series forecasting studies ensure that every piece of information used to generate a forecast would actually have been available at the forecast origin?

| | Sub-question |
|---|---|
| **RQ1** (verifiability) | What proportion of studies report enough information to verify that every pipeline step was causal? |
| **RQ2** (leakage-prone practice) | What proportion describe at least one step that uses information unavailable at the forecast origin, and which steps are most often affected? |
| **RQ3** (consequence) | By how much do such steps inflate reported accuracy under controlled conditions? |
| **RQ4** (correlates; exploratory) | Is leakage risk associated with model family (especially decomposition hybrids), application domain, publication year, journal quartile, or the level of reported accuracy? |

**Expected directions** (stated before data collection; exploratory, not confirmatory): (i) fewer than half of studies will be verifiably causal; (ii) leakage-prone steps will be more frequent in decomposition-hybrid studies than in the general pool; (iii) full-series decomposition will inflate accuracy materially relative to causal decomposition; (iv) higher reported accuracy will be associated with higher leakage risk.

## 3. Definitions

- **Forecast origin *t*:** the last time point whose information may be used to form a forecast.
- **Information set 𝓘ₜ:** all data that would be observable at time *t* in real time, including publication lags of predictors.
- **Out-of-sample forecast:** ŷₜ₊ₕ = f(𝓘ₜ), where every operation used to construct f and its inputs depends only on 𝓘ₜ.
- **Causal step:** a pipeline step whose output at time *t* depends only on 𝓘ₜ (e.g. scaler fitted on training data; trailing moving average; decomposition recomputed on data up to *t*).
- **Leakage:** use, anywhere in the pipeline, of information outside 𝓘ₜ.

**Leakage taxonomy.** Every coded issue is assigned to one of: (1) split; (2) preprocessing (scaling, imputation, smoothing); (3) feature engineering/selection; (4) decomposition; (5) hyperparameter/model selection; (6) target leakage; (7) horizon mismatch; (8) cross-series; (9) evaluation (repeated test-set use, selective reporting); (10) deployment infeasibility.

## 4. Eligibility criteria

**Include** peer-reviewed journal articles that:
1. are published 2018–2025 in English;
2. forecast a time-indexed variable (univariate or multivariate);
3. propose, or empirically compare, forecasting models; and
4. report quantitative out-of-sample accuracy (e.g. RMSE, MAE, MAPE, R², NSE, KGE) on held-out data; and
5. are **open access**: flagged `is_oa` by OpenAlex at the search date, with the full text retrievable free of charge (gold, hybrid, green or bronze). See the limitation in Section 10a.

**Application fields:** any. Leakage is a property of the forecasting pipeline, not of the field, so all fields are eligible (energy, hydrology/climate, finance/economics, agriculture, health, transport, air quality, industry, IT and others). The field is recorded as a descriptor. *(Change from v1, which was limited to four domains; made before any screening decisions were confirmed. The v1 search and screening files are archived in `paper/search/archive_domain_limited_v1/`.)*

**Exclude:** reviews and surveys; purely theoretical or simulation-only papers; classification, anomaly-detection, imputation or nowcasting-only tasks; conference papers, preprints, theses; papers whose full text cannot be obtained; retracted papers.

## 5. Information sources and search

- **Primary database: OpenAlex** (openalex.org), an open bibliographic index whose coverage of peer-reviewed journals is comparable to Scopus and Web of Science. Searches run through its public API on titles and abstracts (`title_and_abstract.search`, stemmed, so "forecast" also matches "forecasting"/"forecasts"), filtered to `publication_year:2018-2025`, `type:article`, `language:en`, `primary_location.source.type:journal`. The search script (`scripts/audit/search_openalex.py`) and the raw downloaded records are archived, so the search is fully reproducible.
- **Abstracts:** where OpenAlex has no abstract (common for some publishers because of licensing), it is retrieved from **Semantic Scholar** by DOI, for screened records only. Records with no abstract are screened on title, then on full text.
- **Journal quartile:** the SCImago Journal Rank (SJR) quartile for the publication year, from the free SCImago download.
- **Why not Scopus or Google Scholar:** there is no subscription access to Scopus. Google Scholar is unsuitable as a primary source for a systematic search, because results are not reproducible, ranking is opaque, at most about 1,000 records are returned per query, and full Boolean search on titles and abstracts is not supported.
- Search date, exact strings and hit counts are recorded and reported (`paper/search/search_log.md`).

**String S1, general pool** (OpenAlex `title_and_abstract.search`):
```
("time series" OR "time-series") AND (forecast OR prediction)
AND ("deep learning" OR "machine learning" OR LSTM OR GRU OR transformer OR "neural network" OR hybrid)
```
(No field terms: 23,768 records at the 7 October 2026 count, 15,381 of them open access.)

**String S2, decomposition stratum:**
```
(forecast OR prediction) AND ("time series" OR "time-series")
AND (wavelet OR "empirical mode decomposition" OR EEMD OR CEEMDAN OR "variational mode decomposition" OR VMD
     OR TQWT OR "singular spectrum analysis" OR "seasonal-trend decomposition")
AND (hybrid OR LSTM OR "neural network" OR "machine learning" OR "deep learning")
```

The bare acronyms EMD, SSA and STL are omitted because they are ambiguous outside this context; the spelled-out forms are included. Strings may be refined once, after a pilot of 50 records, to fix precision problems. Any change is logged with the reason.

## 6. Screening and sampling plan

1. Download S1 and S2 records from OpenAlex (title, abstract, DOI, journal, year, open-access status) and deduplicate by OpenAlex ID, DOI and normalised title. Records found by both strings are assigned to S2.
2. **Random ordering:** shuffle each pool with a fixed seed (`numpy.random.default_rng(2026)`). The shuffled lists are archived.
3. **Sequential screening:** screen records in shuffled order, first on title and abstract, then on full text, until the target number of eligible papers is reached:
   - **General pool (S1): 100 eligible papers**
   - **Decomposition stratum (S2): 40 eligible papers**
4. Every screened record and its exclusion reason is logged (sheet `Screening_Log`), which gives the PRISMA flow counts.
   Only open-access records enter screening. Non-open-access records are removed **after** shuffling, so the remaining order is still a random permutation of the open-access population (`scripts/audit/prepare_screening.py`). A record flagged open access whose full text cannot actually be retrieved is excluded with reason "Full text unavailable".
5. Domain balance is not forced. Domain is recorded and reported, and if one domain has fewer than 10 papers, results for it are pooled.

*Rationale:* screening in random order until a quota is met gives a simple random sample of eligible papers without having to screen the full result set. With n = 100, a proportion near 50% is estimated within about ±10 percentage points (95% CI). The S2 stratum is enriched on purpose and is analysed separately. It is never pooled into prevalence estimates for the general literature.

## 7. Data extraction and coding

Coding uses `audit_form.xlsx` (sheets: `Codebook`, `Audit`, `Screening_Log`, `Level_Rules`). Coders use the main text, appendices, supplementary material and any linked code or data.

### 7.1 Study descriptors
Bibliographic data; domain; target variable; frequency; series length; forecast horizon(s); model family (statistical / ML / DL / decomposition hybrid / other); decomposition method; proposed model; number of benchmarks; journal SJR quartile; reported best accuracy (R²/NSE, or the reported metric).

### 7.2 Core audit items

Each item is coded **C** (causal, explicitly reported), **N** (non-causal, explicitly reported or unambiguously implied), **NR** (not reported or ambiguous), or **NA** (step not used).

| Item | Construct | **C** if… | **N** if… |
|---|---|---|---|
| A1 | Data split | Chronological split with dates, proportions or sizes stated | Random/shuffled split or shuffled k-fold on the time series |
| A2 | Validation / model selection | Separate validation block, rolling origin, or blocked CV before the test period | Model or architecture chosen by test-set performance |
| A3 | Scaling / imputation | Fitted on training data only (or per origin) | Fitted on the full series / before splitting |
| A4 | Decomposition / smoothing / filtering | Recomputed using data up to each origin, applied to the training period only, or causal by construction (e.g. trailing filter, à trous) | Applied to the full series before splitting, or centred filters used |
| A5 | Inputs and horizon | Lags/windows use only past values and the horizon is stated and consistent with the inputs | Inputs include values at or beyond the target time, or a horizon mismatch |
| A6 | Hyperparameter tuning | Tuned on training/validation data only | Tuned on test data, or "best" settings selected by test error |
| A7 | Naive baseline | Persistence, seasonal-naive or climatology benchmark reported | No naive benchmark (coded **NR**) |
| A8 | Statistical comparison | DM test, intervals, or multiple-origin results reported | Claimed but not reported (coded **N**) |
| A9 | Reproducibility | Code **and** data available (repository or supplement) | Neither available, or "on request" (coded **N**) |
| A10 | Metric consistency | Reported metrics internally consistent and correctly labelled | Inconsistency found (see 7.4) |

Ambiguous or "standard" wording (e.g. "data were normalised") without saying which data was used is coded **NR**, never **C**. When in doubt between N and NR, code **NR** and record the passage in `Notes`.

### 7.3 Leakage-risk level (decision rule, computed from A1–A6)

| Level | Label | Rule |
|---|---|---|
| **0** | Verifiably causal | Every applicable item in A1–A6 is **C** |
| **1** | Probably causal | No **N**; 1–2 applicable items **NR** |
| **2** | Indeterminate | No **N**; ≥ 3 applicable items **NR** |
| **3** | Leakage-prone | ≥ 1 item in A1–A6 coded **N** |
| **4** | Demonstrated | Leakage confirmed by inspecting released code or by reproduction (case studies only) |

The level is computed automatically in the spreadsheet. Level 4 is assigned only by documented verification.

### 7.4 Forensic red flags (recorded separately, not part of the level)
- **F1 Metric inconsistency:** e.g. a metric reported under a different name, identical values for different metrics (KGE = R²), RMSE < MAE, or R² outside the possible range.
- **F2 Implausible skill:** for papers using public data, the persistence RMSE is recomputed on the stated test period. A flag is raised if the reported improvement over persistence is far beyond what the controlled experiments (Section 8) produce for causal pipelines. It is also raised, whatever the data source, when accuracy barely decays with horizon, e.g. R² falling by less than about 0.1 from 1-step to 3-step forecasts on a series where persistence skill falls sharply. This is the signature of leakage seen in Experiment 1.
- **F3 Claim–evidence gap:** a validation procedure stated in the methods or a flowchart (e.g. rolling origin, DM test) is absent from the results.

### 7.5 Reliability
Coder 1 codes all papers. Coder 2 independently codes a random 20% subsample (about 28 papers), drawn with seed 2027 after coder 1 finishes the pilot. Agreement on A1–A6 and on the level is summarised with Cohen's κ (unweighted for items, linearly weighted for level). If κ < 0.6 for any item, its codebook definition is revised, and that item is re-coded for all papers. Disagreements in the subsample are resolved by discussion, and the resolved codes are used.

**Pilot:** before formal coding, both coders code the same 10 papers that are not in the sample, and the codebook is refined. Pilot papers are excluded from the analysis.

### 7.6 Use of AI tools
Large language models may be used (i) to suggest title and abstract screening decisions, with a stated reason, and (ii) to find relevant passages in full texts (e.g. where scaling is described). Every screening decision and every audit code is confirmed or corrected by a human coder against the original record or text. Agreement between the AI suggestions and the final human decisions is reported.

## 8. Controlled experiments (RQ3)

Public datasets with different behaviour are used: SPEI-6 and SPEI-12 (two stations, computed from NASA POWER), monthly sunspots, a monthly global wheat price (FRED), and the ETTh1 benchmark (daily). One-step-ahead forecasts are produced at 120 rolling forecast origins with expanding windows. For each mechanism, the leaky and causal versions are identical except for the information the step is allowed to see:

| Experiment | Mechanism | Leaky | Causal |
|---|---|---|---|
| 1 | Decomposition (DWT-MRA, CEEMDAN) + ridge/MLP per component | Full-series decomposition | Re-decomposition on data up to each origin; plus a causal-by-construction à trous wavelet |
| 2 | Scaling | Full-series min-max / z-score | Training-window statistics |
| 3 | Smoothing | Centred moving average | Trailing moving average |
| 4 | Model selection | Hyperparameters chosen by test RMSE | Chosen on a validation block |

**Outcomes:** RMSE, MAE, R²/NSE, correlation, skill relative to persistence. **Inflation** = (RMSE_causal − RMSE_leaky) / RMSE_causal × 100. Diebold–Mariano tests (Harvey–Leybourne–Newbold correction, squared loss) compare leaky vs causal, and causal vs a no-decomposition baseline. Robustness checks: additional seeds and a longer horizon (h = 3).

## 9. Analysis plan

1. PRISMA 2020 flow diagram; sample description by domain, year, model family and journal quartile.
2. **RQ1:** proportion at Level 0, with 95% Wilson CI, reported separately for the general pool and the S2 stratum.
3. **RQ2:** proportion at Level 3, with 95% CI; item-level distribution of C/N/NR/NA (A1–A10); taxonomy counts.
4. **RQ3:** inflation by mechanism × dataset × learner; DM results.
5. **RQ4 (exploratory):** Fisher's exact / χ² tests and one logistic regression of Level 3 (yes/no) on decomposition use, domain, year and journal quartile. Reported accuracy (R²/NSE where available) is compared across levels with a Kruskal–Wallis test. Results are labelled exploratory, with no multiplicity claims.
6. Forensic flags F1–F3: frequencies and illustrative anonymised examples.
7. Inter-coder agreement (κ).

**Synthesis.** Combine audit prevalence (RQ1–RQ2) with experimental inflation (RQ3) to state how much published accuracy claims could be overstated wherever leakage-prone steps are present.

## 10. Reporting and ethics

- Results are reported in aggregate. Paper-level codes are given in the supplementary data file with neutral wording ("not reported", "non-causal as described"). The main text names no individual papers negatively. Illustrative examples are paraphrased and anonymised, unless code-based verification (Level 4) makes attribution appropriate.
- Codes describe what is *reported*, not author intent. "NR" does not mean leakage.
- No human participants; no ethics approval required.

## 10a. Limitation: open-access sample

Without subscription access to most publishers, the audit is restricted to open-access articles. Its findings therefore describe the **open-access forecasting literature** (69% of S1 records and 55% of S2 records at the search date), not all publications. The restriction may bias results in either direction:
- open-access venues are over-represented (e.g. MDPI, Frontiers, Hindawi, Scientific Reports), and these may differ in reporting standards and review practices;
- authors who publish open access may be more inclined to share code and data, which would make our reproducibility estimates (A9) optimistic.

**Mitigation and reporting:**
1. The open-access share and the publisher mix of the eligible sample are reported.
2. Open-access and non-open-access records are compared on the metadata available for all records (year, domain, journal quartile, publisher, and decomposition use from title and abstract) to show whether the populations differ.
3. All prevalence estimates are labelled as estimates for open-access literature.
4. The controlled experiments (RQ3) do not depend on the sample and are unaffected.
5. If full-text access becomes available, a random sample of non-open-access papers can be added as a sensitivity analysis.

## 11. Deviations, data and code sharing

Deviations from this protocol will be listed with reasons in the final paper. The screening log, coded audit sheet, experiment code and outputs will be released under an open licence (OSF and GitHub) on submission.

## 12. Timeline

Weeks 1–3 search, screening and pilot · weeks 2–6 experiments · weeks 3–8 coding · weeks 7–8 second coder and κ · weeks 9–10 analysis · weeks 10–14 writing and submission.

## References (to be verified and completed)

- Du, K., Zhao, Y., & Lei, J. (2017). The incorrect usage of singular spectral analysis and discrete wavelet transform in hybrid models to predict hydrological time series. *Journal of Hydrology*, 552, 44–51.
- Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804.
- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM TKDD*, 6(4), 1–21.
- Quilty, J., & Adamowski, J. (2018). Addressing the incorrect usage of wavelet-based hydrological and water resources forecasting models for real-world applications with best practices and a new forecasting framework. *Journal of Hydrology*, 563, 336–353.
- Renaud, O., Starck, J.-L., & Murtagh, F. (2005). Wavelet-based combined signal filtering and prediction. *IEEE Trans. SMC-B*, 35(6), 1241–1251.
- Vicente-Serrano, S. M., Beguería, S., & López-Moreno, J. I. (2010). A multiscalar drought index sensitive to global warming: the SPEI. *Journal of Climate*, 23(7), 1696–1718.
- Page, M. J., et al. (2021). The PRISMA 2020 statement. *BMJ*, 372, n71.
