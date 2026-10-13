# Search log

**Database:** OpenAlex API (https://api.openalex.org/works), field `title_and_abstract.search` (stemmed)
**Date run:** 2026-10-07
**Filters:** `publication_year:2018-2025,type:article,language:en,primary_location.source.type:journal`
**Script:** `scripts/audit/search_openalex.py` · **Shuffle seed:** 2026 (numpy default_rng, one generator per pool; pools sorted by OpenAlex ID before shuffling)

| | S1 general | S2 decomposition |
|---|---|---|
| Records reported by API | 23768 | 2227 |
| Records downloaded | 23778 | 2230 |
| Unique after deduplication (records in both pools assigned to S2) | 21329 | 2222 |

- Total downloaded: 26008; unique OpenAlex IDs: 23719; in both pools: 2221.
- Records with an OpenAlex abstract: S1 17787, S2 1361.

**S1:** `("time series" OR "time-series") AND (forecast OR prediction) AND ("deep learning" OR "machine learning" OR LSTM OR GRU OR transformer OR "neural network" OR hybrid)`

**S2:** `(forecast OR prediction) AND ("time series" OR "time-series") AND (wavelet OR "empirical mode decomposition" OR EEMD OR CEEMDAN OR "variational mode decomposition" OR VMD OR TQWT OR "singular spectrum analysis" OR "seasonal-trend decomposition") AND (hybrid OR LSTM OR "neural network" OR "machine learning" OR "deep learning")`
