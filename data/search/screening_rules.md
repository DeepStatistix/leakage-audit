# Title/abstract screening rules (v2, 7 October 2026: all application fields)

Applies protocol Section 4 to the shuffled open-access records. **Any application field is eligible.** The question at this stage is: *does this paper forecast future values of a time series and evaluate those forecasts?* When in doubt, mark **Unsure**; the full text decides.

**Include** a journal article that (i) forecasts or predicts a **future** value of a time-indexed variable (any field: energy, hydrology, finance, health, transport, air quality, IT, geotechnics, industry and so on), (ii) proposes or compares forecasting models, and (iii) evaluates the forecasts (error/accuracy metrics, "outperforms", "compared"). At this stage, out-of-sample evaluation is assumed if accuracy is reported; the full text confirms it.

**Exclude reasons** (drop-down list):
- **Not forecasting:** descriptive or causal analysis, nowcasting/estimation of the current value, gap filling, spatial interpolation, downscaling, soft sensors, or regression of an outcome on same-time inputs with no lead time.
- **Not time series**
- **No out-of-sample accuracy**
- **Review/survey**
- **Theory/simulation only**
- **Classification/anomaly task:** fault diagnosis, event or anomaly detection, up/down direction classification with no forecast of values.
- **Not journal article:** editorial, erratum, special-issue introduction, data paper.
- **Not English**, **Full text unavailable**, **Retracted**, **Duplicate**, **Used in pilot**

(*"Out-of-scope domain" is no longer used. It applied only in v1, which was limited to four domains; see `archive_domain_limited_v1/`.*)

**Borderline conventions**
- Remaining-useful-life or state-of-health prediction counts as forecasting **if** a degradation series is extrapolated forward in time. Pure condition classification does not.
- "Movement" or "trend" prediction: Include if values are forecast; Exclude (classification) if only up/down direction is predicted.
- Record the application field in the audit form's `Domain` column. It becomes a descriptor, not an eligibility criterion.
