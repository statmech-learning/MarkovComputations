# Existing-Data Markov Expressivity Reanalysis

Primary outcome is novel-class ICL, aggregated by topology/mask group before inference.
Run-level seeds are not treated as independent topology samples.

## Dataset Coverage

| dataset | rows | groups |
| --- | --- | --- |
| fixed_m20 | 240 | 48 |
| hard_n4_m6 | 60 | 12 |
| hard_n5_m8 | 60 | 12 |
| hard_n5_m12 | 60 | 12 |
| degree_rewire_training | 4 | 4 |
| degree_rewire_library | 32 | 32 |

## Grouped LOOCV Summaries

### fixed_m20

| model | n_groups | loo_r2 | reason |
| --- | --- | --- | --- |
| multiplicity | 48 | 0.095 |  |
| comparison_multiplicity | 48 | 0.046 |  |
| tree_geometry | 48 | 0.158 |  |
| capacity_proxy | 48 | 0.145 |  |

### hard_n4_m6

| model | n_groups | loo_r2 | reason |
| --- | --- | --- | --- |
| multiplicity | 12 | -0.190 |  |
| comparison_multiplicity | 12 | -0.190 |  |
| tree_geometry | 12 | -0.130 |  |
| capacity_proxy | 12 | -0.190 |  |

### hard_n5_m8

| model | n_groups | loo_r2 | reason |
| --- | --- | --- | --- |
| multiplicity | 12 | -0.190 |  |
| comparison_multiplicity | 12 | -0.190 |  |
| tree_geometry | 12 | 0.056 |  |
| capacity_proxy | 12 | -0.190 |  |

### hard_n5_m12

| model | n_groups | loo_r2 | reason |
| --- | --- | --- | --- |
| multiplicity | 12 | -0.190 |  |
| comparison_multiplicity | 12 | -0.190 |  |
| tree_geometry | 12 | 0.437 |  |
| capacity_proxy | 12 | -0.190 |  |

## Limits

- Exact per-coordinate mask arrays are available locally for hard-sweep topology JSON files, but not for the fixed m20 CSV-only rows.
- Existing thermodynamic quantities are diagnostic only; the trained arbitrary directed-rate models are not reversible-edge thermodynamic parameterizations.
- Capacity proxy rows from the first gamma attempt are included as baselines, not accepted as final theory.
