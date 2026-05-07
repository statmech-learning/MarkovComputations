# Topology-ICL Interpretation

Report kind: `input_mask`. Target: `test_novel_classes`.

## Verdict

`structural_positive`: Topology-derived predictors improve over count baselines; mechanism evidence is weaker or unavailable.

Support threshold: candidate minus baseline >= `0.05` using LOO R2 with `n >= 6`.

## Count Control

Fixed physical edge count: `True` with values `[20]`.
Fixed input-coupled parameter count: `True` with values `[200]`.

## Model Comparisons

| scope | candidate | baseline | metric | n | baseline | candidate | delta | supported |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| run | physical_backbone | raw_counts | LOO_R2 | 240 | -0.008 | 0.068 | 0.076 | yes |
| run | mask_family | raw_counts | LOO_R2 | 240 | -0.008 | 0.030 | 0.038 | no |
| run | masked_geometry | raw_counts | LOO_R2 | 240 | -0.008 | 0.116 | 0.124 | yes |
| run | mechanism | raw_counts | missing | 0 | -0.008 | NA | NA | no |
| mask_mean | physical_backbone | raw_counts | LOO_R2 | 48 | -0.043 | 0.172 | 0.215 | yes |
| mask_mean | mask_family | raw_counts | LOO_R2 | 48 | -0.043 | -0.166 | -0.123 | no |
| mask_mean | masked_geometry | raw_counts | LOO_R2 | 48 | -0.043 | 0.189 | 0.232 | yes |
| mask_mean | mechanism | raw_counts | missing | 0 | -0.043 | NA | NA | no |
| mask_best | physical_backbone | raw_counts | LOO_R2 | 48 | -0.043 | 0.023 | 0.066 | yes |
| mask_best | mask_family | raw_counts | LOO_R2 | 48 | -0.043 | -0.047 | -0.004 | no |
| mask_best | masked_geometry | raw_counts | LOO_R2 | 48 | -0.043 | -0.027 | 0.016 | no |
| mask_best | mechanism | raw_counts | missing | 0 | -0.043 | NA | NA | no |
| mask_seed_std | masked_geometry | raw_counts | LOO_R2 | 48 | -0.043 | -0.639 | -0.596 | no |
| mask_seed_std | mechanism | raw_counts | missing | 0 | -0.043 | NA | NA | no |

## Essential Retrain Retention

| experiment | layout | joined | retention mean | retention max | retrain mean | retrain best |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| random | input mask | 16 | 0.625 | 0.684 | 55.368 | 72.200 |
| cycle | input mask | 16 | 0.618 | 0.707 | 54.590 | 73.200 |
| hub | input mask | 16 | 0.637 | 0.702 | 52.398 | 65.200 |

