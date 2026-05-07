# Topology-ICL Interpretation

Report kind: `research`. Target: `test_novel_classes`.

## Verdict

`strong_positive`: Topology-derived predictors and trained functional diagnostics both improve over count baselines.

Support threshold: candidate minus baseline >= `0.05` using LOO R2 with `n >= 6`.

## Model Comparisons

| scope | candidate | baseline | metric | n | baseline | candidate | delta | supported |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| run | edge_plus_drel | edge_count | LOO_R2 | 240 | -0.008 | 0.057 | 0.065 | yes |
| run | input_plus_masked_geometry | edge_count | LOO_R2 | 240 | -0.008 | 0.116 | 0.124 | yes |
| run | edge_plus_mechanism | edge_count | missing | 0 | -0.008 | NA | NA | no |
| run | edge_plus_projection | edge_count | LOO_R2 | 240 | -0.008 | 0.740 | 0.749 | yes |
| topology_mean | edge_plus_drel | edge_count | LOO_R2 | 48 | -0.043 | 0.145 | 0.188 | yes |
| topology_mean | input_plus_masked_geometry | edge_count | LOO_R2 | 48 | -0.043 | 0.189 | 0.232 | yes |
| topology_mean | edge_plus_mechanism | edge_count | missing | 0 | -0.043 | NA | NA | no |
| topology_mean | edge_plus_projection | edge_count | LOO_R2 | 48 | -0.043 | 0.809 | 0.852 | yes |
| topology_best | edge_plus_drel | edge_count | LOO_R2 | 48 | -0.043 | 0.180 | 0.223 | yes |
| topology_best | input_plus_masked_geometry | edge_count | LOO_R2 | 48 | -0.043 | -0.027 | 0.016 | no |
| topology_best | edge_plus_mechanism | edge_count | missing | 0 | -0.043 | NA | NA | no |
| topology_best | edge_plus_projection | edge_count | LOO_R2 | 48 | -0.043 | 0.599 | 0.642 | yes |
| retrain_mean | layout_plus_input_plus_drel | layout_type | LOO_R2 | 96 | 0.510 | 0.821 | 0.311 | yes |
| retrain_mean | layout_plus_input_plus_masked_geometry | layout_type | LOO_R2 | 96 | 0.510 | 0.849 | 0.340 | yes |
| retrain_best | layout_plus_input_plus_drel | layout_type | LOO_R2 | 96 | 0.465 | 0.774 | 0.309 | yes |
| retrain_best | layout_plus_input_plus_masked_geometry | layout_type | LOO_R2 | 96 | 0.465 | 0.799 | 0.334 | yes |

## Essential Retrain Retention

| experiment | layout | joined | retention mean | retention max | retrain mean | retrain best |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| random | physical subgraph | 16 | 0.733 | 0.822 | 65.688 | 90.000 |
| random | input mask | 16 | 0.625 | 0.684 | 55.368 | 72.200 |
| cycle | physical subgraph | 16 | 0.758 | 0.834 | 67.403 | 88.600 |
| cycle | input mask | 16 | 0.618 | 0.707 | 54.590 | 73.200 |
| hub | physical subgraph | 16 | 0.856 | 0.949 | 72.090 | 92.400 |
| hub | input mask | 16 | 0.637 | 0.702 | 52.398 | 65.200 |
