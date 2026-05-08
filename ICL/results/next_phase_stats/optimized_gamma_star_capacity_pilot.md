# Optimized Gamma-Star Capacity Pilot

This pilot tests the sharper Torch-optimized gamma-star proxy on the existing fixed-count `n5_m12_N3_D2` selected topology set. It optimizes bounded edge projections and a bounded decoder against a differentiable branch-min margin surrogate.

## Run

- Regime: `n5_m12_N3_D2`.
- Rows: `12`.
- Samples: `300` train / `300` test exact-copy branch samples.
- Tree feature mode: `logsumexp`.
- Random gamma trials: `8`.
- Optimized gamma restarts/steps: `2 / 80`.
- Norm constraints: projection radius `1.0`, decoder radius `1.0`, edge-bias radius `0.0`.

## Capacity Ranges

- Optimized gamma held-out accuracy fraction: `0.417` to `0.483`.
- Optimized gamma held-out branch p10-min: `-0.7031` to `-0.4275`.
- Random gamma held-out branch p10-min in this logsumexp run: `-0.1524` to `-0.0352`.

## Predictor Comparison

| predictor set | n | R2 | LOO R2 | RMSE |
|---|---:|---:|---:|---:|
| `raw_count` | 12 | 0.000 | -0.190 | 5.881 |
| `input_count_plus_drel` | 12 | 0.000 | -0.190 | 5.881 |
| `tree_geometry` | 12 | 0.733 | 0.324 | 3.038 |
| `trainability_geometry` | 12 | 0.902 | 0.580 | 1.844 |
| `masked_tree_geometry` | 12 | 0.635 | 0.447 | 3.552 |
| `rooted_tree_polytope_capacity` | 12 | 0.588 | 0.081 | 3.774 |
| `normal_fan_capacity` | 12 | 0.608 | -0.072 | 3.684 |
| `gamma_star_capacity` | 12 | 0.408 | -1.623 | 4.524 |
| `gamma_star_optimized_capacity` | 12 | 0.492 | -0.564 | 4.192 |
| `gamma_star_optimized_capacity_plus_drel` | 12 | 0.492 | -0.564 | 4.192 |

## Interpretation

The optimized proxy is sharper infrastructure than the previous random-search gamma probe because it directly optimizes bounded `K` and bounded decoder weights. In this small 12-row pilot it improves over the random gamma features as a regression predictor, but its absolute held-out p10 margins are worse because the current surrogate optimizes smooth branch means plus classification loss rather than the exact p10 objective. Its LOO R2 is still negative and below the tree/trainability geometry baselines. This is not evidence that gamma-star beats `d_rel` or tree geometry yet; it is a working optimizer that exposes the next theoretical task: align the optimization surrogate with the branch-margin statistic and test it on mechanism-isolating contrasts where the capacity object has room to distinguish graphs.

