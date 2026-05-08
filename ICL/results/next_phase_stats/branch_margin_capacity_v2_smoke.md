# Branch-Margin Capacity V2

Finite-sample lower-tail branch-margin probes. These are nonconvex probes, not capacity theorems.

| variant | objective | acc | worst failure | p10 margin | drive range mean | trials |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact | -3.0010 | 0.394 | 1.000 | -1.9762 | 1.9775 | 8 |
| tropical | -2.7996 | 0.400 | 0.895 | -1.5944 | 1.9775 | 8 |
| hard_root | -2.2699 | 0.367 | 0.736 | -1.6552 | 2.0406 | 48 |

## Branch Details

### exact

| branch | n | LCVaR margin | mean margin | failure | accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 53 | -3.0010 | -1.2994 | 0.868 | 0.132 |
| 1 | 57 | -2.0687 | -1.3245 | 1.000 | 0.000 |
| 2 | 70 | -0.2944 | 1.0645 | 0.086 | 0.914 |

### tropical

| branch | n | LCVaR margin | mean margin | failure | accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 53 | -2.7996 | -0.9070 | 0.792 | 0.208 |
| 1 | 57 | -1.6690 | -0.8036 | 0.895 | 0.105 |
| 2 | 70 | -1.0346 | 0.5507 | 0.214 | 0.786 |

### hard_root

| branch | n | LCVaR margin | mean margin | failure | accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 53 | -2.1381 | -0.2719 | 0.736 | 0.264 |
| 1 | 57 | -1.9952 | -0.1689 | 0.596 | 0.404 |
| 2 | 70 | -2.2699 | -0.2359 | 0.586 | 0.414 |

