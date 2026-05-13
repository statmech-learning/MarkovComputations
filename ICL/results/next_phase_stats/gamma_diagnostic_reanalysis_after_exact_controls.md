# Gamma Diagnostic Reanalysis After Exact Controls

## Status

diagnostic_not_selector

## Key Values

| setting | metric | value |
| --- | --- | --- |
| fixed_m20 | best gamma mean LOO R2 | 0.078 |
| fixed_m20 | tree-diff mean LOO R2 | 0.435 |
| prospective | mean gamma+controls LOO R2 | 0.320 |
| prospective | best gamma+controls LOO R2 | 0.571 |
| normal_fan | mean gamma exact LOO R2 | -0.127 |
| normal_fan | best gamma exact LOO R2 | -0.072 |

## Interpretation

Gamma remains a sanity-checked diagnostic. It passed analytic toys, but exact-control trained data do not support using it as a topology selector.
