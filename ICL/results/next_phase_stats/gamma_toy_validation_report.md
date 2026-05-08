# Gamma Toy Validation Report

## Gate Result

Phase 3 gate passed: `False`.

Do not use gamma*_ICL for large topology selection yet.

## Setup

- Probe: `ICL/branch_margin_capacity_v2.py lower_tail_capacity_probe`.
- Samples per toy: `500`.
- Trials per variant: `768`.
- Lower-tail alpha: `0.1`.
- Projection radius: `4.0`.
- Decoder radius: `4.0`.
- Bias labels: `gamma_no_bias` uses `edge_bias_radius=0`; `gamma_with_bias` uses positive edge-bias budget.

## toy_A_two_species_both_branches

fail: gamma*_ICL <= 0 or persistent negative branch lower-tail margin

| bias | variant | case pass | reason | objective | accuracy | worst failure | p10 margin | active branches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gamma_no_bias | exact | True | expected failure observed | -0.086 | 0.494 | 0.527 | -0.061 | [0, 1] |
| gamma_no_bias | tropical | True | expected failure observed | -0.119 | 0.526 | 0.490 | -0.080 | [0, 1] |
| gamma_no_bias | hard_root | True | expected failure observed | -0.554 | 0.518 | 0.490 | -0.371 | [0, 1] |
| gamma_with_bias | exact | True | expected failure observed | -0.173 | 0.460 | 0.559 | -0.105 | [0, 1] |
| gamma_with_bias | tropical | True | expected failure observed | -0.159 | 0.498 | 0.745 | -0.101 | [0, 1] |
| gamma_with_bias | hard_root | True | expected failure observed | -0.555 | 0.468 | 0.642 | -0.381 | [0, 1] |

## toy_B_two_species_one_branch

pass: gamma*_ICL > 0 under reasonable norm budget

| bias | variant | case pass | reason | objective | accuracy | worst failure | p10 margin | active branches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gamma_no_bias | exact | False | one-branch margin was not positive | -0.012 | 0.492 | 0.508 | -0.008 | [0] |
| gamma_no_bias | tropical | False | one-branch margin was not positive | -0.054 | 0.528 | 0.472 | -0.040 | [0] |
| gamma_no_bias | hard_root | False | one-branch margin was not positive | -0.012 | 0.506 | 0.494 | -0.008 | [0] |
| gamma_with_bias | exact | True | positive one-branch margin found | 3.605 | 1.000 | 0.000 | 3.858 | [0] |
| gamma_with_bias | tropical | True | positive one-branch margin found | 3.356 | 1.000 | 0.000 | 3.985 | [0] |
| gamma_with_bias | hard_root | True | positive one-branch margin found | 3.013 | 1.000 | 0.000 | 3.227 | [0] |

## toy_C_three_species_both_branches

pass: gamma*_ICL > 0 or clear improvement over Toy A

| bias | variant | case pass | reason | objective | accuracy | worst failure | p10 margin | active branches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gamma_no_bias | exact | False | three-species case did not improve enough | -0.971 | 0.508 | 0.988 | -0.632 | [0, 1] |
| gamma_no_bias | tropical | False | three-species case did not improve enough | -0.413 | 0.450 | 0.634 | -0.295 | [0, 1] |
| gamma_no_bias | hard_root | False | three-species case did not improve enough | -0.321 | 0.582 | 0.470 | -0.190 | [0, 1] |
| gamma_with_bias | exact | False | three-species case did not improve enough | -0.960 | 0.496 | 0.671 | -0.551 | [0, 1] |
| gamma_with_bias | tropical | False | three-species case did not improve enough | -0.892 | 0.528 | 0.559 | -0.501 | [0, 1] |
| gamma_with_bias | hard_root | False | three-species case did not improve enough | -0.663 | 0.496 | 1.000 | -0.587 | [0, 1] |

## Interpretation

Toy A correctly fails in the no-bias setting. Toy B becomes positive when edge biases are allowed, but the no-bias one-branch case remains negative in this finite-sample probe. Toy C does not pass the configured no-bias analytic check. Therefore lower-tail `gamma*_ICL` should remain gated off for large topology selection until the probe definition or optimizer is repaired.
