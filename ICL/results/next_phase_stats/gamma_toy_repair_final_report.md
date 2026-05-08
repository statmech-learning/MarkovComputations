# Gamma Toy Repair Final Report

Gamma repaired: `True`.

## Gate

| condition | passed |
| --- | --- |
| toy_A_no_bias_fails_as_expected | True |
| toy_B_no_bias_passes_hard_coded_K | True |
| toy_C_no_bias_passes_hard_coded_K | True |
| three_species_tree_sum_orientation_matches_paper | True |
| optimizer_preserves_or_recovers_warm_start | True |
| reports_separate_accuracy_ordering_and_margin | True |

## Diagnosis

The previous failure was localized to branch data and random-probe definition: Toy B was encoded as a single active branch/class rather than the max branch pair M1>,M2>.  Delta-separated analytic datasets and hard-coded K reproduce the original small-system results.

## No-Bias Analytic Summary

| case | accuracy | LCVaR margin | ordering |
| --- | --- | --- | --- |
| Toy A two species both branches | 0.500 | -11.039 | expected failure |
| Toy B two species max branch | 1.000 | 3.513 | pass |
| Toy C three species both branches | 1.000 | 1.516 | 1.000 |

## Optimizer Warm Starts

| case | available | success | initial LCVaR | final LCVaR | initial acc | final acc |
| --- | --- | --- | --- | --- | --- | --- |
| Toy B | True | True | 3.513 | 4.929 | 1.000 | 1.000 |
| Toy C | True | True | 1.516 | 3.035 | 1.000 | 1.000 |

Bias variants are separated: `gamma_no_bias` is the repaired analytic result; `gamma_with_bias` is not used for the repair claim.
