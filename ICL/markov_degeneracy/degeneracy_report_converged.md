# Markov-ICL cross-seed degeneracy test

**Verdict: DIVERGENT**

error-conditioned slot agreement = +0.296 (0 = independent mistakes, 1 = identical function); gauge-quotiented weight cosine = +0.000 vs random-init floor +0.006.

- 16 groups (each = one fixed reaction graph + one fixed input mask), 5 train seeds per group, 80 checkpoints.
- Eval: 600 novel-class ICL queries, identical set for every model.

## Aggregate (mean over groups)

| probe | trained vs trained | trained vs random-init |
|---|---|---|
| error-slot agreement (decisive) | **+0.296** | 0 by construction |
| slot kappa | +0.688 | +0.000 |
| raw prediction agreement | 0.770 | 0.274 |
| attention agreement (1-TV) | 0.743 | 0.365 |
| steady-state agreement (1-TV) | 0.342 | 0.461 |
| **weight cosine (gauge-quotiented)** | **+0.000** | +0.006 |
| - K_eff (col-centered) | -0.016 | - |
| - base_log_rates | +0.076 | - |
| - B | +0.008 | - |

## How to read this

- **error-slot agreement** is the decisive probe. It looks only at queries both models get wrong and asks whether they make the *same* mistake. 0 means independent mistakes (different functions); 1 means identical mistakes (one function). Raw prediction agreement cannot distinguish these because two accurate models agree just by both being right.
- **weight cosine** is measured after removing the model's exact gauge freedoms. ~0 means the seeds' parameters are as unrelated as random initialisations.

## What this means

1. **Parametrically divergent.** Gauge-quotiented weight cosine between two trained seeds is +0.000 -- indistinguishable from the random-init floor (+0.006). No two seeds share *any* parametric structure.
2. **Internally divergent.** Trained-vs-trained steady-state agreement (0.342) is *lower* than trained-vs-random (0.461): training actively pushes each seed's internal state p(z) onto a different region of the simplex. Two trained seeds disagree internally more than a trained model disagrees with noise.
3. **Functionally only weakly shared.** Error-slot agreement is +0.296 of a possible 1.0 -- the seeds share a partial sub-solution but ~70% of their mistake structure is seed-specific. They are not one function.
4. **Undertrained landscape.** Accuracy is 64-83% with within-group spread up to ~11 points, on a task (exact_copy query) that admits a ~100% nearest-context-copy solution. The seeds settle into distinct mediocre optima, not a common one.
5. **Contrast with WTA-ICL.** The autocatalytic WTA model was *functionally constrained* (degenerate: same function, different weights) because it trained above its capacity threshold to the performance ceiling. The first-order Markov model here does the opposite -- different weights AND different functions -- because it never reaches that ceiling on these topologies.
6. **Consequence for the prior topology program.** Per-topology single-seed runs are seed-noise dominated: any topology -> performance regression fitted on one seed per topology was largely fitting cross-seed variance, not a topology law. The degeneracy test makes that quantitative.

## Per-group

| group | acc % (mean+-std) | err-slot | kappa | pred agree | attn agree | weight cosine | rand cosine |
|---|---|---|---|---|---|---|---|
| prospective_balanced_load_0008_c200_seed60626 | 89.6+-5.6 | +0.188 | +0.744 | 0.809 | 0.759 | +0.010 | +0.028 |
| prospective_balanced_load_0024_c200_seed60642 | 94.4+-4.1 | +nan | +0.859 | 0.894 | 0.842 | +0.009 | -0.029 |
| prospective_balanced_load_0035_c200_seed60653 | 89.4+-5.5 | +0.562 | +0.741 | 0.809 | 0.740 | -0.060 | +0.028 |
| prospective_balanced_load_0047_c200_seed60665 | 93.5+-2.2 | +nan | +0.835 | 0.877 | 0.825 | -0.044 | -0.071 |
| prospective_balanced_load_0066_c200_seed60684 | 90.6+-8.2 | +0.250 | +0.772 | 0.830 | 0.805 | +0.008 | -0.009 |
| prospective_balanced_load_0104_c200_seed60722 | 81.9+-7.6 | +0.051 | +0.560 | 0.670 | 0.650 | +0.010 | -0.038 |
| prospective_balanced_load_0114_c200_seed60732 | 92.5+-5.3 | +nan | +0.811 | 0.859 | 0.808 | -0.005 | +0.002 |
| prospective_balanced_load_0135_c200_seed60753 | 91.8+-8.2 | +0.523 | +0.793 | 0.846 | 0.817 | +0.038 | +0.012 |
| prospective_imbalanced_coord_load_0193_c200_seed60651 | 72.0+-11.0 | +0.047 | +0.452 | 0.605 | 0.622 | -0.000 | +0.010 |
| prospective_imbalanced_coord_load_0206_c200_seed60664 | 83.8+-5.5 | +0.438 | +0.661 | 0.750 | 0.717 | -0.070 | -0.019 |
| prospective_imbalanced_coord_load_0207_c200_seed60665 | 77.0+-6.9 | +0.236 | +0.555 | 0.671 | 0.683 | +0.010 | -0.037 |
| prospective_imbalanced_coord_load_0218_c200_seed60676 | 82.5+-9.1 | +0.233 | +0.637 | 0.732 | 0.681 | +0.027 | +0.050 |
| prospective_imbalanced_coord_load_0220_c200_seed60678 | 80.2+-3.5 | +0.205 | +0.648 | 0.744 | 0.716 | -0.005 | +0.039 |
| prospective_imbalanced_coord_load_0252_c200_seed60710 | 81.7+-9.2 | +0.356 | +0.629 | 0.726 | 0.729 | -0.015 | +0.006 |
| prospective_imbalanced_coord_load_0261_c200_seed60719 | 82.9+-7.5 | +0.262 | +0.669 | 0.756 | 0.714 | +0.067 | +0.064 |
| prospective_imbalanced_coord_load_0279_c200_seed60737 | 79.1+-7.0 | +0.503 | +0.648 | 0.742 | 0.774 | +0.026 | +0.063 |
