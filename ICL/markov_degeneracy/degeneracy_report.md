# Markov-ICL cross-seed degeneracy test

**Verdict: DIVERGENT**

error-conditioned slot agreement = +0.229 (0 = independent mistakes, 1 = identical function); gauge-quotiented weight cosine = -0.020 vs random-init floor +0.013.

- 16 groups (each = one fixed reaction graph + one fixed input mask), 5 train seeds per group, 80 checkpoints.
- Eval: 600 novel-class ICL queries, identical set for every model.

## Aggregate (mean over groups)

| probe | trained vs trained | trained vs random-init |
|---|---|---|
| error-slot agreement (decisive) | **+0.229** | 0 by construction |
| slot kappa | +0.465 | +0.000 |
| raw prediction agreement | 0.606 | 0.271 |
| attention agreement (1-TV) | 0.634 | 0.480 |
| steady-state agreement (1-TV) | 0.349 | 0.482 |
| **weight cosine (gauge-quotiented)** | **-0.020** | +0.013 |
| - K_eff (col-centered) | -0.019 | - |
| - base_log_rates | +0.089 | - |
| - B | -0.030 | - |

## How to read this

- **error-slot agreement** is the decisive probe. It looks only at queries both models get wrong and asks whether they make the *same* mistake. 0 means independent mistakes (different functions); 1 means identical mistakes (one function). Raw prediction agreement cannot distinguish these because two accurate models agree just by both being right.
- **weight cosine** is measured after removing the model's exact gauge freedoms. ~0 means the seeds' parameters are as unrelated as random initialisations.

## What this means

1. **Parametrically divergent.** Gauge-quotiented weight cosine between two trained seeds is -0.020 -- indistinguishable from the random-init floor (+0.013). No two seeds share *any* parametric structure.
2. **Internally divergent.** Trained-vs-trained steady-state agreement (0.349) is *lower* than trained-vs-random (0.482): training actively pushes each seed's internal state p(z) onto a different region of the simplex. Two trained seeds disagree internally more than a trained model disagrees with noise.
3. **Functionally only weakly shared.** Error-slot agreement is +0.229 of a possible 1.0 -- the seeds share a partial sub-solution but ~77% of their mistake structure is seed-specific. They are not one function.
4. **Undertrained landscape.** Accuracy is 64-83% with within-group spread up to ~11 points, on a task (exact_copy query) that admits a ~100% nearest-context-copy solution. The seeds settle into distinct mediocre optima, not a common one.
5. **Contrast with WTA-ICL.** The autocatalytic WTA model was *functionally constrained* (degenerate: same function, different weights) because it trained above its capacity threshold to the performance ceiling. The first-order Markov model here does the opposite -- different weights AND different functions -- because it never reaches that ceiling on these topologies.
6. **Consequence for the prior topology program.** Per-topology single-seed runs are seed-noise dominated: any topology -> performance regression fitted on one seed per topology was largely fitting cross-seed variance, not a topology law. The degeneracy test makes that quantitative.

## Per-group

| group | acc % (mean+-std) | err-slot | kappa | pred agree | attn agree | weight cosine | rand cosine |
|---|---|---|---|---|---|---|---|
| prospective_balanced_load_0008_c200_seed60626 | 78.6+-6.5 | +0.257 | +0.525 | 0.648 | 0.637 | -0.005 | -0.034 |
| prospective_balanced_load_0024_c200_seed60642 | 78.3+-6.3 | +0.240 | +0.499 | 0.625 | 0.621 | -0.020 | +0.069 |
| prospective_balanced_load_0035_c200_seed60653 | 76.7+-3.6 | +0.132 | +0.462 | 0.599 | 0.595 | +0.032 | -0.027 |
| prospective_balanced_load_0047_c200_seed60665 | 72.7+-4.9 | +0.194 | +0.395 | 0.550 | 0.585 | -0.024 | +0.003 |
| prospective_balanced_load_0066_c200_seed60684 | 82.0+-4.9 | +0.389 | +0.588 | 0.695 | 0.674 | -0.045 | +0.024 |
| prospective_balanced_load_0104_c200_seed60722 | 73.7+-8.4 | +0.201 | +0.431 | 0.578 | 0.609 | -0.039 | +0.030 |
| prospective_balanced_load_0114_c200_seed60732 | 82.8+-4.4 | +0.451 | +0.614 | 0.716 | 0.675 | -0.056 | +0.010 |
| prospective_balanced_load_0135_c200_seed60753 | 72.6+-11.2 | +0.141 | +0.402 | 0.554 | 0.606 | +0.003 | +0.004 |
| prospective_imbalanced_coord_load_0193_c200_seed60651 | 71.8+-4.3 | +0.198 | +0.523 | 0.654 | 0.659 | -0.043 | +0.029 |
| prospective_imbalanced_coord_load_0206_c200_seed60664 | 72.3+-7.2 | +0.255 | +0.482 | 0.622 | 0.640 | -0.035 | +0.043 |
| prospective_imbalanced_coord_load_0207_c200_seed60665 | 65.7+-2.9 | +0.247 | +0.427 | 0.583 | 0.639 | -0.043 | +0.018 |
| prospective_imbalanced_coord_load_0218_c200_seed60676 | 67.1+-4.6 | +0.112 | +0.372 | 0.536 | 0.616 | +0.004 | -0.004 |
| prospective_imbalanced_coord_load_0220_c200_seed60678 | 64.0+-7.3 | +0.130 | +0.368 | 0.543 | 0.618 | -0.021 | +0.010 |
| prospective_imbalanced_coord_load_0252_c200_seed60710 | 70.6+-3.0 | +0.283 | +0.473 | 0.614 | 0.660 | +0.017 | +0.025 |
| prospective_imbalanced_coord_load_0261_c200_seed60719 | 69.5+-3.4 | +0.247 | +0.440 | 0.588 | 0.648 | -0.010 | -0.013 |
| prospective_imbalanced_coord_load_0279_c200_seed60737 | 66.7+-6.2 | +0.195 | +0.432 | 0.589 | 0.656 | -0.029 | +0.025 |
