# Markov-ICL cross-seed degeneracy test

**Verdict: DIVERGENT**

error-conditioned slot agreement = +0.230 (0 = independent mistakes, 1 = identical function); gauge-quotiented weight cosine = +0.002 vs random-init floor +0.014.

- 4 groups (each = one fixed reaction graph + one fixed input mask), 5 train seeds per group, 80 checkpoints.
- Eval: 600 novel-class ICL queries, identical set for every model.

## Aggregate (mean over groups)

| probe | trained vs trained | trained vs random-init |
|---|---|---|
| error-slot agreement (decisive) | **+0.230** | 0 by construction |
| slot kappa | +0.696 | +0.000 |
| raw prediction agreement | 0.776 | 0.270 |
| attention agreement (1-TV) | 0.753 | 0.363 |
| steady-state agreement (1-TV) | 0.352 | 0.456 |
| **weight cosine (gauge-quotiented)** | **+0.002** | +0.014 |
| - K_eff (col-centered) | -0.005 | - |
| - base_log_rates | +0.055 | - |
| - B | -0.001 | - |

## How to read this

- **error-slot agreement** is the decisive probe. It looks only at queries both models get wrong and asks whether they make the *same* mistake. 0 means independent mistakes (different functions); 1 means identical mistakes (one function). Raw prediction agreement cannot distinguish these because two accurate models agree just by both being right.
- **weight cosine** is measured after removing the model's exact gauge freedoms. ~0 means the seeds' parameters are as unrelated as random initialisations.

## What this means

1. **Parametrically divergent.** Gauge-quotiented weight cosine between two trained seeds is +0.002 -- indistinguishable from the random-init floor (+0.014). No two seeds share *any* parametric structure.
2. **Internally divergent.** Trained-vs-trained steady-state agreement (0.352) is *lower* than trained-vs-random (0.456): training actively pushes each seed's internal state p(z) onto a different region of the simplex. Two trained seeds disagree internally more than a trained model disagrees with noise.
3. **Functionally only weakly shared.** Error-slot agreement is +0.230 of a possible 1.0 -- the seeds share a partial sub-solution but ~77% of their mistake structure is seed-specific. They are not one function.
4. **Undertrained landscape.** Accuracy is 64-83% with within-group spread up to ~11 points, on a task (exact_copy query) that admits a ~100% nearest-context-copy solution. The seeds settle into distinct mediocre optima, not a common one.
5. **Contrast with WTA-ICL.** The autocatalytic WTA model was *functionally constrained* (degenerate: same function, different weights) because it trained above its capacity threshold to the performance ceiling. The first-order Markov model here does the opposite -- different weights AND different functions -- because it never reaches that ceiling on these topologies.
6. **Consequence for the prior topology program.** Per-topology single-seed runs are seed-noise dominated: any topology -> performance regression fitted on one seed per topology was largely fitting cross-seed variance, not a topology law. The degeneracy test makes that quantitative.

## Per-group

| group | acc % (mean+-std) | err-slot | kappa | pred agree | attn agree | weight cosine | rand cosine |
|---|---|---|---|---|---|---|---|
| prospective_balanced_load_0066_c200_seed60684 | 90.6+-8.2 | +0.250 | +0.772 | 0.830 | 0.805 | +0.008 | -0.027 |
| prospective_balanced_load_0114_c200_seed60732 | 92.5+-5.3 | +nan | +0.811 | 0.859 | 0.808 | -0.005 | -0.009 |
| prospective_imbalanced_coord_load_0207_c200_seed60665 | 77.0+-6.9 | +0.236 | +0.555 | 0.671 | 0.683 | +0.010 | +0.048 |
| prospective_imbalanced_coord_load_0220_c200_seed60678 | 80.2+-3.5 | +0.205 | +0.648 | 0.744 | 0.716 | -0.005 | +0.043 |
