# Markov-ICL Expressivity Synthesis

This synthesis separates what the existing artifacts can support from what still requires new controlled training or a new thermodynamic parameterization.

## Expressivity

The exact first-order object is the rooted tree-sum representation. Input multiplicity, comparison-coordinate overlap, branch-aware tree geometry, and lower-tail margin capacity are the right expressivity probes. Existing data support these as useful structural variables, but not as a final scalar law.

In the fixed-count m20 data, grouped LOOCV R2 is 0.095 for aggregate multiplicity variables and 0.158 for tree-geometry variables. In the hard n5_m12 data, tree geometry reaches grouped LOOCV R2 0.437 for mean novel-class ICL. These are existing-data model checks, not held-out theory validation.

The exact-degree normal-fan pilot has only 4 trained groups. It is useful constructively because all groups share the intended fixed-degree/d_rel controls while normal-fan and tree-count summaries vary, but it is not statistically powered.

## Trainability

Best-seed and mean-seed outcomes must remain separate. The existing hard-regime data show enough seed spread that conditioning, redundancy, tree entropy, and post-training branch alignment should be modeled as trainability variables rather than folded into expressivity.

The expressivity/trainability report therefore treats best seed as an envelope proxy, mean seed as reliability, and seed standard deviation as optimization instability. Mechanism correlations are strong in the existing hard data, but they are post-training descriptors and should not be confused with pre-training capacity.

## Mechanism

The strongest existing mechanism evidence remains post-training branch/projection/tree organization and statistic-preserving scrambles. Markov-expressivity metrics should be evaluated by whether they predict or explain that organization, not only average accuracy.

The improved branch-margin capacity probe now reports exact log-sum-exp, tropical, and hard-root lower-tail objectives with branch-wise failures. The smoke run is intentionally small and diagnostic; it validates the measurement path rather than claiming optimized capacity.

## Physical Thermodynamics

Existing arbitrary exponential-rate models may be non-equilibrium, but they do not support thermodynamic force-budget claims. A reversible-edge parameterization and explicit F_max sweep are required before physical thermodynamic conclusions can be made.

The current thermodynamic report status is `no_valid_Fmax_sweep_available`. The local reversible-support audit covers 36 hard topology groups, with mean reversible-edge fraction 0.512. This is an eligibility audit, not an entropy-production or force-budget result.

## Next Controlled Work

The next experiments should be targeted, not broad: input-multiplicity controls with fixed G/count/d_rel; an expanded exact-degree normal-fan panel; a reversible-edge F_max sweep; serial-versus-parallel sharpness controls; and matched expressivity/trainability pairs.
