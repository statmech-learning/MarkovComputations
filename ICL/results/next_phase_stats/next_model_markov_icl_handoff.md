# Next-Model Markov-ICL Handoff

## Use This First

Primary reference:

```text
@/Users/aadarwal/code/statmech/topology/ICL/MARKOV_ICL_NEXT_PHASE_GOAL.md
```

Most important completed synthesis:

```text
@/Users/aadarwal/code/statmech/topology/ICL/results/next_phase_stats/post_gamma_repair_exact_control_synthesis.md
```

Exact-control results commit:

```text
2c4cba6dfd729e8c43d52351288665940fd1e80e
```

Exact-control GitHub:

```text
https://github.com/statmech-learning/MarkovComputations/commit/2c4cba6dfd729e8c43d52351288665940fd1e80e
```

## Concrete Update

The exact-control phase is complete.

```text
Track 0 orientation audit                          complete
Track 1 repaired-gamma existing-data reanalysis    complete
Track 2 prospective tree-diff causal control        complete, 80/80 runs
Track 3 exact-degree normal-fan expansion           complete, 160/160 runs
Track 4 expressivity vs trainability report         complete
Track 5 mechanism and causal scramble follow-up     complete
Track 6 thermodynamic Fmax                          not run, explicitly untested
```

## Main Results

Repaired gamma:

```text
Gamma passes analytic toys.
Gamma does not predict current trained outcomes well.
Do not use gamma_no_bias as a topology selector yet.
```

Fixed-m20 existing data:

```text
tree-difference multiplicity mean-ICL LOO R2 = 0.435
best repaired gamma mean-ICL LOO R2          = 0.078
```

Prospective tree-difference exact control:

```text
balanced high tree-diff mean ICL = 75.360
balanced low tree-diff mean ICL  = 79.330
high-minus-low delta             = -3.970
CI95                             = [-7.9905, 0.6700]

balanced high tree-diff best ICL = 84.900
balanced low tree-diff best ICL  = 89.700
high-minus-low delta             = -4.800
CI95                             = [-9.000, -0.150]
```

Prospective grouped LOO:

```text
mean ICL controls only              R2 = 0.488
mean ICL tree-diff + controls       R2 = 0.447
mean ICL gamma + controls           R2 = 0.320

best ICL controls only              R2 = 0.601
best ICL tree-level + controls      R2 = 0.631
best ICL tree-diff + controls       R2 = 0.545
best ICL gamma + controls           R2 = 0.571
```

Exact-degree normal-fan expansion:

```text
mean ICL tree_count                 R2 = 0.114
mean ICL normal_fan_pair            R2 = 0.112
mean ICL gamma exact                R2 = -0.127

best ICL gamma_plus_normal_fan      R2 = 0.117
best ICL normal_fan_pair            R2 = 0.110
best ICL gamma exact                R2 = -0.072
```

Mechanism:

```text
context_block_shuffle mean accuracy delta                     = -71.458
stat_preserving_projection_scramble mean accuracy delta       = -59.688
stat_preserving_branch_alignment_scramble mean accuracy delta = -57.778
decoder_root_permutation mean accuracy delta                  = -53.125
```

Interpretation:

```text
Trained models use branch/projection organization.
This does not prove tree-difference overlap is a standalone pre-training causal knob.
```

## Best Next Task

Do not rerun the completed exact-control phase. The next model should:

1. Diagnose why the fixed-m20 tree-difference signal failed in the prospective exact-control design.
2. Expand the exact-degree normal-fan experiment across multiple base graphs or degree sequences.
3. Separate rooted-tree count from normal-fan active-tree geometry.
4. Keep repaired gamma as a diagnostic, not a selector.
5. Keep thermodynamics delayed until a reversible-support model exists.

Suggested deliverables:

```text
tree_difference_failure_diagnostic_report.md
tree_difference_failure_diagnostic_report.json
tree_difference_wider_contrast_mask_library.md
tree_difference_wider_contrast_mask_library.json
normal_fan_multibase_exact_control_library.md
normal_fan_multibase_exact_control_library.json
normal_fan_multibase_exact_control_report.md
normal_fan_multibase_exact_control_report.json
gamma_predictor_failure_audit.md
gamma_predictor_failure_audit.json
post_exact_control_next_synthesis.md
post_exact_control_next_synthesis.json
```

## Non-Negotiables

```text
Use novel-class ICL as the primary metric.
Use grouped inference; seeds are nested inside topology/mask groups.
Keep physical topology G separate from input mask Omega.
Do not use bare tree_geometry.
Do not claim motif uniqueness.
Do not claim thermodynamics from arbitrary directed graphs.
Do not claim a universal scalar law from the current weak normal-fan signal.
```
