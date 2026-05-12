# Mechanism and Causal Scramble Follow-Up Report

## Scope

This report uses the prospective exact-control trained models. It joins post-training mechanism diagnostics for all 80 trained runs with statistic-preserving causal scrambles on four selected high-performing trained models, one from each load/contrast family.

## Mechanism Diagnostics

- Mechanism rows: `80`
- Mechanism groups: `16`

| metric | n | mean | min | max |
| --- | --- | --- | --- | --- |
| branch_active_tree_mi | 80 | 0.945 | 0.621 | 1.251 |
| branch_active_root_mi | 80 | 0.608 | 0.292 | 1.029 |
| tree_entropy_mean | 80 | 12.805 | 10.283 | 15.737 |
| root_entropy_mean | 80 | 1.030 | 0.685 | 1.223 |
| target_logprob_margin_branch_mean_min | 80 | 0.128 | -0.751 | 3.125 |
| posterior_matched_comparison_gap_mean | 80 | -0.326 | -0.448 | -0.223 |
| active_tree_matched_comparison_gap_mean | 80 | -0.412 | -0.567 | -0.297 |
| tree_comparison_energy_fraction_mean | 80 | 0.945 | 0.776 | 0.992 |
| edge_importance_gini | 80 | 0.278 | 0.167 | 0.412 |

These diagnostics include branch-active-tree MI, branch-to-root MI, tree/root entropy, trained branch margin, posterior matched comparison gap, input-coupling ablation loss, physical edge ablation loss, and functional edge-importance concentration when available.

## Selected Edge Ablations

- Selected ablation rows: `4`
- Selected ablation groups: `4`

| metric | n | mean | min | max |
| --- | --- | --- | --- | --- |
| input_ablation_max_loss | 4 | 21.146 | 17.083 | 28.750 |
| input_ablation_mean_loss | 4 | 6.125 | 4.812 | 8.167 |
| physical_ablation_max_loss | 4 | 26.563 | 23.333 | 32.917 |
| physical_ablation_mean_loss | 4 | 11.010 | 9.646 | 13.250 |

The ablation panel was run on the same four selected trained models used for causal scrambles. It covers input-coupling and physical-edge loss diagnostics without turning the mechanism follow-up into a new broad sweep.

## Causal Scrambles

- Intervention rows: `48`
- Selected trained runs: `4`
- Repeats per intervention/run: `3`

| intervention | n | mean accuracy delta | min | max | mean margin delta |
| --- | --- | --- | --- | --- | --- |
| context_block_shuffle | 12 | -71.458 | -91.667 | -39.583 | -6.196 |
| decoder_root_permutation | 12 | -53.125 | -75.833 | -7.917 | -4.762 |
| stat_preserving_branch_alignment_scramble | 12 | -57.778 | -89.167 | -11.667 | -4.804 |
| stat_preserving_projection_scramble | 12 | -59.688 | -72.083 | -47.917 | -6.006 |

The largest drops occur under context-block/branch-alignment and projection-direction scrambles. These interventions preserve the physical graph and coarse input-mask support while disrupting trained alignment structure, so they support the mechanism claim that trained first-order CRN-ICL solutions rely on branch/projection organization.

## Interpretation

The mechanism evidence is strong for the selected trained models but remains post-training. It does not rescue the prospective pre-training tree-difference causal contrast, which was negative or inconclusive. The result says that models that train well use branch/projection structure; it does not prove that the prospective tree-difference overlap metric alone is the right pre-training causal knob.
