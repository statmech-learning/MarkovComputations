# Next-Phase Topology ICL Capacity and Causal Probes

This note records the first concrete implementation step after the topology-ICL synthesis: move from raw relative tree rank toward norm-bounded branch-margin capacity, mechanism-isolating graph selection, and statistic-preserving causal scrambles. The scope remains first-order CRNs with exponential input-dependent rates.

## What changed

- Added a bounded gamma-star proxy to `ICL/branch_margin_capacity.py`: sample norm-bounded edge projection matrices respecting the input mask, fit a norm-bounded decoder, and score the worst branch-level margin on held-out branch samples.
- Extended `ICL/collect_branch_margin_capacity.py` and `ICL/regress_topology_results.py` so gamma-star capacity columns can be collected and compared against `d_rel`, masked tree geometry, normal-fan proxies, and raw count baselines.
- Added statistic-preserving causal interventions to `ICL/causal_topology_interventions.py`: `stat_preserving_branch_alignment_scramble` and `stat_preserving_projection_scramble`.
- Added `ICL/make_mechanism_isolation_plan.py` to choose fixed-count contrasts that isolate mechanisms instead of blindly broadening graph sweeps.

## Gamma-Star Proxy

The target theoretical object is

```text
gamma*(G, Omega) = max_{K,B} min_z [q_{label*(z)} - max_{label != label*(z)} q_label(z)]
```

under norm constraints on `K` and `B`. The current implementation is not yet a solved optimization problem; it is a lower-bound/random-search proxy. That is deliberate: the first goal is to create an auditable capacity object that is closer to branch separation than `d_rel`, then improve the optimizer later.

Pilot capacity collection: `12` rows from `n5_m12_N3_D2`, `gamma_star_trials=16`, train/test branch samples `400/400`.
Gamma-star selected held-out branch p10-min range: `-0.1126` to `-0.0872`.
Gamma-star proxy held-out accuracy fraction range: `0.37` to `0.46`.

## Predictor Pilot

Regression target: `test_novel_classes_mean`. Rows: `12`. This is a small fixed-count pilot, so LOO R2 is more important than in-sample R2 and the result should not be overread.

| predictor set | n | R2 | LOO R2 | RMSE |
|---|---:|---:|---:|---:|
| `raw_count` | 12 | 0.000 | -0.190 | 5.881 |
| `input_count_plus_drel` | 12 | 0.000 | -0.190 | 5.881 |
| `tree_geometry` | 12 | 0.733 | 0.324 | 3.038 |
| `trainability_geometry` | 12 | 0.902 | 0.580 | 1.844 |
| `masked_tree_geometry` | 12 | 0.635 | 0.447 | 3.552 |
| `rooted_tree_polytope_capacity` | 12 | 0.588 | 0.081 | 3.774 |
| `normal_fan_capacity` | 12 | 0.673 | -0.284 | 3.362 |
| `gamma_star_capacity` | 12 | 0.567 | -0.377 | 3.869 |
| `gamma_star_capacity_plus_drel` | 12 | 0.567 | -0.377 | 3.869 |

Interpretation: raw count and `d_rel` were flat in this fixed-count subset; tree/trainability geometry generalized best. The new gamma-star proxy has real in-sample signal but negative LOO R2 in this 12-row pilot, so it is not yet better than the existing geometry proxies. This is useful: it says the next step is improving the gamma optimizer and testing it on larger mechanism-isolating contrasts, not claiming victory.

## Mechanism-Isolating Sweep Plan

Generated `5` planned contrasts in `mechanism_isolation_plan.md/json`. These target the exact weaknesses of the current evidence: same count but different bottlenecking, same tree count but different root balance, same degree pattern but different normal-fan proxy, same physical graph with permuted masks, and same mask count with different coordinate-load heterogeneity.
- `same_drel_different_bottleneck_participation`: status `ready`; contrast metric `edge_participation_gini`; group key `['n5_m12_N3_D2', '5.0', '12.0', '88.0']`.
- `same_tree_count_different_root_balance`: status `ready`; contrast metric `root_tree_count_gini`; group key `['n4_m6_N3_D2', '4.0', '6.0', '8.0']`.
- `same_degree_sequence_different_normal_fan`: status `unavailable`; contrast metric `NA`; group key `NA`.
- `same_physical_graph_permuted_input_masks`: status `unavailable`; contrast metric `NA`; group key `NA`.
- `same_mask_count_different_coordinate_load`: status `unavailable`; contrast metric `NA`; group key `NA`.

## Statistic-Preserving Causal Smoke

Run directory: `/home/aadarwal/repos/topology/ICL/results/expanded_hard_sweeps/n5_m12_N3_D2/g0040_cycle_chords_seed41_trainseed1`.
Baseline sampled novel-class target accuracy: `94.00` percent.
- `stat_preserving_branch_alignment_scramble` repeat `0`: accuracy `34.00` percent; delta `-60.00` points; preserves physical graph/root tree counts/d_rel = `True/True/True`.
- `stat_preserving_branch_alignment_scramble` repeat `1`: accuracy `49.00` percent; delta `-45.00` points; preserves physical graph/root tree counts/d_rel = `True/True/True`.
- `stat_preserving_projection_scramble` repeat `0`: accuracy `33.00` percent; delta `-61.00` points; preserves physical graph/root tree counts/d_rel = `True/True/True`.
- `stat_preserving_projection_scramble` repeat `1`: accuracy `35.00` percent; delta `-59.00` points; preserves physical graph/root tree counts/d_rel = `True/True/True`.

Interpretation: this single-run smoke supports the active alignment hypothesis: preserving coarse statistics while scrambling branch/tree projection alignment can destroy ICL. It must be repeated over high-ICL, medium-ICL, and failed models before it becomes a headline result.

## Immediate Next Steps

1. Run the gamma-star proxy over the mechanism-isolating contrasts rather than only `n5_m12_N3_D2`.
2. Replace random-search gamma with a stronger constrained optimizer or alternating max-affine fitting procedure.
3. Run causal scrambles across a stratified model set and compare drops against baseline active-tree MI, soft-posterior MI, margins, and edge-importance concentration.
4. Add clustered/hierarchical inference once the expanded contrast table exists; do not treat seed-level rows as independent topology samples.
