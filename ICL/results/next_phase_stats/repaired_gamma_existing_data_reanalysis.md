# Repaired Gamma Existing-Data Reanalysis

## Status

No new training was launched. This report recomputes no-bias exact, tropical, and hard-root lower-tail gamma probes for existing topology/mask groups and compares them to existing structural metrics.

## Gamma Probe Settings

- `n_samples`: `240`
- `trials`: `8`
- `alpha`: `0.1`
- `projection_radius`: `1.0`
- `decoder_radius`: `1.0`
- `edge_bias_radius`: `0.0`
- `max_root_assignments`: `12`
- `seed_base`: `771`

## Grouped LOO Summary

### fixed_m20_masks_cluster_topology

| outcome | model | groups | LOO R2 | reason |
| --- | --- | --- | --- | --- |
| mean_novel_icl | raw_plus_drel_structural | 48 | 0.145 | NA |
| mean_novel_icl | masked_tree_geometry_structural | 48 | 0.214 | NA |
| mean_novel_icl | tree_geometry_structural_full | 48 | 0.409 | NA |
| mean_novel_icl | tree_geometry_markov_reanalysis_subset | 48 | 0.158 | NA |
| mean_novel_icl | edge_multiplicity_markov_reanalysis | 48 | -0.002 | NA |
| mean_novel_icl | tree_level_multiplicity | 48 | 0.403 | NA |
| mean_novel_icl | tree_difference_multiplicity | 48 | 0.435 | NA |
| mean_novel_icl | repaired_gamma_no_bias_exact | 48 | -0.075 | NA |
| mean_novel_icl | repaired_gamma_no_bias_tropical | 48 | -0.115 | NA |
| mean_novel_icl | repaired_gamma_no_bias_hard_root | 48 | -0.150 | NA |
| mean_novel_icl | gamma_no_bias_plus_tree_difference_multiplicity | 48 | 0.078 | NA |
| best_seed_novel_icl | raw_plus_drel_structural | 48 | 0.180 | NA |
| best_seed_novel_icl | masked_tree_geometry_structural | 48 | -0.005 | NA |
| best_seed_novel_icl | tree_geometry_structural_full | 48 | 0.280 | NA |
| best_seed_novel_icl | tree_geometry_markov_reanalysis_subset | 48 | 0.080 | NA |
| best_seed_novel_icl | edge_multiplicity_markov_reanalysis | 48 | 0.109 | NA |
| best_seed_novel_icl | tree_level_multiplicity | 48 | 0.245 | NA |
| best_seed_novel_icl | tree_difference_multiplicity | 48 | 0.419 | NA |
| best_seed_novel_icl | repaired_gamma_no_bias_exact | 48 | -0.130 | NA |
| best_seed_novel_icl | repaired_gamma_no_bias_tropical | 48 | -0.086 | NA |
| best_seed_novel_icl | repaired_gamma_no_bias_hard_root | 48 | -0.114 | NA |
| best_seed_novel_icl | gamma_no_bias_plus_tree_difference_multiplicity | 48 | 0.063 | NA |
| seed_std_novel_icl | raw_plus_drel_structural | 48 | -0.043 | NA |
| seed_std_novel_icl | masked_tree_geometry_structural | 48 | -0.619 | NA |
| seed_std_novel_icl | tree_geometry_structural_full | 48 | -0.085 | NA |
| seed_std_novel_icl | tree_geometry_markov_reanalysis_subset | 48 | 0.061 | NA |
| seed_std_novel_icl | edge_multiplicity_markov_reanalysis | 48 | -0.142 | NA |
| seed_std_novel_icl | tree_level_multiplicity | 48 | -0.169 | NA |
| seed_std_novel_icl | tree_difference_multiplicity | 48 | -0.047 | NA |
| seed_std_novel_icl | repaired_gamma_no_bias_exact | 48 | 0.071 | NA |
| seed_std_novel_icl | repaired_gamma_no_bias_tropical | 48 | -0.219 | NA |
| seed_std_novel_icl | repaired_gamma_no_bias_hard_root | 48 | -0.091 | NA |
| seed_std_novel_icl | gamma_no_bias_plus_tree_difference_multiplicity | 48 | 0.069 | NA |

Strict modal d_rel subset: `200.0` with `45` groups.
| outcome | model | groups | LOO R2 | reason |
| --- | --- | --- | --- | --- |
| mean_novel_icl | tree_difference_multiplicity | 45 | 0.356 | NA |
| mean_novel_icl | repaired_gamma_no_bias_exact | 45 | -0.095 | NA |
| mean_novel_icl | repaired_gamma_no_bias_tropical | 45 | -0.107 | NA |
| mean_novel_icl | repaired_gamma_no_bias_hard_root | 45 | -0.159 | NA |
| mean_novel_icl | gamma_no_bias_plus_tree_difference_multiplicity | 45 | 0.211 | NA |
| best_seed_novel_icl | tree_difference_multiplicity | 45 | 0.278 | NA |
| best_seed_novel_icl | repaired_gamma_no_bias_exact | 45 | -0.124 | NA |
| best_seed_novel_icl | repaired_gamma_no_bias_tropical | 45 | -0.073 | NA |
| best_seed_novel_icl | repaired_gamma_no_bias_hard_root | 45 | -0.151 | NA |
| best_seed_novel_icl | gamma_no_bias_plus_tree_difference_multiplicity | 45 | 0.071 | NA |

### hard_full_mask_local

| outcome | model | groups | LOO R2 | reason |
| --- | --- | --- | --- | --- |
| mean_novel_icl | raw_plus_drel_structural | 36 | 0.692 | NA |
| mean_novel_icl | masked_tree_geometry_structural | 36 | 0.793 | NA |
| mean_novel_icl | tree_geometry_structural_full | 36 | 0.767 | NA |
| mean_novel_icl | tree_geometry_markov_reanalysis_subset | 36 | 0.785 | NA |
| mean_novel_icl | edge_multiplicity_markov_reanalysis | 36 | 0.689 | NA |
| mean_novel_icl | tree_level_multiplicity | 36 | 0.758 | NA |
| mean_novel_icl | tree_difference_multiplicity | 33 | 0.703 | NA |
| mean_novel_icl | repaired_gamma_no_bias_exact | 36 | 0.006 | NA |
| mean_novel_icl | repaired_gamma_no_bias_tropical | 36 | 0.348 | NA |
| mean_novel_icl | repaired_gamma_no_bias_hard_root | 36 | -0.189 | NA |
| mean_novel_icl | gamma_no_bias_plus_tree_difference_multiplicity | 33 | 0.108 | NA |
| best_seed_novel_icl | raw_plus_drel_structural | 36 | 0.710 | NA |
| best_seed_novel_icl | masked_tree_geometry_structural | 36 | 0.770 | NA |
| best_seed_novel_icl | tree_geometry_structural_full | 36 | 0.754 | NA |
| best_seed_novel_icl | tree_geometry_markov_reanalysis_subset | 36 | 0.782 | NA |
| best_seed_novel_icl | edge_multiplicity_markov_reanalysis | 36 | 0.696 | NA |
| best_seed_novel_icl | tree_level_multiplicity | 36 | 0.753 | NA |
| best_seed_novel_icl | tree_difference_multiplicity | 33 | 0.661 | NA |
| best_seed_novel_icl | repaired_gamma_no_bias_exact | 36 | 0.042 | NA |
| best_seed_novel_icl | repaired_gamma_no_bias_tropical | 36 | 0.397 | NA |
| best_seed_novel_icl | repaired_gamma_no_bias_hard_root | 36 | -0.182 | NA |
| best_seed_novel_icl | gamma_no_bias_plus_tree_difference_multiplicity | 33 | 0.184 | NA |
| seed_std_novel_icl | raw_plus_drel_structural | 36 | 0.002 | NA |
| seed_std_novel_icl | masked_tree_geometry_structural | 36 | -0.111 | NA |
| seed_std_novel_icl | tree_geometry_structural_full | 36 | -0.099 | NA |
| seed_std_novel_icl | tree_geometry_markov_reanalysis_subset | 36 | -0.109 | NA |
| seed_std_novel_icl | edge_multiplicity_markov_reanalysis | 36 | -0.024 | NA |
| seed_std_novel_icl | tree_level_multiplicity | 36 | -0.040 | NA |
| seed_std_novel_icl | tree_difference_multiplicity | 33 | -0.099 | NA |
| seed_std_novel_icl | repaired_gamma_no_bias_exact | 36 | -0.026 | NA |
| seed_std_novel_icl | repaired_gamma_no_bias_tropical | 36 | -0.171 | NA |
| seed_std_novel_icl | repaired_gamma_no_bias_hard_root | 36 | -0.057 | NA |
| seed_std_novel_icl | gamma_no_bias_plus_tree_difference_multiplicity | 33 | -0.145 | NA |
| branch_failure_percent | raw_plus_drel_structural | 36 | 0.779 | NA |
| branch_failure_percent | masked_tree_geometry_structural | 36 | 0.876 | NA |
| branch_failure_percent | tree_geometry_structural_full | 36 | 0.857 | NA |
| branch_failure_percent | tree_geometry_markov_reanalysis_subset | 36 | 0.888 | NA |
| branch_failure_percent | edge_multiplicity_markov_reanalysis | 36 | 0.763 | NA |
| branch_failure_percent | tree_level_multiplicity | 36 | 0.844 | NA |
| branch_failure_percent | tree_difference_multiplicity | 33 | 0.793 | NA |
| branch_failure_percent | repaired_gamma_no_bias_exact | 36 | 0.100 | NA |
| branch_failure_percent | repaired_gamma_no_bias_tropical | 36 | 0.343 | NA |
| branch_failure_percent | repaired_gamma_no_bias_hard_root | 36 | -0.175 | NA |
| branch_failure_percent | gamma_no_bias_plus_tree_difference_multiplicity | 33 | 0.266 | NA |
| trained_branch_margin | raw_plus_drel_structural | 36 | 0.548 | NA |
| trained_branch_margin | masked_tree_geometry_structural | 36 | 0.763 | NA |
| trained_branch_margin | tree_geometry_structural_full | 36 | 0.747 | NA |
| trained_branch_margin | tree_geometry_markov_reanalysis_subset | 36 | 0.774 | NA |
| trained_branch_margin | edge_multiplicity_markov_reanalysis | 36 | 0.544 | NA |
| trained_branch_margin | tree_level_multiplicity | 36 | 0.655 | NA |
| trained_branch_margin | tree_difference_multiplicity | 33 | 0.586 | NA |
| trained_branch_margin | repaired_gamma_no_bias_exact | 36 | -0.026 | NA |
| trained_branch_margin | repaired_gamma_no_bias_tropical | 36 | 0.250 | NA |
| trained_branch_margin | repaired_gamma_no_bias_hard_root | 36 | -0.174 | NA |
| trained_branch_margin | gamma_no_bias_plus_tree_difference_multiplicity | 33 | 0.025 | NA |

Strict modal d_rel subset: `88.0` with `12` groups.
| outcome | model | groups | LOO R2 | reason |
| --- | --- | --- | --- | --- |
| mean_novel_icl | tree_difference_multiplicity | 12 | -0.015 | NA |
| mean_novel_icl | repaired_gamma_no_bias_exact | 12 | -0.668 | NA |
| mean_novel_icl | repaired_gamma_no_bias_tropical | 12 | -0.404 | NA |
| mean_novel_icl | repaired_gamma_no_bias_hard_root | 12 | -0.420 | NA |
| mean_novel_icl | gamma_no_bias_plus_tree_difference_multiplicity | 12 | -0.534 | NA |
| best_seed_novel_icl | tree_difference_multiplicity | 12 | -0.121 | NA |
| best_seed_novel_icl | repaired_gamma_no_bias_exact | 12 | -1.262 | NA |
| best_seed_novel_icl | repaired_gamma_no_bias_tropical | 12 | -0.669 | NA |
| best_seed_novel_icl | repaired_gamma_no_bias_hard_root | 12 | -0.437 | NA |
| best_seed_novel_icl | gamma_no_bias_plus_tree_difference_multiplicity | 12 | -1.162 | NA |

## Key Answers

- Gamma selector gate: `not_cleared_for_large_sweeps`.
- Reason: No repaired gamma model is allowed as a broad selector unless it improves over existing structural metrics.
- `gamma_with_bias` was not computed and is not used for no-bias capacity claims.
- Fixed-m20 branch failures and trained branch margins remain unavailable in aggregate artifacts; hard full-mask groups include those outcomes.

## Interpretation

Treat repaired gamma as a diagnostic predictor unless it survives the prospective exact-control phase. This reanalysis is existing-data only and does not replace the Track 2 causal mask experiment.
