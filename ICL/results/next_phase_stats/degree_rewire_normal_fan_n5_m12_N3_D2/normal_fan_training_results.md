# Degree-Rewired Normal-Fan Training Results

This finishes the targeted same-degree-sequence contrast. All four trained topologies share exact in/out degree sequence, `N_n=5`, `m=12`, `N_c=3`, `D=2`, full input coupling, and `d_rel=88`. They differ only by degree-preserving edge swaps, which change rooted-tree geometry and normal-fan diagnostics.

## Training Coverage

- Groups: `4`
- Runs: `20`
- Expected runs: `20`
- Completed `results.pkl`: `20`

## Topology-Level Outcomes

| group | mean ICL | best ICL | seed std | branch-tree NMI | active tree count | edge participation Gini | rooted trees |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `degree_rewire_n5_m12_base_g0354_degree_balanced_seed9_seed2` | 84.320 | 94.400 | 8.764 | 0.136435 | 39.250 | 0.121212 | 55 |
| `degree_rewire_n5_m12_base_g0354_degree_balanced_seed9_seed3` | 89.360 | 97.600 | 10.209 | 0.160333 | 50.750 | 0.137255 | 85 |
| `degree_rewire_n5_m12_base_g0354_degree_balanced_seed9_seed31` | 97.480 | 98.400 | 1.063 | 0.155424 | 52.750 | 0.137255 | 85 |
| `degree_rewire_n5_m12_base_g0354_degree_balanced_seed9_seed34` | 86.760 | 96.000 | 8.686 | 0.135706 | 45.250 | 0.161538 | 65 |

## Fixed-Statistic Contrasts

| contrast metric | low group | high group | metric delta | mean ICL delta | best ICL delta |
| --- | --- | --- | ---: | ---: | ---: |
| `capacity_normal_fan_branch_tree_nmi_mean` | `degree_rewire_n5_m12_base_g0354_degree_balanced_seed9_seed34` | `degree_rewire_n5_m12_base_g0354_degree_balanced_seed9_seed3` | 0.024627 | 2.600 | 1.600 |
| `capacity_normal_fan_active_tree_count_mean` | `degree_rewire_n5_m12_base_g0354_degree_balanced_seed9_seed2` | `degree_rewire_n5_m12_base_g0354_degree_balanced_seed9_seed31` | 13.500000 | 13.160 | 4.000 |

## Descriptive Correlations

These are descriptive only because `n=4`; they are useful for deciding what to scale next, not for inference.

| metric | Pearson r with mean ICL |
| --- | ---: |
| `capacity_normal_fan_branch_tree_nmi_mean` | 0.691 |
| `capacity_normal_fan_active_tree_count_mean` | 0.869 |
| `capacity_linear_test_accuracy` | NA |
| `edge_participation_gini` | 0.059 |
| `library_n_trees_total_enum` | 0.813 |

## Interpretation

This closes the previously missing mechanism-isolation cell at pilot scale: exact degree sequence and `d_rel` can be held fixed while normal-fan diagnostics vary and trained ICL changes. The sample is deliberately tiny, so this is not a statistical claim. It is a constructive contrast showing that the next larger sweep should target degree-rewired families rather than broad unrelated graph families.
