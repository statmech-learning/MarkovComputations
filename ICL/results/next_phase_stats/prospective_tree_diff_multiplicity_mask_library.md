# Prospective Tree-Difference Multiplicity Mask Library

## Status

This is a prospective exact-control mask library. It uses one fixed physical graph and selects masks before new training.

## Fixed Controls

- Physical graph: `cycle_chords_n6_m20_seed3`
- Source topology JSON: `ssh:engaging:/home/aadarwal/repos/topology/ICL/results/input_mask_fixed_m20_cycle_chords_seed3_c200/cycle_chords_n6_m20_seed3__mask0000_entry_random_c200_seed1_trainseed1/topology.json`
- `N`: `4`
- `D`: `4`
- Input-coupled count: `200`
- Balanced primary contrast: exact edge loads and exact coordinate loads are identical across selected masks.
- Imbalanced secondary contrast: exact edge loads and exact imbalanced coordinate-load distribution are identical across selected masks.
- Selection uses normalized same-root tree-difference comparison overlap, not raw tree-pair counts.

## Selected Categories

| category | masks | d_rel | input counts | edge-load gini | coord-load gini | mean min diff overlap | min diff range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| high_tree_diff_comparison_overlap_balanced_load | 4 | [200] | [200] | 0.000 | 0.000 | 0.936 | [0.9342095655348668, 0.9371069182389937] |
| high_tree_diff_comparison_overlap_imbalanced_coord_load | 4 | [200] | [200] | 0.000 | 0.250 | 0.658 | [0.6485849056603774, 0.6655933214072749] |
| low_tree_diff_comparison_overlap_balanced_load | 4 | [200] | [200] | 0.000 | 0.000 | 0.858 | [0.8497127775190187, 0.8674118925632666] |
| low_tree_diff_comparison_overlap_imbalanced_coord_load | 4 | [200] | [200] | 0.000 | 0.250 | 0.309 | [0.30346841913106976, 0.3131069733479372] |

## Training Files

- Candidate library CSV: `ICL/results/prospective_tree_diff_multiplicity_n6_m20_c200/library.csv`
- Selected training CSV: `ICL/results/prospective_tree_diff_multiplicity_n6_m20_c200/selected.csv`
- Expected training tasks at seeds `1,2,3,4,5`: `80`

## Causal Contrast

The primary causal contrast is high vs low tree-difference comparison overlap in the balanced-load stratum. The imbalanced stratum is secondary and tests whether the direction survives a fixed but imbalanced coordinate-load profile.
