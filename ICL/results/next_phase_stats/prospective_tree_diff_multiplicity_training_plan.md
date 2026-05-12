# Prospective Tree-Difference Multiplicity Training Plan

## Design

- Selected masks: `16`
- Seeds per mask: `5`
- Expected runs: `80`
- Primary outcome: novel-class ICL accuracy.
- Primary contrast: high vs low same-root tree-difference comparison overlap at fixed graph, count, `d_rel`, edge-load distribution, and coordinate-load distribution.

## Submit Command

```bash
python3 ICL/submit_topology_library_sweep.py --library_csv ICL/results/prospective_tree_diff_multiplicity_n6_m20_c200/selected.csv --output_root ICL/results/prospective_tree_diff_multiplicity_training --seeds 1,2,3,4,5 --manifest_csv ICL/results/next_phase_stats/prospective_tree_diff_multiplicity_training_manifest.csv --array --missing_only --max-concurrent 24
```

## Post-Training Collection

```bash
python3 ICL/collect_topology_results.py --input_root ICL/results/prospective_tree_diff_multiplicity_training --output_csv ICL/results/next_phase_stats/prospective_tree_diff_multiplicity_training_results.csv
python3 ICL/prospective_tree_diff_multiplicity_training_report.py
```
