# Multi-Base Normal-Fan / Tree-Count Training Plan

## Status

Ready to submit. This is a targeted exact-control test, not a broad sweep.

## Primary Models

- controls/base_id only
- tree_count_only + base_id
- normal_fan_only + base_id
- tree_count + normal_fan + base_id
- cross_root_rank + normal_fan + tree_count + base_id

## Outcomes

- mean novel-class ICL
- best-seed novel-class ICL
- seed std

## Inference

- grouped LOO
- held-out-base checks
- matched-pair contrasts
- clustered bootstrap by base_id

## Submit

`sbatch /home/aadarwal/repos/topology/ICL/results/next_phase_stats/multibase_normal_fan_tree_count_n5_m12_N3_D2/_array_meta/run_task.sh`
