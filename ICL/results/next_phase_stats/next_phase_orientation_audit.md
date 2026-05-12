# Next-Phase Orientation Audit

## Git State

- Branch: `topology`
- Commit: `ebb0c52781008dbff12f9b0759187c8460511d2c`
- Commit summary: `ebb0c52 Update Markov ICL exact-control goal`

## Required Reports

| path | exists | bytes |
| --- | --- | --- |
| ICL/results/next_phase_stats/post_phase3_markov_icl_synthesis.md | True | 5467 |
| ICL/results/next_phase_stats/gamma_toy_repair_final_report.md | True | 1384 |
| ICL/results/next_phase_stats/input_multiplicity_causal_control_report.md | True | 4198 |
| ICL/results/next_phase_stats/tree_multiplicity_causal_mask_library.md | True | 6370 |
| ICL/results/next_phase_stats/predictor_name_reconciliation.md | True | 13487 |
| ICL/results/next_phase_stats/tree_level_multiplicity_reanalysis.md | True | 5203 |
| ICL/results/next_phase_stats/topology_icl_research_synthesis.md | True | 26719 |

## Code Paths

| purpose | path | exists |
| --- | --- | --- |
| tree_level_and_tree_difference_multiplicity | ICL/tree_level_multiplicity_metrics.py | True |
| repaired_gamma_toy_validation | ICL/analytic_gamma_repair.py | True |
| existing_data_causal_control | ICL/tree_multiplicity_causal_control.py | True |
| fixed_m20_mask_library_generation | ICL/make_input_mask_library.py | True |
| fixed_m20_training_submission | ICL/submit_topology_library_sweep.py | True |
| grouped_loo_inference_structural | ICL/clustered_topology_inference.py | True |
| grouped_loo_inference_markov | ICL/tree_multiplicity_causal_control.py | True |
| lower_tail_gamma_probe | ICL/branch_margin_capacity_v2.py | True |

## Fixed-m20 Learned K Availability

- Location checked: `ssh:engaging:/home/aadarwal/repos/topology`
- Available: `True`
- Note: model.pt stores the learned K_params state_dict; aggregate CSVs do not store learned K tensors.

| root | train dirs | model.pt | results.pkl | topology.json | config.json |
| --- | --- | --- | --- | --- | --- |
| ICL/results/input_mask_fixed_m20_cycle_chords_seed3_c200 | 80 | 80 | 80 | 80 | 80 |
| ICL/results/input_mask_fixed_m20_hub_spoke_seed63_c200 | 80 | 80 | 80 | 80 | 80 |
| ICL/results/input_mask_fixed_m20_random_sc_seed3_c200 | 80 | 80 | 80 | 80 | 80 |

## Conclusion

The repaired gamma and tree-multiplicity reports are present. Fixed-m20 learned tensors are available on Engaging through `model.pt`, but the local aggregate CSV/JSON reports do not contain learned K tensors, so post-training weighted tree-overlap analyses must reload per-run models from Engaging.
