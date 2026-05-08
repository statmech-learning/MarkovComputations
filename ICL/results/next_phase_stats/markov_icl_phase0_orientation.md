# Markov-ICL Phase 0 Orientation

Generated from the local topology worktree after reading the handoff bundle and the three grounding PDFs.

## Git State

- Worktree: `/Users/aadarwal/code/statmech/topology`
- Branch: `topology`
- Commit: `3c9c685`
- Remote: `origin git@github.com:statmech-learning/MarkovComputations`
- Tracked modifications at orientation time: none

The handoff documents were read from `/Users/aadarwal/code/statmech/MarkovComputations/markov_icl_agent_handoff`. The current local `MarkovComputations` checkout is on `main` and dirty, so implementation work is being done in the clean topology worktree.

## Grounding Summary

- The CRN-ICL paper defines the primary task as novel-class context/query matching and the mechanism as subspace projection, not explicit pairwise attention.
- The first-order topology report establishes the exact rooted-tree-sum basis `Theta_T = sum_{e in T} K_e` and shows fixed-count topology/mask effects beyond raw trainable count, while explicitly not claiming a universal `d_rel` law.
- The Markov expressivity paper adds the missing theory layer: input multiplicity, monotonicity failure modes, coefficient constraints among tree-polynomial terms, non-equilibrium driving, sharpness through tree-drive range, and structural compatibility.

## Important Code Paths

- `ICL/topology_metrics.py`: rooted-tree enumeration, relative tree rank, masked rank, input-mask and branch rank metrics.
- `ICL/branch_margin_capacity.py`: existing branch-capacity proxy; useful baseline but not a final lower-tail/worst-branch capacity.
- `ICL/collect_branch_margin_capacity.py`: batch collection for the existing capacity proxy.
- `ICL/clustered_topology_inference.py`: grouped inference over topology/mask groups.
- `ICL/input_mask_utils.py`: input mask validation and load/gini summaries.
- `ICL/models/topology_markov_icl.py`: first-order topology-aware Markov ICL model with masked learned edge projections.
- `ICL/analyze_topology_model.py`, `ICL/collect_mechanism_results.py`, `ICL/summarize_topology_mechanisms.py`: post-training mechanism metrics.
- `ICL/causal_topology_interventions.py`: statistic-preserving causal scrambles.

## Available Results

- `ICL/results/next_phase_stats/pooled_fixed_m20_topology_results.csv`: 240 fixed-count run rows.
- `ICL/results/next_phase_stats/pooled_fixed_m20_with_branch_capacity.csv`: same 240 rows joined to existing capacity proxy.
- `ICL/results/expanded_hard_stats/*_with_branch_capacity.csv`: three hard-regime summaries, 60 rows per regime.
- `ICL/results/expanded_hard_sweeps/*`: local run directories with `model.pt`, `results.pkl`, `config.json`, `topology.json`, `topology_metrics.json`, `mechanism_metrics.json`, and `causal_interventions.json`.
- Artifact counts available locally: 180 `model.pt`, 180 `results.pkl`, 180 `topology_metrics.json`, 180 `mechanism_metrics.json`, 180 `causal_interventions.json`.
- `ICL/results/next_phase_stats/degree_rewire_normal_fan_n5_m12_N3_D2`: exact-degree/d-rel normal-fan pilot with 32 structural candidates and 4 trained groups.
- `ICL/results/next_phase_stats/topology_goal_completion_audit.md` and `ICL/results/next_phase_stats/topology_icl_complete_explanation.md`: prior topology-phase audit and narrative.

## Missing Or Incomplete Artifacts For This Goal

The required Markov-expressivity deliverables are absent at orientation time:

- `markov_icl_expressivity_theory.md`
- `existing_data_markov_expressivity_reanalysis.md`
- `existing_data_markov_expressivity_reanalysis.json`
- `branch_margin_capacity_v2.py`
- `input_multiplicity_control_report.md`
- `input_multiplicity_control_report.json`
- `thermodynamic_force_budget_report.md`
- `thermodynamic_force_budget_report.json`
- `exact_degree_multiplicity_normal_fan_report.md`
- `exact_degree_multiplicity_normal_fan_report.json`
- `expressivity_vs_trainability_report.md`
- `expressivity_vs_trainability_report.json`
- final synthesis separating expressivity, trainability, mechanism, and thermodynamics claims

The local artifacts include arbitrary exponential-rate trained models, but not a reversible-edge thermodynamic model parameterization or an existing `F_max` force-budget sweep.

## Proposed Additions

- Add `ICL/markov_icl_expressivity_theory.md`.
- Add `ICL/markov_expressivity_reanalysis.py` to compute Markov-expressivity metrics and generate required existing-data reports from committed CSV/JSON artifacts.
- Add `ICL/branch_margin_capacity_v2.py` with exact log-sum-exp, tropical/max-over-trees, and hard-root variants targeting lower-tail/worst-branch margins under norm and mask constraints.
- Add generated reports/JSON under `ICL/results/next_phase_stats/`.
- Add tests for reusable Markov-expressivity metric helpers and capacity-v2 objective behavior where feasible without training.

## Phase 1 Command Sequence

Run existing-data analysis only; no new broad training sweeps:

```bash
cd /Users/aadarwal/code/statmech/topology
python3 ICL/markov_expressivity_reanalysis.py \
  --output_dir ICL/results/next_phase_stats

python3 ICL/branch_margin_capacity_v2.py \
  --topology_family cycle_chords \
  --n_nodes 5 \
  --n_edges 12 \
  --n_context 3 \
  --z_dim 2 \
  --n_samples 180 \
  --trials 8 \
  --max_root_assignments 6 \
  --output_json ICL/results/next_phase_stats/branch_margin_capacity_v2_smoke.json \
  --output_md ICL/results/next_phase_stats/branch_margin_capacity_v2_smoke.md

python3 -m py_compile $(find ICL -name '*.py' -not -path '*/__pycache__/*')
python3 -m unittest discover -s ICL/tests
git diff --check
```

If the full unittest suite is too slow locally, run focused tests for the new files and record the limitation.
