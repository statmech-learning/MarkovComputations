# Markov-ICL Goal Completion Audit

Branch: `topology`
Base commit: `3c9c6858ef22a88ecc459cc21fbee94e1045cdf1`

## Deliverables

- `ICL/markov_icl_expressivity_theory.md`
- `ICL/markov_expressivity_reanalysis.py`
- `ICL/branch_margin_capacity_v2.py`
- `ICL/results/next_phase_stats/markov_icl_phase0_orientation.md`
- `ICL/results/next_phase_stats/existing_data_markov_expressivity_reanalysis.md`
- `ICL/results/next_phase_stats/existing_data_markov_expressivity_reanalysis.json`
- `ICL/results/next_phase_stats/input_multiplicity_control_report.md`
- `ICL/results/next_phase_stats/input_multiplicity_control_report.json`
- `ICL/results/next_phase_stats/thermodynamic_force_budget_report.md`
- `ICL/results/next_phase_stats/thermodynamic_force_budget_report.json`
- `ICL/results/next_phase_stats/exact_degree_multiplicity_normal_fan_report.md`
- `ICL/results/next_phase_stats/exact_degree_multiplicity_normal_fan_report.json`
- `ICL/results/next_phase_stats/expressivity_vs_trainability_report.md`
- `ICL/results/next_phase_stats/expressivity_vs_trainability_report.json`
- `ICL/results/next_phase_stats/markov_icl_final_synthesis.md`
- `ICL/results/next_phase_stats/branch_margin_capacity_v2_smoke.md`
- `ICL/results/next_phase_stats/branch_margin_capacity_v2_smoke.json`
- `ICL/tests/test_branch_margin_capacity_v2.py`
- `ICL/tests/test_markov_expressivity_reanalysis.py`

## Evidence Boundary

The existing-data reports use committed CSV/JSON artifacts only. No broad training sweep was launched.

The thermodynamic force-budget report is intentionally conservative: existing arbitrary directed exponential-rate runs are not a reversible-edge thermodynamic parameterization, so the report gives a reversible-support audit and required F_max implementation steps rather than claiming entropy-production or force-budget results.

The exact-degree normal-fan report uses the four locally available trained pilot groups. It is constructive but underpowered.

## Verification

Commands run successfully:

```bash
python3 ICL/markov_expressivity_reanalysis.py --output_dir ICL/results/next_phase_stats
python3 ICL/branch_margin_capacity_v2.py --topology_family cycle_chords --n_nodes 5 --n_edges 12 --n_context 3 --z_dim 2 --n_samples 180 --trials 8 --max_root_assignments 6 --seed 4 --output_json ICL/results/next_phase_stats/branch_margin_capacity_v2_smoke.json --output_md ICL/results/next_phase_stats/branch_margin_capacity_v2_smoke.md
python3 -m unittest ICL.tests.test_branch_margin_capacity_v2 ICL.tests.test_markov_expressivity_reanalysis
python3 -m unittest discover -s ICL/tests
python3 -m py_compile $(find ICL -name '*.py' -not -path '*/__pycache__/*')
git diff --check
```

Full unit-test result: 149 tests passed.
