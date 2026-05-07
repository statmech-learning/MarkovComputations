# Topology-ICL Status Audit

Current branch: `topology`

For a prompt-to-artifact completion audit, see
`TOPOLOGY_COMPLETION_AUDIT.md`.

For the next-phase implementation/theory audit requested by the critique, see
`TOPOLOGY_THEORY_AUDIT.md`.

## Objective

Build a theory and simulation framework for testing whether first-order CRN
topology controls in-context learning beyond raw trainable degree count, using
the matrix-tree theorem to connect reaction topology to relative spanning-tree
projection geometry.

## Deliverable Checklist

| Requirement | Artifact evidence | Status |
| --- | --- | --- |
| First-order CRN topology represented as an explicit directed graph | `models/topology_markov_icl.py`, `run_topology_icl.py` | Implemented; Torch training requires cluster or Torch-enabled env |
| Separate physical topology from input-encoding topology | `input_mask_utils.py`, `make_input_mask_library.py`, `run_topology_icl.py --input_mask_json` | Implemented and covered by pure-Python tests |
| Matrix-tree structural metrics for arbitrary strongly connected graphs | `topology_metrics.py` | Implemented: arborescence enumeration, relative tree rank, spectra, root balance, edge participation |
| First mandatory theory audit | `TOPOLOGY_THEORY_AUDIT.md` | Completed for the current first-order implementation; main caveats are trainable tree intercepts, nested seed dependence, and unmatched motif controls |
| Branch-aware comparison-capacity metrics | `comparison_branch_common_d_rel_*`, `comparison_branch_d_rel_*`, `comparison_branch_input_overlap_*`, and `comparison_branch_input_count_*` in `topology_metrics.py` | Implemented, collected, regressed, and reported |
| Sampled branch-margin capacity probe | `branch_margin_capacity.py` | Implemented as a conservative pre-training proxy; it gates exact-copy comparison features by common context/query tree-contrast support and reports oracle plus norm-controlled linear margins |
| Controlled physical topology libraries at fixed `N_n,N_c,D,m` | `make_topology_library.py`, `submit_topology_library_sweep.py` | Implemented and dry-run/tested |
| Expanded matched topology families | `topology_metrics.py`, `make_topology_library.py` | Added `degree_balanced`, `bottleneck_bridge`, and `redundant_paths` families beyond random/cycle/hub/two-module baselines |
| Multi-regime expanded sweep plan | `make_topology_sweep_plan.py` | Implemented and tested; generates CSV and dry-run command script across `N_n`, edge regime, `N_c`, and `D` |
| Controlled fixed-physical-graph input-mask libraries | `make_input_mask_library.py`, `make_input_mask_report.py` | Implemented and tested |
| Run-level and topology-level regressions against raw count and tree geometry | `regress_topology_results.py`, `aggregate_topology_seeds.py` | Implemented and tested |
| Cluster-aware inference for nested seed rows | `clustered_topology_inference.py` | Implemented and tested: group-level regressions, cluster bootstrap deltas, leave-one-family-out prediction, residual decomposition |
| Novel-class ICL remains the primary collected metric | `collect_topology_results.py` field `test_novel_classes`; report defaults | Implemented |
| Expressivity vs trainability split | `aggregate_topology_seeds.py` outputs `target_max`, `target_mean`, `target_std` | Implemented and tested |
| Post-training active tree/root, branch MI, margins, sensitivities, and ablations | `topology_analysis.py`, `analyze_topology_model.py`, `collect_mechanism_results.py`, `summarize_topology_mechanisms.py` | Implemented; execution requires Torch-enabled trained runs |
| Causal branch/tree-alignment interventions | `causal_topology_interventions.py`, `submit_causal_interventions.py`, `collect_causal_interventions.py` | Implemented and pure-Python helpers/submitter/collector tested; cluster execution on trained high-ICL runs still required |
| Essential physical subgraph extraction and retraining | `extract_essential_subgraphs.py`, `finalize_essential_physical_retrains.py`, `recover_essential_physical_retrains.py`, `compare_essential_retrains.py` | Implemented and tested |
| Matched essential-motif controls | `make_matched_motif_controls.py` | Implemented and tested for retrainable random/degree-rewired physical controls matched on coarse tree-geometry features |
| Essential input-mask extraction and retraining | `extract_essential_input_masks.py`, `finalize_essential_inputmask_retrains.py`, `recover_essential_inputmask_retrains.py` | Implemented and tested |
| Consolidated research report | `make_topology_research_report.py`, `finalize_topology_research_report.py` | Implemented; finalizer verifies both `essential_input50` and `essential_inputmask50` layouts before interpretation |
| Artifact audit and interrupted-array recovery | `audit_topology_artifacts.py`, `recover_essential_physical_retrains.py`, `recover_essential_inputmask_retrains.py` | Implemented and tested |

## Verification Gates

Local, dependency-light checks:

```bash
python3 -m unittest discover -s ICL/tests
python3 -m py_compile $(find ICL -name '*.py' -not -path '*/__pycache__/*')
git diff --check
```

As of the latest local run, the unittest suite has 83 tests and passes. Local
Python does not have Torch, so training and mechanism smoke tests must run on
the cluster or another Torch-enabled environment.

## Current Cluster Blocker

The intended Engaging tmux pane `icl:13.2` is currently local, not on Engaging:

```text
Aadarshs-Mac-mini.local
/Users/aadarwal/code/projects/InContextLearning
ssh engaging password prompt
```

Do not enter a password or use other panes that may belong to other agents.
Once `icl:13.2` is restored to Engaging, use the cluster worktree:

```bash
cd ~/repos/topology/ICL
git pull --ff-only
```

Then recover the essential input-mask retrains in two passes.

First inspect:

```bash
python3 recover_essential_inputmask_retrains.py \
  --experiment random=results/input_mask_fixed_m20_random_sc_seed3_c200 \
  --experiment cycle=results/input_mask_fixed_m20_cycle_chords_seed3_c200 \
  --experiment hub=results/input_mask_fixed_m20_hub_spoke_seed63_c200 \
  --seeds 1,2,3,4,5 \
  --submit_missing \
  --finalize_if_complete \
  --output_md results/input_mask_topology_report.md \
  --output_json results/input_mask_topology_report.json \
  --max-concurrent 16 \
  --dry-run
```

Then submit missing retrains only:

```bash
python3 recover_essential_inputmask_retrains.py \
  --experiment random=results/input_mask_fixed_m20_random_sc_seed3_c200 \
  --experiment cycle=results/input_mask_fixed_m20_cycle_chords_seed3_c200 \
  --experiment hub=results/input_mask_fixed_m20_hub_spoke_seed63_c200 \
  --seeds 1,2,3,4,5 \
  --submit_missing \
  --max-concurrent 16
```

After those jobs finish, finalize:

```bash
python3 recover_essential_inputmask_retrains.py \
  --experiment random=results/input_mask_fixed_m20_random_sc_seed3_c200 \
  --experiment cycle=results/input_mask_fixed_m20_cycle_chords_seed3_c200 \
  --experiment hub=results/input_mask_fixed_m20_hub_spoke_seed63_c200 \
  --seeds 1,2,3,4,5 \
  --finalize_if_complete \
  --output_md results/input_mask_topology_report.md \
  --output_json results/input_mask_topology_report.json
```

Then run the final non-mutating completion gate:

```bash
python3 verify_topology_completion.py \
  --experiment random=results/input_mask_fixed_m20_random_sc_seed3_c200 \
  --experiment cycle=results/input_mask_fixed_m20_cycle_chords_seed3_c200 \
  --experiment hub=results/input_mask_fixed_m20_hub_spoke_seed63_c200 \
  --seeds 1,2,3,4,5 \
  --report_kind input_mask \
  --report_md results/input_mask_topology_report.md \
  --report_json results/input_mask_topology_report.json
```

The strict input-mask audit/finalizer should stop before report overwrite if
retrains are incomplete. The completion verifier should pass before
interpreting or sharing the scientific result.

For physical essential subgraphs, first use the physical guarded finalizer to
refresh `essential_input50_retrain/topology_seed_aggregates.csv` and
`essential_input50/retrain_comparison.json`:

```bash
python3 recover_essential_physical_retrains.py \
  --experiment random=results/input_mask_fixed_m20_random_sc_seed3_c200 \
  --experiment cycle=results/input_mask_fixed_m20_cycle_chords_seed3_c200 \
  --experiment hub=results/input_mask_fixed_m20_hub_spoke_seed63_c200 \
  --seeds 1,2,3,4,5 \
  --submit_missing \
  --max-concurrent 16
```

After any missing physical retrain jobs finish, finalize through the same
guarded path:

```bash
python3 recover_essential_physical_retrains.py \
  --experiment random=results/input_mask_fixed_m20_random_sc_seed3_c200 \
  --experiment cycle=results/input_mask_fixed_m20_cycle_chords_seed3_c200 \
  --experiment hub=results/input_mask_fixed_m20_hub_spoke_seed63_c200 \
  --seeds 1,2,3,4,5 \
  --finalize_if_complete
```

Then generate, verify, and interpret the consolidated research report:

```bash
python3 finalize_topology_research_report.py \
  --experiment random=results/input_mask_fixed_m20_random_sc_seed3_c200 \
  --experiment cycle=results/input_mask_fixed_m20_cycle_chords_seed3_c200 \
  --experiment hub=results/input_mask_fixed_m20_hub_spoke_seed63_c200 \
  --seeds 1,2,3,4,5 \
  --output_md results/topology_research_report.md \
  --output_json results/topology_research_report.json
```

The report finalizer runs `make_topology_research_report.py`,
`verify_topology_completion.py --report_kind research`, and
`interpret_topology_report.py --report_kind research` in that order. The
research verifier audits both input-mask and physical-essential retrain layouts
and requires both layouts to be present in the report JSON.

For the focused input-mask report, create the conservative H0/H1 interpretation
after verification passes:

```bash
python3 interpret_topology_report.py \
  --report_json results/input_mask_topology_report.json \
  --report_kind input_mask \
  --output_md results/input_mask_topology_interpretation.md \
  --output_json results/input_mask_topology_interpretation.json
```

The recovery wrapper runs this interpretation step automatically after a
successful `--finalize_if_complete` unless `--skip_interpretation` is supplied.

## Not Yet Complete

The project is not complete until cluster-side retrains and final reports are
completed, verified, and interpreted. Local tests prove the analysis/control
plane is wired, but they are not evidence that final trained CRN results exist
or answer the scientific question.
