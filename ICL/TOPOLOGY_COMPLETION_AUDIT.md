# Topology-ICL Completion Audit

Date: 2026-05-07

Branch: `topology`

Latest inspected commits:

```text
e586667 Harden input-mask provenance and recovery
c96e32e Normalize branch metric source loaders
d5b20bb Propagate aggregate metric provenance
00f738d Track common branch metric provenance
f57b2bb Validate essential mask audit references
b0ed933 Clarify pooled retrain reporting
6cac50f Harden essential retrain finalization
5ca545e Backfill common branch metrics in reports
```

## Objective Restated As Deliverables

Build a first-order CRN topology research framework that can test whether
reaction topology controls in-context learning beyond raw trainable degree
count. The concrete deliverables are:

1. A first-order topology-controlled CRN implementation with explicit physical
   graph and input-encoding mask support.
2. Matrix-tree structural metrics that expose relative spanning-tree projection
   geometry, including branch-aware comparison capacity.
3. Controlled topology and input-mask libraries at matched raw counts.
4. Training, collection, aggregation, and regression scripts centered on
   novel-class ICL accuracy.
5. Post-training mechanism analysis: active trees/roots, branch margins,
   mutual information, tree-projection alignment, sensitivities, ablations, and
   essential subgraphs/masks.
6. Cluster orchestration, recovery, audit, and report-generation scripts that
   are safe around interrupted SLURM arrays and other agents' work.
7. Actual cluster-side trained results and final reports that answer the
   scientific question.

Completion requires all seven deliverables. The local implementation can be
complete while the scientific project remains incomplete if trained cluster
artifacts and reports are missing.

## Evidence Inspected

Local commands inspected in this audit:

```bash
git status --short --branch
git log --oneline -8
tmux list-panes -t icl:13 -F '#{pane_index} #{pane_active} #{pane_current_command} #{pane_current_path} #{pane_title}'
tmux capture-pane -t icl:13.2 -p -S -24
find ICL -maxdepth 1 -type f -name '*.py' -o -name '*.md' | sort
find ICL/tests -maxdepth 1 -type f -name 'test_*.py' | sort
find ICL/results -maxdepth 2 -type f 2>/dev/null | head -80
python3 -m unittest discover -s ICL/tests
python3 -m py_compile $(find ICL -name '*.py' -not -path '*/__pycache__/*')
git diff --check
python3 verify_topology_completion.py ... --report_kind input_mask --report_md ... --report_json ...
python3 verify_topology_completion.py ... --report_kind research --report_md ... --report_json ...
```

Observed state:

- Local branch is clean and tracking `origin/topology`.
- Local unit suite has 67 tests and passes.
- Python syntax compilation passes.
- `git diff --check` passes.
- Local `ICL/results` contains only older `markov_icl_gmm_*.pt` files, not the
  topology/input-mask sweep reports needed for the current project.
- The only allowed cluster pane, `icl:13.2`, is local after an SSH disconnect
  and password prompt, not an active `login005` shell.

## Prompt-To-Artifact Checklist

| Requirement from research plan | Concrete artifact or evidence | Status |
| --- | --- | --- |
| Use first-order CRNs as the mathematically clean foundation | `models/topology_markov_icl.py`, `run_topology_icl.py`, `topology_metrics.py` | Implemented locally |
| Use exponential rate encoding so matrix-tree projection theory is exact | `run_topology_icl.py`, `models/topology_markov_icl.py` | Implemented locally; cluster smoke still required |
| Represent physical reaction topology as a directed graph | `topology_metrics.py`, `make_topology_library.py`, `run_topology_icl.py --edge_json` | Implemented and tested |
| Keep physical topology separate from input-encoding topology | `input_mask_utils.py`, `make_input_mask_library.py`, `run_topology_icl.py --input_mask_json` | Implemented and tested |
| Track functional post-training topology | `topology_analysis.py`, `analyze_topology_model.py`, `collect_mechanism_results.py` | Implemented; requires trained Torch runs |
| Enforce strong connectivity for first-order graph comparisons | `topology_metrics.py`, `make_topology_library.py`, `extract_essential_subgraphs.py` | Implemented and tested |
| Compute rooted spanning trees and matrix-tree metrics | `topology_metrics.py` | Implemented and tested |
| Compute relative tree-difference rank and spectra | `rank_D`, `d_rel`, `effective_rank_D`, `condition_number_D` fields | Implemented and tested |
| Compute masked input-encoding relative geometry | `effective_rank_D_masked`, `condition_number_D_masked`, `input_*` metrics | Implemented and tested |
| Add strict branch-aware common subspace rank | `comparison_branch_common_d_rel_*` | Implemented, collected, aggregated, reported |
| Preserve provenance for common-branch fallback metrics | `comparison_branch_common_d_rel_source` in collection, aggregation, reports, essential comparisons | Implemented and tested |
| Add input-overlap branch support metric | `comparison_branch_input_overlap_*` | Implemented, collected, aggregated, reported |
| Preserve provenance for input-overlap fallback metrics | `comparison_branch_input_overlap_source` | Implemented and tested |
| Generate controlled physical topology libraries | `make_topology_library.py` | Implemented and tested |
| Generate controlled fixed-physical input-mask libraries | `make_input_mask_library.py` | Implemented and tested |
| Train matched systems through SLURM arrays | `submit_topology_phase1.py`, `submit_topology_library_sweep.py` | Submitters implemented and dry-run tested; actual cluster execution incomplete |
| Use novel-class ICL accuracy as the primary metric | `collect_topology_results.py` field `test_novel_classes`; report defaults | Implemented |
| Keep train/validation accuracy secondary | Collection keeps `train_acc_final`, `val_acc_final`, and `test_novel_classes` separately | Implemented |
| Aggregate mean, max, and seed variance | `aggregate_topology_seeds.py` outputs `target_mean`, `target_max`, `target_std` | Implemented and tested |
| Regress accuracy on raw counts and topology metrics | `regress_topology_results.py`, `aggregate_topology_seeds.py`, report builders | Implemented and tested |
| Separate expressivity from trainability | best-seed, mean-seed, and seed-std aggregate fields | Implemented; scientific interpretation awaits cluster results |
| Compute active tree/root assignments | `topology_analysis.py`, `analyze_topology_model.py` | Implemented; requires trained runs |
| Compute branch-to-tree/root MI and purity | `collect_mechanism_results.py`, `summarize_topology_mechanisms.py` | Implemented and tested with synthetic data |
| Compute branch margins and branch accuracies | `collect_mechanism_results.py`, `make_input_mask_report.py`, `make_topology_research_report.py` | Implemented and tested |
| Compute tree-projection alignment to comparison directions | `topology_analysis.py`, `collect_mechanism_results.py` | Implemented; requires trained runs |
| Compute edge sensitivities and ablations | `topology_analysis.py`, `analyze_topology_model.py --ablate_input --ablate_physical` | Implemented; requires trained runs |
| Extract essential physical subgraphs | `extract_essential_subgraphs.py` | Implemented and tested |
| Extract essential input masks while preserving physical graph | `extract_essential_input_masks.py` | Implemented and tested |
| Retrain extracted motifs/masks from scratch | `submit_topology_library_sweep.py`, `finalize_essential_inputmask_retrains.py`, `compare_essential_retrains.py` | Orchestration implemented and tested; cluster retrains incomplete |
| Do not conflate input-coupling ablation with physical edge ablation | Separate ablation fields and `extract_essential_input_masks.py` vs `extract_essential_subgraphs.py` | Implemented |
| Avoid clustering as the primary hypothesis | Reports emphasize regressions, branch metrics, mechanisms, and essential motifs | Implemented in report structure |
| Keep nonlinear autocatalytic/WTA outside first-order tree claims | Current implementation and docs are scoped to first-order topology | Satisfied by scope |
| Provide final consolidated report | `make_topology_research_report.py`, `make_input_mask_report.py` | Implemented; final cluster reports missing |
| Provide artifact audit and safe recovery | `audit_topology_artifacts.py`, `recover_essential_inputmask_retrains.py` | Implemented and tested |
| Audit physical essential-subgraph retrain artifacts | `audit_topology_artifacts.py --essential_kind physical` | Implemented and tested |
| Verify final report and artifact consistency in one non-mutating command | `verify_topology_completion.py --report_kind input_mask` and `--report_kind research` | Implemented and tested; research mode audits both input-mask and physical-essential layouts |
| Avoid interfering with other agents on Engaging | Current blocker documents that only `icl:13.2` should be used | Satisfied locally; cluster work paused |

## Verification Coverage

The local test suite covers the analysis/control plane:

- topology metrics and branch ranks,
- input-mask validation,
- topology and input-mask library generation,
- collection and regression scripts,
- seed aggregation and provenance fallback,
- mechanism collection summaries,
- essential subgraph and input-mask extraction,
- retrain comparison,
- report generation,
- artifact audit,
- SLURM dry-run wrappers,
- guarded essential input-mask recovery/finalization.

The local test suite does not prove:

- Torch training runs succeed on Engaging,
- novel-class ICL accuracies are scientifically meaningful,
- matrix-tree predictors explain trained model accuracy,
- post-training mechanism metrics have been computed for real trained models,
- essential subgraphs or input masks actually retrain successfully,
- final Markdown/JSON reports exist for the target cluster experiments.

## Required Cluster Evidence Still Missing

The following expected artifacts were not present locally and could not be
inspected through the allowed cluster pane:

```text
results/input_mask_fixed_m20_random_sc_seed3_c200
results/input_mask_fixed_m20_cycle_chords_seed3_c200
results/input_mask_fixed_m20_hub_spoke_seed63_c200
results/input_mask_topology_report.md
results/input_mask_topology_report.json
```

For each fixed-input-count experiment, completion requires:

- `topology_results.csv`
- `topology_seed_aggregates.csv`
- `mechanism_results.csv`
- `mechanism_summary.json`
- `essential_inputmask50/selected.csv`
- `essential_inputmask50/retrain_comparison.csv`
- `essential_inputmask50/retrain_comparison.json`
- `essential_inputmask50_retrain/topology_seed_aggregates.csv`
- exact retrain run directories for every selected `topology_id` and seed,
  each containing `results.pkl`, `topology_metrics.json`, and `config.json`.

The strict audit command that should pass before interpreting results is:

```bash
python3 audit_topology_artifacts.py \
  --experiment random=results/input_mask_fixed_m20_random_sc_seed3_c200 \
  --experiment cycle=results/input_mask_fixed_m20_cycle_chords_seed3_c200 \
  --experiment hub=results/input_mask_fixed_m20_hub_spoke_seed63_c200 \
  --seeds 1,2,3,4,5 \
  --require_source_results \
  --require_mechanisms \
  --require_essential_inputmask \
  --require_essential_retrains \
  --strict
```

For consolidated reports that include physical essential subgraphs, the physical
layout audit should also pass:

```bash
python3 audit_topology_artifacts.py \
  --experiment random=results/input_mask_fixed_m20_random_sc_seed3_c200 \
  --experiment cycle=results/input_mask_fixed_m20_cycle_chords_seed3_c200 \
  --experiment hub=results/input_mask_fixed_m20_hub_spoke_seed63_c200 \
  --seeds 1,2,3,4,5 \
  --essential_directory essential_input50 \
  --retrain_directory essential_input50_retrain \
  --essential_kind physical \
  --require_source_results \
  --require_mechanisms \
  --require_essential \
  --require_essential_retrains \
  --strict
```

## Completion Verdict

Not complete.

The local framework is substantially implemented and covered by dependency-light
tests, but the project objective asks whether topology controls real first-order
CRN ICL. That requires cluster-side trained artifacts, mechanism summaries,
essential retrains, and final reports. Those artifacts are not available through
the currently allowed pane.

The next required action is to restore `icl:13.2` to an active Engaging
`login005` shell, pull `origin/topology`, run the guarded recovery/finalization
commands from `TOPOLOGY_STATUS.md`, and inspect the resulting Markdown/JSON
reports before drawing scientific conclusions.
