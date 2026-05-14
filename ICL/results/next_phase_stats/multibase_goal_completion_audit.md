# Multibase Goal Completion Audit

## Objective

Complete `ICL/MARKOV_ICL_NEXT_PHASE_GOAL.md` for the multi-base normal-fan / tree-count phase using the existing Engaging tmux pane `icl:13.2`, without launching a broad new sweep.

## Checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| Track 0 live-job audit | `multibase_live_job_audit.md/json`; 185 expected tasks, 185 completed, 37/37 groups complete, no missing task IDs | complete |
| Track 1 consume multi-base results | `multibase_exact_control_results_report.md/json`, group table CSV, pairwise contrast CSV | complete |
| Grouped inference, not seed-independent inference | result report uses 37 topology-group rows and grouped LOO models; seed rows are aggregated into mean/best/std outcomes | complete |
| Arm A fixed-tree-count normal-fan contrast | pairwise contrast CSV and report include 11 Arm A pairs | complete |
| Arm B variable-tree-count matched-normal-fan contrast | pairwise contrast CSV and report include 9 Arm B pairs | complete |
| Track 2 cross-root / decoder-aware metrics | `ICL/cross_root_decoder_contrast_metrics.py`; `cross_root_decoder_contrast_reanalysis.md/json` | complete |
| Compare cross-root metrics to normal fan, tree count, and gamma | result report model table includes cross-root + tree-count + normal-fan + base model | complete |
| Track 3 retrospective/prospective tree-difference diagnosis | `retrospective_vs_prospective_tree_difference_diagnostic.md/json` and feature table CSV | complete |
| Track 4 mechanism follow-up | targeted 8-model mechanism follow-up; `multibase_mechanism_followup_report.md/json`, diagnostics CSV, scramble CSV | complete |
| Track 5 gamma diagnostic | `gamma_multibase_diagnostic_report.md/json` and `gamma_multibase_rows.json`; gamma remains diagnostic | complete |
| Final synthesis | `post_multibase_exact_control_synthesis.md/json` | complete |
| Thermodynamics delayed | synthesis explicitly states thermodynamics remains untested and no `F_max` claim is supported | complete |
| Verification | `python3 -m py_compile ICL/cross_root_decoder_contrast_metrics.py ICL/multibase_exact_control_analysis.py`; all required artifacts are nonempty | complete |

## Main Results

The multi-base run completed all 185 training tasks over 37 topology groups. In grouped LOO, `tree_count_plus_base` was the best mean-ICL and best-seed model in this analysis (`R2 = 0.561` for mean novel-class ICL and `R2 = 0.834` for best-seed ICL). Arm B, which varies tree count at matched normal-fan score, had a positive paired mean-ICL contrast (`+4.067`, bootstrap 95% CI `[0.876, 7.364]`). Arm A, which varies normal-fan score at nearly fixed tree count, was weaker (`+1.193`, 95% CI `[-1.291, 4.007]`).

Targeted post-training mechanism follow-up remained strong: 8 selected models produced 64 causal-scramble rows, with mean accuracy drops of `-79.562` for context-block shuffle, `-55.031` for stat-preserving projection scramble, `-54.406` for stat-preserving branch-alignment scramble, and `-48.187` for decoder-root permutation.

## Completion Judgment

The required deliverables in `MARKOV_ICL_NEXT_PHASE_GOAL.md` are present and populated. The only caveat is interpretive: the positive multibase signal is currently strongest for rooted-tree abundance plus base controls, not for a universal scalar topology law. The synthesis states that distinction explicitly.
