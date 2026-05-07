# Topology-ICL Status Audit

Current branch: `topology`

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
| Branch-aware comparison-capacity metrics | `comparison_branch_common_d_rel_*`, `comparison_branch_d_rel_*`, `comparison_branch_input_overlap_*`, and `comparison_branch_input_count_*` in `topology_metrics.py` | Implemented, collected, regressed, and reported |
| Controlled physical topology libraries at fixed `N_n,N_c,D,m` | `make_topology_library.py`, `submit_topology_library_sweep.py` | Implemented and dry-run/tested |
| Controlled fixed-physical-graph input-mask libraries | `make_input_mask_library.py`, `make_input_mask_report.py` | Implemented and tested |
| Run-level and topology-level regressions against raw count and tree geometry | `regress_topology_results.py`, `aggregate_topology_seeds.py` | Implemented and tested |
| Novel-class ICL remains the primary collected metric | `collect_topology_results.py` field `test_novel_classes`; report defaults | Implemented |
| Expressivity vs trainability split | `aggregate_topology_seeds.py` outputs `target_max`, `target_mean`, `target_std` | Implemented and tested |
| Post-training active tree/root, branch MI, margins, sensitivities, and ablations | `topology_analysis.py`, `analyze_topology_model.py`, `collect_mechanism_results.py`, `summarize_topology_mechanisms.py` | Implemented; execution requires Torch-enabled trained runs |
| Essential physical subgraph extraction and retraining | `extract_essential_subgraphs.py`, `compare_essential_retrains.py` | Implemented and tested |
| Essential input-mask extraction and retraining | `extract_essential_input_masks.py`, `finalize_essential_inputmask_retrains.py`, `recover_essential_inputmask_retrains.py` | Implemented and tested |
| Consolidated research report | `make_topology_research_report.py` | Implemented; supports both `essential_input50` and `essential_inputmask50` layouts |
| Artifact audit and interrupted-array recovery | `audit_topology_artifacts.py`, `recover_essential_inputmask_retrains.py` | Implemented and tested |

## Verification Gates

Local, dependency-light checks:

```bash
python3 -m unittest discover -s ICL/tests
python3 -m py_compile $(find ICL -name '*.py' -not -path '*/__pycache__/*')
git diff --check
```

As of the latest local run, the unittest suite has 42 tests and passes. Local
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

The strict audit/finalizer should stop before report overwrite if retrains are
incomplete.

## Not Yet Complete

The project is not complete until cluster-side retrains and final reports are
completed and inspected. Local tests prove the analysis/control plane is wired,
but they are not evidence that final trained CRN results exist or answer the
scientific question.
