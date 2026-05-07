# Topology-ICL Goal Completion Audit

Generated: 2026-05-07. The hard-pilot evidence was added in commit `22de6ec` (`Add hard topology pilot evidence`). The branch now also includes follow-up orchestration, capacity-probe, and inference-verification updates: explicit submitted-job Python control, causal finalizer orchestration, guarded hard-pilot follow-up orchestration, tropical rooted-tree random-feature capacity fields, derived graph-family holdout, family-cluster bootstrap diagnostics, and a targeted next-phase report refresh utility for cluster follow-up artifacts.

This audit maps the active research objective to concrete repository artifacts. It is deliberately conservative: a requirement is marked complete only when the current branch contains inspectable evidence for it.

## Current Verification

- Branch: `topology`, synchronized with `origin/topology` through the follow-up orchestration commits.
- Hard-pilot artifact paths on branch: `889` files under `ICL/results/expanded_hard_libraries`, `ICL/results/expanded_hard_sweeps`, and `ICL/results/expanded_hard_stats`.
- Raw-output exclusion check: no `results.pkl`, `model.pt`, `__pycache__`, or `_array_meta` paths under the committed hard-pilot artifact set.
- Local unit tests: `python3 -m unittest discover -s ICL/tests`, `128` tests passed.
- Next-phase report verifier: `python3 ICL/verify_topology_completion.py --experiment next=ICL/results/next_phase_stats --report_md ICL/results/next_phase_stats/next_phase_evidence_report.md --report_json ICL/results/next_phase_stats/next_phase_evidence_report.json --report_kind next_phase`, passed. The verifier now requires hard-regime reports to expose `derived_graph_family` holdout and family-cluster bootstrap metrics. An optional stricter gate, `--require_expanded_followups`, intentionally fails on the current report until hard-pilot mechanism and causal follow-up counts are nonzero.
- Main next-phase report: `ICL/results/next_phase_stats/next_phase_evidence_report.md`, refreshed with hard-regime rank-weighted and tropical rooted-tree random-feature capacity metrics. Hard-regime held-out rows now use derived graph-family labels rather than full topology-instance names, and hard-regime uncertainty tables include both topology-cluster and family-cluster bootstrap deltas. `ICL/run_expanded_hard_followups.py` now provides a guarded submit/collect/refresh/strict-verify path for the hard regimes and refuses to finalize source-light roots with no raw `results.pkl` files. `ICL/refresh_next_phase_report.py` can update only selected labeled sections after new cluster artifacts land, while preserving existing sections if the active checkout lacks raw per-run files.

## Completion Criteria Audit

The active objective is not a single software feature. It requires a
statistically and theoretically defensible first-order CRN topology theory with
inspectable evidence for every link in the chain below. Passing unit tests or a
non-strict report verifier is not sufficient by itself, because those checks do
not create missing mechanism or causal evidence for the expanded hard regimes.

| Objective clause | Required evidence for completion | Current inspected evidence | Completion judgment |
| --- | --- | --- | --- |
| Audit the implementation | Orientation, bias, strong-connectivity, selection, and novel-class metric checks must be documented and enforced by verifiers/tests. | `TOPOLOGY_THEORY_AUDIT.md`, `audit_topology_artifacts.py`, `verify_topology_completion.py`, and their tests are present. | Complete for committed artifacts. |
| Upgrade inference to hierarchical/clustered analysis | Group-level or clustered analysis must avoid treating nested seeds as fully independent and must report uncertainty beyond naive run-level regression. | `clustered_topology_inference.py`, topology-cluster bootstrap, family-cluster bootstrap, random-intercept residual decomposition, and next-phase report tables are present. | Complete for tested regimes. |
| Expand matched topology sweeps | There must be matched fixed-count studies beyond the original `N_n=6,m=20` regime, with fixed trainable-count controls and novel-class ICL outputs. | Hard-pilot summaries exist for `n4_m6_N3_D2`, `n5_m8_N3_D2`, and `n5_m12_N3_D2`; each has `60` topology rows in `run_expanded_hard_followups.py --status`. | Partial: pilot coverage exists, but not the full planned grid. |
| Replace `d_rel` with branch-margin/tree-polytope probes | A branch-separation capacity proxy or theory must be implemented and compared against `d_rel`. | `branch_margin_capacity.py` includes rank-weighted and tropical rooted-tree random-feature diagnostics; hard-regime branch-capacity regressions are reported. | Partial: useful lower-bound probes exist, but this is not yet a solved tree-polytope capacity law. |
| Test held-out-family generalization | Regression must leave out graph families, not individual topology-instance labels, and report held-out-family performance. | Hard clustered reports use `family_col == "derived_graph_family"`; verifier checks this and Markdown exposes the family holdout. | Complete for current hard pilots; still limited by small family counts. |
| Validate mechanisms with matched motifs | Extracted motifs must be compared against matched controls rather than only raw sparse masks. | `make_matched_motif_controls.py`, `compare_matched_motif_controls.py`, and matched-control report sections are present. | Complete for original fixed-count backbones; not a universal motif law. |
| Validate mechanisms with active-tree diagnostics | Expanded hard regimes must have post-training mechanism metrics, not only pre-training structural predictors. | Original regimes have mechanism diagnostics. Hard status currently reports `mechanisms=0` for all three hard regimes. | Incomplete. |
| Validate mechanisms with causal tree-branch interventions | Expanded hard regimes must have causal intervention summaries showing whether tree/branch alignment matters after training. | Original regimes have causal summaries. Hard status currently reports `causal=0` for all three hard regimes. | Incomplete. |
| Preserve first-order scope | Claims must be limited to first-order CRNs with exponential input-dependent rates unless separate nonlinear theory is developed. | Next-phase report scope and interpretation guardrails explicitly limit the claim. | Complete. |

The strict completion gate encodes the two currently missing expanded-regime
mechanism requirements:

```bash
python3 ICL/verify_topology_completion.py \
  --experiment next=ICL/results/next_phase_stats \
  --report_md ICL/results/next_phase_stats/next_phase_evidence_report.md \
  --report_json ICL/results/next_phase_stats/next_phase_evidence_report.json \
  --report_kind next_phase \
  --require_expanded_followups
```

Current strict-gate failure:

| Expanded hard regime | Training rows | Mechanism files | Causal files | Status |
| --- | ---: | ---: | ---: | --- |
| `hard_n4_m6_N3_D2` | 60 | 0 | 0 | blocked on raw cluster run outputs |
| `hard_n5_m8_N3_D2` | 60 | 0 | 0 | blocked on raw cluster run outputs |
| `hard_n5_m12_N3_D2` | 60 | 0 | 0 | blocked on raw cluster run outputs |

The next concrete completion step must run on the Engaging worktree containing
the raw `results.pkl`/`model.pt` files and a Torch-enabled Python:

```bash
cd ~/repos/topology/ICL
git fetch statmech topology
git checkout topology
git pull --ff-only statmech topology

python3 run_expanded_hard_followups.py --status

python3 run_expanded_hard_followups.py \
  --submit_followups \
  --device cpu \
  --job_python "$TOPOLOGY_PYTHON" \
  --max-concurrent 20

# After the SLURM arrays finish:
python3 run_expanded_hard_followups.py \
  --collect_followups \
  --refresh_report \
  --strict_verify \
  --device cpu \
  --job_python "$TOPOLOGY_PYTHON"
```

## Checklist

| Requirement | Current artifact/evidence | Status | Notes |
| --- | --- | --- | --- |
| Audit implementation details before strengthening claims | `ICL/verify_topology_completion.py`, `ICL/audit_topology_artifacts.py`, tests under `ICL/tests/test_audit_topology_artifacts.py` and `ICL/tests/test_verify_topology_completion.py` | Complete for current report artifacts | The verifier now includes a `next_phase` report kind and passes on `next_phase_evidence_report.md/json`. |
| Upgrade inference to group-aware / clustered analysis | `ICL/clustered_topology_inference.py`; `ICL/results/next_phase_stats/next_phase_evidence_report.md`; `ICL/results/expanded_hard_stats/*clustered_inference.json` | Complete for tested regimes | Report includes group LOO R2 and cluster-bootstrap delta R2 for original, pilot, and hard regimes. Hard regimes now also include a two-stage family-cluster bootstrap using derived graph-family labels. |
| Expand matched topology sweeps beyond the original fixed-count regime | `ICL/results/expanded_hard_sweeps/n4_m6_N3_D2`, `n5_m8_N3_D2`, `n5_m12_N3_D2`; earlier `n5_m7_N2_D1`, `n5_m12_N2_D1` summaries in the report | Partial | Adds harder `N_c=3,D=2` pilots and sparse/intermediate/dense regimes, but does not yet cover the full planned grid `N_n in {4,5,6,7,8}`, `N_c in {2,3}`, `D in {1,2}`. |
| Replace raw `d_rel` with branch-margin / tree-polytope capacity probes | `ICL/branch_margin_capacity.py`; `ICL/collect_branch_margin_capacity.py`; `ICL/results/*branch_margin_capacity*`; hard summaries in `ICL/results/expanded_hard_libraries/*/branch_margin_capacity_summary.json`; tropical models in `ICL/results/expanded_hard_stats/*branch_capacity_clustered_inference.json` | Partial | Probe now includes rank-weighted fields and a tropical rooted-tree random-feature separability diagnostic. The tropical diagnostic gives non-flat hard-regime bootstrap signals, but it remains a sampled lower-bound proxy rather than a solved branch-margin/tree-polytope capacity theory. |
| Test held-out-family / held-out-backbone generalization | `heldout R2` columns in `ICL/results/next_phase_stats/next_phase_evidence_report.md`; clustered inference JSONs; `ICL/clustered_topology_inference.py --derive_graph_family` | Partial | Hard-regime heldout now leaves out graph-family labels parsed from topology names (`cycle_chords`, `random_sc`, etc.) instead of individual topology instances. Coverage remains limited by the small number of families and groups per regime. |
| Validate mechanisms through active-tree/projection diagnostics | `ICL/analyze_topology_model.py`; `ICL/summarize_topology_mechanisms.py`; projection diagnostics in `ICL/results/topology_research_report.md`; next-phase report notes hard pilots have `mechanisms=0`; `ICL/finalize_topology_sweep.py` and `ICL/TOPOLOGY_README.md` now provide an explicit-Python hard-pilot submission path; `verify_topology_completion.py --require_expanded_followups` enforces nonzero hard mechanism counts | Partial | Original fixed-count runs have projection-alignment diagnostics; expanded hard pilots do not yet have mechanism decomposition. The blocker is execution in a Torch-enabled cluster environment. |
| Validate mechanisms through causal tree-branch alignment interventions | `ICL/causal_topology_interventions.py`; `ICL/results/next_phase_stats/*causal_interventions_summary.json`; causal section in `next_phase_evidence_report.md`; `ICL/finalize_topology_sweep.py` now submits and collects causal reports; `verify_topology_completion.py --require_expanded_followups` enforces nonzero hard causal counts | Complete for original fixed-count backbones; partial for expanded pilots | Original random/cycle/hub runs show large accuracy collapses under projection/context/decoder scrambling. Expanded hard pilots have no causal runs yet. |
| Add matched essential-motif baselines | `ICL/make_matched_motif_controls.py`; `ICL/compare_matched_motif_controls.py`; `ICL/results/input_mask_fixed_m20_*_c200/essential_input50_matched_controls/*`; matched motif section in `next_phase_evidence_report.md` | Complete for original fixed-count backbones | Current evidence weakens the unique extracted-motif superiority claim: matched controls are comparable or better in several cases. |
| State scientifically scoped conclusions only for first-order CRNs | `ICL/results/next_phase_stats/next_phase_evidence_report.md` scope/header and interpretation guardrails | Complete | Report explicitly limits claims to first-order CRNs with exponential input-dependent rates. |

## Current Scientific State

The current branch supports the statement:

> In the tested first-order fixed-count regimes, topology-associated structural and functional variables explain residual novel-class ICL variation that raw count does not explain.

The current branch does not yet support the stronger statement:

> A complete topology capacity law for first-order CRN ICL has been established.

The strongest remaining gaps are:

1. Extend matched sweeps across the full planned parameter grid, not just selected pilot regimes.
2. Replace the current branch-margin proxy with a more discriminating tree-polytope / branch-coverage capacity probe.
3. Run mechanism decomposition and causal interventions on the expanded hard pilots.
4. Extend the verifier further if future reports add new evidence sections beyond the current next-phase schema.
