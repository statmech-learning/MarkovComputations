# Topology-ICL Goal Completion Audit

Generated: 2026-05-07. The hard-pilot evidence was added in commit `22de6ec` (`Add hard topology pilot evidence`); this audit is tracked in the subsequent audit commit.

This audit maps the active research objective to concrete repository artifacts. It is deliberately conservative: a requirement is marked complete only when the current branch contains inspectable evidence for it.

## Current Verification

- Branch: `topology`, synchronized with `origin/topology` after the hard-pilot evidence commit.
- Hard-pilot artifact paths on branch: `889` files under `ICL/results/expanded_hard_libraries`, `ICL/results/expanded_hard_sweeps`, and `ICL/results/expanded_hard_stats`.
- Raw-output exclusion check: no `results.pkl`, `model.pt`, `__pycache__`, or `_array_meta` paths under the committed hard-pilot artifact set.
- Local unit tests: `python3 -m unittest discover -s ICL/tests`, `112` tests passed.
- Main next-phase report: `ICL/results/next_phase_stats/next_phase_evidence_report.md`.

## Checklist

| Requirement | Current artifact/evidence | Status | Notes |
| --- | --- | --- | --- |
| Audit implementation details before strengthening claims | `ICL/verify_topology_completion.py`, `ICL/audit_topology_artifacts.py`, tests under `ICL/tests/test_audit_topology_artifacts.py` and `ICL/tests/test_verify_topology_completion.py` | Partial | The audit tooling exists and is tested for older report schemas. The current next-phase report is not covered by this verifier schema. |
| Upgrade inference to group-aware / clustered analysis | `ICL/clustered_topology_inference.py`; `ICL/results/next_phase_stats/next_phase_evidence_report.md`; `ICL/results/expanded_hard_stats/*clustered_inference.json` | Complete for tested regimes | Report includes group LOO R2 and cluster-bootstrap delta R2 for original, pilot, and hard regimes. |
| Expand matched topology sweeps beyond the original fixed-count regime | `ICL/results/expanded_hard_sweeps/n4_m6_N3_D2`, `n5_m8_N3_D2`, `n5_m12_N3_D2`; earlier `n5_m7_N2_D1`, `n5_m12_N2_D1` summaries in the report | Partial | Adds harder `N_c=3,D=2` pilots and sparse/intermediate/dense regimes, but does not yet cover the full planned grid `N_n in {4,5,6,7,8}`, `N_c in {2,3}`, `D in {1,2}`. |
| Replace raw `d_rel` with branch-margin / tree-polytope capacity probes | `ICL/branch_margin_capacity.py`; `ICL/collect_branch_margin_capacity.py`; `ICL/results/*branch_margin_capacity*`; hard summaries in `ICL/results/expanded_hard_libraries/*/branch_margin_capacity_summary.json` | Partial | Probe is implemented and reported, but hard-regime results are flat (`0.892` linear accuracy), so it is not yet a discriminating capacity theory. |
| Test held-out-family / held-out-backbone generalization | `heldout R2` columns in `ICL/results/next_phase_stats/next_phase_evidence_report.md`; clustered inference JSONs | Partial | Heldout metrics are reported, but family coverage remains limited, especially for the original three-backbone fixed-count setting. |
| Validate mechanisms through active-tree/projection diagnostics | `ICL/analyze_topology_model.py`; `ICL/summarize_topology_mechanisms.py`; projection diagnostics in `ICL/results/topology_research_report.md`; next-phase report notes hard pilots have `mechanisms=0` | Partial | Original fixed-count runs have projection-alignment diagnostics; expanded hard pilots do not yet have mechanism decomposition. |
| Validate mechanisms through causal tree-branch alignment interventions | `ICL/causal_topology_interventions.py`; `ICL/results/next_phase_stats/*causal_interventions_summary.json`; causal section in `next_phase_evidence_report.md` | Complete for original fixed-count backbones | Original random/cycle/hub runs show large accuracy collapses under projection/context/decoder scrambling. Expanded hard pilots have no causal runs yet. |
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
4. Add a verifier for the next-phase report schema, since the existing completion verifier targets the older consolidated report schemas.
