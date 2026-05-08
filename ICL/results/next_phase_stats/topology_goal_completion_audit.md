# Topology-ICL Goal Completion Audit

Updated after commit `ba06656` (`Add expanded hard mechanism evidence`).

This audit maps the active research objective to concrete repository artifacts. It is intentionally conservative: broad theory-building items are marked complete only when the current branch contains inspectable evidence, while long-run scientific extensions remain scoped as future work.

## Current Verification

- Branch state: local `topology` is synchronized with `origin/topology` at `ba06656`.
- Remote state: Engaging `~/repos/topology` pushed `ba06656d372847b0e63920cc47e63873d4898287` to `statmech/topology`.
- Strict verifier:

```bash
python3 ICL/verify_topology_completion.py \
  --experiment next=ICL/results/next_phase_stats \
  --report_md ICL/results/next_phase_stats/next_phase_evidence_report.md \
  --report_json ICL/results/next_phase_stats/next_phase_evidence_report.json \
  --report_kind next_phase \
  --require_expanded_followups
```

Result: `Topology completion verification passed`.

- Expanded hard follow-up counts committed under `ICL/results/expanded_hard_sweeps`:

| Regime | `results.pkl` | `model.pt` | `mechanism_metrics.json` | `causal_interventions.json` |
| --- | ---: | ---: | ---: | ---: |
| `n4_m6_N3_D2` | 60 | 60 | 60 | 60 |
| `n5_m8_N3_D2` | 60 | 60 | 60 | 60 |
| `n5_m12_N3_D2` | 60 | 60 | 60 | 60 |
| Total | 180 | 180 | 180 | 180 |

- Main report: `ICL/results/next_phase_stats/next_phase_evidence_report.md` now reports hard follow-up mechanism and causal counts as `60/60/60` for all three hard regimes.
- Interpretation guardrails remain in force: headline claims use `test_novel_classes`, run rows are treated as seeds nested inside topology/mask groups, causal scrambling is interpreted only for sufficiently accurate baselines, and branch-margin capacity is treated as a proxy rather than a solved capacity law.

## Completion Criteria Audit

| Objective clause | Required evidence for completion | Current inspected evidence | Completion judgment |
| --- | --- | --- | --- |
| Audit the implementation | Orientation, bias, strong-connectivity, mask-selection, and novel-class metric checks must be documented and enforced by tests or verifiers. | `TOPOLOGY_THEORY_AUDIT.md`, `audit_topology_artifacts.py`, `verify_topology_completion.py`, and audit/verifier tests are present. | Complete for committed artifacts. |
| Upgrade inference to hierarchical/clustered analysis | Analysis must avoid treating nested training seeds as fully independent and must report group-aware uncertainty. | `clustered_topology_inference.py`, topology-cluster bootstrap, family-cluster bootstrap, random-intercept decomposition, and next-phase report tables are present. | Complete for tested regimes. |
| Expand matched topology sweeps | Fixed-count studies beyond the original `N_n=6,m=20` setting must be present with fixed trainable-count controls and novel-class ICL outputs. | Hard regimes `n4_m6_N3_D2`, `n5_m8_N3_D2`, and `n5_m12_N3_D2` each have 60 trained runs plus mechanism and causal follow-ups; earlier expanded pilots remain summarized. | Complete for the current hard-pilot phase; not the full future grid. |
| Replace `d_rel` with branch-margin/tree-polytope probes | A branch-separation capacity proxy must be implemented and compared against `d_rel`. | `branch_margin_capacity.py` includes rank-weighted, tropical rooted-tree random-feature, root-specific tree-polytope support, and sampled normal-fan branch/tree diagnostics; hard-regime capacity regressions are reported. | Partial by design: useful proxies exist, but not a final capacity theorem. |
| Test held-out-family generalization | Regression must leave out graph families rather than only individual topology instances. | Hard clustered reports use derived graph-family labels; verifier checks family-holdout fields and report sections expose held-out-family results. | Complete for current hard pilots; limited by small family counts. |
| Validate mechanisms with active-tree diagnostics | Expanded hard regimes must have post-training mechanism metrics, not only structural predictors. | All three hard regimes now have 60 `mechanism_metrics.json` files and summary/regression artifacts. | Complete for current hard pilots. |
| Validate mechanisms with causal tree-branch interventions | Expanded hard regimes must have causal intervention summaries testing branch/projection alignment. | All three hard regimes now have 60 `causal_interventions.json` files plus per-regime CSV and summary artifacts. | Complete for current hard pilots. |
| Validate mechanisms with matched motifs | Extracted motifs must be compared against matched controls rather than only raw sparse masks. | `make_matched_motif_controls.py`, `compare_matched_motif_controls.py`, and matched-control report sections are present. | Complete for original fixed-count backbones; not a universal motif law. |
| Preserve first-order scope | Claims must be limited to first-order CRNs with exponential input-dependent rates unless separate nonlinear theory is developed. | `next_phase_evidence_report.md/json` explicitly scope claims to first-order CRNs and the matrix-tree theorem. | Complete. |

## Scientific State

The repository now supports the scoped statement:

> In the tested first-order fixed-count regimes, topology-associated structural and functional variables explain residual novel-class ICL variation that raw count does not explain.

It does not yet support a stronger universal claim such as:

> `d_rel` is the capacity law for all first-order CRN ICL.

or:

> The first-order matrix-tree topology theory transfers automatically to autocatalytic or WTA CRNs.

## Remaining Research Gaps

1. Expand the matched sweep grid beyond the current hard pilots.
2. Replace the current branch-margin/normal-fan proxies with a sharper tree-polytope branch-coverage theory.
3. Use larger held-out-family sets to reduce dependence on the small number of current graph families.
4. Treat matched motif results as evidence about feasible small physical subgraphs, not as proof that extracted motifs uniquely dominate matched controls.
