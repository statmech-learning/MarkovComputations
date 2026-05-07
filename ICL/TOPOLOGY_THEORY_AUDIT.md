# First-Order Topology-ICL Theory Audit

Date: 2026-05-07

Scope: first-order CRNs with exponential input-dependent rates and explicit physical reaction graphs plus input-encoding masks. This audit checks whether the current implementation supports the matrix-tree/topology interpretation and where the current claims must be softened before broader sweeps.

## Verdict

The implementation is sound enough to extend the first-order topology-ICL study, with two required caveats:

1. Edge biases are trainable by default, so branch-margin and tree-polytope analyses must include tree intercepts, not only tree-sum projection directions.
2. Essential physical motif results are post-training, performance-conditioned, and not yet matched against random controls. They should be treated as mechanism evidence, not as a controlled proof that physical motifs intrinsically dominate input masks.

## Audit Table

| Item | Status | Evidence | Required interpretation |
| --- | --- | --- | --- |
| Tree orientation | Pass | `TopologyMatrixTreeMarkovICL` declares edges as `(source, target)` and uses `W[target, source]`; the implementation writes rates at `W_batch[..., target, source]` and subtracts column sums. `topology_metrics.py` uses the same `generator[target, source]` convention for determinant counts and weighted numerators. The explicit arborescence enumerator defines one outgoing edge per non-root and follows edges to the root. `test_topology_metrics.py` checks enumeration against cofactors for weighted rates. | The tree-sum interpretation is using in-arborescences toward the root, consistent with the steady-state convention. Keep orientation tests in the verification path for every future refactor. |
| Bias treatment | Pass with caveat | `base_log_rates` is an `nn.Parameter`; `run_topology_icl.py` defaults `--learn_base_rates` to true and passes it into the model. | The theoretical object is `beta_T + Theta_T^T z`, not just `Theta_T^T z`. Any branch-margin probe or tropical capacity argument must allow tree intercepts or explicitly run a frozen-bias control. |
| Strong connectivity | Pass | `make_topology_library.py` filters candidates whose metrics are not strongly connected. `run_topology_icl.py` rejects non-strongly-connected topologies unless `--allow_not_strongly_connected` is deliberately set for diagnostics. `extract_essential_subgraphs.py` defaults to `--ensure_strongly_connected` and greedily augments selected edges until strong connectivity is restored, otherwise it drops the candidate. | First-order matrix-tree comparisons should use only strongly connected physical graphs. Any run with `--allow_not_strongly_connected` is diagnostic and should not enter headline regressions. |
| Physical topology versus input-encoding topology | Pass | Physical edge JSON and input-mask JSON are loaded separately in `run_topology_icl.py`; `make_input_mask_library.py` holds the physical graph fixed while varying binary input masks; separate extraction scripts exist for physical essential subgraphs and essential input masks. | Keep physical edge deletion, input-coupling ablation, and input-mask retraining as separate mechanisms. Do not describe input-mask sparsity as physical edge deletion. |
| Mask/topology selection leakage | Pass for pre-training libraries | `make_topology_library.py` selects diverse physical graphs using structural metrics only. `make_input_mask_library.py` selects masks using pre-training structural/input-load metrics only. Neither reads training accuracy, validation accuracy, novel-class accuracy, or mechanism outputs. | Fixed-count mask/topology regression claims are not biased by performance-based selection at library construction time. |
| Essential motif selection leakage | Expected by design | `extract_essential_subgraphs.py` ranks candidate motifs by source novel-class accuracy and mechanism accuracy after extracting functional edge scores. This is post-training mechanism extraction, not a pre-training topology-library sampler. | Motif retraining claims must say "extracted from trained high-performing mechanisms" unless matched controls are added. They are not currently independent topology samples. |
| Novel-class metric | Pass | `evaluation.py` returns `novel_classes`; `collect_topology_results.py` stores it as `test_novel_classes`; aggregation, regression, clustered inference, final report, and verifier defaults use `test_novel_classes`. | Headline scientific claims should use `test_novel_classes`. Training, validation, and in-distribution test accuracy remain diagnostics for memorization or optimization failure. |
| Motif counts and matching | Partial | Extracted physical subgraph rows report `n_edges`, `p`, `d_rel`, branch common ranks, effective rank, root imbalance, edge participation, bottleneck fraction, and source accuracies. Current comparisons do not yet force matching on `m`, `d_rel`, effective rank, root balance, edge participation, and input-coupled count. | Phrase current motif result as protocol-specific mechanism evidence. The next phase needs matched random motif controls before making causal claims about motif superiority. |
| Statistical independence | Partial | Existing run-level data are nested seeds within topology/mask groups. The new `clustered_topology_inference.py` adds group-level regressions, clustered bootstrap, leave-family-out prediction, and random-intercept-style residual decomposition. | Prefer group-level and clustered/hierarchical evidence over naive run-level p-values or R2. Report mean, max, and seed variance separately. |
| Coarse rank versus branch capacity | Partial by construction | `d_rel` and branch common ranks are implemented structural proxies. The new `branch_margin_capacity.py` adds a conservative sampled branch-support and norm-controlled branch-separation probe, but it is not yet a full nonconvex `max_{K,B}` tree-polytope capacity solver. | Do not call `d_rel` "the capacity." Use it as a pre-training proxy and compare it against branch-margin capacity probes in the next sweeps. |

## File References

- `ICL/models/topology_markov_icl.py`: generator convention, trainable `K_params`, trainable `base_log_rates`, masked input couplings, and steady-state solvers.
- `ICL/topology_metrics.py`: strong connectivity, rooted arborescence enumeration, matrix-tree determinant counts, incidence matrices, relative tree matrices, spectra, and branch-rank proxies.
- `ICL/run_topology_icl.py`: experiment entry point, strong-connectivity guard, default exponential encoding, trainable base-rate flag, and novel-class testing call.
- `ICL/make_topology_library.py`: pre-training structural selection of strongly connected physical topologies.
- `ICL/make_input_mask_library.py`: fixed-physical-graph input-mask generation and structural diversity selection.
- `ICL/extract_essential_subgraphs.py`: post-training functional physical motif extraction with strong-connectivity augmentation.
- `ICL/extract_essential_input_masks.py`: post-training functional input-mask extraction without physical edge deletion.
- `ICL/collect_topology_results.py`, `ICL/aggregate_topology_seeds.py`, `ICL/regress_topology_results.py`, `ICL/clustered_topology_inference.py`: result collection and count/topology predictor analysis centered on `test_novel_classes`.
- `ICL/branch_margin_capacity.py`: first conservative branch-separation capacity probe beyond raw `d_rel`.

## Implications For The Next Phase

The current safe claim is:

> In the tested first-order fixed-count regimes, topology-associated structural and functional variables explain residual variation in novel-class ICL that raw count does not explain.

The current evidence does not yet justify:

- a universal topology law for CRN ICL,
- applying matrix-tree conclusions to autocatalytic or WTA CRNs,
- treating `d_rel` as the full capacity,
- claiming extracted physical motifs outperform input masks under fully matched controls.

The immediate next work should therefore prioritize clustered/hierarchical inference, held-out-family prediction, branch-margin capacity probes, matched motif baselines, and causal branch/tree-alignment interventions.
