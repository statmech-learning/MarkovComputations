# Input-Multiplicity Causal Control Training Plan

## Status

The current pass uses the already-trained fixed-m20 mask library, so no broad new sweep was launched. The existing artifact already provides 48 mask/topology groups with 5 training seeds per group.

## Existing-Data Control Used Now

- Unit of inference: mask/topology group; seed rows are summarized as mean, best, and standard deviation.
- Matched exactly on input-coupled parameter count `200` and aggregate `M_mean=10`.
- Physical graph identity is held fixed in matched high/low comparisons and included as a categorical control in regressions.
- `d_rel=200` for 45 of 48 groups; the three `d_rel=190` groups are included with a `d_rel` covariate and checked by a strict `d_rel=200` sensitivity analysis.
- The selector is normalized same-root tree-difference comparison overlap, not raw tree-pair counts.

## Follow-Up Training If More Cluster Time Is Allocated

1. For each physical graph, materialize the selected mask groups from `tree_multiplicity_causal_mask_library.json`.
2. Add replacement masks if needed so every high/low category has exact `d_rel`, exact input-coupled edge count, and exact input-coupled coordinate count.
3. Train at least 5 seeds per mask group; use more seeds for matched pairs whose current seed standard deviation exceeds 8 novel-ICL points.
4. Save branch-wise novel-class accuracy, branch failures, trained branch margin, and post-training tree/posterior diagnostics in addition to the existing mean/best/std summaries.
5. Analyze only group-level or hierarchical models; do not treat seed rows as independent topology samples.

## Primary Contrast

High normalized tree-difference comparison overlap versus low normalized tree-difference comparison overlap, under fixed physical graph, fixed input parameter count, fixed aggregate multiplicity, and matched or covaried `d_rel`.
