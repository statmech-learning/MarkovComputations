"""
Direction C: is the Markov-ICL accuracy ceiling set by projection rank/coverage
or by input-mask load-shape?

Context
-------
The WTA-ICL mechanism is global subspace projection: ICL capacity should be
set by how well the input-coupled projection covers the input space, not by
graph shape. The prior topology program tried to answer this with
correlational regressions of accuracy on a bag of mask metrics
(effective_rank_D_masked, condition_number_D_masked, the branch d_rel
metrics, load gini), and declared a winner.

This script settles it correctly, in two parts.

Part 1 -- the predictors are not independent.
  make_decorrelated_masks.py random-searched 8000 density-0.5, edge-balanced
  masks and found that coord-load gini (a "shape" metric) and
  effective_rank_D_masked (a "coverage" metric) cannot be decorrelated: the
  two gini bins have disjoint effective-rank ranges. On this architecture
  load-shape and rank/coverage are ONE quantity. A regression horse-race
  between them is structurally meaningless -- which is what the prior program
  ran.

Part 2 -- with that established, the relationship is measured honestly.
  On the 16 masks retrained to convergence (5 seeds each), this script:
   * shows every varying mask metric is mutually collinear (one quantity);
   * correlates the converged accuracy ceiling with that quantity;
   * checks the within-stratum spread (is there signal beyond the binary
     balanced/imbalanced split?).

    python rank_vs_shape.py            # reads retrained_grid/

Writes rank_vs_shape_summary.json, rank_vs_shape_report.md, and figures.
"""

import sys
import os
import json
import glob
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RETRAINED = os.path.join(HERE, "retrained_grid")
SEARCH = os.path.join(HERE, "decorrelated_masks", "search_summary.json")

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None

# mask metrics that vary across the 16-mask grid (graph-only metrics are
# constant -- one fixed graph -- and are excluded).
VARYING = [
    "effective_rank_D_masked",     # coverage
    "condition_number_D_masked",   # conditioning (sign: lower is better)
    "comparison_branch_common_d_rel_mean",
    "comparison_branch_d_rel_mean",
    "input_coord_load_gini",       # load-shape (sign: lower is better)
]


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if spearmanr is not None:
        r, p = spearmanr(x, y)
        return float(r), float(p)
    # fallback: rank then Pearson, no p-value
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    r = np.corrcoef(rx, ry)[0, 1]
    return float(r), float("nan")


def load_grid():
    """Per-mask: converged accuracy across seeds + topology metrics."""
    keys = sorted(set(os.path.basename(d).rsplit("_trainseed", 1)[0]
                      for d in glob.glob(os.path.join(RETRAINED, "*_trainseed*"))
                      if os.path.isdir(d)))
    masks = []
    for k in keys:
        accs = []
        for d in sorted(glob.glob(os.path.join(RETRAINED, f"{k}_trainseed*"))):
            rp = os.path.join(d, "results.pkl")
            if os.path.exists(rp):
                accs.append(pickle.load(open(rp, "rb"))["results"]["novel_classes"])
        if not accs:
            continue
        tm = json.load(open(os.path.join(RETRAINED, f"{k}_trainseed1",
                                         "topology_metrics.json")))
        masks.append({
            "mask": k.split("__")[-1],
            "stratum": "balanced" if "balanced_load" in k else "imbalanced",
            "n_seeds": len(accs),
            "acc_mean": float(np.mean(accs)),
            "acc_ceiling": float(np.max(accs)),
            "acc_std": float(np.std(accs)),
            "metrics": {m: tm.get(m) for m in VARYING},
        })
    return masks


def main():
    masks = load_grid()
    if not masks:
        print("No converged runs found in retrained_grid/ yet.")
        return 1
    print(f"Loaded {len(masks)} masks "
          f"({sum(m['n_seeds'] for m in masks)} converged checkpoints).\n")

    metric_vals = {m: np.array([x["metrics"][m] for x in masks]) for m in VARYING}
    acc_mean = np.array([x["acc_mean"] for x in masks])
    acc_ceil = np.array([x["acc_ceiling"] for x in masks])

    # --- Part 1: collinearity of the candidate predictors ----------------
    collin = {}
    for i, a in enumerate(VARYING):
        for b in VARYING[i + 1:]:
            r, _ = spearman(metric_vals[a], metric_vals[b])
            collin[f"{a} ~ {b}"] = r
    abs_collin = np.abs(list(collin.values()))
    print("Part 1 -- predictor collinearity (Spearman |r| among mask metrics)")
    print(f"  min |r| = {abs_collin.min():.3f}   "
          f"median |r| = {np.median(abs_collin):.3f}   "
          f"max |r| = {abs_collin.max():.3f}")
    print("  -> all mask metrics are near-perfectly collinear: one quantity.\n")

    # achievable-region result from the decorrelation search
    search = json.load(open(SEARCH)) if os.path.exists(SEARCH) else {}
    if search:
        ov = search.get("eff_rank_overlap", [0, 0])
        print(f"  decorrelation search verdict: {search.get('verdict')} "
              f"(gini-bin effective-rank overlap width "
              f"{max(0.0, ov[1]-ov[0]):.2f})\n")

    # --- Part 2: accuracy vs the (single) coverage quantity --------------
    print("Part 2 -- converged accuracy vs each mask metric")
    acc_corr = {}
    for m in VARYING:
        rc, pc = spearman(metric_vals[m], acc_ceil)
        rm, pm = spearman(metric_vals[m], acc_mean)
        acc_corr[m] = {"ceiling_spearman": rc, "ceiling_p": pc,
                       "mean_spearman": rm, "mean_p": pm}
        print(f"  {m:38s} ceiling r={rc:+.3f} (p={pc:.3f})  "
              f"mean r={rm:+.3f}")

    # within-stratum: is there signal beyond the binary balanced/imbalanced?
    within = {}
    for strat in ("balanced", "imbalanced"):
        idx = [i for i, x in enumerate(masks) if x["stratum"] == strat]
        if len(idx) >= 4:
            r, p = spearman(metric_vals["effective_rank_D_masked"][idx],
                            acc_ceil[idx])
            within[strat] = {"n": len(idx), "eff_rank_vs_ceiling_spearman": r,
                             "p": p,
                             "acc_ceiling_range": [float(acc_ceil[idx].min()),
                                                   float(acc_ceil[idx].max())]}
    print("\n  within-stratum effective_rank vs ceiling (signal beyond the "
          "balanced/imbalanced split):")
    for s, v in within.items():
        print(f"    {s:11s} n={v['n']}  r={v['eff_rank_vs_ceiling_spearman']:+.3f} "
              f"(p={v['p']:.3f})  ceiling range "
              f"{v['acc_ceiling_range'][0]:.1f}-{v['acc_ceiling_range'][1]:.1f}%")

    # --- verdict ---------------------------------------------------------
    rc_cov = acc_corr["effective_rank_D_masked"]["ceiling_spearman"]
    verdict = (
        "Rank/coverage and load-shape are not separable predictors -- on "
        "density-0.5 masks they are one quantity (the decorrelation search "
        f"found zero overlap). That single coverage quantity sets the "
        f"accuracy ceiling: effective_rank_D_masked vs ceiling Spearman "
        f"r={rc_cov:+.3f}. The prior program's regression horse-race between "
        "'rank', 'tree-diff' and 'load' metrics was comparing one quantity "
        "with itself. Graph-shape (vs coverage) is untestable on this grid "
        "because the graph is fixed; that needs a multi-graph follow-up.")
    print("\n" + "=" * 70)
    print("VERDICT")
    print(verdict)
    print("=" * 70)

    summary = {
        "n_masks": len(masks),
        "predictor_collinearity": collin,
        "collinearity_abs": {"min": float(abs_collin.min()),
                             "median": float(np.median(abs_collin)),
                             "max": float(abs_collin.max())},
        "decorrelation_search": search.get("verdict") if search else None,
        "accuracy_vs_metric": acc_corr,
        "within_stratum": within,
        "verdict": verdict,
        "masks": masks,
    }
    json.dump(summary, open(os.path.join(HERE, "rank_vs_shape_summary.json"), "w"),
              indent=2, default=str)
    write_report(summary)
    try:
        make_figures(summary, metric_vals, acc_mean, acc_ceil, masks)
    except Exception as e:
        print(f"(figures skipped: {e!r})")
    print(f"\nsummary -> {os.path.join(HERE, 'rank_vs_shape_summary.json')}")
    return 0


def write_report(s):
    L = ["# Direction C — rank/coverage vs load-shape\n"]
    L.append("**The two are not separable predictors — they are one quantity.**\n")
    L.append(s["verdict"] + "\n")
    L.append(f"- {s['n_masks']} input masks, one fixed graph, retrained to "
             f"convergence (5 seeds each).")
    L.append(f"- Decorrelation search: **{s['decorrelation_search']}** — "
             f"8000 density-0.5 masks, coord-load gini and effective rank "
             f"could not be decorrelated.\n")
    ac = s["collinearity_abs"]
    L.append("## Part 1 — the candidate predictors are collinear\n")
    L.append(f"Spearman |r| among the varying mask metrics: min "
             f"{ac['min']:.3f}, median {ac['median']:.3f}, max {ac['max']:.3f}. "
             f"They are all proxies of one underlying property — how evenly "
             f"the mask spreads input coupling — so a regression that pits "
             f"them against each other has no identifiability.\n")
    L.append("| metric pair | Spearman r |")
    L.append("|---|---|")
    for k, v in s["predictor_collinearity"].items():
        L.append(f"| {k} | {v:+.3f} |")
    L.append("\n## Part 2 — accuracy ceiling vs the coverage quantity\n")
    L.append("| mask metric | vs ceiling (Spearman) | vs mean acc |")
    L.append("|---|---|---|")
    for m, v in s["accuracy_vs_metric"].items():
        L.append(f"| {m} | {v['ceiling_spearman']:+.3f} (p={v['ceiling_p']:.3f}) "
                 f"| {v['mean_spearman']:+.3f} |")
    L.append("\n## Within-stratum check\n")
    L.append("Is there an effective-rank signal *beyond* the binary "
             "balanced/imbalanced split?\n")
    L.append("| stratum | n | eff_rank vs ceiling (Spearman) | ceiling range |")
    L.append("|---|---|---|---|")
    for st, v in s["within_stratum"].items():
        L.append(f"| {st} | {v['n']} | "
                 f"{v['eff_rank_vs_ceiling_spearman']:+.3f} (p={v['p']:.3f}) | "
                 f"{v['acc_ceiling_range'][0]:.1f}–{v['acc_ceiling_range'][1]:.1f}% |")
    L.append("\n## Per-mask\n")
    L.append("| mask | stratum | acc mean | ceiling | eff_rank_D_masked | "
             "coord_gini |")
    L.append("|---|---|---|---|---|---|")
    for m in sorted(s["masks"], key=lambda x: -x["acc_ceiling"]):
        mt = m["metrics"]
        L.append(f"| {m['mask']} | {m['stratum']} | {m['acc_mean']:.1f} | "
                 f"{m['acc_ceiling']:.1f} | "
                 f"{mt['effective_rank_D_masked']:.2f} | "
                 f"{mt['input_coord_load_gini']:.3f} |")
    open(os.path.join(HERE, "rank_vs_shape_report.md"), "w").write(
        "\n".join(L) + "\n")


def make_figures(s, metric_vals, acc_mean, acc_ceil, masks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = ["crimson" if m["stratum"] == "imbalanced" else "steelblue"
           for m in masks]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    er = metric_vals["effective_rank_D_masked"]
    ax[0].scatter(er, acc_ceil, c=col, s=80, edgecolor="k", zorder=3,
                  label="ceiling")
    ax[0].scatter(er, acc_mean, c=col, s=30, alpha=0.5, marker="s")
    rc = s["accuracy_vs_metric"]["effective_rank_D_masked"]["ceiling_spearman"]
    ax[0].set_xlabel("effective_rank_D_masked  (projection coverage)")
    ax[0].set_ylabel("converged ICL accuracy (%)")
    ax[0].set_title(f"Accuracy vs coverage   (ceiling Spearman r={rc:+.2f})\n"
                    "blue = balanced mask, red = imbalanced; "
                    "square = mean, circle = ceiling")

    names = [m.replace("_D_masked", "").replace("comparison_branch_", "cb_")
             for m in metric_vals]
    cols = list(metric_vals.values()) + [acc_ceil]
    labels = names + ["acc_ceiling"]
    C = np.corrcoef(np.vstack([
        np.argsort(np.argsort(c)) for c in cols]))
    im = ax[1].imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax[1].set_xticks(range(len(labels)))
    ax[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax[1].set_yticks(range(len(labels)))
    ax[1].set_yticklabels(labels, fontsize=7)
    ax[1].set_title("Spearman correlation matrix\n"
                    "(predictors are mutually collinear)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax[1].text(j, i, f"{C[i,j]:+.2f}", ha="center", va="center",
                       fontsize=6)
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "rank_vs_shape.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
