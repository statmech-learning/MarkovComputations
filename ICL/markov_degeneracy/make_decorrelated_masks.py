"""
Build a decorrelated input-mask grid to separate rank/coverage from load-shape.

The problem
-----------
The prior `prospective_tree_diff_multiplicity` grid varies only the input mask
on one fixed reaction graph, in two strata: balanced-load (coord-load
gini = 0) and imbalanced-load (gini = 0.25). Every candidate predictor of
accuracy -- effective_rank_D_masked, condition_number_D_masked, the branch
d_rel metrics -- separates cleanly by that stratum and is nearly constant
within it. So across the 16 masks all predictors are mutually collinear: the
grid CANNOT tell which mask property causes the accuracy ceiling. The prior
program regressed accuracy on these confounded predictors and picked a winner.

The fix
-------
We want masks that break the collinearity: low-gini masks with low effective
rank, and high-gini masks with high effective rank. If such masks exist, a
2x2 factorial {coord-load gini} x {effective rank} identifies the causal
variable. If they do NOT exist -- if gini and rank are inseparable on
density-0.5 masks -- that is itself the answer (the two are one quantity).

This script:
  1. builds the centered tree matrix D for the fixed graph;
  2. random-searches density-0.5, edge-balanced masks (each of 20 edges
     couples to exactly 10 of 20 input coordinates -- matching the grid);
  3. scores each by coord-load gini and effective_rank_D_masked;
  4. characterises the achievable (gini, rank) region and selects a
     decorrelated 2x2 mask set;
  5. saves the selected masks + a scatter of the achievable region.

    python make_decorrelated_masks.py
"""

import sys
import os
import json

import numpy as np

TOPO_ICL = "/Users/aadarwal/code/statmech/topology/ICL"
GRID = os.path.join(TOPO_ICL, "results", "prospective_tree_diff_multiplicity_training")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "decorrelated_masks")

sys.path.insert(0, TOPO_ICL)
from topology_metrics import (  # noqa: E402
    normalize_edges, enumerate_arborescences, incidence_matrix,
    centered_tree_matrix, masked_relative_svd_metrics, gini,
)

N_EDGES = 20
P = 20            # (N+1)*z_dim = 5*4
ROW_SUM = 10      # each edge couples to exactly 10 of 20 coords (density 0.5)
N_SEARCH = 8000
RNG = np.random.default_rng(20260518)


def build_D():
    """Centered tree matrix of the fixed grid graph (mask-independent)."""
    any_run = sorted(os.listdir(GRID))
    src = next(os.path.join(GRID, d) for d in any_run if "_trainseed1" in d)
    topo = json.load(open(os.path.join(src, "topology.json")))
    edges = normalize_edges(topo["n_nodes"], [tuple(e) for e in topo["edges"]])
    arbs = enumerate_arborescences(topo["n_nodes"], edges)
    M = incidence_matrix(arbs, len(edges))
    return centered_tree_matrix(M), topo


def random_mask(weights=None):
    """20x20 binary mask; every row a 10-subset of columns (edge-balanced)."""
    m = np.zeros((N_EDGES, P), dtype=int)
    for r in range(N_EDGES):
        cols = RNG.choice(P, size=ROW_SUM, replace=False, p=weights)
        m[r, cols] = 1
    return m


def score(mask, D):
    s = masked_relative_svd_metrics(D, mask.astype(float), P)
    return {
        "coord_gini": gini(mask.sum(axis=0)),
        "edge_gini": gini(mask.sum(axis=1)),
        "eff_rank": s["effective_rank"],
        "cond": s["condition_number"],
        "d_rel": s["rank"],
    }


def search(D):
    """Random + load-biased search; return list of (mask, score)."""
    pool = []
    # uniform random -> low-gini region
    for _ in range(N_SEARCH // 2):
        m = random_mask()
        pool.append((m, score(m, D)))
    # load-biased random -> higher-gini region
    for _ in range(N_SEARCH // 2):
        # a random subset of coords made "popular" by a random bias strength
        w = np.ones(P)
        n_pop = RNG.integers(3, 9)
        pop = RNG.choice(P, size=n_pop, replace=False)
        w[pop] = RNG.uniform(2.0, 9.0)
        w = w / w.sum()
        m = random_mask(weights=w)
        pool.append((m, score(m, D)))
    return pool


def pick(pool, gini_lo, gini_hi, rank_lo, rank_hi, tol_g, tol_r, k):
    """Select k masks nearest each (gini, rank) target cell."""
    cells = {
        "loG_loR": (gini_lo, rank_lo), "loG_hiR": (gini_lo, rank_hi),
        "hiG_loR": (gini_hi, rank_lo), "hiG_hiR": (gini_hi, rank_hi),
    }
    chosen = {}
    for name, (gt, rt) in cells.items():
        cand = [(m, s) for m, s in pool
                if abs(s["coord_gini"] - gt) < tol_g
                and abs(s["eff_rank"] - rt) < tol_r]
        cand.sort(key=lambda ms: (abs(ms[1]["coord_gini"] - gt) / tol_g
                                  + abs(ms[1]["eff_rank"] - rt) / tol_r))
        chosen[name] = cand[:k]
    return chosen


def main():
    os.makedirs(OUT, exist_ok=True)
    D, topo = build_D()
    print(f"D (centered tree matrix): {D.shape}\n")
    pool = search(D)

    g = np.array([s["coord_gini"] for _, s in pool])
    r = np.array([s["eff_rank"] for _, s in pool])
    assert max(s["edge_gini"] for _, s in pool) < 1e-9, "edge load not balanced"
    print(f"searched {len(pool)} masks   "
          f"coord_gini {g.min():.3f}-{g.max():.3f}   "
          f"eff_rank {r.min():.2f}-{r.max():.2f}")

    # gini bins
    lo_mask = g < np.quantile(g, 0.10)
    hi_mask = g > np.quantile(g, 0.90)
    gini_lo, gini_hi = float(g[lo_mask].mean()), float(g[hi_mask].mean())
    rlo, rhi = r[lo_mask], r[hi_mask]
    print(f"  low-gini  bin: gini~{gini_lo:.3f}  eff_rank {rlo.min():.2f}-{rlo.max():.2f}")
    print(f"  high-gini bin: gini~{gini_hi:.3f}  eff_rank {rhi.min():.2f}-{rhi.max():.2f}")

    # decorrelation is possible iff the two gini bins overlap in eff_rank
    overlap_lo = max(rlo.min(), rhi.min())
    overlap_hi = min(rlo.max(), rhi.max())
    print(f"  eff_rank overlap of the two gini bins: "
          f"[{overlap_lo:.2f}, {overlap_hi:.2f}]  "
          f"width {max(0.0, overlap_hi-overlap_lo):.2f}")

    summary = {
        "search": {"n": len(pool),
                   "coord_gini_range": [float(g.min()), float(g.max())],
                   "eff_rank_range": [float(r.min()), float(r.max())]},
        "gini_bins": {"low": gini_lo, "high": gini_hi,
                      "low_eff_rank_range": [float(rlo.min()), float(rlo.max())],
                      "high_eff_rank_range": [float(rhi.min()), float(rhi.max())]},
        "eff_rank_overlap": [float(overlap_lo), float(overlap_hi)],
    }

    if overlap_hi - overlap_lo < 0.5:
        summary["verdict"] = "INSEPARABLE"
        summary["note"] = ("coord-load gini and effective_rank_D_masked cannot "
                           "be decorrelated on density-0.5 edge-balanced masks: "
                           "they are effectively one quantity.")
        print("\nVERDICT: gini and eff_rank are INSEPARABLE on this graph.")
    else:
        # two common rank levels inside the overlap band
        rank_lo = overlap_lo + 0.25 * (overlap_hi - overlap_lo)
        rank_hi = overlap_lo + 0.75 * (overlap_hi - overlap_lo)
        tol_r = 0.20 * (overlap_hi - overlap_lo)
        chosen = pick(pool, gini_lo, gini_hi, rank_lo, rank_hi,
                      tol_g=0.5 * gini_hi if gini_hi > 0 else 0.03,
                      tol_r=max(tol_r, 0.4), k=3)
        cells = {}
        for name, lst in chosen.items():
            cells[name] = []
            for i, (m, s) in enumerate(lst):
                mid = f"{name}_{i}"
                json.dump({"input_mask": m.tolist(), "score": s,
                           "n_nodes": topo["n_nodes"], "edges": topo["edges"]},
                          open(os.path.join(OUT, f"{mid}.json"), "w"), indent=1)
                cells[name].append({"id": mid, **{k: s[k] for k in
                                    ("coord_gini", "eff_rank", "cond", "d_rel")}})
        summary["verdict"] = "DECORRELATED"
        summary["targets"] = {"rank_lo": rank_lo, "rank_hi": rank_hi}
        summary["cells"] = cells
        print(f"\nVERDICT: DECORRELATED 2x2 built  "
              f"(rank levels {rank_lo:.2f} / {rank_hi:.2f})")
        for name, lst in cells.items():
            gv = np.mean([c["coord_gini"] for c in lst])
            rv = np.mean([c["eff_rank"] for c in lst])
            print(f"  {name}: {len(lst)} masks  gini~{gv:.3f}  eff_rank~{rv:.2f}")

    json.dump(summary, open(os.path.join(OUT, "search_summary.json"), "w"),
              indent=2)
    try:
        scatter(pool, summary)
    except Exception as e:
        print(f"(scatter skipped: {e!r})")
    print(f"\n-> {OUT}")
    return 0


def scatter(pool, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = [s["coord_gini"] for _, s in pool]
    r = [s["eff_rank"] for _, s in pool]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(g, r, s=6, c="lightgray", alpha=0.5, label="searched masks")
    if summary.get("verdict") == "DECORRELATED":
        col = {"loG_loR": "steelblue", "loG_hiR": "navy",
               "hiG_loR": "salmon", "hiG_hiR": "crimson"}
        for name, lst in summary["cells"].items():
            ax.scatter([c["coord_gini"] for c in lst],
                       [c["eff_rank"] for c in lst], s=90, c=col[name],
                       edgecolor="k", zorder=3, label=name)
    ax.set_xlabel("coord-load gini  (load-shape)")
    ax.set_ylabel("effective_rank_D_masked  (rank / coverage)")
    ax.set_title("Achievable (load-shape, rank) region of density-0.5 masks")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "achievable_region.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
