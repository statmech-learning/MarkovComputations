"""
M4 -- Decision geometry.

Question: how is the input space partitioned by the WTA-ICL model?

This module visualises the decision geometry of the learned chemical reaction
network in three complementary ways:

  1. SCORE-SPACE EMBEDDING. The model's mechanism is global subspace
     projection -> soft winner-take-all. The natural coordinate system is
     therefore the per-species score matrix S = {W_j . X} (and its softplus
     image f). We PCA-embed each example in score space and colour by
     `dom_species` and `true_pos`. Raw `z_flat` PCA is shown only as a
     (deliberately weak) contrast -- the input itself carries no obvious
     partition; the structure lives in the *projected* score space.

  2. SIMPLEX / TERNARY plot of `Y_frac` restricted to the top-3 species. The
     soft WTA places every example on the n-simplex; the top-3 marginal shows
     how graded the mixture is and how examples cluster by dominant species.

  3. QUERY-INTERPOLATION SWEEP. We fix four context class means and slide the
     query linearly between two of them. For each interpolation parameter we
     build the probe input ourselves and call `core.instrument`, then plot
     `dom_species`, `peak_share` and `attention` vs the sweep parameter. The
     decision boundary is piecewise-smooth (curved near softplus knees), not
     an exact hyperplane.

Built on the frozen `core.py`. Handles any number of checkpoints and any
`n_nodes` -- nothing about n=8/12 or specific labels is hard-coded.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

import numpy as np
import torch


# ===========================================================================
# Small numpy helpers (kept local; core.py is frozen)
# ===========================================================================
def _pca(X, k=2):
    """Plain PCA: return (scores (M,k), explained_variance_ratio (k,))."""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    # economy SVD
    U, s, _ = np.linalg.svd(Xc, full_matrices=False)
    scores = U[:, :k] * s[:k]
    var = s ** 2
    evr = (var[:k] / (var.sum() + 1e-12))
    return scores, evr


def _silhouette_like(scores, labels):
    """Cheap cluster-separation score in [-1,1]: mean over points of
    (b - a) / max(a, b), a = mean dist to own-label centroid,
    b = mean dist to nearest other-label centroid. Higher => cleaner
    partition. Robust to any number of labels."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return float("nan")
    cents = {u: scores[labels == u].mean(axis=0) for u in uniq}
    vals = []
    for u in uniq:
        pts = scores[labels == u]
        if len(pts) == 0:
            continue
        a = np.linalg.norm(pts - cents[u], axis=1).mean()
        others = [np.linalg.norm(pts - cents[v], axis=1).mean()
                  for v in uniq if v != u]
        b = min(others)
        vals.append((b - a) / (max(a, b) + 1e-12))
    return float(np.mean(vals)) if vals else float("nan")


def _ternary_xy(top3):
    """Map rows of a (M,3) array of barycentric coords to 2-D (M,2) for an
    equilateral-triangle ternary plot."""
    top3 = np.asarray(top3, dtype=float)
    s = top3.sum(axis=1, keepdims=True)
    bc = top3 / (s + 1e-12)
    # vertices of an equilateral triangle
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])
    xy = bc[:, 0:1] * v0 + bc[:, 1:2] * v1 + bc[:, 2:3] * v2
    return xy


# ===========================================================================
# Probe construction for the query-interpolation sweep
# ===========================================================================
def _build_interpolation_sweep(n_steps=121, n_replicates=24, seed=4):
    """Build probe inputs for a 1-D query-interpolation sweep.

    Strategy (generic in N, D -- read from core.SHARED_CONFIG):
      * fix N context class means (one example per class, B=1, exact_copy
        style geometry but with the query *displaced*),
      * slide the query along the straight line from context-mean 0 to
        context-mean 1, parameterised by alpha in [0, 1],
      * `n_replicates` independent random context layouts are generated so
        the sweep curves are averaged over context geometry, not a single
        accidental arrangement.

    Returns:
      alphas   : (n_steps,) interpolation parameter
      eval_set : list of (z_seq, labels, target) probe tuples,
                 length n_steps * n_replicates, ordered replicate-major
      reshape  : helper info dict
    """
    c = core.SHARED_CONFIG
    N, D = c["N"], c["D"]
    rng = np.random.default_rng(seed)
    alphas = np.linspace(0.0, 1.0, n_steps)

    eval_set = []
    for _ in range(n_replicates):
        # N context class means, paper-style scale (~ randn / sqrt(D))
        means = rng.standard_normal((N, D)) / np.sqrt(D)
        # context labels: arbitrary distinct integer labels
        labels = torch.tensor(
            rng.permutation(np.arange(1, N + 1)).astype(np.float32))
        # endpoints of the interpolation line (context 0 -> context 1)
        m0, m1 = means[0], means[1]
        for a in alphas:
            z_ctx = means.copy()                       # (N, D)
            z_query = (1.0 - a) * m0 + a * m1          # slide the query
            z_seq = np.vstack([z_ctx, z_query[None, :]])  # (N+1, D)
            # target: label of whichever context the query is closest to
            dist = ((z_ctx - z_query[None, :]) ** 2).sum(1)
            tgt = float(labels[int(dist.argmin())].item())
            eval_set.append((torch.tensor(z_seq, dtype=torch.float32),
                             labels.clone(), tgt))
    info = dict(N=N, D=D, n_steps=n_steps, n_replicates=n_replicates)
    return alphas, eval_set, info


# ===========================================================================
# Per-checkpoint figure: score-space + simplex + raw contrast
# ===========================================================================
def _fig_score_space(plt, ck, tr, outdir):
    """Score-space PCA (coloured by dom_species and true_pos), top-3 simplex,
    and a raw-z_flat PCA contrast. Returns (path, metrics dict)."""
    # --- score matrix S = W . X -------------------------------------------
    # core gives us f (softplus(W.X)/K); recover the raw linear projection
    # W.X directly so the embedding is in the cleanest score coordinates.
    phys = core.physical_params(ck.model)
    W = phys["W"]                                  # (n, (N+1)*D)
    S = tr.z_flat @ W.T                            # (M, n)  raw scores W_j.X
    f = tr.f                                       # (M, n)  softplus scores

    sc_S, evr_S = _pca(S, 2)
    sc_f, evr_f = _pca(f, 2)
    sc_z, evr_z = _pca(tr.z_flat, 2)

    sil_dom = _silhouette_like(sc_S, tr.dom_species)
    sil_pos = _silhouette_like(sc_S, tr.true_pos)
    sil_zpos = _silhouette_like(sc_z, tr.true_pos)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6))

    def _scatter(ax, sc, evr, color, title, cmap, cbar_label):
        sca = ax.scatter(sc[:, 0], sc[:, 1], c=color, cmap=cmap, s=8,
                         alpha=0.7, linewidths=0)
        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({100*evr[0]:.0f}%)")
        ax.set_ylabel(f"PC2 ({100*evr[1]:.0f}%)")
        cb = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cbar_label)

    _scatter(axes[0, 0], sc_S, evr_S, tr.dom_species,
             f"score space  W.X  | colour = dom_species\n(silhouette={sil_dom:.2f})",
             "tab10", "dom_species")
    _scatter(axes[0, 1], sc_S, evr_S, tr.true_pos,
             f"score space  W.X  | colour = true_pos\n(silhouette={sil_pos:.2f})",
             "tab10", "true context pos")
    _scatter(axes[0, 2], sc_f, evr_f, tr.dom_species,
             "softplus-rate space  f  | colour = dom_species",
             "tab10", "dom_species")

    # --- top-3 simplex (ternary) ------------------------------------------
    ax = axes[1, 0]
    # global top-3 species by mean Y_frac (consistent axes for all examples).
    # Pad with zero columns when n_nodes < 3 so the ternary map still works.
    order = np.argsort(-tr.Y_frac.mean(axis=0))
    top3_idx = order[:3]
    top3 = tr.Y_frac[:, top3_idx]
    if top3.shape[1] < 3:
        top3 = np.concatenate(
            [top3, np.zeros((top3.shape[0], 3 - top3.shape[1]))], axis=1)
    xy = _ternary_xy(top3)
    sca = ax.scatter(xy[:, 0], xy[:, 1], c=tr.dom_species, cmap="tab10",
                     s=9, alpha=0.7, linewidths=0)
    # triangle frame + vertex labels
    tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.plot(tri[:, 0], tri[:, 1], color="k", lw=1.0)
    vlabels = [f"sp{top3_idx[i]}" if i < len(top3_idx) else "(none)"
               for i in range(3)]
    for (vx, vy), lab, ha in zip(tri[:3], vlabels,
                                 ("right", "left", "center")):
        ax.annotate(lab, (vx, vy), ha=ha,
                    va="top" if ha != "center" else "bottom", fontsize=8)
    ax.set_title(f"Y_frac top-3 simplex (mean peak share={tr.peak_share.mean():.2f})")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04).set_label("dom_species")

    # --- peak-share histogram ---------------------------------------------
    ax = axes[1, 1]
    ax.hist(tr.peak_share, bins=30, color="#4477aa", alpha=0.85)
    ax.axvline(tr.peak_share.mean(), color="crimson", lw=1.5,
               label=f"mean={tr.peak_share.mean():.2f}")
    ax.axvline(1.0 / tr.n_nodes, color="gray", ls="--", lw=1.0,
               label=f"uniform=1/n={1.0/tr.n_nodes:.2f}")
    ax.set_title("dominant-species share of Y  (soft-WTA hardness)")
    ax.set_xlabel("peak_share")
    ax.set_ylabel("count")
    ax.legend(fontsize=7)

    # --- raw z_flat PCA contrast ------------------------------------------
    _scatter(axes[1, 2], sc_z, evr_z, tr.true_pos,
             f"CONTRAST: raw z_flat PCA | colour = true_pos\n"
             f"(silhouette={sil_zpos:.2f} -- weak)",
             "tab10", "true context pos")

    fig.suptitle(f"M4 decision geometry -- {ck.label}  "
                 f"(n_nodes={ck.n_nodes}, novel split, M={tr.n}, "
                 f"acc={tr.accuracy:.1f}%)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = core.save_fig(fig, outdir, f"score_space_{ck.label}.png")

    metrics = dict(
        sil_score_dom=sil_dom, sil_score_pos=sil_pos, sil_zflat_pos=sil_zpos,
        evr_score=float(evr_S.sum()), evr_zflat=float(evr_z.sum()),
        mean_peak_share=float(tr.peak_share.mean()),
        top3_species=[int(i) for i in top3_idx])
    return path, metrics


# ===========================================================================
# Per-checkpoint figure: query-interpolation sweep
# ===========================================================================
def _fig_interpolation(plt, ck, outdir):
    """Build probe inputs, instrument, and plot sweep curves. Returns
    (path, metrics dict)."""
    alphas, eval_set, info = _build_interpolation_sweep()
    n_steps, n_rep = info["n_steps"], info["n_replicates"]
    temperature = float(ck.params["temperature"])
    tr = core.instrument(ck.model, eval_set, temperature)

    # reshape replicate-major: (n_rep, n_steps, ...)
    def _r(a):
        return np.asarray(a).reshape(n_rep, n_steps, *np.asarray(a).shape[1:])

    dom = _r(tr.dom_species)             # (rep, step)
    peak = _r(tr.peak_share)             # (rep, step)
    attn = _r(tr.attention)             # (rep, step, N)
    predp = _r(tr.pred_pos)              # (rep, step)
    truep = _r(tr.true_pos)              # (rep, step)

    N = info["N"]
    # mean attention on context 0 and context 1 (the sweep endpoints)
    attn0 = attn[:, :, 0].mean(axis=0)
    attn1 = attn[:, :, 1].mean(axis=0)
    peak_mean = peak.mean(axis=0)
    peak_std = peak.std(axis=0)

    # mid-sweep boundary sharpness: width of the alpha window over which
    # mean attention-on-ctx0 falls from 0.75 -> 0.25 (transition region).
    hi = np.where(attn0 >= 0.75)[0]
    lo = np.where(attn0 <= 0.25)[0]
    if len(hi) and len(lo):
        boundary_width = float(abs(alphas[lo.min()] - alphas[hi.max()]))
    else:
        boundary_width = float("nan")

    # number of distinct dom_species visited along the sweep, per replicate
    n_dom_visited = float(np.mean([len(np.unique(dom[r]))
                                   for r in range(n_rep)]))
    # routing accuracy on the probe set (sanity)
    sweep_acc = 100.0 * float((tr.pred_pos == tr.true_pos).mean())

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))

    # (1) attention vs alpha
    ax = axes[0, 0]
    ax.plot(alphas, attn0, color="#1f77b4", lw=2,
            label="attn -> ctx 0 (sweep start)")
    ax.plot(alphas, attn1, color="#d62728", lw=2,
            label="attn -> ctx 1 (sweep end)")
    for i in range(2, N):
        ax.plot(alphas, attn[:, :, i].mean(axis=0), color="gray", lw=1,
                alpha=0.6, label="attn -> other ctx" if i == 2 else None)
    ax.axvline(0.5, color="k", ls=":", lw=1)
    ax.set_title("decoder attention vs query interpolation")
    ax.set_xlabel(r"interpolation $\alpha$  (query: ctx0 $\to$ ctx1)")
    ax.set_ylabel("mean attention")
    ax.legend(fontsize=7)

    # (2) dom_species vs alpha (heatmap over replicates)
    ax = axes[0, 1]
    im = ax.imshow(dom, aspect="auto", cmap="tab10",
                   extent=[0, 1, n_rep, 0], interpolation="nearest",
                   vmin=0, vmax=max(9, ck.n_nodes - 1))
    ax.set_title("dom_species along the sweep (rows = context replicates)")
    ax.set_xlabel(r"interpolation $\alpha$")
    ax.set_ylabel("replicate")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("dom_species")

    # (3) peak_share vs alpha
    ax = axes[1, 0]
    ax.plot(alphas, peak_mean, color="#2ca02c", lw=2)
    ax.fill_between(alphas, peak_mean - peak_std, peak_mean + peak_std,
                    color="#2ca02c", alpha=0.25)
    ax.axvline(0.5, color="k", ls=":", lw=1)
    ax.axhline(1.0 / ck.n_nodes, color="gray", ls="--", lw=1,
               label=f"uniform=1/n={1.0/ck.n_nodes:.2f}")
    ax.set_title("WTA hardness (peak_share) along the sweep\n"
                 "dip near the boundary = soft, graded mixture")
    ax.set_xlabel(r"interpolation $\alpha$")
    ax.set_ylabel("peak_share  (mean +/- sd)")
    ax.legend(fontsize=7)

    # (4) predicted vs true position fraction (the decision flip)
    ax = axes[1, 1]
    frac_pred0 = (predp == 0).mean(axis=0)
    frac_true0 = (truep == 0).mean(axis=0)
    ax.plot(alphas, frac_true0, color="k", ls="--", lw=1.5,
            label="true: nearest = ctx 0")
    ax.plot(alphas, frac_pred0, color="#9467bd", lw=2,
            label="model: pred_pos = ctx 0")
    ax.axvline(0.5, color="k", ls=":", lw=1)
    ax.set_title(f"decision flip  (boundary width 0.75->0.25 = "
                 f"{boundary_width:.3f})\npiecewise-smooth, not a hard "
                 f"hyperplane")
    ax.set_xlabel(r"interpolation $\alpha$")
    ax.set_ylabel("fraction choosing ctx 0")
    ax.legend(fontsize=7)

    fig.suptitle(f"M4 query-interpolation sweep -- {ck.label}  "
                 f"(n_nodes={ck.n_nodes}, {n_rep} context replicates x "
                 f"{n_steps} steps, probe acc={sweep_acc:.1f}%)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = core.save_fig(fig, outdir, f"interp_sweep_{ck.label}.png")

    metrics = dict(
        boundary_width=boundary_width,
        n_dom_visited=n_dom_visited,
        sweep_probe_accuracy=sweep_acc,
        peak_share_min=float(peak_mean.min()),
        peak_share_max=float(peak_mean.max()))
    return path, metrics


# ===========================================================================
# Module entry point
# ===========================================================================
def run(checkpoints, traces, outdir):
    """Build the M4 decision-geometry analysis.

    checkpoints : list[core.Checkpoint]
    traces      : {label: {'in_dist': Trace, 'novel': Trace}}
    outdir      : Path for this module's figures
    Returns a dict of JSON-serialisable scalar findings.
    """
    plt = core.setup_style()
    findings = {"module": "m4_geometry", "n_checkpoints": len(checkpoints),
                "per_checkpoint": {}, "figures": []}

    sil_score_dom, sil_score_pos, sil_zflat_pos = [], [], []
    boundary_widths, peak_shares = [], []

    for ck in checkpoints:
        tr = traces[ck.label]["novel"]          # novel split = true ICL
        sf_path, sf_m = _fig_score_space(plt, ck, tr, outdir)
        iv_path, iv_m = _fig_interpolation(plt, ck, outdir)

        findings["figures"].extend([str(sf_path), str(iv_path)])
        cp = {**sf_m, **iv_m, "n_nodes": ck.n_nodes,
              "novel_accuracy": float(tr.accuracy)}
        findings["per_checkpoint"][ck.label] = cp

        sil_score_dom.append(sf_m["sil_score_dom"])
        sil_score_pos.append(sf_m["sil_score_pos"])
        sil_zflat_pos.append(sf_m["sil_zflat_pos"])
        boundary_widths.append(iv_m["boundary_width"])
        peak_shares.append(sf_m["mean_peak_share"])

        print(f"  [{ck.label:20s}] score-sil(dom)={sf_m['sil_score_dom']:.2f} "
              f"score-sil(pos)={sf_m['sil_score_pos']:.2f} "
              f"zflat-sil(pos)={sf_m['sil_zflat_pos']:.2f}  "
              f"boundary_width={iv_m['boundary_width']:.3f}  "
              f"peak_share={sf_m['mean_peak_share']:.2f}")

    def _m(x):
        x = [v for v in x if np.isfinite(v)]
        return float(np.mean(x)) if x else float("nan")

    # Empirical note: score space organises cleanly by *dom_species* (the
    # active detector), NOT by true_pos -- the partition is by which species
    # fires, and decoding to a position happens downstream via B. Raw z_flat
    # carries essentially no partition for either label. The headline gain is
    # therefore score-space-by-species vs raw-z_flat-by-species/pos.
    findings["summary"] = dict(
        mean_silhouette_score_space_dom=_m(sil_score_dom),
        mean_silhouette_score_space_pos=_m(sil_score_pos),
        mean_silhouette_zflat_pos=_m(sil_zflat_pos),
        score_space_species_separation_gain=_m(
            [s - z for s, z in zip(sil_score_dom, sil_zflat_pos)]),
        mean_boundary_width=_m(boundary_widths),
        mean_peak_share=_m(peak_shares))
    return findings


if __name__ == "__main__":
    print("M4 decision geometry -- loading checkpoints ...")
    checkpoints, traces = core.load_all()
    outdir = core.module_outdir("m4_geometry")
    print(f"Discovered {len(checkpoints)} checkpoint(s); "
          f"figures -> {outdir}\n")
    result = run(checkpoints, traces, outdir)
    print("\n=== M4 summary ===")
    for k, v in result["summary"].items():
        print(f"  {k:36s} = {v:.4f}")
    print(f"\n{len(result['figures'])} figure(s) written to {outdir}")
