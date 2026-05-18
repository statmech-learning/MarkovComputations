"""
M6 -- cross-model comparison: is the learned topology canonical?

Question (AGENTS.md section 3, W2): when two WTA-ICL models are trained with
the SAME architecture (same n_nodes) but differ in init seed / hardware, do
they learn the SAME topology -- the same set of detector species, just
re-labelled (a permutation gauge) -- or genuinely different (degenerate)
solutions?

This module aligns species across same-n_nodes checkpoints and quantifies how
well they match. It keeps two comparison regimes STRICTLY SEPARATE:

  * SAME-SEED pair  (e.g. paper-n8-s30 vs engaging-n8-s30): identical init,
    differ only by CPU/BLAS numerical noise. Agreement here measures TRAINING
    STABILITY -- it is NOT evidence of canonicality.
  * CROSS-SEED pair (e.g. grid seed 31 vs seed 30): genuinely independent
    inits. Agreement here IS the canonical-vs-degenerate result.

Alignment uses GAUGE-INVARIANT quantities only -- never raw B or raw K/beta:
  * parametric : W-row cosine similarity, and core.comparison_scores(W)
                 (the subspace-projection signature of each species).
  * functional : dom_species co-occurrence on the SHARED eval set -- two
                 species "match" if they dominate Y on the same inputs.
Species are matched by a one-to-one assignment (Hungarian) maximising a
combined gauge-invariant similarity; the resulting match quality is the
canonicality score.

It also reports n=8 vs n=12 (effective node count, accuracy) and the
capacity-sweep accuracy curve across all n_nodes present.

Handles ANY number of checkpoints and ANY n_nodes -- nothing is hard-coded.
"""

import sys
import os
import itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

import numpy as np


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _hungarian(cost):
    """Minimum-cost one-to-one assignment for a SQUARE cost matrix.

    Returns (row_idx, col_idx). Uses scipy if available, else a brute-force
    permutation search (fine for the small n_nodes here, <= ~12).
    """
    cost = np.asarray(cost, dtype=float)
    n = cost.shape[0]
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment(cost)
    except Exception:
        best, best_perm = np.inf, None
        for perm in itertools.permutations(range(n)):
            c = cost[np.arange(n), perm].sum()
            if c < best:
                best, best_perm = c, perm
        return np.arange(n), np.array(best_perm)


def _wrow_cosine(Wa, Wb):
    """(na, nb) cosine similarity between every W-row of model a and b.

    W is gauge-invariant; its rows are the species' input projections.
    """
    na = Wa / (np.linalg.norm(Wa, axis=1, keepdims=True) + 1e-12)
    nb = Wb / (np.linalg.norm(Wb, axis=1, keepdims=True) + 1e-12)
    return na @ nb.T


def _compscore_cosine(Ca, Cb):
    """(na, nb) cosine similarity between species' comparison-score signatures.

    Ca, Cb come from core.comparison_scores -- each row is a species' (N_c,)
    'which context position do I compare against the query' signature, also
    fully gauge-invariant.
    """
    na = Ca / (np.linalg.norm(Ca, axis=1, keepdims=True) + 1e-12)
    nb = Cb / (np.linalg.norm(Cb, axis=1, keepdims=True) + 1e-12)
    return na @ nb.T


def _dom_cooccurrence(dom_a, dom_b, na, nb):
    """(na, nb) functional co-occurrence: fraction of shared eval examples on
    which species i dominates Y in model a AND species j dominates in model b.

    dom_a, dom_b are dom_species arrays over the SAME (fixed-seed) eval set,
    so they are directly comparable example-by-example.
    """
    M = len(dom_a)
    out = np.zeros((na, nb))
    for i in range(na):
        ai = (dom_a == i)
        if not ai.any():
            continue
        for j in range(nb):
            out[i, j] = np.logical_and(ai, dom_b == j).sum() / M
    return out


def _normalised_match(sim, weight):
    """Given a similarity matrix `sim` (rows=model a species, cols=model b),
    pad to square, solve the assignment that MAXIMISES total similarity, and
    return (row_idx, col_idx, per_pair_sim, mean_sim).

    `weight` only labels which similarity this is (for callers); not used here.
    """
    na, nb = sim.shape
    n = max(na, nb)
    padded = np.zeros((n, n))
    padded[:na, :nb] = sim
    r, c = _hungarian(-padded)            # maximise similarity
    keep = [(ri, ci) for ri, ci in zip(r, c) if ri < na and ci < nb]
    pair_sim = np.array([padded[ri, ci] for ri, ci in keep])
    return keep, pair_sim, float(pair_sim.mean()) if len(pair_sim) else 0.0


def _participation_ratio(p):
    """Effective count = (sum p)^2 / sum p^2 for a non-negative vector p."""
    p = np.asarray(p, dtype=float)
    s1, s2 = p.sum(), (p * p).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 0.0


# ---------------------------------------------------------------------------
# core comparison of one pair of checkpoints (same n_nodes)
# ---------------------------------------------------------------------------
def _compare_pair(ck_a, ck_b, tr_a, tr_b, N, D):
    """Align species of two same-n_nodes checkpoints on gauge-invariant
    quantities. tr_a / tr_b are the `novel`-split Traces (shared eval set).

    Returns a dict of scalar match metrics + the matched-row similarity used
    for the figure.
    """
    pa = core.physical_params(ck_a.model)
    pb = core.physical_params(ck_b.model)
    Wa, Wb = pa["W"], pb["W"]
    Ca = core.comparison_scores(Wa, N, D)
    Cb = core.comparison_scores(Wb, N, D)

    # three gauge-invariant similarity matrices ----------------------------
    sim_w = _wrow_cosine(Wa, Wb)                    # parametric: W rows
    sim_c = _compscore_cosine(Ca, Cb)               # parametric: subspace sig
    cooc = _dom_cooccurrence(tr_a.dom_species, tr_b.dom_species,
                             Wa.shape[0], Wb.shape[0])   # functional

    # combined parametric score, then one assignment that all metrics report
    # under (so the numbers describe ONE consistent species correspondence).
    combined = 0.5 * sim_w + 0.5 * sim_c
    keep, _, _ = _normalised_match(combined, "combined")

    def _score_under(sim_mat):
        vals = np.array([sim_mat[ri, ci] for ri, ci in keep])
        return float(vals.mean()) if len(vals) else 0.0, vals

    w_mean, w_vals = _score_under(sim_w)
    c_mean, c_vals = _score_under(sim_c)
    # functional: co-occurrence normalised by per-species occupancy so it is a
    # conditional-agreement probability, robust to uneven species usage.
    occ_a = np.array([(tr_a.dom_species == ri).mean() for ri, _ in keep])
    cooc_vals = np.array([cooc[ri, ci] for ri, ci in keep])
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.where(occ_a > 1e-6, cooc_vals / occ_a, 0.0)
    func_mean = float(np.nansum(cooc_vals))         # total fraction of agreeing examples
    func_cond = float(cond[occ_a > 1e-6].mean()) if (occ_a > 1e-6).any() else 0.0

    same_seed = (ck_a.seed == ck_b.seed)
    return dict(
        label_a=ck_a.label, label_b=ck_b.label, n_nodes=ck_a.n_nodes,
        regime="same_seed" if same_seed else "cross_seed",
        w_cos_mean=round(w_mean, 4),
        compscore_cos_mean=round(c_mean, 4),
        dom_cooccur_frac=round(func_mean, 4),
        dom_cooccur_conditional=round(func_cond, 4),
        # arrays kept for plotting only (not returned to summary.json):
        _w_vals=w_vals, _c_vals=c_vals, _cooc_vals=cooc_vals,
        _sim_w=sim_w, _keep=keep)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def run(checkpoints, traces, outdir):
    """checkpoints: list[Checkpoint]; traces: {label: {'in_dist','novel': Trace}};
    outdir: Path. Returns a dict of JSON-serialisable scalar findings."""
    plt = core.setup_style()
    N = core.SHARED_CONFIG["N"]
    D = core.SHARED_CONFIG["D"]

    findings = {"n_checkpoints": len(checkpoints)}
    if not checkpoints:
        return findings

    # ----- group checkpoints by n_nodes -----------------------------------
    by_n = {}
    for ck in checkpoints:
        by_n.setdefault(ck.n_nodes, []).append(ck)
    findings["n_nodes_present"] = sorted(by_n.keys())

    # =====================================================================
    # 1. pairwise species alignment within each n_nodes group
    # =====================================================================
    pairs = []
    for n_nodes, group in sorted(by_n.items()):
        for ck_a, ck_b in itertools.combinations(group, 2):
            tr_a = traces[ck_a.label]["novel"]
            tr_b = traces[ck_b.label]["novel"]
            pairs.append(_compare_pair(ck_a, ck_b, tr_a, tr_b, N, D))

    same_seed = [p for p in pairs if p["regime"] == "same_seed"]
    cross_seed = [p for p in pairs if p["regime"] == "cross_seed"]

    def _agg(group, key):
        v = [p[key] for p in group]
        return round(float(np.mean(v)), 4) if v else None

    findings["n_pairs_total"] = len(pairs)
    findings["n_pairs_same_seed"] = len(same_seed)
    findings["n_pairs_cross_seed"] = len(cross_seed)

    # --- TRAINING STABILITY (same-seed): NOT a canonicality claim ---------
    findings["stability_w_cos"] = _agg(same_seed, "w_cos_mean")
    findings["stability_compscore_cos"] = _agg(same_seed, "compscore_cos_mean")
    findings["stability_dom_cooccur"] = _agg(same_seed, "dom_cooccur_frac")
    findings["stability_dom_conditional"] = _agg(same_seed, "dom_cooccur_conditional")

    # --- CANONICALITY (cross-seed): the real result -----------------------
    findings["canonical_w_cos"] = _agg(cross_seed, "w_cos_mean")
    findings["canonical_compscore_cos"] = _agg(cross_seed, "compscore_cos_mean")
    findings["canonical_dom_cooccur"] = _agg(cross_seed, "dom_cooccur_frac")
    findings["canonical_dom_conditional"] = _agg(cross_seed, "dom_cooccur_conditional")
    if cross_seed:
        c = findings["canonical_w_cos"]
        findings["canonical_verdict"] = (
            "canonical" if c is not None and c > 0.9 else
            "partially-canonical" if c is not None and c > 0.6 else
            "degenerate")
    else:
        findings["canonical_verdict"] = "untested (no cross-seed pairs yet)"

    # per-pair detail for the figure / summary
    findings["pairs"] = [
        {k: v for k, v in p.items() if not k.startswith("_")} for p in pairs]

    # =====================================================================
    # 2. n=8 vs n=12: effective node count + accuracy per n_nodes
    # =====================================================================
    per_n = {}
    for n_nodes, group in sorted(by_n.items()):
        accs, effs = [], []
        for ck in group:
            tr = traces[ck.label]["novel"]
            accs.append(tr.accuracy)
            mean_yf = tr.Y_frac.mean(axis=0)            # (n,) mean simplex use
            effs.append(_participation_ratio(mean_yf))
        per_n[n_nodes] = dict(
            n_checkpoints=len(group),
            mean_accuracy=round(float(np.mean(accs)), 3),
            mean_effective_nodes=round(float(np.mean(effs)), 3),
            capacity_utilisation=round(float(np.mean(effs)) / n_nodes, 3))
    findings["per_n_nodes"] = {str(k): v for k, v in per_n.items()}

    # =====================================================================
    # 3. figures
    # =====================================================================
    figs = []

    # ---- Fig A: cross-model match summary --------------------------------
    if pairs:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        # (left) bar chart of the 3 gauge-invariant match metrics per pair
        ax = axes[0]
        labels = [f"{p['label_a']}\nvs {p['label_b']}\n[{p['regime']}]"
                  for p in pairs]
        x = np.arange(len(pairs))
        w = 0.26
        ax.bar(x - w, [p["w_cos_mean"] for p in pairs], w,
               label="W-row cosine", color="#3367d6")
        ax.bar(x, [p["compscore_cos_mean"] for p in pairs], w,
               label="comparison-score cosine", color="#37a169")
        ax.bar(x + w, [p["dom_cooccur_conditional"] for p in pairs], w,
               label="dom_species cond. agreement", color="#dd6b20")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_ylabel("gauge-invariant similarity")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.9, ls="--", lw=0.8, color="grey")
        ax.set_title("Cross-model species alignment\n(same-seed = stability, "
                     "cross-seed = canonicality)")
        ax.legend(fontsize=6.5, loc="lower right")

        # (right) matched-species W-cosine heatmap for the first pair
        ax = axes[1]
        p0 = pairs[0]
        sim_w, keep = p0["_sim_w"], p0["_keep"]
        order_b = [ci for _, ci in keep]
        order_a = [ri for ri, _ in keep]
        M = sim_w[np.ix_(order_a, order_b)]
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_xlabel(f"{p0['label_b']} species (matched order)")
        ax.set_ylabel(f"{p0['label_a']} species")
        ax.set_title(f"W-row cosine, aligned\n{p0['label_a']} vs {p0['label_b']}"
                     f" [{p0['regime']}]")
        for t, (ri, ci) in enumerate(keep):
            ax.text(t, t, f"{sim_w[ri, ci]:.2f}", ha="center", va="center",
                    fontsize=6, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        figs.append(str(core.save_fig(fig, outdir, "m6_cross_model_match.png")))

    # ---- Fig B: capacity sweep -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ns = sorted(per_n.keys())
    ax = axes[0]
    ax.plot(ns, [per_n[n]["mean_accuracy"] for n in ns], "o-",
            color="#3367d6")
    ax.set_xlabel("n_nodes (species count)")
    ax.set_ylabel("novel-split accuracy (%)")
    ax.set_title("Capacity sweep: accuracy")
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(ns, [per_n[n]["mean_effective_nodes"] for n in ns], "s-",
            color="#37a169", label="effective node count")
    ax.plot(ns, ns, "k--", lw=0.8, label="n_nodes (all-used)")
    ax.set_xlabel("n_nodes (species count)")
    ax.set_ylabel("effective node count (participation ratio)")
    ax.set_title("Capacity sweep: effective utilisation")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figs.append(str(core.save_fig(fig, outdir, "m6_capacity_sweep.png")))

    findings["figures"] = figs
    return findings


# ---------------------------------------------------------------------------
# standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    checkpoints, traces = core.load_all()
    outdir = core.module_outdir("m6_comparison")
    result = run(checkpoints, traces, outdir)
    print(json.dumps(result, indent=2, default=str))
