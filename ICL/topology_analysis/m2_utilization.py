"""
M2 -- Node utilization.

Question: how many species does the model actually USE?

The WTA is soft (see core.py): Y is a simplex point spread over ~2.5-3
species, not a one-hot vector. So "how many nodes are used" is not a count of
non-zero species but a graded, measured quantity. This module measures:

  * dom_species histogram (argmax_j Y_j -- the dominant species per example)
  * mean Y_frac per species (graded, time-averaged species usage)
  * EFFECTIVE NODE COUNT = participation ratio (Sum p_i)^2 / Sum p_i^2 of the
    mean Y_frac vector p. A model that splits Y evenly over k species has
    effective count ~k; a model that always uses one species has count ~1.
  * WTA hardness: the peak_share distribution and the mean count of species
    with Y_frac > 5%.
  * effective node count vs n_nodes across the capacity sweep.

Interpretation guide (from AGENTS.md, against N_c = number of context
positions): ~2*N_c active species => branch-covering solution; ~N_c =>
compressed / default-threshold solution; < N_c => suspect.

Builds exactly this file; imports only `core`. Run standalone:
    python m2_utilization.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

import numpy as np


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def participation_ratio(p) -> float:
    """Effective number of components of a non-negative weight vector p.

    PR = (sum p_i)^2 / sum p_i^2.  Equals k for k equal weights, 1 for a
    one-hot vector, n_nodes for a perfectly uniform vector.
    """
    p = np.asarray(p, dtype=float)
    s1 = p.sum()
    s2 = (p * p).sum()
    if s2 <= 0.0:
        return 0.0
    return float(s1 * s1 / s2)


def utilization_stats(trace, active_thresh=0.05):
    """Per-checkpoint/split utilization metrics from a core.Trace.

    Returns a dict of JSON-serializable scalars + a few arrays for plotting.
    """
    Y_frac = trace.Y_frac                       # (M, n)
    n_nodes = int(trace.n_nodes)
    M = trace.n

    # graded, time-averaged usage of each species
    mean_Y_frac = Y_frac.mean(axis=0)           # (n,)

    # effective node count = participation ratio of the mean usage vector
    eff_node_count = participation_ratio(mean_Y_frac)

    # dom_species histogram (which species dominates Y, per example)
    dom_hist = np.bincount(trace.dom_species, minlength=n_nodes)  # (n,)
    dom_frac = dom_hist / max(M, 1)
    # number of species that ever dominate at least one example
    n_ever_dom = int((dom_hist > 0).sum())
    # effective count of *dominant-species* distribution (occupancy spread)
    eff_dom_count = participation_ratio(dom_frac)

    # WTA hardness
    peak_share = trace.peak_share               # (M,)
    n_active_per_ex = (Y_frac > active_thresh).sum(axis=1)   # (M,)
    mean_n_active = float(n_active_per_ex.mean())

    # species that are "active" on average (mean usage above threshold)
    n_active_mean = int((mean_Y_frac > active_thresh).sum())

    return dict(
        n_nodes=n_nodes,
        M=M,
        eff_node_count=round(eff_node_count, 4),
        eff_dom_count=round(eff_dom_count, 4),
        n_ever_dom=n_ever_dom,
        n_active_mean=n_active_mean,
        mean_n_active=round(mean_n_active, 4),
        mean_peak_share=round(float(peak_share.mean()), 4),
        median_peak_share=round(float(np.median(peak_share)), 4),
        min_peak_share=round(float(peak_share.min()), 4),
        max_mean_Y_frac=round(float(mean_Y_frac.max()), 4),
        accuracy=round(trace.accuracy, 2),
        # arrays kept only for plotting (stripped before returning from run)
        _mean_Y_frac=mean_Y_frac,
        _dom_frac=dom_frac,
        _peak_share=peak_share,
        _n_active_per_ex=n_active_per_ex,
    )


def classify_solution(eff_node_count, N_c):
    """Map effective node count to the AGENTS.md interpretation buckets."""
    if eff_node_count < 0.75 * N_c:
        return "suspect (<N_c)"
    if eff_node_count < 1.5 * N_c:
        return "compressed/default-threshold (~N_c)"
    return "branch-covering (~2*N_c)"


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def _plot_per_checkpoint(plt, label, stats_novel, stats_indist, N_c, outdir):
    """One figure per checkpoint: usage bars, dom histogram, peak-share hist,
    active-count hist. `novel` split is primary; `in_dist` shown as contrast.
    """
    s = stats_novel
    n = s["n_nodes"]
    species = np.arange(n)

    fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.2))
    fig.suptitle(f"M2 node utilization -- {label}  (n_nodes={n}, N_c={N_c})",
                 fontsize=11)

    # (0,0) mean Y_frac per species: novel vs in_dist
    w = 0.4
    ax[0, 0].bar(species - w / 2, s["_mean_Y_frac"], w, label="novel",
                 color="#2c7fb8")
    ax[0, 0].bar(species + w / 2, stats_indist["_mean_Y_frac"], w,
                 label="in_dist", color="#bdbdbd")
    ax[0, 0].axhline(1.0 / n, ls=":", c="k", lw=0.8,
                     label=f"uniform (1/n={1.0/n:.3f})")
    ax[0, 0].axhline(0.05, ls="--", c="#d95f02", lw=0.8, label="5% active")
    ax[0, 0].set_title("mean Y_frac per species (graded usage)")
    ax[0, 0].set_xlabel("species j")
    ax[0, 0].set_ylabel("mean Y_frac")
    ax[0, 0].set_xticks(species)
    ax[0, 0].legend(fontsize=7)

    # (0,1) dom_species histogram (novel)
    ax[0, 1].bar(species, s["_dom_frac"], color="#41ab5d")
    ax[0, 1].set_title(f"dom_species occupancy (novel)  "
                       f"-- {s['n_ever_dom']}/{n} ever dominate")
    ax[0, 1].set_xlabel("species j = argmax Y_j")
    ax[0, 1].set_ylabel("fraction of examples")
    ax[0, 1].set_xticks(species)

    # (1,0) peak_share distribution (novel)
    ax[1, 0].hist(s["_peak_share"], bins=30, range=(0, 1), color="#7570b3")
    ax[1, 0].axvline(s["mean_peak_share"], c="r", lw=1.2,
                     label=f"mean={s['mean_peak_share']:.3f}")
    ax[1, 0].set_title("WTA hardness: peak_share (dominant species fraction)")
    ax[1, 0].set_xlabel("Y_frac of dominant species")
    ax[1, 0].set_ylabel("# examples")
    ax[1, 0].legend(fontsize=8)

    # (1,1) number of species with Y_frac > 5% per example (novel)
    maxk = int(s["_n_active_per_ex"].max())
    bins = np.arange(0, max(maxk + 2, 3)) - 0.5
    ax[1, 1].hist(s["_n_active_per_ex"], bins=bins, color="#e7298a")
    ax[1, 1].axvline(s["mean_n_active"], c="k", lw=1.2,
                     label=f"mean={s['mean_n_active']:.2f}")
    ax[1, 1].set_title("species with Y_frac > 5% per example")
    ax[1, 1].set_xlabel("# active species")
    ax[1, 1].set_ylabel("# examples")
    ax[1, 1].legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return core.save_fig(fig, outdir, f"utilization_{label}.png")


def _plot_sweep(plt, rows, outdir):
    """Effective node count vs n_nodes across the capacity sweep (novel)."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))

    n_nodes = np.array([r["n_nodes"] for r in rows], dtype=float)
    eff = np.array([r["eff_node_count"] for r in rows], dtype=float)
    n_active = np.array([r["mean_n_active"] for r in rows], dtype=float)
    peak = np.array([r["mean_peak_share"] for r in rows], dtype=float)
    acc = np.array([r["accuracy"] for r in rows], dtype=float)
    labels = [r["label"] for r in rows]
    N_c = rows[0]["N_c"] if rows else 4

    # (0) effective node count vs n_nodes
    ax[0].plot(n_nodes, n_nodes, ls=":", c="k", lw=0.9, label="y=n_nodes")
    ax[0].axhline(N_c, ls="--", c="#1b9e77", lw=0.9, label=f"N_c={N_c}")
    ax[0].axhline(2 * N_c, ls="--", c="#d95f02", lw=0.9, label=f"2*N_c={2*N_c}")
    ax[0].scatter(n_nodes, eff, s=70, c="#2c7fb8", zorder=3,
                  label="effective node count")
    ax[0].scatter(n_nodes, n_active, s=45, c="#e7298a", marker="s",
                  zorder=3, label="mean # active (>5%)")
    for x, y, lab in zip(n_nodes, eff, labels):
        ax[0].annotate(lab, (x, y), fontsize=6, xytext=(4, 4),
                       textcoords="offset points")
    ax[0].set_title("capacity sweep: utilization vs n_nodes (novel)")
    ax[0].set_xlabel("n_nodes (model capacity)")
    ax[0].set_ylabel("effective node count")
    ax[0].legend(fontsize=7)

    # (1) WTA hardness vs accuracy
    sc = ax[1].scatter(peak, acc, s=70, c=n_nodes, cmap="viridis", zorder=3)
    for x, y, lab in zip(peak, acc, labels):
        ax[1].annotate(lab, (x, y), fontsize=6, xytext=(4, 4),
                       textcoords="offset points")
    cb = fig.colorbar(sc, ax=ax[1])
    cb.set_label("n_nodes")
    ax[1].set_title("WTA hardness vs novel accuracy")
    ax[1].set_xlabel("mean peak_share (soft <-> hard WTA)")
    ax[1].set_ylabel("novel accuracy (%)")

    fig.tight_layout()
    return core.save_fig(fig, outdir, "utilization_sweep.png")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def run(checkpoints, traces, outdir) -> dict:
    """M2 node utilization.

    checkpoints: list[core.Checkpoint]
    traces:      {label: {'in_dist': Trace, 'novel': Trace}}
    outdir:      Path for this module's figures
    Returns a dict of JSON-serializable scalar findings.
    """
    plt = core.setup_style()
    outdir = core.module_outdir("m2_utilization") if outdir is None else outdir

    per_ck = {}
    sweep_rows = []
    figures = []

    for ck in checkpoints:
        tr_novel = traces[ck.label]["novel"]
        tr_indist = traces[ck.label]["in_dist"]
        # N_c = number of context positions = q-shape of the model's B / trace
        N_c = int(tr_novel.q.shape[1])

        s_novel = utilization_stats(tr_novel)
        s_indist = utilization_stats(tr_indist)
        s_novel["solution_class"] = classify_solution(
            s_novel["eff_node_count"], N_c)

        # Gauge-invariant effective node count. M5 found Y_frac (hence the
        # eff_node_count above) is gauge-dependent: Y_r carries the per-species
        # gauge factor. The per-species OUTPUT CONTRIBUTION  Y_r * ||B[r,:]||
        # IS gauge-invariant (under (Y_r, B_r) -> (lam*Y_r, B_r/lam)). Report
        # both: the Y_frac version is the trained network's actual concentration
        # spread; this version is the function-level capacity measure.
        Bn = np.linalg.norm(ck.model.B.detach().cpu().numpy(), axis=1)  # (n,)
        contrib = (tr_novel.Y * Bn[None, :]).mean(axis=0)               # (n,)
        s_novel["eff_node_count_gauge_inv"] = round(
            participation_ratio(contrib), 4)

        fig_path = _plot_per_checkpoint(
            plt, ck.label, s_novel, s_indist, N_c, outdir)
        figures.append(str(fig_path))

        # public (JSON-serializable) record -- strip private arrays
        rec = {k: v for k, v in s_novel.items() if not k.startswith("_")}
        rec["N_c"] = N_c
        rec["in_dist_eff_node_count"] = s_indist["eff_node_count"]
        rec["in_dist_mean_peak_share"] = s_indist["mean_peak_share"]
        rec["in_dist_accuracy"] = s_indist["accuracy"]
        per_ck[ck.label] = rec

        sweep_rows.append(dict(label=ck.label, **rec))

    # capacity-sweep figure (sorted by n_nodes for a clean curve)
    if sweep_rows:
        sweep_rows.sort(key=lambda r: (r["n_nodes"], r["label"]))
        fig_path = _plot_sweep(plt, sweep_rows, outdir)
        figures.append(str(fig_path))

    # ---- aggregate findings -------------------------------------------------
    eff_counts = [r["eff_node_count"] for r in per_ck.values()]
    eff_counts_gi = [r["eff_node_count_gauge_inv"] for r in per_ck.values()]
    peak_shares = [r["mean_peak_share"] for r in per_ck.values()]
    n_actives = [r["mean_n_active"] for r in per_ck.values()]

    findings = dict(
        module="m2_utilization",
        n_checkpoints=len(checkpoints),
        per_checkpoint=per_ck,
        mean_effective_node_count=round(float(np.mean(eff_counts)), 4)
        if eff_counts else None,
        mean_effective_node_count_gauge_inv=round(float(np.mean(eff_counts_gi)), 4)
        if eff_counts_gi else None,
        mean_peak_share=round(float(np.mean(peak_shares)), 4)
        if peak_shares else None,
        mean_n_active=round(float(np.mean(n_actives)), 4)
        if n_actives else None,
        figures=figures,
    )
    return findings


# ---------------------------------------------------------------------------
def _main():
    print("[m2_utilization] loading checkpoints + traces via core.load_all() ...")
    checkpoints, traces = core.load_all()
    print(f"[m2_utilization] {len(checkpoints)} checkpoint(s) discovered.")
    outdir = core.module_outdir("m2_utilization")
    findings = run(checkpoints, traces, outdir)

    print("\n=== M2 node utilization -- findings ===")
    for label, rec in findings["per_checkpoint"].items():
        print(f"\n  {label}  (n_nodes={rec['n_nodes']}, N_c={rec['N_c']}, "
              f"novel acc={rec['accuracy']}%)")
        print(f"    effective node count : {rec['eff_node_count']:6.3f}  "
              f"-> {rec['solution_class']}")
        print(f"    mean # active (>5%)  : {rec['mean_n_active']:6.3f}    "
              f"species ever dominant: {rec['n_ever_dom']}/{rec['n_nodes']}")
        print(f"    mean peak_share      : {rec['mean_peak_share']:6.3f}  "
              f"(median {rec['median_peak_share']:.3f}, "
              f"min {rec['min_peak_share']:.3f})  -- SOFT WTA")
    print(f"\n  AGGREGATE: mean effective node count = "
          f"{findings['mean_effective_node_count']}, "
          f"mean peak_share = {findings['mean_peak_share']}, "
          f"mean # active = {findings['mean_n_active']}")
    print(f"  figures written: {len(findings['figures'])}")
    for f in findings["figures"]:
        print(f"    {f}")
    return findings


if __name__ == "__main__":
    _main()
