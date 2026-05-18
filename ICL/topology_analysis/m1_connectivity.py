"""
M1 -- connectivity & subspace projection.

Question: what does each species' input projection W_j detect?

The encoder projects the WHOLE flattened input X (= [ctx0..ctx_{N-1}, query])
through each species: W_j . X. The paper's claim is that species act as
"comparison-subspace detectors" -- a clean comparator for context position i
has equal-and-opposite weight on context-block i and the query block, i.e.
W_j[block i] ~ -W_j[query block]. core.comparison_scores() measures exactly
this (cosine of context-block-i weights vs negated query-block weights).

This module produces, per checkpoint:
  - a heatmap of comparison_scores  (n x N_c)  -- which species are clean
    "position i vs query" comparators;
  - a grid of W_j reshaped to (N+1 positions x D dims) heatmaps;
  - an SVD of W -> effective rank (energy-based) + a singular-value spectrum;
  - identification of near-zero-norm "default" species.

It returns gauge-invariant scalars only: W is untouched by the gauge symmetry,
so every quantity here is identifiable.

Generality: nothing is hard-coded to n=8/12 or to N=4 -- shapes come from each
checkpoint's W and params.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

import numpy as np


# --- tunable thresholds -----------------------------------------------------
COMPARATOR_COS = 0.6      # cosine >= this on some context position => comparator
DEFAULT_W_FRAC = 0.15     # ||W_j|| <= this * median(||W||) => default species
SVD_ENERGY = 0.95         # effective rank = #SV to reach this energy fraction


# ===========================================================================
# Per-checkpoint analysis
# ===========================================================================
def _analyze(W, N, D):
    """Compute all M1 diagnostics for one checkpoint's W.

    W: (n, (N+1)*D). Returns a dict of arrays/scalars (numpy / python).
    """
    n = W.shape[0]
    Wr = W.reshape(n, N + 1, D)                       # (n, N+1, D)

    # --- per-species projection norm ---------------------------------------
    w_norm = np.linalg.norm(W, axis=1)                # (n,)
    med_norm = float(np.median(w_norm))
    # "default" species: near-zero projection -> near-constant score, wins
    # by default. Guard against an all-zero W (median 0).
    norm_thresh = DEFAULT_W_FRAC * med_norm if med_norm > 0 else 1e-9
    is_default = w_norm <= max(norm_thresh, 1e-9)

    # --- comparison (subspace-projection) scores ---------------------------
    cmp_scores = core.comparison_scores(W, N, D)      # (n, N)
    # best context position each species comparates, and how cleanly
    best_pos = cmp_scores.argmax(axis=1)              # (n,)
    best_cos = cmp_scores.max(axis=1)                 # (n,)
    # a comparator is a non-default species with a clean equal-and-opposite
    # signature on at least one context position
    is_comparator = (~is_default) & (best_cos >= COMPARATOR_COS)

    # positions covered by >=1 comparator
    covered = np.zeros(N, dtype=bool)
    for j in np.where(is_comparator)[0]:
        for i in range(N):
            if cmp_scores[j, i] >= COMPARATOR_COS:
                covered[i] = True
    # comparators-per-position (a species can cover several positions)
    comps_per_pos = np.zeros(N, dtype=int)
    for i in range(N):
        comps_per_pos[i] = int(
            ((cmp_scores[:, i] >= COMPARATOR_COS) & is_comparator).sum())

    # --- SVD / effective rank ----------------------------------------------
    sv = np.linalg.svd(W, compute_uv=False)           # (min(n,(N+1)D),)
    energy = np.cumsum(sv ** 2) / (np.sum(sv ** 2) + 1e-12)
    eff_rank_energy = int(np.searchsorted(energy, SVD_ENERGY) + 1)
    # participation-ratio rank (smooth, gauge-free alternative)
    s2 = sv ** 2
    eff_rank_pr = float((s2.sum() ** 2) / ((s2 ** 2).sum() + 1e-12))

    return dict(
        n=n, N=N, D=D, Wr=Wr,
        w_norm=w_norm, med_norm=med_norm, is_default=is_default,
        cmp_scores=cmp_scores, best_pos=best_pos, best_cos=best_cos,
        is_comparator=is_comparator, covered=covered,
        comps_per_pos=comps_per_pos,
        sv=sv, energy=energy,
        eff_rank_energy=eff_rank_energy, eff_rank_pr=eff_rank_pr,
    )


# ===========================================================================
# Figures
# ===========================================================================
def _fig_comparison_heatmap(plt, results, outdir):
    """Heatmap grid of comparison_scores (n x N_c) for every checkpoint."""
    items = list(results.items())
    ncol = min(len(items), 2)
    nrow = int(np.ceil(len(items) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 3.6 * nrow),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (label, r) in zip(axes.flat, items):
        ax.axis("on")
        cs = r["cmp_scores"]
        im = ax.imshow(cs, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"{label}\ncomparison scores  (n={r['n']})")
        ax.set_xlabel("context position i")
        ax.set_ylabel("species j")
        ax.set_xticks(range(r["N"]))
        ax.set_yticks(range(r["n"]))
        # annotate cells + flag comparator / default species
        for j in range(r["n"]):
            tag = ""
            if r["is_default"][j]:
                tag = " D"
            elif r["is_comparator"][j]:
                tag = " *"
            if tag:
                ax.text(r["N"] - 0.5 + 0.05, j, tag, va="center", ha="left",
                        fontsize=8, fontweight="bold")
            for i in range(r["N"]):
                ax.text(i, j, f"{cs[j, i]:.2f}", va="center", ha="center",
                        fontsize=6,
                        color="white" if abs(cs[j, i]) > 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="cos(W_ctx_i, -W_query)")
    fig.suptitle("M1: subspace-projection comparators  "
                 "(* = comparator, D = default species)", y=1.02)
    fig.tight_layout()
    return core.save_fig(fig, outdir, "comparison_scores.png")


def _fig_W_grids(plt, results, outdir):
    """For each checkpoint: grid of W_j reshaped to (N+1 positions x D dims)."""
    paths = []
    for label, r in results.items():
        n, N, D = r["n"], r["N"], r["D"]
        Wr = r["Wr"]
        vmax = float(np.abs(Wr).max()) or 1.0
        ncol = min(n, 4)
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(2.4 * ncol, 2.1 * nrow),
                                 squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for j in range(n):
            ax = axes.flat[j]
            ax.axis("on")
            im = ax.imshow(Wr[j], aspect="auto", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax)
            kind = ("default" if r["is_default"][j]
                    else ("comparator" if r["is_comparator"][j] else "mixed"))
            ax.set_title(f"W[{j}]  {kind}\n||W||={r['w_norm'][j]:.2f}",
                         fontsize=8)
            ax.set_xlabel("dim", fontsize=7)
            ax.set_ylabel("position", fontsize=7)
            ax.set_xticks(range(D))
            yt = [f"ctx{i}" for i in range(N)] + ["query"]
            ax.set_yticks(range(N + 1))
            ax.set_yticklabels(yt, fontsize=6)
            ax.tick_params(labelsize=6)
        fig.suptitle(f"M1: W_j reshaped (position x dim) -- {label}", y=1.01)
        fig.tight_layout()
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        paths.append(core.save_fig(fig, outdir,
                                   f"W_grid_{label}.png"))
    return paths


def _fig_svd(plt, results, outdir):
    """Singular-value spectra + cumulative energy for every checkpoint."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for label, r in results.items():
        sv = r["sv"]
        x = np.arange(1, len(sv) + 1)
        ax1.plot(x, sv, "o-", label=f"{label} (rank~{r['eff_rank_energy']})")
        ax2.plot(x, r["energy"], "o-", label=label)
    ax1.set_title("M1: singular values of W")
    ax1.set_xlabel("index")
    ax1.set_ylabel("singular value")
    ax1.legend(fontsize=7)
    ax2.axhline(SVD_ENERGY, color="gray", ls="--", lw=1,
                label=f"{SVD_ENERGY:.0%} energy")
    ax2.set_title("M1: cumulative energy of W")
    ax2.set_xlabel("# singular values")
    ax2.set_ylabel("cumulative energy fraction")
    ax2.set_ylim(0, 1.02)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    return core.save_fig(fig, outdir, "svd_spectrum.png")


# ===========================================================================
# Module entry point
# ===========================================================================
def run(checkpoints, traces, outdir):
    """Build the M1 connectivity / subspace-projection analysis.

    checkpoints: list[core.Checkpoint]
    traces:      {label: {'in_dist': Trace, 'novel': Trace}}  (unused here --
                 M1 is purely a weight-space analysis, gauge-invariant)
    outdir:      Path for this module's figures.
    Returns a dict of JSON-serializable per-checkpoint scalar findings.
    """
    plt = core.setup_style()
    outdir = core.module_outdir("m1_connectivity") if outdir is None else outdir

    results = {}
    for ck in checkpoints:
        pp = core.physical_params(ck.model)
        W = pp["W"]
        N = int(ck.params["N"])
        D = int(ck.params["D"])
        # sanity: W is (n, (N+1)*D)
        if W.shape[1] != (N + 1) * D:
            # fall back to inferring D from W if params disagree
            D = W.shape[1] // (N + 1)
        results[ck.label] = _analyze(W, N, D)

    # --- figures -----------------------------------------------------------
    fig_paths = {}
    if results:
        fig_paths["comparison_scores"] = str(
            _fig_comparison_heatmap(plt, results, outdir))
        fig_paths["W_grids"] = [str(p) for p in
                                _fig_W_grids(plt, results, outdir)]
        fig_paths["svd_spectrum"] = str(_fig_svd(plt, results, outdir))

    # --- findings dict -----------------------------------------------------
    findings = {}
    for label, r in results.items():
        n_comp = int(r["is_comparator"].sum())
        n_def = int(r["is_default"].sum())
        n_mixed = r["n"] - n_comp - n_def
        findings[label] = dict(
            n_species=int(r["n"]),
            n_context_positions=int(r["N"]),
            eff_rank_energy=int(r["eff_rank_energy"]),
            eff_rank_participation=round(float(r["eff_rank_pr"]), 3),
            n_comparator_species=n_comp,
            n_default_species=n_def,
            n_mixed_species=int(n_mixed),
            n_positions_covered=int(r["covered"].sum()),
            all_positions_covered=bool(r["covered"].all()),
            comparators_per_position=[int(x) for x in r["comps_per_pos"]],
            max_comparison_cosine=round(float(r["best_cos"].max()), 3),
            mean_best_comparison_cosine=round(float(r["best_cos"].mean()), 3),
            median_W_norm=round(float(r["med_norm"]), 4),
            min_W_norm=round(float(r["w_norm"].min()), 4),
            default_species_idx=[int(j) for j in
                                 np.where(r["is_default"])[0]],
            comparator_species_idx=[int(j) for j in
                                    np.where(r["is_comparator"])[0]],
        )

    summary = dict(
        module="m1_connectivity",
        n_checkpoints=len(results),
        figures=fig_paths,
        per_checkpoint=findings,
    )

    # --- console report ----------------------------------------------------
    print(f"\n=== M1 connectivity & subspace projection "
          f"({len(results)} checkpoint(s)) ===")
    for label, f in findings.items():
        print(f"\n  {label}:  n_species={f['n_species']}  "
              f"eff_rank(energy)={f['eff_rank_energy']}  "
              f"eff_rank(PR)={f['eff_rank_participation']}")
        print(f"    comparators={f['n_comparator_species']}  "
              f"default={f['n_default_species']}  "
              f"mixed={f['n_mixed_species']}")
        print(f"    positions covered: {f['n_positions_covered']}/"
              f"{f['n_context_positions']}  "
              f"(all={f['all_positions_covered']})  "
              f"comparators/pos={f['comparators_per_position']}")
        print(f"    max comparison cosine={f['max_comparison_cosine']}  "
              f"mean-best={f['mean_best_comparison_cosine']}")
    print()
    return summary


# ===========================================================================
# Standalone
# ===========================================================================
if __name__ == "__main__":
    checkpoints, traces = core.load_all()
    outdir = core.module_outdir("m1_connectivity")
    out = run(checkpoints, traces, outdir)
    print(f"M1 done. Figures in: {outdir}")
