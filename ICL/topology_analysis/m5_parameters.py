"""
M5 -- physical reaction parameters.

Question: what do the (gauge-invariant) reaction parameters actually do?

The WTA model has an exact per-species gauge freedom
    (K_r, beta_r, B[r,:]) -> (lambda*K_r, beta_r/lambda, B[r,:]/lambda)
so raw K_r and beta_r are individually UNPHYSICAL. The identifiable
quantities are:
    W              the input projections (gauge-invariant)
    Kbeta = K*beta the winner / softmin score scale
    eff_decoder    K_r * B[r,:], the effective decoder weight

This module:
  * plots the gauge-invariant quantities (W-row norms, K*beta, K*B) per
    checkpoint; raw K and beta are shown only as an explicitly-labelled
    "gauge-dependent" diagnostic;
  * empirically confirms that the winner ranking / softmin selection depends
    ONLY on K*beta (a random gauge transform leaves winner/Y unchanged, and
    the softmin weights are reproduced from K*beta + f alone);
  * scatters K*beta and W-row norm against species utilization (mean Y_frac)
    to test whether low-score / high-W species win more, and flags default
    (near-zero-W) species.

Entry point: run(checkpoints, traces, outdir) -> dict of scalar findings.
Standalone:  python m5_parameters.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _spearman(a, b):
    """Spearman rank correlation; nan-safe, returns 0.0 for degenerate input."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def _pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def _verify_gauge_invariance(checkpoint, eval_set):
    """Apply a random per-species gauge transform to a copy of the model and
    measure what changes and what does not.

    The gauge is (K_r, beta_r, B[r,:]) -> (lambda*K_r, beta_r/lambda, B[r,:]/lambda).
    Predictions:
      * The selection rule (ratio = beta/f, winner, softmin) depends only on
        K*beta -- invariant -> winner / softmin unchanged.
      * Y itself carries a K_r factor (Y_potential_j = K_j*softplus(...)), so
        raw Y AND its simplex-normalised Y_frac are GAUGE-DEPENDENT internal
        quantities -- they DO change.
      * But q = Y @ B has the K_r factor cancelled by B -> B/lambda, so q,
        attention, and the prediction -- the model's actual FUNCTION -- are
        invariant.

    Returns a dict of max-absolute-deviations: function quantities ~0,
    Y_frac deliberately reported so the gauge-dependence of the internal
    representation is explicit.
    """
    import copy
    import torch

    model = checkpoint.model
    temperature = float(checkpoint.params["temperature"])
    base = core.instrument(model, eval_set, temperature)

    n = int(model.n_nodes)
    rng = np.random.default_rng(12345)
    lam = rng.uniform(0.3, 3.0, size=n).astype(np.float32)   # gauge factors

    gm = copy.deepcopy(model)
    with torch.no_grad():
        # (K, beta, B) -> (lam*K, beta/lam, B/lam)
        gm.log_K.add_(torch.from_numpy(np.log(lam)).to(gm.log_K.dtype))
        gm.log_beta.add_(torch.from_numpy(-np.log(lam)).to(gm.log_beta.dtype))
        gm.B.mul_(torch.from_numpy(1.0 / lam[:, None]).to(gm.B.dtype))
    gm.eval()
    pert = core.instrument(gm, eval_set, temperature)

    return dict(
        # selection rule -- gauge-INVARIANT (depends only on K*beta)
        winner_changed_frac=float((base.winner != pert.winner).mean()),
        softmin_max_dev=float(np.abs(base.softmin_w - pert.softmin_w).max()),
        # model function (decoded output) -- gauge-INVARIANT
        q_max_dev=float(np.abs(base.q - pert.q).max()),
        attention_max_dev=float(np.abs(base.attention - pert.attention).max()),
        pred_changed_frac=float((base.pred_pos != pert.pred_pos).mean()),
        # internal representation -- gauge-DEPENDENT (expected to change)
        Yfrac_max_dev=float(np.abs(base.Y_frac - pert.Y_frac).max()),
        dom_changed_frac=float((base.dom_species != pert.dom_species).mean()),
    )


def _selection_from_gauge_invariants(model, z_flat, Kbeta, tau):
    """Reconstruct the WTA selection score, winner, and softmin weights using
    ONLY gauge-invariant quantities (W and K*beta).

    Algebra (from compute_reaction_rates):
        f_j   = alpha * softplus(W_j . X) / K_j
        ratio = beta_j / f_j
              = beta_j * K_j / (alpha * softplus(W_j . X))
              = (K_j*beta_j) / (alpha * softplus(W_j . X))
    The raw K_j and beta_j cancel into the gauge-invariant product K*beta, so
    the selection score -- and hence winner = argmin ratio and the softmin
    weights softmax(-ratio/tau) -- depend ONLY on W and K*beta. This function
    rebuilds them from those two and is compared against the true trace to
    confirm the claim empirically.

    Returns (ratio, winner, softmin_w) computed from gauge-invariants.
    """
    import torch

    Wm = (model.W * model.W_mask).detach()
    w0 = model.w0.detach()
    alpha = float(model.alpha)
    bsp = float(model.beta_softplus)

    z = torch.from_numpy(np.asarray(z_flat)).float()
    lin = z @ Wm.T + w0.unsqueeze(0)
    sp = torch.nn.functional.softplus(lin, beta=bsp).numpy()      # (M, n)
    sp = np.clip(sp, 1e-12, None)

    ratio = Kbeta[None, :] / (alpha * sp)                         # (M, n)
    winner = ratio.argmin(axis=1)
    z_score = -(ratio - ratio.min(axis=1, keepdims=True)) / tau
    w = np.exp(z_score)
    softmin_w = w / w.sum(axis=1, keepdims=True)
    return ratio, winner, softmin_w


# ---------------------------------------------------------------------------
# per-checkpoint analysis
# ---------------------------------------------------------------------------
def _analyze_one(checkpoint, traces, eval_set, plt, outdir):
    label = checkpoint.label
    n = checkpoint.n_nodes
    N, D = core.SHARED_CONFIG["N"], core.SHARED_CONFIG["D"]
    tau = float(checkpoint.model.get_annealed_params()["tau"])

    pp = core.physical_params(checkpoint.model)
    W = pp["W"]                       # (n, (N+1)*D)
    Kbeta = pp["Kbeta"]               # (n,)
    eff_dec = pp["eff_decoder"]       # (n, N)
    K_raw, beta_raw = pp["K"], pp["beta"]

    w_norm = np.linalg.norm(W, axis=1)            # gauge-invariant
    eff_dec_norm = np.linalg.norm(eff_dec, axis=1)

    # default species: near-zero W relative to the per-checkpoint max.
    w_rel = w_norm / (w_norm.max() + 1e-12)
    default_mask = w_rel < 0.15
    default_species = [int(j) for j in np.where(default_mask)[0]]

    # utilization on the novel split (true ICL)
    tr = traces[label]["novel"]
    mean_Yfrac = tr.Y_frac.mean(axis=0)                       # (n,)
    dom_counts = np.bincount(tr.dom_species, minlength=n)
    win_counts = np.bincount(tr.winner, minlength=n)

    # --- gauge invariance check -------------------------------------------
    gauge = _verify_gauge_invariance(checkpoint, eval_set)

    # --- selection reconstructed from gauge-invariants (W, K*beta) only ----
    ratio_gi, winner_gi, sm_gi = _selection_from_gauge_invariants(
        checkpoint.model, tr.z_flat, Kbeta, tau)
    recon_winner_agree = float((winner_gi == tr.winner).mean())
    softmin_recon_dev = float(np.abs(sm_gi - tr.softmin_w).max())
    # rank correlation of the reconstructed vs true selection score, per example
    ratio_rank_corr = float(np.mean([
        _spearman(ratio_gi[i], tr.ratios[i]) for i in range(min(500, tr.n))]))

    # --- correlations: does parameter X predict utilization? --------------
    corr_kbeta_util = _spearman(Kbeta, mean_Yfrac)
    corr_wnorm_util = _spearman(w_norm, mean_Yfrac)
    corr_effdec_util = _spearman(eff_dec_norm, mean_Yfrac)
    # low score should win more -> negative corr expected if score == K*beta
    corr_kbeta_domfrac = _spearman(Kbeta, dom_counts / max(1, dom_counts.sum()))

    # ------------------------------------------------------------------
    # FIGURE 1: gauge-invariant parameters per species
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    sp = np.arange(n)
    cols = ["#d62728" if d else "#1f77b4" for d in default_mask]

    ax = axes[0, 0]
    ax.bar(sp, w_norm, color=cols)
    ax.set_title("W-row norm  (gauge-invariant)")
    ax.set_xlabel("species"); ax.set_ylabel("||W_j||")

    ax = axes[0, 1]
    ax.bar(sp, Kbeta, color=cols)
    ax.set_title("K*beta  (gauge-invariant: softmin score)")
    ax.set_xlabel("species"); ax.set_ylabel("K_j * beta_j")
    ax.set_yscale("log")

    ax = axes[0, 2]
    ax.bar(sp, eff_dec_norm, color=cols)
    ax.set_title("||K*B||  (gauge-invariant: eff. decoder)")
    ax.set_xlabel("species"); ax.set_ylabel("||K_j * B_j||")

    ax = axes[1, 0]
    im = ax.imshow(eff_dec, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(eff_dec).max(), vmax=np.abs(eff_dec).max())
    ax.set_title("effective decoder  K_j * B[j,:]  (gauge-invariant)")
    ax.set_xlabel("context position"); ax.set_ylabel("species")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    width = 0.4
    ax.bar(sp - width / 2, K_raw / K_raw.max(), width,
           color="#999999", label="K (norm.)")
    ax.bar(sp + width / 2, beta_raw / beta_raw.max(), width,
           color="#cccccc", label="beta (norm.)")
    ax.set_title("raw K, beta  --  GAUGE-DEPENDENT (unphysical)")
    ax.set_xlabel("species"); ax.legend(fontsize=7)

    ax = axes[1, 2]
    ax.bar(sp, mean_Yfrac, color=cols)
    ax.set_title("mean Y_frac  (utilization, novel split)")
    ax.set_xlabel("species"); ax.set_ylabel("mean Y_frac")

    fig.suptitle(f"M5 gauge-invariant parameters -- {label} "
                 f"(n_nodes={n}, tau={tau:.3g}); red = default (||W||<0.15 max)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    core.save_fig(fig, outdir, f"m5_params_{label}.png")

    # ------------------------------------------------------------------
    # FIGURE 2: parameter vs utilization scatter
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    ax.scatter(Kbeta, mean_Yfrac, c=cols, s=60, edgecolor="k")
    for j in range(n):
        ax.annotate(str(j), (Kbeta[j], mean_Yfrac[j]), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("K*beta  (softmin score)")
    ax.set_ylabel("mean Y_frac (utilization)")
    ax.set_title(f"K*beta vs use   Spearman={corr_kbeta_util:+.2f}")

    ax = axes[1]
    ax.scatter(w_norm, mean_Yfrac, c=cols, s=60, edgecolor="k")
    for j in range(n):
        ax.annotate(str(j), (w_norm[j], mean_Yfrac[j]), fontsize=7)
    ax.set_xlabel("||W_j||")
    ax.set_ylabel("mean Y_frac (utilization)")
    ax.set_title(f"W-norm vs use   Spearman={corr_wnorm_util:+.2f}")

    ax = axes[2]
    ax.scatter(eff_dec_norm, mean_Yfrac, c=cols, s=60, edgecolor="k")
    for j in range(n):
        ax.annotate(str(j), (eff_dec_norm[j], mean_Yfrac[j]), fontsize=7)
    ax.set_xlabel("||K*B_j||  (eff. decoder)")
    ax.set_ylabel("mean Y_frac (utilization)")
    ax.set_title(f"eff-decoder vs use   Spearman={corr_effdec_util:+.2f}")

    fig.suptitle(f"M5 parameter vs utilization -- {label}  "
                 f"(red = default species)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    core.save_fig(fig, outdir, f"m5_scatter_{label}.png")

    # ------------------------------------------------------------------
    # per-checkpoint findings
    # ------------------------------------------------------------------
    return dict(
        n_nodes=n,
        n_species=int(n),
        tau=tau,
        n_default_species=int(default_mask.sum()),
        default_species=default_species,
        kbeta_min=float(Kbeta.min()),
        kbeta_max=float(Kbeta.max()),
        kbeta_ratio=float(Kbeta.max() / (Kbeta.min() + 1e-30)),
        wnorm_min=float(w_norm.min()),
        wnorm_max=float(w_norm.max()),
        # gauge invariance verification:
        #   function quantities (winner/softmin/q/attention/pred) -> ~0
        #   internal representation (Y_frac/dom) -> changes (gauge-dependent)
        gauge_winner_changed_frac=gauge["winner_changed_frac"],
        gauge_softmin_max_dev=gauge["softmin_max_dev"],
        gauge_q_max_dev=gauge["q_max_dev"],
        gauge_attention_max_dev=gauge["attention_max_dev"],
        gauge_pred_changed_frac=gauge["pred_changed_frac"],
        gauge_Yfrac_max_dev=gauge["Yfrac_max_dev"],
        gauge_dom_changed_frac=gauge["dom_changed_frac"],
        # winner ranking depends only on W and K*beta (gauge-invariants)
        softmin_from_kbeta_max_dev=softmin_recon_dev,
        winner_from_kbeta_agreement=recon_winner_agree,
        ratio_rank_corr_from_kbeta=ratio_rank_corr,
        # parameter vs utilization
        corr_kbeta_utilization=corr_kbeta_util,
        corr_wnorm_utilization=corr_wnorm_util,
        corr_effdec_utilization=corr_effdec_util,
        corr_kbeta_dom_frac=corr_kbeta_domfrac,
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def run(checkpoints, traces, outdir):
    """M5: physical (gauge-invariant) reaction-parameter analysis.

    checkpoints: list[core.Checkpoint]
    traces:      {label: {'in_dist': Trace, 'novel': Trace}}
    outdir:      Path for this module's figures.
    Returns a dict of JSON-serializable scalar findings.
    """
    plt = core.setup_style()
    outdir = core.module_outdir(os.path.basename(str(outdir))) \
        if not hasattr(outdir, "mkdir") else outdir

    # one shared eval set for the gauge-invariance check (novel split)
    eval_sets = core.make_eval_sets()
    eval_set = eval_sets["novel"]

    per_ck = {}
    for ck in checkpoints:
        per_ck[ck.label] = _analyze_one(ck, traces, eval_set, plt, outdir)

    # ----- cross-checkpoint summary figure --------------------------------
    if per_ck:
        labels = list(per_ck.keys())
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        x = np.arange(len(labels))

        ax = axes[0]
        ax.bar(x - 0.2, [per_ck[l]["gauge_attention_max_dev"] for l in labels],
               0.4, color="#2ca02c", label="attention (function)")
        ax.bar(x + 0.2, [per_ck[l]["gauge_Yfrac_max_dev"] for l in labels],
               0.4, color="#d62728", label="Y_frac (internal repr.)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("max |dev| after gauge transform")
        ax.set_title("function is gauge-invariant; internal Y_frac is not")
        ax.legend(fontsize=7)

        ax = axes[1]
        ax.bar(x - 0.2, [per_ck[l]["corr_kbeta_utilization"] for l in labels],
               0.4, color="#1f77b4", label="corr(K*beta, use)")
        ax.bar(x + 0.2, [per_ck[l]["corr_wnorm_utilization"] for l in labels],
               0.4, color="#ff7f0e", label="corr(||W||, use)")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Spearman correlation")
        ax.set_title("parameter vs utilization")
        ax.legend(fontsize=7)

        fig.suptitle("M5 cross-checkpoint summary", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        core.save_fig(fig, outdir, "m5_summary.png")

    # ----- aggregate scalars ----------------------------------------------
    def _mean(key):
        vals = [v[key] for v in per_ck.values()]
        return float(np.mean(vals)) if vals else 0.0

    findings = {
        "n_checkpoints": len(per_ck),
        "per_checkpoint": per_ck,
        # gauge invariance of the model FUNCTION: these should all be ~0 ->
        # confirms raw K, beta unphysical but the computation is well-defined
        "gauge_invariance_max_softmin_dev":
            float(max((v["gauge_softmin_max_dev"]
                       for v in per_ck.values()), default=0.0)),
        "gauge_invariance_max_attention_dev":
            float(max((v["gauge_attention_max_dev"]
                       for v in per_ck.values()), default=0.0)),
        "gauge_invariance_max_winner_changed_frac":
            float(max((v["gauge_winner_changed_frac"]
                       for v in per_ck.values()), default=0.0)),
        "gauge_invariance_max_pred_changed_frac":
            float(max((v["gauge_pred_changed_frac"]
                       for v in per_ck.values()), default=0.0)),
        # Y_frac is a GAUGE-DEPENDENT internal quantity -- it DOES change
        "gauge_dependent_max_Yfrac_dev":
            float(max((v["gauge_Yfrac_max_dev"]
                       for v in per_ck.values()), default=0.0)),
        # winner ranking depends only on W and K*beta (gauge-invariants)
        "mean_winner_from_kbeta_agreement": _mean("winner_from_kbeta_agreement"),
        "mean_ratio_rank_corr_from_kbeta": _mean("ratio_rank_corr_from_kbeta"),
        "max_softmin_from_kbeta_dev":
            float(max((v["softmin_from_kbeta_max_dev"]
                       for v in per_ck.values()), default=0.0)),
        # parameter vs utilization
        "mean_corr_kbeta_utilization": _mean("corr_kbeta_utilization"),
        "mean_corr_wnorm_utilization": _mean("corr_wnorm_utilization"),
        "mean_corr_effdec_utilization": _mean("corr_effdec_utilization"),
        "total_default_species":
            int(sum(v["n_default_species"] for v in per_ck.values())),
    }
    return findings


if __name__ == "__main__":
    import json

    checkpoints, traces = core.load_all()
    outdir = core.module_outdir("m5_parameters")
    result = run(checkpoints, traces, outdir)

    print("\n" + "=" * 70)
    print("M5 -- physical reaction parameters: findings")
    print("=" * 70)
    for ck_label, v in result["per_checkpoint"].items():
        print(f"\n{ck_label}  (n_nodes={v['n_nodes']}, tau={v['tau']:.3g})")
        print(f"  default species (||W|| < 0.15*max): "
              f"{v['n_default_species']}  {v['default_species']}")
        print(f"  K*beta range: [{v['kbeta_min']:.3g}, {v['kbeta_max']:.3g}]  "
              f"(ratio {v['kbeta_ratio']:.1f}x)")
        print(f"  GAUGE CHECK -- after random gauge transform:")
        print(f"    [function -- should be ~0]")
        print(f"    winner changed frac : {v['gauge_winner_changed_frac']:.2e}")
        print(f"    softmin max dev     : {v['gauge_softmin_max_dev']:.2e}")
        print(f"    attention max dev   : {v['gauge_attention_max_dev']:.2e}")
        print(f"    pred changed frac   : {v['gauge_pred_changed_frac']:.2e}")
        print(f"    [internal repr -- gauge-DEPENDENT, expected nonzero]")
        print(f"    Y_frac max dev      : {v['gauge_Yfrac_max_dev']:.2e}")
        print(f"  winner from (W,K*beta) agreement : "
              f"{v['winner_from_kbeta_agreement']*100:.1f}%  "
              f"(score rank-corr {v['ratio_rank_corr_from_kbeta']:+.3f})")
        print(f"  corr(K*beta, utilization)    : "
              f"{v['corr_kbeta_utilization']:+.3f}")
        print(f"  corr(||W||,  utilization)    : "
              f"{v['corr_wnorm_utilization']:+.3f}")
        print(f"  corr(eff-dec, utilization)   : "
              f"{v['corr_effdec_utilization']:+.3f}")

    print("\n" + "-" * 70)
    print("AGGREGATE")
    print(f"  checkpoints analyzed                 : {result['n_checkpoints']}")
    print(f"  gauge: max softmin deviation         : "
          f"{result['gauge_invariance_max_softmin_dev']:.2e}  (~0 => invariant)")
    print(f"  gauge: max attention deviation       : "
          f"{result['gauge_invariance_max_attention_dev']:.2e}  (~0 => invariant)")
    print(f"  gauge: max pred-changed fraction     : "
          f"{result['gauge_invariance_max_pred_changed_frac']:.2e}  (~0 => invariant)")
    print(f"  gauge: max Y_frac deviation          : "
          f"{result['gauge_dependent_max_Yfrac_dev']:.2e}  "
          f"(nonzero => Y_frac is gauge-dependent)")
    print(f"  mean winner from (W,K*beta) agreement : "
          f"{result['mean_winner_from_kbeta_agreement']*100:.1f}%  "
          f"(should be ~100%)")
    print(f"  mean selection-score rank correlation : "
          f"{result['mean_ratio_rank_corr_from_kbeta']:+.3f}")
    print(f"  mean corr(K*beta, utilization)       : "
          f"{result['mean_corr_kbeta_utilization']:+.3f}")
    print(f"  mean corr(||W||,  utilization)       : "
          f"{result['mean_corr_wnorm_utilization']:+.3f}")
    print(f"  total default species across runs   : "
          f"{result['total_default_species']}")
    print("\nfigures ->", outdir)
    print(json.dumps({k: v for k, v in result.items()
                      if k != "per_checkpoint"}, indent=2))
