"""
M3 -- the routing algorithm.

Question: how does species activity (the soft Y-mixture) decode to the answer?

The decoder is  q = Y @ B  ->  attention = softmax(q / temp)  -> argmax over the
N context positions. This module dissects that map:

  (a) Per-species decoder pattern  softmax(B[j,:]/temp): each species, considered
      alone, "votes" for one context position. Group species by their decoded
      position argmax_i B[j,i] and TEST the branch hypothesis -- do ~2-3 species
      share each of the N targets? If they do, probe whether their input
      projections W_j differ systematically (an empirical "branch" axis), rather
      than assuming a sign label.
  (b) dom_species x true_pos confusion matrix -- which dominant species fire on
      which copy positions.
  (c) Soft-mixture check: compare the model's true pred_pos to a
      dominant-species-ONLY prediction (q built from that single B-row). Low
      agreement => the graded Y-mixture is essential, not a tie-break.

All quantities come through core; nothing here is hard-coded to n=8/12 or a
fixed number of checkpoints.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)


def _species_decoder(ck):
    """Per-species decoder vote: softmax(B[j,:]/temp) over the N context slots.

    Uses the GAUGE-INVARIANT effective decoder K_r*B[r,:]: the attention temp
    and softmax make the per-row scale matter, and only K*B is identifiable.
    Returns (votes (n,N), decoded_pos (n,), eff_decoder (n,N))."""
    phys = core.physical_params(ck.model)
    eff = phys["eff_decoder"]                       # (n, N) = K_r * B[r,:]
    temp = float(ck.params["temperature"])
    votes = _softmax(eff / temp, axis=1)            # per-species attention
    decoded_pos = eff.argmax(axis=1)                # position each species votes
    return votes, decoded_pos, eff


def _dom_only_pred(tr, model):
    """Prediction if ONLY the dominant CONTRIBUTOR species were active.

    The dominant species is the one with the largest GAUGE-INVARIANT output
    contribution  Y_r * ||B[r,:]||  (M5 found argmax-Y / Y_frac are gauge-
    dependent; the product Y_r*||B_r|| is invariant under the gauge
    (Y_r, B_r) -> (lam*Y_r, B_r/lam)). Builds q from that single B-row and runs
    the same softmax-attention decode. Returns pred_pos_domonly (M,)."""
    B = model.B.detach().cpu().numpy()              # (n, N)
    Bnorm = np.linalg.norm(B, axis=1) + 1e-12       # (n,)
    contrib = tr.Y * Bnorm[None, :]                 # (M, n) gauge-invariant
    dom = contrib.argmax(axis=1)                    # (M,) dominant contributor
    temp = float(model_temperature(model))
    q_dom = B[dom, :]                               # (M, N) -- one B-row each
    att = _softmax(q_dom / temp, axis=1)
    return att.argmax(axis=1)


def model_temperature(model):
    """Decoder temperature -- attribute on the model if present, else fallback."""
    for attr in ("temperature", "temp", "decode_temperature"):
        if hasattr(model, attr):
            v = getattr(model, attr)
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 1.0


# ---------------------------------------------------------------------------
# per-checkpoint analysis
# ---------------------------------------------------------------------------
def _analyze_checkpoint(ck, tr):
    """Routing analysis for one checkpoint (on its `novel` Trace)."""
    n = ck.model.n_nodes
    N = tr.true_pos.max(initial=0) + 1
    N = int(max(N, tr.pred_pos.max(initial=0) + 1))
    # B is (n, N): the real N
    B = ck.model.B.detach().cpu().numpy()
    N = B.shape[1]
    temp = float(ck.params["temperature"])

    # --- (a) per-species decoder votes & target grouping --------------------
    votes, decoded_pos, eff = _species_decoder(ck)
    # how peaked is each species' own vote (max softmax weight)?
    species_vote_peak = votes.max(axis=1)
    # group species -> decoded position
    groups = {i: np.where(decoded_pos == i)[0].tolist() for i in range(N)}
    species_per_target = np.array([len(groups[i]) for i in range(N)], float)

    # --- branch probe: for shared-target species, do W rows differ? ---------
    # For each target with >=2 species, take the mean pairwise (1-cos) of their
    # W rows. High value => the co-decoding species detect genuinely different
    # input subspaces (an empirical "branch"); low => near-redundant copies.
    W = core.physical_params(ck.model)["W"]         # (n, (N+1)*D)
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    branch_dists = []
    branch_detail = {}
    for i in range(N):
        idx = groups[i]
        if len(idx) < 2:
            continue
        sub = Wn[idx]                               # (k, dim)
        cos = sub @ sub.T                           # (k, k)
        iu = np.triu_indices(len(idx), k=1)
        pair_dist = 1.0 - cos[iu]                   # (k*(k-1)/2,)
        branch_dists.extend(pair_dist.tolist())
        branch_detail[i] = float(np.mean(pair_dist))
    mean_branch_dist = float(np.mean(branch_dists)) if branch_dists else 0.0

    # --- (b) dom_species x true_pos confusion --------------------------------
    conf = np.zeros((n, N), float)
    for s, p in zip(tr.dom_species, tr.true_pos):
        conf[s, p] += 1.0
    # routing accuracy = pred_pos hits the true copy position
    routing_acc = 100.0 * float((tr.pred_pos == tr.true_pos).mean())

    # for each species, the position it MOST OFTEN dominates on
    dom_counts = conf.sum(axis=1)
    used = dom_counts > 0
    dom_fav_pos = np.full(n, -1, int)
    dom_fav_pos[used] = conf[used].argmax(axis=1)
    # consistency: do a species' functional favourite & B-vote agree?
    agree_mask = used & (dom_fav_pos == decoded_pos)
    decode_consistency = (100.0 * agree_mask.sum() / max(1, used.sum()))

    # --- (c) soft-mixture check ---------------------------------------------
    dom_only = _dom_only_pred(tr, ck.model)
    dom_only_agree = 100.0 * float((dom_only == tr.pred_pos).mean())
    dom_only_acc = 100.0 * float((dom_only == tr.true_pos).mean())
    # how often the mixture FLIPS the answer relative to the dominant row
    mixture_flip = 100.0 * float((dom_only != tr.pred_pos).mean())
    # of the flips, how often the mixture is the one that's right
    flips = dom_only != tr.pred_pos
    if flips.any():
        mix_rescue = 100.0 * float(
            ((tr.pred_pos == tr.true_pos) & flips).mean()
            / max(1e-9, flips.mean()))
    else:
        mix_rescue = 0.0

    # soft-mixture concentration of the attention itself
    att_peak = tr.attention.max(axis=1)

    return dict(
        label=ck.label, n_nodes=int(n), N=int(N),
        accuracy=float(tr.accuracy),
        routing_accuracy=routing_acc,
        peak_share_mean=float(tr.peak_share.mean()),
        attention_peak_mean=float(att_peak.mean()),
        # grouping / branch hypothesis
        groups=groups,
        species_per_target=species_per_target,
        species_per_target_mean=float(species_per_target.mean()),
        species_per_target_min=int(species_per_target.min()),
        species_per_target_max=int(species_per_target.max()),
        n_targets_with_branch=int((species_per_target >= 2).sum()),
        mean_branch_W_dist=mean_branch_dist,
        branch_detail=branch_detail,
        species_vote_peak_mean=float(species_vote_peak.mean()),
        decode_consistency=float(decode_consistency),
        # soft-mixture check
        dom_only_agreement=dom_only_agree,
        dom_only_accuracy=dom_only_acc,
        mixture_flip_rate=mixture_flip,
        mixture_rescue_rate=mix_rescue,
        # raw arrays for plotting
        _votes=votes, _decoded_pos=decoded_pos, _conf=conf,
        _eff=eff, _used=used,
    )


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def _plot_checkpoint(plt, res, outdir):
    label = res["label"]
    n, N = res["n_nodes"], res["N"]
    votes, decoded_pos = res["_votes"], res["_decoded_pos"]
    conf = res["_conf"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # (1) per-species decoder votes, rows sorted by decoded position
    order = np.argsort(decoded_pos)
    ax = axes[0]
    im = ax.imshow(votes[order], aspect="auto", cmap="magma",
                   vmin=0, vmax=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"s{j}" for j in order], fontsize=7)
    ax.set_xticks(range(N))
    ax.set_xlabel("decoded context position")
    ax.set_ylabel("species (sorted by vote)")
    ax.set_title(f"{label}\nper-species decoder vote  softmax(K·B/temp)")
    # separator lines between target groups
    boundaries = np.where(np.diff(decoded_pos[order]) != 0)[0]
    for b in boundaries:
        ax.axhline(b + 0.5, color="cyan", lw=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (2) dom_species x true_pos confusion (row-normalised)
    ax = axes[1]
    rn = conf / (conf.sum(axis=1, keepdims=True) + 1e-12)
    im = ax.imshow(rn, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"s{j}" for j in range(n)], fontsize=7)
    ax.set_xticks(range(N))
    ax.set_xlabel("true copy position")
    ax.set_ylabel("dominant species")
    ax.set_title("dom_species × true_pos\n(row-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (3) species-per-target bar + branch distances
    ax = axes[2]
    spt = res["species_per_target"]
    ax.bar(range(N), spt, color="steelblue", label="species sharing target")
    ax.axhline(2.0, color="red", ls="--", lw=1,
               label="branch hypothesis (~2)")
    ax.set_xticks(range(N))
    ax.set_xlabel("context position (target)")
    ax.set_ylabel("# species decoding to it")
    ax.set_title(f"target grouping\nmean branch W-dist = "
                 f"{res['mean_branch_W_dist']:.3f}")
    ax.legend(fontsize=7)

    fig.tight_layout()
    core.save_fig(fig, outdir, f"m3_routing_{label}.png")


def _plot_summary(plt, results, outdir):
    """Cross-checkpoint summary of the soft-mixture check."""
    labels = [r["label"] for r in results]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    w = 0.38
    ax.bar(x - w / 2, [r["routing_accuracy"] for r in results], w,
           label="routing accuracy", color="seagreen")
    ax.bar(x + w / 2, [r["dom_only_accuracy"] for r in results], w,
           label="dominant-species-only accuracy", color="indianred")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("soft mixture vs dominant-species-only")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(x - w / 2, [r["dom_only_agreement"] for r in results], w,
           label="dom-only ↔ model agreement", color="slateblue")
    ax.bar(x + w / 2, [r["mixture_rescue_rate"] for r in results], w,
           label="mixture-rescue rate (of flips)", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("%")
    ax.set_title("how much the graded mixture matters")
    ax.legend(fontsize=8)

    fig.tight_layout()
    core.save_fig(fig, outdir, "m3_routing_summary.png")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def run(checkpoints, traces, outdir):
    """M3 routing analysis.

    checkpoints: list[Checkpoint]
    traces:      {label: {'in_dist': Trace, 'novel': Trace}}
    outdir:      Path for figures.  Returns a dict of scalar findings.
    """
    plt = core.setup_style()
    results = []

    for ck in checkpoints:
        tr = traces[ck.label]["novel"]            # novel split = true ICL
        res = _analyze_checkpoint(ck, tr)
        results.append(res)
        _plot_checkpoint(plt, res, outdir)

        print(f"[{res['label']}] n={res['n_nodes']} N={res['N']}  "
              f"acc={res['accuracy']:.1f}%  routing_acc={res['routing_accuracy']:.1f}%")
        print(f"    species/target: mean={res['species_per_target_mean']:.2f} "
              f"min={res['species_per_target_min']} max={res['species_per_target_max']}  "
              f"targets with branch(>=2)={res['n_targets_with_branch']}/{res['N']}")
        print(f"    mean branch W-dist (1-cos of co-deciders)={res['mean_branch_W_dist']:.3f}  "
              f"decode-consistency={res['decode_consistency']:.1f}%")
        print(f"    soft-mixture: dom-only agreement={res['dom_only_agreement']:.1f}%  "
              f"dom-only acc={res['dom_only_accuracy']:.1f}%  "
              f"mixture-flip={res['mixture_flip_rate']:.1f}%  "
              f"mixture-rescue={res['mixture_rescue_rate']:.1f}%")

    if results:
        _plot_summary(plt, results, outdir)

    # ------ aggregate, JSON-serializable findings ---------------------------
    def _avg(key):
        return float(np.mean([r[key] for r in results])) if results else 0.0

    per_checkpoint = {}
    for r in results:
        per_checkpoint[r["label"]] = {
            "n_nodes": r["n_nodes"], "N": r["N"],
            "accuracy": round(r["accuracy"], 2),
            "routing_accuracy": round(r["routing_accuracy"], 2),
            "species_per_target_mean": round(r["species_per_target_mean"], 3),
            "species_per_target_min": r["species_per_target_min"],
            "species_per_target_max": r["species_per_target_max"],
            "n_targets_with_branch": r["n_targets_with_branch"],
            "mean_branch_W_dist": round(r["mean_branch_W_dist"], 4),
            "decode_consistency": round(r["decode_consistency"], 2),
            "dom_only_agreement": round(r["dom_only_agreement"], 2),
            "dom_only_accuracy": round(r["dom_only_accuracy"], 2),
            "mixture_flip_rate": round(r["mixture_flip_rate"], 2),
            "mixture_rescue_rate": round(r["mixture_rescue_rate"], 2),
            "peak_share_mean": round(r["peak_share_mean"], 3),
            "attention_peak_mean": round(r["attention_peak_mean"], 3),
            "species_per_target": [int(v) for v in r["species_per_target"]],
        }

    findings = {
        "m3_n_checkpoints": len(results),
        "m3_mean_routing_accuracy": round(_avg("routing_accuracy"), 2),
        "m3_mean_species_per_target": round(_avg("species_per_target_mean"), 3),
        "m3_mean_branch_W_dist": round(_avg("mean_branch_W_dist"), 4),
        "m3_mean_dom_only_agreement": round(_avg("dom_only_agreement"), 2),
        "m3_mean_dom_only_accuracy": round(_avg("dom_only_accuracy"), 2),
        "m3_mean_mixture_flip_rate": round(_avg("mixture_flip_rate"), 2),
        "m3_mean_mixture_rescue_rate": round(_avg("mixture_rescue_rate"), 2),
        "m3_mean_decode_consistency": round(_avg("decode_consistency"), 2),
        # branch hypothesis verdict: ~2-3 species/target supports it
        "m3_branch_hypothesis_supported": bool(
            results and 1.6 <= _avg("species_per_target_mean") <= 3.6),
        # soft-mixture verdict: a notable gap from one-hot decoding
        "m3_soft_mixture_essential": bool(
            results and _avg("dom_only_agreement") < 92.0),
        "m3_per_checkpoint": per_checkpoint,
    }
    return findings


if __name__ == "__main__":
    cks, traces = core.load_all()
    out = core.module_outdir("m3_routing")
    findings = run(cks, traces, out)
    print("\n=== M3 findings ===")
    for k, v in findings.items():
        if k != "m3_per_checkpoint":
            print(f"  {k}: {v}")
    print(f"  figures -> {out}")
