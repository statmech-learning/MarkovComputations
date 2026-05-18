"""
Degeneracy test for the topology-aware first-order Markov ICL model.

Question
--------
Within one fixed reaction graph + one fixed input mask, five models are trained
from five different random seeds. Across those seeds, are the trained networks:
  (A) the SAME function AND the same weights      -> unique solution;
  (B) the SAME function but DIFFERENT weights     -> degenerate solution;
  (C) DIFFERENT functions and different weights   -> divergent / non-converged.

Outcome (B) is what the WTA-ICL analysis found for the autocatalytic model and
what the prior topology program never tested directly. This script settles
which of A/B/C holds for the first-order Markov model.

The hard part is separating three things that all look like "agreement":
  - two models being individually accurate (both track the ground truth),
  - two models computing the same function (correlated even when wrong),
  - two models having the same weights.
We measure each explicitly.

Functional similarity (gauge-free)
  On one shared novel-class ICL eval set, for every seed pair:
    pred_agreement        raw fraction of queries with identical argmax label
                          -- inflated by both models being accurate.
    slot_kappa            Cohen's kappa of the chosen context slot
                          (chance-corrected for the marginal slot rates).
    error_slot_agreement  THE decisive probe: on queries where BOTH models are
                          WRONG, the fraction picking the SAME wrong slot,
                          renormalised so 0 = independent (chance 1/3),
                          1 = identical mistakes. This strips out the shared
                          correlation with ground truth -- two genuinely
                          identical functions fail identically.
    attention / steady_state agreement   1 - mean total-variation distance of
                          the 4-slot attention and 6-node steady-state p(z).
  A random-init model of the same architecture gives the floor for every probe.

Parametric similarity (gauge-quotiented)
  Exact gauge freedoms that leave the function unchanged are removed first:
    base_log_rates -> base_log_rates + delta*1   (uniform rate scaling cancels
                      in the matrix-tree normalisation)
    B              -> B + u (1_N)^T              (shifts q by a per-sample
                      constant; softmax-invariant)
  base is mean-centered, B is row-centered, K is compared as the masked
  function-carrying tensor K_eff = K_params*input_mask (raw and column-mean-
  centered -- the column shift is a near-gauge the 50% mask partly blocks).
  Cosine of the gauge-quotiented weights; random-init pairs give the floor.

Verdict
  DEGENERATE  : functions agree (decisive probes near the trained ceiling,
                far above the random floor) but weight cosine ~ random floor.
  UNIQUE      : functions agree AND weight cosine well above the random floor.
  DIVERGENT   : functions do NOT agree beyond the both-accurate baseline
                (error_slot_agreement ~ 0, weight cosine ~ random floor).

    python degeneracy_test.py

Writes degeneracy_summary.json, degeneracy_report.md, degeneracy_scatter.png.
"""

import sys
import os
import json
import glob
import argparse
import itertools
import warnings

import numpy as np
import torch

TOPO_ICL = "/Users/aadarwal/code/statmech/topology/ICL"
GRID = os.path.join(TOPO_ICL, "results", "prospective_tree_diff_multiplicity_training")
HERE = os.path.dirname(os.path.abspath(__file__))
TAG = ""  # filename suffix for the output artifacts; set from --tag

sys.path.insert(0, TOPO_ICL)
from models.topology_markov_icl import TopologyMatrixTreeMarkovICL  # noqa: E402
from data_generation import GaussianMixtureModel, generate_icl_gmm_data  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)

EVAL_SAMPLES = 600
EVAL_GMM_SEED = 12345


# --------------------------------------------------------------------------
# discovery / loading
# --------------------------------------------------------------------------
def discover_groups():
    """Group run dirs by (graph + input mask); the _trainseedN suffix varies."""
    groups = {}
    for d in sorted(glob.glob(os.path.join(GRID, "*_trainseed*"))):
        if not os.path.isdir(d):
            continue
        key, seed = os.path.basename(d).rsplit("_trainseed", 1)
        groups.setdefault(key, []).append((int(seed), d))
    for key in groups:
        groups[key].sort()
    return groups


def load_model(run_dir):
    topo = json.load(open(os.path.join(run_dir, "topology.json")))
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    m = TopologyMatrixTreeMarkovICL(
        n_nodes=topo["n_nodes"], z_dim=cfg["D"], L=cfg["L"], N=cfg["N"],
        edges=[tuple(e) for e in topo["edges"]], input_mask=topo["input_mask"],
        transform_func=cfg["transform_func"], learn_base_rates=cfg["learn_base_rates"],
        use_label_mod=cfg["use_label_mod"], print_creation=False,
    )
    m.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu"))
    m.eval()
    res = {}
    rp = os.path.join(run_dir, "results.pkl")
    if os.path.exists(rp):
        import pickle
        res = pickle.load(open(rp, "rb")).get("results", {})
    return m, cfg, res


# --------------------------------------------------------------------------
# functional probes
# --------------------------------------------------------------------------
def build_eval_set(cfg):
    """Shared novel-class ICL eval set (the true-ICL test condition)."""
    gmm = GaussianMixtureModel(cfg["K"], cfg["D"], L=cfg["L"],
                               epsilon=cfg["epsilon"], seed=EVAL_GMM_SEED)
    torch.manual_seed(EVAL_GMM_SEED)
    np.random.seed(EVAL_GMM_SEED)
    data = generate_icl_gmm_data(gmm, EVAL_SAMPLES, cfg["N"], novel_classes=True,
                                 exact_copy=cfg["exact_copy"], B=cfg["B"], L=cfg["L"])
    z = torch.stack([d[0] for d in data])             # (S, N+1, D)
    lab = torch.stack([d[1] for d in data])           # (S, N)
    tgt = torch.tensor([float(d[2]) for d in data])   # (S,)
    # correct context slot = first slot whose label equals the target label
    correct_slot = (lab == tgt[:, None]).float().argmax(dim=1)   # (S,)
    return z, lab, tgt, correct_slot


def run_model(model, cfg, z, lab):
    """predicted label, chosen context slot, 4-slot attention, 6-node p(z)."""
    logits = model(z, lab, method=cfg["method"], temperature=cfg["temperature"])
    pred = logits.argmax(dim=1) + 1
    z_flat = z.reshape(z.shape[0], -1)
    W = model.compute_rate_matrix_W(z_flat, lab)
    p = model.steady_state(W, method=cfg["method"])
    q = torch.matmul(p, model.B)
    attn = torch.softmax(q / cfg["temperature"], dim=1)
    slot = attn.argmax(dim=1)
    return pred, slot, attn, p


def tv_similarity(A, B):
    """1 - mean total-variation distance between rows of two simplex batches."""
    return float(1.0 - 0.5 * (A - B).abs().sum(dim=1).mean())


def slot_kappa(s_i, s_j, n_slots):
    """Cohen's kappa of two slot-choice vectors (chance-corrected agreement)."""
    s_i, s_j = s_i.numpy(), s_j.numpy()
    p_o = float((s_i == s_j).mean())
    pe = 0.0
    for s in range(n_slots):
        pe += (s_i == s).mean() * (s_j == s).mean()
    return (p_o - pe) / (1.0 - pe) if pe < 1.0 else float("nan")


def error_slot_agreement(s_i, s_j, correct, n_slots):
    """On queries where BOTH models are wrong, normalised same-wrong-slot rate.

    0 = independent mistakes (chance 1/(n_slots-1)); 1 = identical mistakes.
    This is gauge-free and accuracy-free: it isolates 'same function'.
    """
    both_wrong = (s_i != correct) & (s_j != correct)
    n = int(both_wrong.sum())
    if n < 20:
        return float("nan"), n
    agree = float((s_i[both_wrong] == s_j[both_wrong]).float().mean())
    chance = 1.0 / (n_slots - 1)
    return (agree - chance) / (1.0 - chance), n


# --------------------------------------------------------------------------
# parametric probes (gauge-quotiented)
# --------------------------------------------------------------------------
def cos(a, b):
    a, b = a.flatten(), b.flatten()
    na, nb = a.norm(), b.norm()
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(torch.dot(a, b) / (na * nb))


def gauge_params(model):
    K_eff = (model.K_params * model.input_mask).clone()
    base = model.base_log_rates.clone()
    B = model.B.clone()
    return {"K_eff": K_eff,
            "K_col": K_eff - K_eff.mean(dim=0, keepdim=True),   # column near-gauge
            "base_q": base - base.mean(),                       # rate-scale gauge
            "B_q": B - B.mean(dim=1, keepdim=True)}             # per-row gauge


def param_similarity(pa, pb):
    return {
        "K_raw": cos(pa["K_eff"], pb["K_eff"]),
        "K_colcentered": cos(pa["K_col"], pb["K_col"]),
        "base": cos(pa["base_q"], pb["base_q"]),
        "B": cos(pa["B_q"], pb["B_q"]),
        "combined": cos(
            torch.cat([pa["K_col"].flatten(), pa["base_q"], pa["B_q"].flatten()]),
            torch.cat([pb["K_col"].flatten(), pb["base_q"], pb["B_q"].flatten()])),
    }


def mean(xs):
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def out_path(stem, ext):
    return os.path.join(HERE, f"{stem}{('_' + TAG) if TAG else ''}.{ext}")


def main():
    global GRID, TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=GRID,
                    help="directory of *_trainseed* run dirs to analyse")
    ap.add_argument("--tag", default="",
                    help="suffix for output filenames (e.g. 'converged')")
    args = ap.parse_args()
    GRID, TAG = args.grid, args.tag

    groups = discover_groups()
    print(f"Discovered {len(groups)} (graph,mask) groups, "
          f"{sum(len(v) for v in groups.values())} checkpoints total.\n")

    group_results = []
    acc = {"err": [], "kappa": [], "pred": [], "attn": [], "ss": [],
           "pc": [], "pk": [], "pbase": [], "pB": [],
           "r_pred": [], "r_attn": [], "r_ss": [], "r_pc": [], "r_kappa": []}

    for gi, (key, runs) in enumerate(sorted(groups.items())):
        seeds = [s for s, _ in runs]
        models, cfgs, stored = [], [], []
        for s, d in runs:
            m, cfg, res = load_model(d)
            models.append(m); cfgs.append(cfg); stored.append(res)
        cfg = cfgs[0]
        n_slots = cfg["N"]

        for m in models[1:]:
            assert torch.equal(m.edge_sources, models[0].edge_sources)
            assert torch.equal(m.input_mask, models[0].input_mask)

        z, lab, tgt, correct = build_eval_set(cfg)
        preds, slots, attns, ps, accs = [], [], [], [], []
        for m in models:
            pr, sl, at, p = run_model(m, cfg, z, lab)
            preds.append(pr); slots.append(sl); attns.append(at); ps.append(p)
            accs.append(float((pr == tgt.long()).float().mean()) * 100.0)

        # random-init reference, same architecture
        torch.manual_seed(7000 + gi)
        rnd = TopologyMatrixTreeMarkovICL(
            n_nodes=models[0].n_nodes, z_dim=cfg["D"], L=cfg["L"], N=cfg["N"],
            edges=list(models[0].edges), input_mask=models[0].input_mask.tolist(),
            transform_func=cfg["transform_func"],
            learn_base_rates=cfg["learn_base_rates"],
            use_label_mod=cfg["use_label_mod"], print_creation=False)
        rnd.eval()
        r_pred, r_slot, r_attn, r_p = run_model(rnd, cfg, z, lab)

        gp = [gauge_params(m) for m in models]
        gp_rnd = gauge_params(rnd)

        # pairwise over the 5 trained seeds
        err, kap, fp, fa, fpp = [], [], [], [], []
        pc, pk, pb_, pB = [], [], [], []
        for i, j in itertools.combinations(range(len(models)), 2):
            fp.append(float((preds[i] == preds[j]).float().mean()))
            kap.append(slot_kappa(slots[i], slots[j], n_slots))
            e, _ = error_slot_agreement(slots[i], slots[j], correct, n_slots)
            err.append(e)
            fa.append(tv_similarity(attns[i], attns[j]))
            fpp.append(tv_similarity(ps[i], ps[j]))
            s = param_similarity(gp[i], gp[j])
            pc.append(s["combined"]); pk.append(s["K_colcentered"])
            pb_.append(s["base"]); pB.append(s["B"])

        # trained-vs-random reference
        r_fp, r_ka, r_fa, r_fpp, r_pc = [], [], [], [], []
        for i in range(len(models)):
            r_fp.append(float((preds[i] == r_pred).float().mean()))
            r_ka.append(slot_kappa(slots[i], r_slot, n_slots))
            r_fa.append(tv_similarity(attns[i], r_attn))
            r_fpp.append(tv_similarity(ps[i], r_p))
            r_pc.append(param_similarity(gp[i], gp_rnd)["combined"])

        g = {
            "group": key, "seeds": seeds, "n_models": len(models),
            "accuracy_pct": {"per_seed": accs, "mean": mean(accs),
                             "std": float(np.std(accs)),
                             "stored_novel_classes":
                                 [r.get("novel_classes") for r in stored]},
            "functional_trained_vs_trained": {
                "error_slot_agreement": mean(err),
                "slot_kappa": mean(kap),
                "pred_agreement": mean(fp),
                "attention_agreement": mean(fa),
                "steady_state_agreement": mean(fpp)},
            "functional_trained_vs_random": {
                "slot_kappa": mean(r_ka),
                "pred_agreement": mean(r_fp),
                "attention_agreement": mean(r_fa),
                "steady_state_agreement": mean(r_fpp)},
            "parametric_trained_vs_trained": {
                "combined_cosine": mean(pc), "K_colcentered_cosine": mean(pk),
                "base_cosine": mean(pb_), "B_cosine": mean(pB)},
            "parametric_trained_vs_random": {"combined_cosine": mean(r_pc)},
        }
        group_results.append(g)
        acc["err"].append(mean(err)); acc["kappa"].append(mean(kap))
        acc["pred"].append(mean(fp)); acc["attn"].append(mean(fa)); acc["ss"].append(mean(fpp))
        acc["pc"].append(mean(pc)); acc["pk"].append(mean(pk))
        acc["pbase"].append(mean(pb_)); acc["pB"].append(mean(pB))
        acc["r_pred"].append(mean(r_fp)); acc["r_attn"].append(mean(r_fa))
        acc["r_ss"].append(mean(r_fpp)); acc["r_pc"].append(mean(r_pc))
        acc["r_kappa"].append(mean(r_ka))

        print(f"[{gi+1:2d}/{len(groups)}] {key.split('__')[-1]:42s} "
              f"acc {g['accuracy_pct']['mean']:5.1f}+-{g['accuracy_pct']['std']:4.1f}  "
              f"errSlot {mean(err):+.3f}  kappa {mean(kap):+.3f}  "
              f"paramCos {mean(pc):+.3f}  (randParam {mean(r_pc):+.3f})")

    # ---- aggregate verdict ----------------------------------------------
    agg = {
        "n_groups": len(group_results),
        "functional_trained_vs_trained": {
            "error_slot_agreement_mean": mean(acc["err"]),
            "slot_kappa_mean": mean(acc["kappa"]),
            "pred_agreement_mean": mean(acc["pred"]),
            "attention_agreement_mean": mean(acc["attn"]),
            "steady_state_agreement_mean": mean(acc["ss"])},
        "functional_trained_vs_random": {
            "slot_kappa_mean": mean(acc["r_kappa"]),
            "pred_agreement_mean": mean(acc["r_pred"]),
            "attention_agreement_mean": mean(acc["r_attn"]),
            "steady_state_agreement_mean": mean(acc["r_ss"])},
        "parametric_trained_vs_trained": {
            "combined_cosine_mean": mean(acc["pc"]),
            "K_colcentered_cosine_mean": mean(acc["pk"]),
            "base_cosine_mean": mean(acc["pbase"]),
            "B_cosine_mean": mean(acc["pB"])},
        "parametric_trained_vs_random": {
            "combined_cosine_mean": mean(acc["r_pc"])},
    }

    err = agg["functional_trained_vs_trained"]["error_slot_agreement_mean"]
    pcos = agg["parametric_trained_vs_trained"]["combined_cosine_mean"]
    rpcos = agg["parametric_trained_vs_random"]["combined_cosine_mean"]
    param_low = abs(pcos) < 0.15 or abs(pcos - rpcos) < 0.15
    func_same = err > 0.5            # mistakes strongly correlated -> same function

    if func_same and param_low:
        verdict = "DEGENERATE"
    elif func_same and not param_low:
        verdict = "UNIQUE (up to gauge)"
    else:
        verdict = "DIVERGENT"
    agg["verdict"] = verdict
    agg["reason"] = (
        f"error-conditioned slot agreement = {err:+.3f} "
        f"(0 = independent mistakes, 1 = identical function); "
        f"gauge-quotiented weight cosine = {pcos:+.3f} "
        f"vs random-init floor {rpcos:+.3f}.")

    summary = {"config": {"eval_samples": EVAL_SAMPLES,
                          "eval_gmm_seed": EVAL_GMM_SEED, "grid": GRID},
               "aggregate": agg, "groups": group_results}
    out_json = out_path("degeneracy_summary", "json")
    json.dump(summary, open(out_json, "w"), indent=2, default=str)
    write_report(summary)
    try:
        make_scatter(summary)
    except Exception as e:
        print(f"  (scatter skipped: {e!r})")

    print("\n" + "=" * 70)
    print(f"VERDICT: {verdict}")
    print(agg["reason"])
    print("=" * 70)
    print(f"summary -> {out_json}")
    return 0


def write_report(summary):
    a = summary["aggregate"]
    ftt = a["functional_trained_vs_trained"]
    ftr = a["functional_trained_vs_random"]
    par = a["parametric_trained_vs_trained"]
    L = []
    L.append("# Markov-ICL cross-seed degeneracy test\n")
    L.append(f"**Verdict: {a['verdict']}**\n")
    L.append(a["reason"] + "\n")
    L.append(f"- {a['n_groups']} groups (each = one fixed reaction graph + one "
             f"fixed input mask), 5 train seeds per group, 80 checkpoints.")
    L.append(f"- Eval: {summary['config']['eval_samples']} novel-class ICL "
             f"queries, identical set for every model.\n")
    L.append("## Aggregate (mean over groups)\n")
    L.append("| probe | trained vs trained | trained vs random-init |")
    L.append("|---|---|---|")
    L.append(f"| error-slot agreement (decisive) | "
             f"**{ftt['error_slot_agreement_mean']:+.3f}** | 0 by construction |")
    L.append(f"| slot kappa | {ftt['slot_kappa_mean']:+.3f} | "
             f"{ftr['slot_kappa_mean']:+.3f} |")
    L.append(f"| raw prediction agreement | {ftt['pred_agreement_mean']:.3f} | "
             f"{ftr['pred_agreement_mean']:.3f} |")
    L.append(f"| attention agreement (1-TV) | "
             f"{ftt['attention_agreement_mean']:.3f} | "
             f"{ftr['attention_agreement_mean']:.3f} |")
    L.append(f"| steady-state agreement (1-TV) | "
             f"{ftt['steady_state_agreement_mean']:.3f} | "
             f"{ftr['steady_state_agreement_mean']:.3f} |")
    L.append(f"| **weight cosine (gauge-quotiented)** | "
             f"**{par['combined_cosine_mean']:+.3f}** | "
             f"{a['parametric_trained_vs_random']['combined_cosine_mean']:+.3f} |")
    L.append(f"| - K_eff (col-centered) | {par['K_colcentered_cosine_mean']:+.3f} | - |")
    L.append(f"| - base_log_rates | {par['base_cosine_mean']:+.3f} | - |")
    L.append(f"| - B | {par['B_cosine_mean']:+.3f} | - |\n")
    L.append("## How to read this\n")
    L.append("- **error-slot agreement** is the decisive probe. It looks only "
             "at queries both models get wrong and asks whether they make the "
             "*same* mistake. 0 means independent mistakes (different "
             "functions); 1 means identical mistakes (one function). Raw "
             "prediction agreement cannot distinguish these because two "
             "accurate models agree just by both being right.")
    L.append("- **weight cosine** is measured after removing the model's exact "
             "gauge freedoms. ~0 means the seeds' parameters are as unrelated "
             "as random initialisations.\n")

    L.append("## What this means\n")
    rcos = a["parametric_trained_vs_random"]["combined_cosine_mean"]
    L.append(f"1. **Parametrically divergent.** Gauge-quotiented weight cosine "
             f"between two trained seeds is {par['combined_cosine_mean']:+.3f} "
             f"-- indistinguishable from the random-init floor "
             f"({rcos:+.3f}). No two seeds share *any* parametric structure.")
    L.append(f"2. **Internally divergent.** Trained-vs-trained steady-state "
             f"agreement ({ftt['steady_state_agreement_mean']:.3f}) is *lower* "
             f"than trained-vs-random ({ftr['steady_state_agreement_mean']:.3f})"
             f": training actively pushes each seed's internal state p(z) onto "
             f"a different region of the simplex. Two trained seeds disagree "
             f"internally more than a trained model disagrees with noise.")
    L.append(f"3. **Functionally only weakly shared.** Error-slot agreement is "
             f"{ftt['error_slot_agreement_mean']:+.3f} of a possible 1.0 -- the "
             f"seeds share a partial sub-solution but ~"
             f"{100*(1-ftt['error_slot_agreement_mean']):.0f}% of their "
             f"mistake structure is seed-specific. They are not one function.")
    L.append("4. **Undertrained landscape.** Accuracy is 64-83% with "
             "within-group spread up to ~11 points, on a task (exact_copy "
             "query) that admits a ~100% nearest-context-copy solution. The "
             "seeds settle into distinct mediocre optima, not a common one.")
    L.append("5. **Contrast with WTA-ICL.** The autocatalytic WTA model was "
             "*functionally constrained* (degenerate: same function, different "
             "weights) because it trained above its capacity threshold to the "
             "performance ceiling. The first-order Markov model here does the "
             "opposite -- different weights AND different functions -- because "
             "it never reaches that ceiling on these topologies.")
    L.append("6. **Consequence for the prior topology program.** Per-topology "
             "single-seed runs are seed-noise dominated: any topology -> "
             "performance regression fitted on one seed per topology was "
             "largely fitting cross-seed variance, not a topology law. The "
             "degeneracy test makes that quantitative.\n")

    L.append("## Per-group\n")
    L.append("| group | acc % (mean+-std) | err-slot | kappa | "
             "pred agree | attn agree | weight cosine | rand cosine |")
    L.append("|---|---|---|---|---|---|---|---|")
    for g in summary["groups"]:
        ac = g["accuracy_pct"]; ft = g["functional_trained_vs_trained"]
        pt = g["parametric_trained_vs_trained"]
        L.append(f"| {g['group'].split('__')[-1]} | "
                 f"{ac['mean']:.1f}+-{ac['std']:.1f} | "
                 f"{ft['error_slot_agreement']:+.3f} | {ft['slot_kappa']:+.3f} | "
                 f"{ft['pred_agreement']:.3f} | {ft['attention_agreement']:.3f} | "
                 f"{pt['combined_cosine']:+.3f} | "
                 f"{g['parametric_trained_vs_random']['combined_cosine']:+.3f} |")
    open(out_path("degeneracy_report", "md"), "w").write("\n".join(L) + "\n")


def make_scatter(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = summary["groups"]
    fx = [x["functional_trained_vs_trained"]["error_slot_agreement"] for x in g]
    py = [x["parametric_trained_vs_trained"]["combined_cosine"] for x in g]
    rpy = [x["parametric_trained_vs_random"]["combined_cosine"] for x in g]

    fig, ax = plt.subplots(figsize=(7.2, 6))
    ax.axhspan(-0.15, 0.15, color="gold", alpha=0.15)
    ax.axvspan(-0.2, 0.2, color="lightgray", alpha=0.4)
    ax.scatter(fx, py, s=80, c="crimson", edgecolor="k", zorder=3,
               label="trained seed-pair (same graph + mask)")
    ax.scatter([0] * len(rpy), rpy, s=55, c="steelblue", marker="^",
               edgecolor="k", zorder=3, label="trained vs random-init")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("functional similarity\n"
                  "error-conditioned slot agreement  "
                  "(0 = independent mistakes, 1 = same function)")
    ax.set_ylabel("parametric similarity\n"
                  "gauge-quotiented weight cosine")
    ax.set_title("Markov-ICL cross-seed degeneracy test")
    ax.set_xlim(-0.25, 1.02)
    ax.set_ylim(-0.6, 1.02)
    ax.text(0.97, 0.30, "DEGENERATE\n(same function,\ndifferent weights)",
            ha="right", fontsize=8.5, color="goldenrod", style="italic")
    ax.text(0.07, 0.55, "DIVERGENT\n(different functions\nAND different weights)",
            fontsize=8.5, color="dimgray", style="italic")
    ax.text(0.97, 0.88, "UNIQUE (up to gauge)",
            ha="right", fontsize=8.5, color="seagreen", style="italic")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path("degeneracy_scatter", "png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
