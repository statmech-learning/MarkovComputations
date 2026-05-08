#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Serif",
    "font.size": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

INK = "#222222"
BLUE = "#2f5f8f"
TEAL = "#237a77"
ORANGE = "#b46b28"
RED = "#9f3a38"
GRAY = "#777777"
LIGHT = "#eeeeee"


def save(fig, name: str):
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def annotate_bars(ax, bars, fmt="{:.2f}", dx=0.015):
    for b in bars:
        w = b.get_width()
        x = w + (dx if w >= 0 else -dx)
        ha = "left" if w >= 0 else "right"
        ax.text(x, b.get_y() + b.get_height() / 2, fmt.format(w), ha=ha, va="center", fontsize=8)


def fig_tree_basis():
    fig, ax = plt.subplots(figsize=(7.3, 3.0))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    nodes = {"1": (1.0, 2.8), "2": (2.6, 2.8), "3": (2.6, 1.2), "4": (1.0, 1.2)}
    edges = [("1", "2", r"$K_{12}$"), ("2", "3", r"$K_{23}$"), ("3", "4", r"$K_{34}$"),
             ("4", "1", r"$K_{41}$"), ("1", "3", r"$K_{13}$"), ("4", "2", r"$K_{42}$")]
    for a, b, lab in edges:
        x1, y1 = nodes[a]
        x2, y2 = nodes[b]
        arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
                              linewidth=1.2, color=INK, shrinkA=12, shrinkB=12,
                              connectionstyle="arc3,rad=0.06")
        ax.add_patch(arr)
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.12, lab, fontsize=8, ha="center", va="center", color=BLUE)
    for n, (x, y) in nodes.items():
        ax.add_patch(Circle((x, y), 0.22, facecolor="white", edgecolor=INK, linewidth=1.2))
        ax.text(x, y, n, ha="center", va="center", fontsize=9)

    ax.text(1.8, 3.72, r"physical graph $G=(V,E)$", ha="center", va="center", fontsize=10)
    ax.text(1.8, 0.28, r"edges carry learned vectors $K_e$", ha="center", va="center", fontsize=9, color=GRAY)

    ax.annotate("", xy=(4.45, 2.0), xytext=(3.25, 2.0), arrowprops=dict(arrowstyle="->", lw=1.4, color=INK))
    ax.text(3.85, 2.26, "matrix-tree", ha="center", fontsize=8.5)

    box = FancyBboxPatch((4.7, 0.7), 4.85, 2.65, boxstyle="round,pad=0.25,rounding_size=0.05",
                         facecolor="#f8f8f8", edgecolor="#bbbbbb", linewidth=0.9)
    ax.add_patch(box)
    ax.text(7.1, 3.05, r"computational basis", ha="center", fontsize=10)
    ax.text(7.1, 2.55, r"$au_r(z)=\sum_{T\in\mathrm{Trees}_r(G)} e^{\beta_T+\Theta_T^\top z}$", ha="center", fontsize=11)
    ax.text(7.1, 2.0, r"$\Theta_T=\sum_{e\in T}K_e$", ha="center", fontsize=13, color=BLUE)
    ax.text(7.1, 1.42, r"basis is $\{\Theta_T:T\in\mathrm{Trees}_r(G),\ r\in V\}$", ha="center", fontsize=10)
    ax.text(7.1, 0.95, r"not the isolated edge set $\{K_e:e\in E\}$", ha="center", fontsize=10, color=RED)
    save(fig, "fig_tree_sum_basis")


def fig_predictors():
    fixed = [
        ("raw count", -0.04300588501584368),
        (r"raw+$d_{rel}$", 0.14495126498743327),
        ("masked tree", 0.18868381225741682),
        ("tree geometry", 0.4087607151863111),
    ]
    hard = [
        ("raw count", -0.19008264462809876),
        (r"raw+$d_{rel}$", -0.19008264462809876),
        ("tree geometry", 0.3238604298943035),
        ("masked tree", 0.44659016077696834),
        ("rooted polytope", 0.08074626727683243),
        (r"opt. $\gamma$ proxy", -0.5638231537760179),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharex=True)
    for ax, data, title in zip(axes, [fixed, hard], [r"fixed $N_n=6,m=20$ groups", r"hard $N_n=5,m=12,N_c=3,D=2$"]):
        labs = [x[0] for x in data][::-1]
        vals = [x[1] for x in data][::-1]
        colors = [BLUE if v >= 0 else RED for v in vals]
        bars = ax.barh(range(len(vals)), vals, color=colors, alpha=0.92)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_yticks(range(len(vals)), labs)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"group LOO $R^2$")
        ax.set_xlim(-0.65, 0.52)
        annotate_bars(ax, bars)
        ax.grid(axis="x", color=LIGHT, lw=0.8)
    fig.suptitle("Structural predictors improve over raw count, but are regime-dependent", y=1.03, fontsize=11)
    save(fig, "fig_structural_predictors")


def fig_functional():
    data = [("run level", 0.740), ("group mean", 0.809), ("group best", 0.599)]
    fig, ax = plt.subplots(figsize=(4.2, 2.45))
    vals = [v for _, v in data]
    bars = ax.bar(range(len(vals)), vals, color=[TEAL, BLUE, ORANGE], width=0.55)
    ax.set_xticks(range(len(vals)), [k for k, _ in data])
    ax.set_ylabel(r"LOO $R^2$")
    ax.set_ylim(0, 0.92)
    ax.set_title("Post-training projection organization is the strongest diagnostic", fontsize=10)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.025, f"{v:.3f}", ha="center", fontsize=8.5)
    ax.grid(axis="y", color=LIGHT, lw=0.8)
    save(fig, "fig_post_training_alignment")


def fig_mechanism_contrasts():
    mech_path = ROOT / "ICL/results/next_phase_stats/mechanism_isolation_evidence.csv"
    rows = list(csv.DictReader(open(mech_path, newline="")))
    selected = []
    # Two structural controls plus the random fixed-graph input-mask contrasts.
    for i, row in enumerate(rows, 1):
        if i in (1, 2, 7, 8):
            selected.append(row)
    nf = json.load(open(ROOT / "ICL/results/next_phase_stats/degree_rewire_normal_fan_n5_m12_N3_D2/normal_fan_training_summary.json"))
    nf_contrasts = nf["contrasts"]
    labels = [
        r"same $d_{rel}$: edge participation",
        "same tree count: root balance",
        "same physical graph: edge-load mask",
        "same physical graph: coordinate-load mask",
        "exact degree: branch-tree NMI",
        "exact degree: active-tree count",
    ]
    vals = [float(r["delta_icl_mean"]) for r in selected] + [float(c["delta_mean"]) for c in nf_contrasts]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    y = list(range(len(vals)))[::-1]
    colors = [TEAL if v >= 0 else RED for v in vals]
    bars = ax.barh(y, vals, color=colors, alpha=0.92)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel("mean novel-class ICL change (points)")
    ax.set_title("Controlled contrasts show that rank alone is incomplete", fontsize=10)
    ax.set_xlim(-19, 16)
    annotate_bars(ax, bars, fmt="{:.1f}", dx=0.5)
    ax.grid(axis="x", color=LIGHT, lw=0.8)
    save(fig, "fig_mechanism_contrasts")


def fig_causal_scrambles():
    path = ROOT / "ICL/results/next_phase_stats/stat_preserving_causal_stratified/summary.json"
    rows = json.load(open(path))
    by = defaultdict(list)
    for r in rows:
        by[r["bucket"]].append(r)
    buckets = ["low", "mid", "high"]
    branch = []
    proj = []
    base = []
    for b in buckets:
        items = by[b]
        branch.append(sum(float(x["stat_preserving_branch_alignment_scramble_delta_mean"]) for x in items)/len(items))
        proj.append(sum(float(x["stat_preserving_projection_scramble_delta_mean"]) for x in items)/len(items))
        base.append(sum(float(x["sampled_baseline_accuracy"]) for x in items)/len(items))
    x = list(range(len(buckets)))
    w = 0.34
    fig, ax = plt.subplots(figsize=(5.3, 3.0))
    b1 = ax.bar([i-w/2 for i in x], branch, width=w, color=RED, label="branch alignment")
    b2 = ax.bar([i+w/2 for i in x], proj, width=w, color=ORANGE, label="projection scramble")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x, [f"{b}\nbase {base[i]:.1f}" for i,b in enumerate(buckets)])
    ax.set_ylabel("accuracy change (points)")
    ax.set_title("Statistic-preserving scrambles destroy ICL", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.set_ylim(-105, 8)
    for bars in [b1, b2]:
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, v-4, f"{v:.1f}", ha="center", va="top", fontsize=8, color="white")
    ax.grid(axis="y", color=LIGHT, lw=0.8)
    save(fig, "fig_causal_scrambles")


def fig_degree_rewire():
    path = ROOT / "ICL/results/next_phase_stats/degree_rewire_normal_fan_n5_m12_N3_D2/normal_fan_training_joined.csv"
    rows = list(csv.DictReader(open(path, newline="")))
    xs = [float(r["capacity_normal_fan_active_tree_count_mean"]) for r in rows]
    ys = [float(r["test_novel_classes_mean"]) for r in rows]
    nmi = [float(r["capacity_normal_fan_branch_tree_nmi_mean"]) for r in rows]
    labels = [r["topology_name"].split("_seed")[-1] for r in rows]
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    sc = ax.scatter(xs, ys, c=nmi, s=70, cmap="viridis", edgecolor=INK, linewidth=0.45)
    for x, y, lab in zip(xs, ys, labels):
        ax.text(x+0.35, y+0.25, f"seed {lab}", fontsize=7.5)
    ax.set_xlabel("active tree count")
    ax.set_ylabel("mean novel-class ICL")
    ax.set_title(r"Exact-degree pilot: $d_{rel}$ fixed at 88", fontsize=10)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("branch-tree NMI", fontsize=8.5)
    ax.grid(color=LIGHT, lw=0.8)
    save(fig, "fig_degree_rewire_pilot")


def fig_motif_controls():
    report = json.load(open(ROOT / "ICL/results/next_phase_stats/next_phase_evidence_report.json"))
    rows = report["matched_motif_controls"]
    labels = [r["label"] for r in rows]
    source = [float(r["overall"]["source_retrain_target_mean_mean"]) for r in rows]
    control = [float(r["overall"]["control_target_mean_mean"]) for r in rows]
    x = list(range(len(labels)))
    w = 0.35
    fig, ax = plt.subplots(figsize=(4.9, 2.8))
    ax.bar([i-w/2 for i in x], source, width=w, color=BLUE, label="extracted motif")
    ax.bar([i+w/2 for i in x], control, width=w, color=GRAY, label="matched controls")
    ax.set_xticks(x, labels)
    ax.set_ylabel("mean retrain ICL")
    ax.set_ylim(58, 78)
    ax.set_title("Extracted motifs work, but are not uniquely optimal", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    for i,(s,c) in enumerate(zip(source, control)):
        ax.text(i-w/2, s+0.45, f"{s:.1f}", ha="center", fontsize=8)
        ax.text(i+w/2, c+0.45, f"{c:.1f}", ha="center", fontsize=8)
    ax.grid(axis="y", color=LIGHT, lw=0.8)
    save(fig, "fig_motif_controls")


def fig_markov_reanalysis():
    path = ROOT / "ICL/results/next_phase_stats/existing_data_markov_expressivity_reanalysis.json"
    payload = json.load(open(path))
    datasets = ["fixed_m20", "hard_n4_m6", "hard_n5_m8", "hard_n5_m12"]
    labels = ["fixed\nm20", "hard\nn4 m6", "hard\nn5 m8", "hard\nn5 m12"]
    models = ["multiplicity", "comparison_multiplicity", "tree_geometry", "capacity_proxy"]
    model_labels = ["input\nmultiplicity", "comparison\nmultiplicity", "tree\ngeometry", "capacity\nproxy"]
    colors = [GRAY, ORANGE, BLUE, TEAL]

    values = {model: [] for model in models}
    for dataset in datasets:
        rows = payload["analyses"][dataset]["models"]
        for model in models:
            match = [
                row for row in rows
                if row["outcome"] == "mean_novel_icl" and row["model"] == model
            ][0]
            values[model].append(float(match["loo_r2"]))

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = list(range(len(datasets)))
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for model, label, color, offset in zip(models, model_labels, colors, offsets):
        bars = ax.bar([i + offset for i in x], values[model], width=width, color=color, label=label)
        for bar, val in zip(bars, values[model]):
            if val >= 0.08:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + (0.025 if val >= 0 else -0.035),
                    f"{val:.2f}",
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=7.3,
                )
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"grouped LOO $R^2$")
    ax.set_title("Markov-expressivity reanalysis: useful variables, no final scalar law", fontsize=10)
    ax.set_ylim(-0.25, 0.52)
    ax.legend(frameon=False, fontsize=7.8, ncol=4, loc="upper left")
    ax.grid(axis="y", color=LIGHT, lw=0.8)
    save(fig, "fig_markov_reanalysis")


def fig_input_multiplicity_controls():
    path = ROOT / "ICL/results/next_phase_stats/input_multiplicity_control_report.json"
    payload = json.load(open(path))
    rows = payload["family_summary"]
    rows = sorted(rows, key=lambda row: float(row["mean_group_mean_icl"]))
    label_map = {
        "balanced": "balanced",
        "coord_block": "coord\nblock",
        "edge_block": "edge\nblock",
        "entry_random": "entry\nrandom",
        "high_participation_edges": "high\npart. edges",
        "low_participation_edges": "low\npart. edges",
    }
    labels = [label_map.get(row["input_mask_family"], row["input_mask_family"].replace("_", "\n")) for row in rows]
    mean_icl = [float(row["mean_group_mean_icl"]) for row in rows]
    best_icl = [float(row["mean_group_best_icl"]) for row in rows]
    overlap = [float(row["mean_branch_input_overlap_min"]) for row in rows]
    gini = [float(row["mean_M_gini"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    y = list(range(len(rows)))
    bars = axes[0].barh([i - 0.18 for i in y], mean_icl, height=0.34, color=BLUE, label="mean seed")
    axes[0].barh([i + 0.18 for i in y], best_icl, height=0.34, color=TEAL, label="best seed")
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel("novel-class ICL")
    axes[0].set_title("Mask family changes ICL at fixed count", fontsize=10)
    axes[0].set_xlim(55, 90)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].grid(axis="x", color=LIGHT, lw=0.8)
    for bar in bars:
        axes[0].text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.1f}", va="center", fontsize=7.5)

    ax2 = axes[1]
    sc = ax2.scatter(overlap, mean_icl, c=gini, cmap="magma_r", s=72, edgecolor=INK, linewidth=0.45)
    for xval, yval, label in zip(overlap, mean_icl, labels):
        display = label.replace("\n", " ")
        if xval > 30:
            dy = 0.35 if display.startswith("low") else (-0.35 if display.startswith("edge") else 0.0)
            ax2.text(xval - 0.7, yval + dy, display, fontsize=6.9, va="center", ha="right")
        else:
            ax2.text(xval + 0.6, yval, display, fontsize=6.9, va="center", ha="left")
    ax2.set_xlabel("mean minimum comparison overlap")
    ax2.set_ylabel("mean novel-class ICL")
    ax2.set_title("Branch overlap is the direct risk variable", fontsize=10)
    ax2.grid(color=LIGHT, lw=0.8)
    cb = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.03)
    cb.set_label("M Gini", fontsize=8.5)
    save(fig, "fig_input_multiplicity_controls")


def fig_expressivity_trainability():
    path = ROOT / "ICL/results/next_phase_stats/expressivity_vs_trainability_report.json"
    payload = json.load(open(path))
    outcomes = ["best_seed_novel_icl", "mean_novel_icl", "seed_std_novel_icl"]
    outcome_labels = ["best seed\nexpressivity", "mean seed\nreliability", "seed std.\ninstability"]
    predictor_labels = {
        "comparison_branch_common_d_rel_min": "branch common rank",
        "effective_rank_D_masked": "masked eff. rank",
        "condition_number_D_masked_log10": "conditioning",
        "capacity_linear_test_margin_p10": "old capacity p10",
        "mechanism_branch_active_tree_nmi_mean": "branch-tree NMI",
        "mechanism_target_logprob_margin_branch_mean_min": "trained margin",
        "mechanism_tree_entropy_mean": "tree entropy",
    }
    predictors = [row["predictor"] for row in payload["correlations"][outcomes[0]]]
    matrix = []
    for predictor in predictors:
        row_values = []
        for outcome in outcomes:
            match = [row for row in payload["correlations"][outcome] if row["predictor"] == predictor][0]
            value = match.get("pearson_r")
            row_values.append(float(value) if value is not None else math.nan)
        matrix.append(row_values)

    arr = [[0.0 if math.isnan(value) else value for value in row] for row in matrix]
    fig, ax = plt.subplots(figsize=(5.3, 3.4))
    im = ax.imshow(arr, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(outcomes)), outcome_labels)
    ax.set_yticks(range(len(predictors)), [predictor_labels[p] for p in predictors])
    ax.set_title("Best seed, mean seed, and seed variance are different targets", fontsize=10)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            text = "" if math.isnan(value) else f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="white" if abs(arr[i][j]) > 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Pearson r", fontsize=8.5)
    save(fig, "fig_expressivity_trainability")


def fig_capacity_v2():
    path = ROOT / "ICL/results/next_phase_stats/branch_margin_capacity_v2_smoke.json"
    payload = json.load(open(path))
    results = payload["results"]
    variants = [row["variant"] for row in results]
    objective = [float(row["best"]["objective"]) for row in results]
    failure = [float(row["best"]["branch_failure_rate_max"]) for row in results]
    accuracy = [float(row["best"]["accuracy"]) for row in results]

    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.0))
    fig.subplots_adjust(wspace=0.42)
    x = list(range(len(variants)))
    colors = [BLUE, TEAL, ORANGE]
    bars = axes[0].bar(x, objective, color=colors, width=0.58)
    axes[0].axhline(0, color=INK, lw=0.8)
    axes[0].set_xticks(x, variants)
    axes[0].set_ylabel(r"worst-branch LCVaR margin")
    axes[0].set_title(r"Lower-tail capacity objective", fontsize=10)
    axes[0].grid(axis="y", color=LIGHT, lw=0.8)
    for bar, val in zip(bars, objective):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val - 0.12, f"{val:.2f}", ha="center", va="top", fontsize=8, color="white")

    w = 0.34
    axes[1].bar([i - w / 2 for i in x], accuracy, width=w, color=TEAL, label="accuracy")
    axes[1].bar([i + w / 2 for i in x], failure, width=w, color=RED, label="worst branch failure")
    axes[1].set_xticks(x, variants)
    axes[1].set_ylim(0, 1.08)
    axes[1].set_title("Smoke-run branch failures", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    axes[1].grid(axis="y", color=LIGHT, lw=0.8)
    for i, (acc, fail) in enumerate(zip(accuracy, failure)):
        axes[1].text(i - w / 2, acc + 0.03, f"{acc:.2f}", ha="center", fontsize=7.5)
        axes[1].text(i + w / 2, fail + 0.03, f"{fail:.2f}", ha="center", fontsize=7.5)
    save(fig, "fig_capacity_v2")


def main():
    fig_tree_basis()
    fig_predictors()
    fig_functional()
    fig_mechanism_contrasts()
    fig_causal_scrambles()
    fig_degree_rewire()
    fig_motif_controls()
    fig_markov_reanalysis()
    fig_input_multiplicity_controls()
    fig_expressivity_trainability()
    fig_capacity_v2()
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
