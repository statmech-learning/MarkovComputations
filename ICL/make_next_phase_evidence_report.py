"""Build a compact next-phase topology-ICL evidence report.

This report is intentionally narrower than ``make_topology_research_report``.
It summarizes the follow-up artifacts requested after the first critique:

* clustered/group-aware inference for nested seed rows,
* optional branch-margin capacity enriched regressions,
* causal branch/tree-alignment interventions,
* branch-margin capacity probe summaries,
* matched essential-motif control retrain summaries,
* expanded pilot sweep status.

The script is read-only with respect to experiment outputs. It does not submit
or collect jobs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from typing import Iterable, List, Sequence, Tuple


CORE_MODELS = [
    "raw_count",
    "raw_plus_drel",
    "input_count_plus_drel",
    "input_count_plus_branch_drel",
    "tree_geometry",
    "masked_tree_geometry",
    "branch_margin_capacity",
    "branch_margin_capacity_plus_drel",
    "branch_rank_weighted_capacity",
    "branch_rank_weighted_capacity_plus_drel",
    "tropical_tree_capacity",
    "tropical_tree_capacity_plus_drel",
]


def load_json(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def parse_labeled_path(raw: str) -> Tuple[str, str]:
    if "=" not in raw:
        label = os.path.splitext(os.path.basename(raw))[0]
        return label, raw
    label, path = raw.split("=", 1)
    return label.strip(), path.strip()


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def load_labeled_jsons(items: Sequence[str]) -> List[dict]:
    loaded = []
    for item in items:
        label, path = parse_labeled_path(item)
        payload = load_json(path)
        loaded.append({"label": label, "path": path, "payload": payload})
    return loaded


def clustered_summary(entry: dict) -> dict:
    payload = entry["payload"]
    group = payload.get("group_level", {}).get("target_mean", {})
    bootstrap = payload.get("cluster_bootstrap_run_level", {})
    family_bootstrap = payload.get("family_cluster_bootstrap_run_level", {})
    holdout = payload.get("leave_family_out_group_target_mean", {})
    models = {}
    for name in CORE_MODELS:
        fit = group.get(name)
        boot = bootstrap.get(name)
        held = holdout.get(name)
        if not fit and not boot and not held:
            continue
        models[name] = {
            "group_n": (fit or {}).get("n") or (fit or {}).get("n_groups_or_rows"),
            "group_loo_r2": (fit or {}).get("leave_one_out_r2"),
            "bootstrap_delta_mean": (boot or {}).get("delta_mean"),
            "bootstrap_delta_ci95": (boot or {}).get("delta_ci95"),
            "bootstrap_prob_positive": (boot or {}).get("prob_delta_positive"),
            "family_bootstrap_delta_mean": (family_bootstrap.get(name) or {}).get("delta_mean"),
            "family_bootstrap_prob_positive": (family_bootstrap.get(name) or {}).get("prob_delta_positive"),
            "heldout_family_pooled_r2": (held or {}).get("pooled_r2"),
            "heldout_family_rmse": (held or {}).get("pooled_rmse"),
        }
    return {
        "label": entry["label"],
        "path": entry["path"],
        "n_run_rows": payload.get("n_run_rows"),
        "n_clusters": payload.get("n_clusters"),
        "n_families": payload.get("n_families"),
        "family_col": payload.get("family_col"),
        "models": models,
    }


def causal_summary(entry: dict) -> dict:
    payload = entry["payload"]
    interventions = []
    for name, stats in sorted(payload.get("interventions", {}).items()):
        interventions.append(
            {
                "intervention": name,
                "n": stats.get("n"),
                "target_accuracy_delta_mean": stats.get("target_accuracy_delta_mean"),
                "target_accuracy_delta_min": stats.get("target_accuracy_delta_min"),
                "target_accuracy_delta_max": stats.get("target_accuracy_delta_max"),
            }
        )
    return {
        "label": entry["label"],
        "path": entry["path"],
        "n_rows": payload.get("n_rows"),
        "n_runs": payload.get("n_runs"),
        "interventions": interventions,
    }


def capacity_summary(entry: dict) -> dict:
    payload = entry["payload"]
    families = []
    for name, stats in sorted(payload.get("families", {}).items()):
        families.append(
            {
                "family": name,
                "n": stats.get("n"),
                "linear_test_accuracy_mean": stats.get("linear_test_accuracy_mean"),
                "linear_test_accuracy_max": stats.get("linear_test_accuracy_max"),
                "rank_weighted_linear_test_accuracy_mean": stats.get("rank_weighted_linear_test_accuracy_mean"),
                "rank_weighted_linear_test_accuracy_max": stats.get("rank_weighted_linear_test_accuracy_max"),
                "tropical_linear_test_accuracy_mean": stats.get("tropical_linear_test_accuracy_mean"),
                "tropical_linear_test_accuracy_max": stats.get("tropical_linear_test_accuracy_max"),
                "tropical_root_feature_effective_rank_mean": stats.get(
                    "tropical_root_feature_effective_rank_mean"
                ),
            }
        )
    return {
        "label": entry["label"],
        "path": entry["path"],
        "n_rows": payload.get("n_rows"),
        "families": families,
    }


def matched_motif_summary(entry: dict) -> dict:
    payload = entry["payload"]
    return {
        "label": entry["label"],
        "path": entry["path"],
        "n_joined": payload.get("n_joined"),
        "overall": payload.get("overall", {}),
        "by_control_kind": payload.get("by_control_kind", {}),
    }


def matched_motif_interpretation(entry: dict) -> str:
    kind_stats = list((entry.get("by_control_kind") or {}).values())
    deltas = [
        stats.get("control_minus_source_retrain_mean_mean")
        for stats in kind_stats
        if stats.get("control_minus_source_retrain_mean_mean") is not None
    ]
    win_rates = [
        stats.get("control_win_rate_mean")
        for stats in kind_stats
        if stats.get("control_win_rate_mean") is not None
    ]
    if deltas and all(float(delta) < 0.0 for delta in deltas):
        if not win_rates or all(float(rate) < 0.5 for rate in win_rates):
            return (
                "Extracted motifs beat these matched controls on mean retrain ICL, "
                "supporting a functionally specific motif-retention interpretation "
                "within this tested first-order regime."
            )
    if deltas and all(float(delta) > 0.0 for delta in deltas):
        return (
            "Matched controls retrain above the extracted motifs here, so this "
            "backbone does not support a unique extracted-motif superiority claim."
        )
    return (
        "Matched controls are mixed or comparable to the extracted motifs here; "
        "interpret motif retraining as evidence that small matched physical "
        "subgraphs can support ICL, not that the extracted edge set is uniquely "
        "superior."
    )


def read_count(root: str, filename: str) -> int:
    total = 0
    for _, _, files in os.walk(root):
        if filename in files:
            total += 1
    return total


def expanded_status(items: Sequence[str]) -> List[dict]:
    rows = []
    for item in items:
        label, root = parse_labeled_path(item)
        rows.append(
            {
                "label": label,
                "root": root,
                "results_pkl_count": read_count(root, "results.pkl") if os.path.exists(root) else 0,
                "mechanism_count": read_count(root, "mechanism_metrics.json") if os.path.exists(root) else 0,
                "causal_count": read_count(root, "causal_interventions.json") if os.path.exists(root) else 0,
            }
        )
    return rows


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return out


def build_report(args) -> dict:
    clustered = [clustered_summary(entry) for entry in load_labeled_jsons(args.clustered_json)]
    causal = [causal_summary(entry) for entry in load_labeled_jsons(args.causal_json)]
    capacity = [capacity_summary(entry) for entry in load_labeled_jsons(args.branch_capacity_json)]
    matched_motifs = [
        matched_motif_summary(entry)
        for entry in load_labeled_jsons(args.matched_motif_json)
    ]
    expanded = expanded_status(args.expanded_root)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "First-order CRNs with exponential input-dependent rates; matrix-tree theorem "
            "controls the topology-to-steady-state map."
        ),
        "clustered_inference": clustered,
        "causal_interventions": causal,
        "branch_margin_capacity": capacity,
        "matched_motif_controls": matched_motifs,
        "expanded_pilot_status": expanded,
    }


def build_markdown(report: dict) -> str:
    lines = [
        "# Next-Phase Topology-ICL Evidence Report",
        "",
        f"Generated: `{report['generated_at']}`.",
        "",
        "Scope: first-order CRNs with exponential input-dependent rates. These results do not claim a topology theory for autocatalytic or WTA CRNs.",
        "",
        "Conservative headline: in the tested fixed-count regimes, topology-associated structural and functional variables explain residual novel-class ICL variation that raw count does not explain.",
        "",
        "## Clustered And Group-Aware Inference",
        "",
    ]
    if not report["clustered_inference"]:
        lines.append("No clustered inference artifacts were supplied.")
    for entry in report["clustered_inference"]:
        family_col = entry.get("family_col")
        family_text = (
            f"Families: `{entry['n_families']}` via `{family_col}`."
            if family_col
            else f"Families: `{entry['n_families']}`."
        )
        lines.extend(
            [
                f"### {entry['label']}",
                "",
                (
                    f"Rows: `{entry['n_run_rows']}`. Groups: `{entry['n_clusters']}`. "
                    f"{family_text}"
                ),
                "",
            ]
        )
        rows = []
        for name, stats in entry["models"].items():
            ci = stats.get("bootstrap_delta_ci95") or [None, None]
            rows.append(
                [
                    name,
                    str(stats.get("group_n") or "NA"),
                    fmt(stats.get("group_loo_r2")),
                    fmt(stats.get("bootstrap_delta_mean")),
                    fmt(stats.get("family_bootstrap_delta_mean")),
                    f"[{fmt(ci[0])}, {fmt(ci[1])}]",
                    fmt(stats.get("bootstrap_prob_positive")),
                    fmt(stats.get("heldout_family_pooled_r2")),
                ]
            )
        lines.extend(
            markdown_table(
                [
                    "model",
                    "n",
                    "group LOO R2",
                    "boot delta R2",
                    "family boot delta R2",
                    "CI95",
                    "P(delta>0)",
                    "heldout R2",
                ],
                rows,
            )
        )
        lines.append("")

    lines.extend(["## Causal Alignment Interventions", ""])
    if not report["causal_interventions"]:
        lines.append("No causal intervention summaries were supplied.")
    for entry in report["causal_interventions"]:
        lines.extend(
            [
                f"### {entry['label']}",
                "",
                f"Rows: `{entry['n_rows']}`. Runs: `{entry['n_runs']}`.",
                "",
            ]
        )
        rows = [
            [
                item["intervention"],
                str(item.get("n") or "NA"),
                fmt(item.get("target_accuracy_delta_mean"), 2),
                fmt(item.get("target_accuracy_delta_min"), 2),
                fmt(item.get("target_accuracy_delta_max"), 2),
            ]
            for item in entry["interventions"]
        ]
        lines.extend(markdown_table(["intervention", "n", "mean delta", "min", "max"], rows))
        lines.append("")

    lines.extend(["## Branch-Margin Capacity Probes", ""])
    if not report["branch_margin_capacity"]:
        lines.append("No branch-margin capacity summaries were supplied.")
    for entry in report["branch_margin_capacity"]:
        lines.extend([f"### {entry['label']}", "", f"Rows: `{entry['n_rows']}`.", ""])
        rows = [
            [
                item["family"],
                str(item.get("n") or "NA"),
                fmt(item.get("linear_test_accuracy_mean")),
                fmt(item.get("linear_test_accuracy_max")),
                fmt(item.get("rank_weighted_linear_test_accuracy_mean")),
                fmt(item.get("rank_weighted_linear_test_accuracy_max")),
                fmt(item.get("tropical_linear_test_accuracy_mean")),
                fmt(item.get("tropical_linear_test_accuracy_max")),
                fmt(item.get("tropical_root_feature_effective_rank_mean")),
            ]
            for item in entry["families"]
        ]
        lines.extend(
            markdown_table(
                [
                    "family",
                    "n",
                    "linear accuracy mean",
                    "linear accuracy max",
                    "rank-weighted linear mean",
                    "rank-weighted linear max",
                    "tropical accuracy mean",
                    "tropical accuracy max",
                    "tropical root eff-rank",
                ],
                rows,
            )
        )
        lines.append("")

    lines.extend(["## Matched Essential-Motif Controls", ""])
    if not report["matched_motif_controls"]:
        lines.append("No matched essential-motif control summaries were supplied.")
    for entry in report["matched_motif_controls"]:
        overall = entry.get("overall") or {}
        lines.extend(
            [
                f"### {entry['label']}",
                "",
                f"Joined controls: `{entry['n_joined']}`. Source motifs represented: `{overall.get('n_sources', 'NA')}`.",
                "",
                matched_motif_interpretation(entry),
                "",
            ]
        )
        rows = []
        for kind, stats in sorted((entry.get("by_control_kind") or {}).items()):
            rows.append(
                [
                    kind,
                    str(stats.get("n") or "NA"),
                    str(stats.get("n_sources") or "NA"),
                    fmt(stats.get("control_target_mean_mean"), 2),
                    fmt(stats.get("source_retrain_target_mean_mean"), 2),
                    fmt(stats.get("control_minus_source_retrain_mean_mean"), 2),
                    fmt(stats.get("control_win_rate_mean"), 3),
                    fmt(stats.get("match_score_mean"), 3),
                ]
            )
        lines.extend(
            markdown_table(
                [
                    "control kind",
                    "controls",
                    "sources",
                    "control mean ICL",
                    "source motif mean ICL",
                    "control-source delta",
                    "control win rate",
                    "match score",
                ],
                rows,
            )
        )
        lines.append("")

    lines.extend(["## Expanded Pilot Status", ""])
    if not report["expanded_pilot_status"]:
        lines.append("No expanded pilot roots were supplied.")
    else:
        rows = [
            [
                item["label"],
                f"`{item['root']}`",
                str(item["results_pkl_count"]),
                str(item["mechanism_count"]),
                str(item["causal_count"]),
            ]
            for item in report["expanded_pilot_status"]
        ]
        lines.extend(markdown_table(["regime", "root", "results.pkl", "mechanisms", "causal"], rows))
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- Treat run rows as seeds nested inside topology/mask groups; group-level and clustered summaries are the safer evidence.",
            "- Use `test_novel_classes` as the headline ICL metric.",
            "- Interpret causal scrambling as evidence for branch/projection alignment only when baseline accuracy is high enough to make a collapse meaningful.",
            "- Treat branch-margin capacity as a proxy for tree-polytope branch coverage, not as the final capacity theory.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clustered_json", action="append", default=[])
    parser.add_argument("--causal_json", action="append", default=[])
    parser.add_argument("--branch_capacity_json", action="append", default=[])
    parser.add_argument("--matched_motif_json", action="append", default=[])
    parser.add_argument("--expanded_root", action="append", default=[])
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    report = build_report(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_md)), exist_ok=True)
    with open(args.output_md, "w") as handle:
        handle.write(build_markdown(report))
    with open(args.output_json, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote next-phase evidence report to {args.output_md}")
    print(f"Wrote next-phase evidence JSON to {args.output_json}")


if __name__ == "__main__":
    main()
