"""Validate lower-tail gamma* probes on analytic Markov-ICL toy cases."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from branch_margin_capacity_v2 import json_ready, lower_tail_capacity_probe
from topology_metrics import complete_digraph


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "ICL" / "results" / "next_phase_stats"


TOYS = [
    {
        "name": "toy_A_two_species_both_branches",
        "n_nodes": 2,
        "n_context": 2,
        "z_dim": 1,
        "branch_subset": None,
        "expectation": "fail: gamma*_ICL <= 0 or persistent negative branch lower-tail margin",
    },
    {
        "name": "toy_B_two_species_one_branch",
        "n_nodes": 2,
        "n_context": 2,
        "z_dim": 1,
        "branch_subset": [0],
        "expectation": "pass: gamma*_ICL > 0 under reasonable norm budget",
    },
    {
        "name": "toy_C_three_species_both_branches",
        "n_nodes": 3,
        "n_context": 2,
        "z_dim": 1,
        "branch_subset": None,
        "expectation": "pass: gamma*_ICL > 0 or clear improvement over Toy A",
    },
]

VARIANTS = ["exact", "tropical", "hard_root"]
BIAS_SETTINGS = [
    ("gamma_no_bias", 0.0),
    ("gamma_with_bias", 2.0),
]


def compact_probe(result: dict[str, Any]) -> dict[str, Any]:
    best = result["best"]
    return {
        "variant": result["variant"],
        "active_branches": result.get("active_branches"),
        "edge_bias_radius": result["edge_bias_radius"],
        "n_samples": result["n_samples"],
        "total_trials": result["total_trials"],
        "n_trees_total": result["n_trees_total"],
        "best_objective": best["objective"],
        "best_accuracy": best["accuracy"],
        "best_branch_failure_rate_max": best["branch_failure_rate_max"],
        "best_margin_p10": best["margin_p10"],
        "best_branch_margin_lcvar_min": best["branch_margin_lcvar_min"],
        "best_branch_accuracy_min": best["branch_accuracy_min"],
        "best_root_assignment": result.get("best_root_assignment"),
        "by_branch": best["by_branch"],
    }


def toy_pass_status(toy_name: str, rows: Sequence[dict[str, Any]], toy_a_reference: float | None) -> tuple[bool, str]:
    objectives = [float(row["best_objective"]) for row in rows if row.get("best_objective") is not None]
    failures = [float(row["best_branch_failure_rate_max"]) for row in rows if row.get("best_branch_failure_rate_max") is not None]
    best_obj = max(objectives) if objectives else float("-inf")
    min_failure = min(failures) if failures else 1.0
    if toy_name.startswith("toy_A"):
        ok = all(value <= 0.0 for value in objectives) or min_failure > 0.0
        return ok, "expected failure observed" if ok else "unexpected positive lower-tail objective"
    if toy_name.startswith("toy_B"):
        ok = best_obj > 0.0 and min_failure == 0.0
        return ok, "positive one-branch margin found" if ok else "one-branch margin was not positive"
    if toy_name.startswith("toy_C"):
        improved = toy_a_reference is not None and best_obj > toy_a_reference + 0.05
        ok = best_obj > 0.0 or improved
        return ok, "positive or improved three-species margin found" if ok else "three-species case did not improve enough"
    return False, "unknown toy"


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    toy_rows = []
    toy_a_best_no_bias = None
    for toy in TOYS:
        edges = complete_digraph(toy["n_nodes"]).edges
        case = {
            "name": toy["name"],
            "expectation": toy["expectation"],
            "n_nodes": toy["n_nodes"],
            "n_context": toy["n_context"],
            "z_dim": toy["z_dim"],
            "branch_subset": toy["branch_subset"],
            "results": {},
        }
        for bias_label, edge_bias_radius in BIAS_SETTINGS:
            rows = []
            for variant in VARIANTS:
                result = lower_tail_capacity_probe(
                    n_nodes=toy["n_nodes"],
                    edges=edges,
                    n_context=toy["n_context"],
                    z_dim=toy["z_dim"],
                    variant=variant,
                    n_samples=args.n_samples,
                    trials=args.trials,
                    seed=args.seed + 37 * len(toy_rows) + 11 * VARIANTS.index(variant),
                    alpha=args.alpha,
                    projection_radius=args.projection_radius,
                    decoder_radius=args.decoder_radius,
                    edge_bias_radius=edge_bias_radius,
                    max_root_assignments=args.max_root_assignments,
                    branch_subset=toy["branch_subset"],
                )
                rows.append(compact_probe(result))
            reference = toy_a_best_no_bias if bias_label == "gamma_no_bias" else None
            passed, reason = toy_pass_status(toy["name"], rows, reference)
            case["results"][bias_label] = {
                "edge_bias_radius": edge_bias_radius,
                "passed_expected_direction": passed,
                "reason": reason,
                "variant_rows": rows,
            }
            if toy["name"].startswith("toy_A") and bias_label == "gamma_no_bias":
                toy_a_best_no_bias = max(float(row["best_objective"]) for row in rows)
        toy_rows.append(case)

    toy_b_bias = next(item for item in toy_rows if item["name"].startswith("toy_B"))["results"]["gamma_with_bias"]
    toy_a_no_bias = next(item for item in toy_rows if item["name"].startswith("toy_A"))["results"]["gamma_no_bias"]
    toy_c_no_bias = next(item for item in toy_rows if item["name"].startswith("toy_C"))["results"]["gamma_no_bias"]
    phase_passed = (
        bool(toy_a_no_bias["passed_expected_direction"])
        and bool(toy_b_bias["passed_expected_direction"])
        and bool(toy_c_no_bias["passed_expected_direction"])
    )
    return {
        "schema": "gamma_toy_validation_report.v1",
        "probe": "ICL/branch_margin_capacity_v2.py lower_tail_capacity_probe",
        "n_samples": args.n_samples,
        "trials": args.trials,
        "alpha": args.alpha,
        "projection_radius": args.projection_radius,
        "decoder_radius": args.decoder_radius,
        "phase_3_gate_passed": phase_passed,
        "phase_3_gate_reason": (
            "Do not use gamma*_ICL for large topology selection yet."
            if not phase_passed
            else "Toy validation passed the configured checks."
        ),
        "toy_cases": toy_rows,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        return f"{value:.3f}"
    return str(value)


def markdown_table(rows: list[list[Any]], headers: list[str]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(out)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Gamma Toy Validation Report",
        "",
        "## Gate Result",
        "",
        f"Phase 3 gate passed: `{report['phase_3_gate_passed']}`.",
        "",
        report["phase_3_gate_reason"],
        "",
        "## Setup",
        "",
        f"- Probe: `{report['probe']}`.",
        f"- Samples per toy: `{report['n_samples']}`.",
        f"- Trials per variant: `{report['trials']}`.",
        f"- Lower-tail alpha: `{report['alpha']}`.",
        f"- Projection radius: `{report['projection_radius']}`.",
        f"- Decoder radius: `{report['decoder_radius']}`.",
        "- Bias labels: `gamma_no_bias` uses `edge_bias_radius=0`; `gamma_with_bias` uses positive edge-bias budget.",
        "",
    ]
    for toy in report["toy_cases"]:
        lines.extend([f"## {toy['name']}", "", toy["expectation"], ""])
        rows = []
        for bias_label, block in toy["results"].items():
            for row in block["variant_rows"]:
                rows.append(
                    [
                        bias_label,
                        row["variant"],
                        block["passed_expected_direction"],
                        block["reason"],
                        row["best_objective"],
                        row["best_accuracy"],
                        row["best_branch_failure_rate_max"],
                        row["best_margin_p10"],
                        row["active_branches"],
                    ]
                )
        lines.extend(
            [
                markdown_table(
                    rows,
                    [
                        "bias",
                        "variant",
                        "case pass",
                        "reason",
                        "objective",
                        "accuracy",
                        "worst failure",
                        "p10 margin",
                        "active branches",
                    ],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "Toy A correctly fails in the no-bias setting. Toy B becomes positive when edge biases are allowed, but the no-bias one-branch case remains negative in this finite-sample probe. Toy C does not pass the configured no-bias analytic check. Therefore lower-tail `gamma*_ICL` should remain gated off for large topology selection until the probe definition or optimizer is repaired.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--trials", type=int, default=768)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--projection-radius", type=float, default=4.0)
    parser.add_argument("--decoder-radius", type=float, default=4.0)
    parser.add_argument("--max-root-assignments", type=int, default=24)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = run_validation(args)
    json_path = out_dir / "gamma_toy_validation_report.json"
    md_path = out_dir / "gamma_toy_validation_report.md"
    json_path.write_text(json.dumps(json_ready(report), indent=2, sort_keys=True) + "\n")
    write_markdown(json_ready(report), md_path)
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
