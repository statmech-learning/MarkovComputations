"""Reconcile predictor family names used in the Markov-ICL reports.

The next-phase goal identified a naming collision: fixed-m20 "tree_geometry"
appears with different leave-one-out R2 values in the structural report and the
Markov reanalysis.  This script reads the existing artifacts and emits an
explicit name/feature/target/unit ledger so later reports do not compare
different regressions under the same label.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

from clustered_topology_inference import DEFAULT_MODELS


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "ICL" / "results" / "next_phase_stats"
STRUCTURAL_ARTIFACT = DEFAULT_OUT_DIR / "pooled_fixed_m20_branch_capacity_clustered_inference.json"
MARKOV_ARTIFACT = DEFAULT_OUT_DIR / "existing_data_markov_expressivity_reanalysis.json"

STRUCTURAL_RENAMES = {
    "tree_geometry": "tree_geometry_structural_full",
    "masked_tree_geometry": "masked_tree_geometry_structural",
    "raw_count": "raw_count_structural",
    "raw_plus_drel": "raw_plus_drel_structural",
    "input_count": "input_count_structural",
    "input_count_plus_drel": "input_count_plus_drel_structural",
    "input_count_plus_branch_drel": "input_count_plus_branch_drel_structural",
    "branch_margin_capacity": "branch_margin_capacity_structural",
    "branch_margin_capacity_plus_drel": "branch_margin_capacity_plus_drel_structural",
    "branch_rank_weighted_capacity": "branch_rank_weighted_capacity_structural",
    "branch_rank_weighted_capacity_plus_drel": "branch_rank_weighted_capacity_plus_drel_structural",
    "tropical_tree_capacity": "tropical_tree_capacity_structural",
    "tropical_tree_capacity_plus_drel": "tropical_tree_capacity_plus_drel_structural",
    "rooted_tree_polytope_capacity": "rooted_tree_polytope_capacity_structural",
    "rooted_tree_polytope_capacity_plus_drel": "rooted_tree_polytope_capacity_plus_drel_structural",
    "normal_fan_capacity": "normal_fan_capacity_structural",
    "normal_fan_capacity_plus_drel": "normal_fan_capacity_plus_drel_structural",
}

MARKOV_RENAMES = {
    "multiplicity": "edge_multiplicity_markov_reanalysis",
    "comparison_multiplicity": "comparison_edge_multiplicity_markov_reanalysis",
    "tree_geometry": "tree_geometry_markov_reanalysis_subset",
    "capacity_proxy": "capacity_proxy_markov_reanalysis",
}

TARGET_UNITS = {
    "target_mean": "group-mean",
    "target_max": "group-best",
    "target_std": "seed-std",
    "mean_novel_icl": "group-mean",
    "best_seed_novel_icl": "group-best",
    "seed_std_novel_icl": "seed-std",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def clean_float(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def structural_records(data: dict[str, Any], artifact: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    group_level = data.get("group_level", {})
    for target, models in sorted(group_level.items()):
        for old_name, fit in sorted(models.items()):
            records.append(
                {
                    "source": "structural_clustered_inference",
                    "old_name": old_name,
                    "recommended_name": STRUCTURAL_RENAMES.get(old_name, f"{old_name}_structural"),
                    "feature_columns": list(fit.get("predictors", [])),
                    "target_variable": target,
                    "unit_of_analysis": TARGET_UNITS.get(target, "group-level"),
                    "grouping_loo_scheme": (
                        "Seed runs are aggregated by topology/mask cluster; group-level "
                        "ordinary least squares uses one-row leave-one-out over those clusters."
                    ),
                    "n_rows": clean_float(fit.get("n")),
                    "n_groups": clean_float(fit.get("n_groups_or_rows")),
                    "n_run_rows_source": data.get("n_run_rows"),
                    "n_aggregate_rows_source": data.get("n_aggregate_rows"),
                    "regularization_or_standardization": (
                        "OLS with intercept; non-intercept columns standardized once on the "
                        "full design before leave-one-out scoring; no ridge regularization."
                    ),
                    "source_script": "ICL/clustered_topology_inference.py; ICL/regress_topology_results.py",
                    "source_artifact": str(artifact.relative_to(REPO_ROOT)),
                    "r2": clean_float(fit.get("r2")),
                    "loo_r2": clean_float(fit.get("leave_one_out_r2")),
                    "rmse": clean_float(fit.get("rmse")),
                    "discrepancy_reason": (
                        "Structural predictor set from regress_topology_results.PREDICTOR_SETS; "
                        "compare only to rows with the same recommended_name, target, unit, and LOO scheme."
                    ),
                }
            )
    return records


def markov_records(data: dict[str, Any], artifact: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for analysis_name, analysis in sorted(data.get("analyses", {}).items()):
        for model in analysis.get("models", []):
            old_name = model.get("model")
            outcome = model.get("outcome")
            records.append(
                {
                    "source": "markov_expressivity_reanalysis",
                    "analysis": analysis_name,
                    "old_name": old_name,
                    "recommended_name": MARKOV_RENAMES.get(old_name, f"{old_name}_markov_reanalysis"),
                    "feature_columns": list(model.get("predictors", [])),
                    "target_variable": outcome,
                    "unit_of_analysis": TARGET_UNITS.get(str(outcome), "group-level"),
                    "grouping_loo_scheme": (
                        "Seed runs are grouped by topology/mask group; leave-one-group-out "
                        "prediction standardizes predictors inside each training fold and "
                        "fits an intercept plus ridge-stabilized linear model."
                    ),
                    "n_rows": analysis.get("n_rows"),
                    "n_groups": model.get("n_groups"),
                    "n_groups_source": analysis.get("n_groups"),
                    "regularization_or_standardization": (
                        "Fold-wise predictor centering/scaling; intercept unpenalized; ridge=1e-6."
                    ),
                    "source_script": "ICL/markov_expressivity_reanalysis.py",
                    "source_artifact": str(artifact.relative_to(REPO_ROOT)),
                    "r2": None,
                    "loo_r2": clean_float(model.get("loo_r2")),
                    "rmse": None,
                    "reason": model.get("reason"),
                    "discrepancy_reason": (
                        "Markov reanalysis subset; compare only to rows with the same "
                        "recommended_name, target, unit, and LOO scheme."
                    ),
                }
            )
    return records


def first_record(
    records: Iterable[dict[str, Any]],
    *,
    source: str,
    old_name: str,
    target: str,
    analysis: str | None = None,
) -> dict[str, Any] | None:
    for record in records:
        if record.get("source") != source:
            continue
        if record.get("old_name") != old_name:
            continue
        if record.get("target_variable") != target:
            continue
        if analysis is not None and record.get("analysis") != analysis:
            continue
        return record
    return None


def discrepancy_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    structural = first_record(
        records,
        source="structural_clustered_inference",
        old_name="tree_geometry",
        target="target_mean",
    )
    markov = first_record(
        records,
        source="markov_expressivity_reanalysis",
        old_name="tree_geometry",
        target="mean_novel_icl",
        analysis="fixed_m20",
    )
    if structural is None or markov is None:
        return []
    s_loo = structural.get("loo_r2")
    m_loo = markov.get("loo_r2")
    return [
        {
            "reported_collision": 'fixed m20 "tree_geometry"',
            "structural_record": structural["recommended_name"],
            "markov_record": markov["recommended_name"],
            "structural_loo_r2": s_loo,
            "markov_loo_r2": m_loo,
            "loo_r2_difference": None if s_loo is None or m_loo is None else float(s_loo - m_loo),
            "structural_features": structural["feature_columns"],
            "markov_features": markov["feature_columns"],
            "reason": (
                "The 0.409 structural number and 0.158 Markov-reanalysis number come from "
                "different feature columns, different standardization/regularization schemes, "
                "and differently named target fields.  They should not share the bare name "
                "'tree_geometry'."
            ),
        }
    ]


def markdown_table(rows: list[list[Any]], headers: list[str]) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, float):
            return f"{value:.3f}"
        if isinstance(value, list):
            return "`" + "`, `".join(str(item) for item in value) + "`"
        return str(value)

    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(item).replace("\n", " ") for item in row) + " |")
    return "\n".join(out)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    records = report["predictor_families"]
    fixed_rows = [
        [
            r.get("source"),
            r.get("analysis", "fixed_m20"),
            r.get("old_name"),
            r.get("recommended_name"),
            r.get("target_variable"),
            r.get("unit_of_analysis"),
            r.get("n_groups"),
            r.get("loo_r2"),
            r.get("feature_columns"),
        ]
        for r in records
        if r.get("source") == "structural_clustered_inference"
        or (r.get("source") == "markov_expressivity_reanalysis" and r.get("analysis") == "fixed_m20")
    ]
    compact_rows = sorted(fixed_rows, key=lambda row: (str(row[0]), str(row[2]), str(row[4])))

    lines = [
        "# Predictor Name Reconciliation",
        "",
        "## Gate Result",
        "",
        report["gate_result"],
        "",
        "## Numerical Collision",
        "",
    ]
    for item in report["discrepancies"]:
        lines.extend(
            [
                f"- Collision: `{item['reported_collision']}`.",
                f"- Structural rename: `{item['structural_record']}`; LOO R2 `{item['structural_loo_r2']:.3f}`.",
                f"- Markov rename: `{item['markov_record']}`; LOO R2 `{item['markov_loo_r2']:.3f}`.",
                f"- Difference: `{item['loo_r2_difference']:.3f}`.",
                f"- Reason: {item['reason']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Fixed-m20 Predictor Ledger",
            "",
            markdown_table(
                compact_rows,
                [
                    "source",
                    "analysis",
                    "old_name",
                    "recommended_name",
                    "target",
                    "unit",
                    "groups",
                    "loo_r2",
                    "feature_columns",
                ],
            ),
            "",
            "## Regression Definitions",
            "",
            "- Structural clustered inference: group rows are topology/mask aggregates, OLS is fit after full-design standardization, and one row is left out at a time.",
            "- Markov reanalysis: group rows are topology/mask aggregates, predictors are standardized inside each training fold, and a ridge of `1e-6` stabilizes the fold-wise linear solve.",
            "- Run-level seed rows remain grouped by topology/mask; seed rows should not be treated as independent topologies.",
            "",
            "## Required Rename Map",
            "",
            markdown_table(
                [[old, new] for old, new in sorted(report["rename_map"]["structural"].items())],
                ["structural old name", "recommended name"],
            ),
            "",
            markdown_table(
                [[old, new] for old, new in sorted(report["rename_map"]["markov_reanalysis"].items())],
                ["markov old name", "recommended name"],
            ),
            "",
            "## No New Claim",
            "",
            "This artifact only resolves naming and regression definitions. It does not add a new scientific claim about which predictor family is better.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def build_report(structural_path: Path, markov_path: Path) -> dict[str, Any]:
    structural = load_json(structural_path)
    markov = load_json(markov_path)
    records = structural_records(structural, structural_path) + markov_records(markov, markov_path)
    return {
        "schema": "predictor_name_reconciliation.v1",
        "gate_result": (
            "Phase 1 passes only after the ambiguous bare names are replaced by the recommended names. "
            "The fixed-m20 0.409 and 0.158 LOO R2 values are not contradictory because they are different regressions."
        ),
        "source_model_lists": {
            "clustered_topology_inference.DEFAULT_MODELS": list(DEFAULT_MODELS),
            "markov_expressivity_reanalysis.complete_predictors": sorted(MARKOV_RENAMES),
        },
        "rename_map": {
            "structural": STRUCTURAL_RENAMES,
            "markov_reanalysis": MARKOV_RENAMES,
        },
        "predictor_families": records,
        "discrepancies": discrepancy_summary(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--structural-artifact", default=str(STRUCTURAL_ARTIFACT))
    parser.add_argument("--markov-artifact", default=str(MARKOV_ARTIFACT))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(Path(args.structural_artifact), Path(args.markov_artifact))
    json_path = out_dir / "predictor_name_reconciliation.json"
    md_path = out_dir / "predictor_name_reconciliation.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown(report, md_path)
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
