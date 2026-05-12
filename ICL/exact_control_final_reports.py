"""Write final exact-control synthesis reports for the post-gamma Markov-ICL phase."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "next_phase_stats"


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def fnum(value: Any, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def finite_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def model_r2(report: dict[str, Any], outcome: str, model: str) -> float | None:
    for row in report.get("model_results", {}).get(outcome, []):
        if row.get("model") == model:
            return finite_float(row.get("loo_r2"))
    return None


def best_model(report: dict[str, Any], outcome: str) -> dict[str, Any] | None:
    rows = [
        row
        for row in report.get("model_results", {}).get(outcome, [])
        if finite_float(row.get("loo_r2")) is not None
    ]
    if not rows:
        return None
    return max(rows, key=lambda row: float(row["loo_r2"]))


def lookup_contrast(report: dict[str, Any], load: str, outcome: str) -> dict[str, Any] | None:
    for row in report.get("contrasts", []):
        if row.get("load_stratum") == load and row.get("outcome") == outcome:
            return row
    return None


def summarize_mechanisms(rows: list[dict[str, str]]) -> dict[str, Any]:
    fields = [
        "branch_active_tree_mi",
        "branch_active_root_mi",
        "tree_entropy_mean",
        "root_entropy_mean",
        "target_logprob_margin_branch_mean_min",
        "posterior_matched_comparison_gap_mean",
        "active_tree_matched_comparison_gap_mean",
        "tree_comparison_energy_fraction_mean",
        "input_ablation_max_loss",
        "physical_ablation_max_loss",
        "edge_importance_gini",
    ]
    summary: dict[str, Any] = {
        "n_rows": len(rows),
        "n_groups": len({row.get("topology_name", "") for row in rows}),
        "fields": {},
    }
    for field in fields:
        values = [finite_float(row.get(field)) for row in rows]
        values = [value for value in values if value is not None]
        summary["fields"][field] = {
            "available": bool(values),
            "n": len(values),
            "mean": mean(values) if values else None,
            "std": pstdev(values) if len(values) > 1 else 0.0 if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    by_prefix = defaultdict(list)
    for row in rows:
        label = row.get("label", "")
        if "balanced_load" in label:
            key = "balanced_load"
        elif "imbalanced_coord_load" in label:
            key = "imbalanced_coord_load"
        else:
            key = "unknown"
        value = finite_float(row.get("target_logprob_margin_branch_mean_min"))
        if value is not None:
            by_prefix[key].append(value)
    summary["trained_margin_by_load_family"] = {
        key: {"n": len(values), "mean": mean(values)}
        for key, values in sorted(by_prefix.items())
        if values
    }
    return summary


def summarize_selected_ablations(rows: list[dict[str, str]]) -> dict[str, Any]:
    fields = [
        "input_ablation_max_loss",
        "input_ablation_mean_loss",
        "physical_ablation_max_loss",
        "physical_ablation_mean_loss",
    ]
    summary: dict[str, Any] = {
        "n_rows": len(rows),
        "n_groups": len({row.get("topology_name", "") for row in rows}),
        "fields": {},
    }
    for field in fields:
        values = [finite_float(row.get(field)) for row in rows]
        values = [value for value in values if value is not None]
        summary["fields"][field] = {
            "available": bool(values),
            "n": len(values),
            "mean": mean(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return summary


def summarize_interventions(csv_rows: list[dict[str, str]], summary: dict[str, Any]) -> dict[str, Any]:
    by_intervention: dict[str, list[float]] = defaultdict(list)
    by_intervention_margin: dict[str, list[float]] = defaultdict(list)
    for row in csv_rows:
        intervention = row.get("intervention", "")
        delta = finite_float(row.get("target_accuracy_delta"))
        margin_delta = finite_float(row.get("target_logprob_margin_mean_delta"))
        if delta is not None:
            by_intervention[intervention].append(delta)
        if margin_delta is not None:
            by_intervention_margin[intervention].append(margin_delta)
    result = {
        "n_rows": len(csv_rows),
        "n_runs": len({row.get("run_dir", "") for row in csv_rows}),
        "interventions": {},
    }
    for name in sorted(by_intervention):
        values = by_intervention[name]
        margins = by_intervention_margin.get(name, [])
        result["interventions"][name] = {
            "n": len(values),
            "accuracy_delta_mean": mean(values),
            "accuracy_delta_min": min(values),
            "accuracy_delta_max": max(values),
            "margin_delta_mean": mean(margins) if margins else None,
        }
    result["collector_summary"] = summary
    return result


def md_table(headers: list[str], rows: Iterable[list[Any]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def make_track4(
    gamma_report: dict[str, Any],
    prospective_report: dict[str, Any],
    normal_report: dict[str, Any],
) -> dict[str, Any]:
    fixed_interp = gamma_report.get("interpretation", {})
    payload = {
        "schema": "expressivity_vs_trainability_exact_control_report.v1",
        "source_artifacts": [
            "repaired_gamma_existing_data_reanalysis.md",
            "prospective_tree_diff_multiplicity_causal_report.md",
            "exact_degree_exact_drel_exact_multiplicity_normal_fan_training_report.md",
        ],
        "fixed_m20_gamma": {
            "best_gamma_model": fixed_interp.get("fixed_m20_best_gamma_model"),
            "best_gamma_mean_loo_r2": fixed_interp.get("fixed_m20_best_gamma_mean_loo_r2"),
            "tree_difference_mean_loo_r2": fixed_interp.get("fixed_m20_mean_tree_difference_loo_r2"),
            "selector_gate": fixed_interp.get("gamma_selector_gate"),
        },
        "prospective_exact_control": {
            "mean_controls_r2": model_r2(prospective_report, "mean_seed_novel_icl", "controls_only"),
            "mean_tree_diff_r2": model_r2(
                prospective_report,
                "mean_seed_novel_icl",
                "tree_difference_multiplicity_plus_controls",
            ),
            "mean_gamma_r2": model_r2(prospective_report, "mean_seed_novel_icl", "gamma_no_bias_plus_controls"),
            "best_controls_r2": model_r2(prospective_report, "best_seed_novel_icl", "controls_only"),
            "best_tree_diff_r2": model_r2(
                prospective_report,
                "best_seed_novel_icl",
                "tree_difference_multiplicity_plus_controls",
            ),
            "best_gamma_r2": model_r2(prospective_report, "best_seed_novel_icl", "gamma_no_bias_plus_controls"),
            "seed_std_best_model": best_model(prospective_report, "seed_std_novel_icl"),
        },
        "normal_fan_exact_control": {
            "mean_best_model": best_model(normal_report, "mean_seed_novel_icl"),
            "best_best_model": best_model(normal_report, "best_seed_novel_icl"),
            "seed_std_best_model": best_model(normal_report, "seed_std_novel_icl"),
            "gamma_mean_r2": model_r2(normal_report, "mean_seed_novel_icl", "gamma_no_bias_exact"),
            "gamma_best_r2": model_r2(normal_report, "best_seed_novel_icl", "gamma_no_bias_exact"),
            "gamma_plus_normal_fan_mean_r2": model_r2(
                normal_report, "mean_seed_novel_icl", "gamma_plus_normal_fan"
            ),
            "gamma_plus_normal_fan_best_r2": model_r2(
                normal_report, "best_seed_novel_icl", "gamma_plus_normal_fan"
            ),
        },
        "interpretation": {
            "expressivity": (
                "No tested repaired gamma_no_bias variant currently predicts best-seed ICL strongly. "
                "Normal-fan active-tree/tree-count variables give weak best-seed signal under exact controls."
            ),
            "trainability": (
                "Mean-seed prediction tracks the same weak normal-fan/tree-count signal, while seed-std "
                "models remain negative or small; the current controls do not isolate a clean trainability metric."
            ),
            "gamma_status": "candidate_diagnostic_not_selector",
        },
    }
    return payload


def write_track4(payload: dict[str, Any]) -> None:
    fixed = payload["fixed_m20_gamma"]
    p = payload["prospective_exact_control"]
    n = payload["normal_fan_exact_control"]
    text = f"""# Expressivity vs Trainability Exact-Control Report

## Scope

This report separates best-seed ICL as an expressivity upper envelope from mean-seed ICL and seed-standard deviation as trainability and reliability diagnostics. It uses only the repaired-gamma existing-data reanalysis, the prospective tree-difference exact-control experiment, and the exact-degree normal-fan expansion.

## Repaired Gamma Existing-Data Result

In the fixed-m20 mask library, the best repaired no-bias gamma model for mean ICL was `{fixed.get('best_gamma_model')}` with grouped LOO `R2 = {fnum(fixed.get('best_gamma_mean_loo_r2'))}`. The tree-difference multiplicity model remained stronger with grouped LOO `R2 = {fnum(fixed.get('tree_difference_mean_loo_r2'))}`. The selector gate is `{fixed.get('selector_gate')}`.

## Prospective Tree-Difference Control

{md_table(
        ['outcome', 'controls', 'tree-diff + controls', 'gamma + controls'],
        [
            [
                'mean seed ICL',
                fnum(p.get('mean_controls_r2')),
                fnum(p.get('mean_tree_diff_r2')),
                fnum(p.get('mean_gamma_r2')),
            ],
            [
                'best seed ICL',
                fnum(p.get('best_controls_r2')),
                fnum(p.get('best_tree_diff_r2')),
                fnum(p.get('best_gamma_r2')),
            ],
        ],
    )}

The prospective exact-control result does not show a clean expressivity/trainability split in favor of tree-difference overlap or repaired gamma. Controls-only already explain more than the added tree-difference and gamma models for mean ICL; tree-level overlap, not tree-difference overlap, was the only small best-seed improvement in that report.

## Exact-Degree Normal-Fan Expansion

{md_table(
        ['diagnostic', 'model', 'grouped LOO R2'],
        [
            ['mean seed ICL best model', n.get('mean_best_model', {}).get('model'), fnum(n.get('mean_best_model', {}).get('loo_r2'))],
            ['best seed ICL best model', n.get('best_best_model', {}).get('model'), fnum(n.get('best_best_model', {}).get('loo_r2'))],
            ['seed std best model', n.get('seed_std_best_model', {}).get('model'), fnum(n.get('seed_std_best_model', {}).get('loo_r2'))],
            ['gamma exact, mean ICL', 'gamma_no_bias_exact', fnum(n.get('gamma_mean_r2'))],
            ['gamma exact, best ICL', 'gamma_no_bias_exact', fnum(n.get('gamma_best_r2'))],
            ['gamma + normal fan, best ICL', 'gamma_plus_normal_fan', fnum(n.get('gamma_plus_normal_fan_best_r2'))],
        ],
    )}

The exact-degree normal-fan expansion gives a weak signal for active-tree/tree-count geometry on both mean and best seed ICL. Repaired gamma alone is not predictive here, and adding gamma to normal-fan features only reaches the weak normal-fan range.

## Interpretation

- Expressivity: weak evidence that normal-fan active-tree/tree-count variables predict the best-seed upper envelope under exact controls.
- Trainability: no clean metric yet explains seed variance; the tested seed-std LOO models are negative or small.
- Gamma: repaired `gamma_no_bias` passed analytic toys but is not yet useful as a selector in these existing-data or exact-control analyses.
"""
    write_text(OUT / "expressivity_vs_trainability_exact_control_report.md", text)
    write_json(OUT / "expressivity_vs_trainability_exact_control_report.json", payload)


def make_track5(
    mechanism_rows: list[dict[str, str]],
    selected_ablation_rows: list[dict[str, str]],
    intervention_rows: list[dict[str, str]],
    intervention_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "mechanism_and_causal_scramble_followup_report.v1",
        "source_artifacts": [
            "prospective_tree_diff_multiplicity_mechanism_results.csv",
            "prospective_tree_diff_multiplicity_causal_interventions.csv",
            "prospective_tree_diff_multiplicity_causal_interventions.json",
        ],
        "mechanism_summary": summarize_mechanisms(mechanism_rows),
        "selected_ablation_summary": summarize_selected_ablations(selected_ablation_rows),
        "causal_intervention_summary": summarize_interventions(intervention_rows, intervention_summary),
        "interpretation": {
            "mechanism": (
                "The trained prospective-control models expose branch-active tree/root organization, "
                "posterior matched comparison gaps, input ablation losses, physical edge ablation losses, "
                "and edge-importance concentration. These are post-training diagnostics, not selectors."
            ),
            "scramble_result": (
                "Selected-model statistic-preserving branch-alignment and projection scrambles reduce "
                "novel-class ICL sharply, supporting the branch/projection mechanism in trained models."
            ),
            "scope": (
                "The scrambles were run on four selected prospective-control trained models, with "
                "three repeats per intervention. They are mechanism checks, not an additional training sweep."
            ),
        },
    }


def write_track5(payload: dict[str, Any]) -> None:
    mech = payload["mechanism_summary"]
    ablations = payload["selected_ablation_summary"]
    interventions = payload["causal_intervention_summary"]
    mech_rows = []
    for field, stats in mech["fields"].items():
        if field in {"input_ablation_max_loss", "physical_ablation_max_loss"}:
            continue
        mech_rows.append(
            [
                field,
                stats["n"],
                fnum(stats["mean"]),
                fnum(stats["min"]),
                fnum(stats["max"]),
            ]
        )
    ablation_rows = []
    for field, stats in ablations["fields"].items():
        ablation_rows.append(
            [
                field,
                stats["n"],
                fnum(stats["mean"]),
                fnum(stats["min"]),
                fnum(stats["max"]),
            ]
        )
    intervention_rows = []
    for name, stats in interventions["interventions"].items():
        intervention_rows.append(
            [
                name,
                stats["n"],
                fnum(stats["accuracy_delta_mean"]),
                fnum(stats["accuracy_delta_min"]),
                fnum(stats["accuracy_delta_max"]),
                fnum(stats.get("margin_delta_mean")),
            ]
        )
    text = f"""# Mechanism and Causal Scramble Follow-Up Report

## Scope

This report uses the prospective exact-control trained models. It joins post-training mechanism diagnostics for all 80 trained runs with statistic-preserving causal scrambles on four selected high-performing trained models, one from each load/contrast family.

## Mechanism Diagnostics

- Mechanism rows: `{mech['n_rows']}`
- Mechanism groups: `{mech['n_groups']}`

{md_table(['metric', 'n', 'mean', 'min', 'max'], mech_rows)}

These diagnostics include branch-active-tree MI, branch-to-root MI, tree/root entropy, trained branch margin, posterior matched comparison gap, input-coupling ablation loss, physical edge ablation loss, and functional edge-importance concentration when available.

## Selected Edge Ablations

- Selected ablation rows: `{ablations['n_rows']}`
- Selected ablation groups: `{ablations['n_groups']}`

{md_table(['metric', 'n', 'mean', 'min', 'max'], ablation_rows)}

The ablation panel was run on the same four selected trained models used for causal scrambles. It covers input-coupling and physical-edge loss diagnostics without turning the mechanism follow-up into a new broad sweep.

## Causal Scrambles

- Intervention rows: `{interventions['n_rows']}`
- Selected trained runs: `{interventions['n_runs']}`
- Repeats per intervention/run: `3`

{md_table(['intervention', 'n', 'mean accuracy delta', 'min', 'max', 'mean margin delta'], intervention_rows)}

The largest drops occur under context-block/branch-alignment and projection-direction scrambles. These interventions preserve the physical graph and coarse input-mask support while disrupting trained alignment structure, so they support the mechanism claim that trained first-order CRN-ICL solutions rely on branch/projection organization.

## Interpretation

The mechanism evidence is strong for the selected trained models but remains post-training. It does not rescue the prospective pre-training tree-difference causal contrast, which was negative or inconclusive. The result says that models that train well use branch/projection structure; it does not prove that the prospective tree-difference overlap metric alone is the right pre-training causal knob.
"""
    write_text(OUT / "mechanism_and_causal_scramble_followup_report.md", text)
    write_json(OUT / "mechanism_and_causal_scramble_followup_report.json", payload)


def make_final_synthesis(
    gamma_report: dict[str, Any],
    prospective_report: dict[str, Any],
    normal_report: dict[str, Any],
    track4: dict[str, Any],
    track5: dict[str, Any],
) -> dict[str, Any]:
    fixed = gamma_report.get("interpretation", {})
    balanced_mean = lookup_contrast(prospective_report, "balanced_load", "mean_seed_novel_icl")
    balanced_best = lookup_contrast(prospective_report, "balanced_load", "best_seed_novel_icl")
    return {
        "schema": "post_gamma_repair_exact_control_synthesis.v1",
        "source_artifacts": [
            "post_phase3_markov_icl_synthesis.md",
            "gamma_toy_repair_final_report.md",
            "input_multiplicity_causal_control_report.md",
            "tree_multiplicity_causal_mask_library.md",
            "predictor_name_reconciliation.md",
            "tree_level_multiplicity_reanalysis.md",
            "repaired_gamma_existing_data_reanalysis.md",
            "prospective_tree_diff_multiplicity_causal_report.md",
            "exact_degree_exact_drel_exact_multiplicity_training_report.md",
            "expressivity_vs_trainability_exact_control_report.md",
            "mechanism_and_causal_scramble_followup_report.md",
        ],
        "answers": {
            "gamma_predicts_existing_data_better": "no",
            "prospective_tree_difference_causal_effect": "not_supported_in_first_exact_control",
            "tree_difference_survives_graph_mask_identity": "not_yet",
            "normal_fan_predicts_under_exact_controls": "weak_positive",
            "gamma_predicts_best_or_mean": "neither_in_current_data",
            "recommended_next_metric": (
                "normal-fan active-tree/tree-count variables plus grouped controls; keep tree-difference "
                "overlap as a mask-design covariate and gamma_no_bias as a diagnostic, not a selector."
            ),
            "thermodynamic_tested": "no",
        },
        "evidence": {
            "fixed_m20_tree_difference_mean_loo_r2": fixed.get("fixed_m20_mean_tree_difference_loo_r2"),
            "fixed_m20_best_gamma_mean_loo_r2": fixed.get("fixed_m20_best_gamma_mean_loo_r2"),
            "balanced_tree_diff_mean_delta": balanced_mean.get("delta_high_minus_low") if balanced_mean else None,
            "balanced_tree_diff_mean_ci95": balanced_mean.get("bootstrap_ci95") if balanced_mean else None,
            "balanced_tree_diff_best_delta": balanced_best.get("delta_high_minus_low") if balanced_best else None,
            "balanced_tree_diff_best_ci95": balanced_best.get("bootstrap_ci95") if balanced_best else None,
            "normal_fan_mean_best_model": best_model(normal_report, "mean_seed_novel_icl"),
            "normal_fan_best_best_model": best_model(normal_report, "best_seed_novel_icl"),
            "normal_fan_gamma_exact_mean_r2": model_r2(normal_report, "mean_seed_novel_icl", "gamma_no_bias_exact"),
            "normal_fan_gamma_exact_best_r2": model_r2(normal_report, "best_seed_novel_icl", "gamma_no_bias_exact"),
            "causal_scramble_interventions": track5["causal_intervention_summary"]["interventions"],
        },
        "claims": {
            "expressivity": {
                "supported": [
                    "First-order tree-sum basis remains the correct scoped theory for Markov-ICL steady states.",
                    "Normal-fan active-tree/tree-count variables show weak predictive signal for best-seed ICL under exact degree/d_rel/multiplicity controls.",
                ],
                "not_supported": [
                    "Repaired gamma_no_bias is not yet a reliable expressivity selector.",
                    "No universal scalar capacity law is supported."
                ],
            },
            "trainability": {
                "supported": [
                    "Mean-seed ICL broadly tracks the weak normal-fan/tree-count signal in the exact-degree expansion."
                ],
                "not_supported": [
                    "The current variables do not explain seed variance well.",
                    "No clean expressivity-vs-trainability split has been isolated."
                ],
            },
            "mechanism": {
                "supported": [
                    "Trained selected models are sensitive to branch-alignment, context-block, projection, and decoder-root scrambles.",
                    "Post-training mechanism metrics show branch/root/tree organization and ablation-sensitive edges."
                ],
                "not_supported": [
                    "Mechanism scrambles are not pre-training causal proof of tree-difference overlap."
                ],
            },
            "causal_evidence": {
                "supported": [
                    "The prospective mask library fixed G, input count, d_rel, aggregate/load distributions, and varied tree-difference overlap."
                ],
                "weakened": [
                    "High same-root tree-difference overlap did not improve ICL in the clean balanced-load contrast."
                ],
                "open": [
                    "Whether another mask family or larger prospective design recovers the older fixed-m20 tree-difference signal."
                ],
            },
            "thermodynamic_physics": {
                "supported": [],
                "not_tested": [
                    "No reversible-support thermodynamic Fmax experiment was run in this phase."
                ],
            },
        },
    }


def write_final(payload: dict[str, Any]) -> None:
    e = payload["evidence"]
    answers = payload["answers"]
    normal_mean = e.get("normal_fan_mean_best_model") or {}
    normal_best = e.get("normal_fan_best_best_model") or {}
    text = f"""# Post-Gamma-Repair Exact-Control Synthesis

## Bottom Line

Gamma is repaired on analytic toys, but it is not yet predictive enough to use as a selector. The prospective same-root tree-difference causal control produced a useful negative/inconclusive result: high tree-difference comparison overlap did not improve novel-class ICL under the first exact balanced-load control. The exact-degree normal-fan expansion produced the strongest new pre-training signal, but it is weak rather than a universal law.

## Direct Answers

{md_table(
        ['question', 'answer'],
        [
            ['Does repaired gamma_no_bias beat current tree/mask metrics?', answers['gamma_predicts_existing_data_better']],
            ['Does prospective tree-difference overlap causally improve ICL?', answers['prospective_tree_difference_causal_effect']],
            ['Does tree-difference survive mask-family/graph identity controls?', answers['tree_difference_survives_graph_mask_identity']],
            ['Does exact-control normal-fan geometry predict ICL?', answers['normal_fan_predicts_under_exact_controls']],
            ['Does gamma predict best seed, mean seed, both, or neither?', answers['gamma_predicts_best_or_mean']],
            ['Was thermodynamics tested?', answers['thermodynamic_tested']],
        ],
    )}

## Key Evidence

- Fixed-m20 existing data: tree-difference multiplicity mean-ICL LOO `R2 = {fnum(e.get('fixed_m20_tree_difference_mean_loo_r2'))}`, best repaired gamma mean-ICL LOO `R2 = {fnum(e.get('fixed_m20_best_gamma_mean_loo_r2'))}`.
- Prospective balanced-load high-minus-low tree-difference overlap: mean ICL delta `{fnum(e.get('balanced_tree_diff_mean_delta'))}` with CI `{e.get('balanced_tree_diff_mean_ci95')}`; best ICL delta `{fnum(e.get('balanced_tree_diff_best_delta'))}` with CI `{e.get('balanced_tree_diff_best_ci95')}`.
- Exact-degree normal-fan expansion: best mean-ICL model `{normal_mean.get('model')}` with LOO `R2 = {fnum(normal_mean.get('loo_r2'))}`; best best-seed model `{normal_best.get('model')}` with LOO `R2 = {fnum(normal_best.get('loo_r2'))}`.
- Repaired gamma in the exact-degree expansion: mean ICL LOO `R2 = {fnum(e.get('normal_fan_gamma_exact_mean_r2'))}`, best ICL LOO `R2 = {fnum(e.get('normal_fan_gamma_exact_best_r2'))}`.
- Mechanism scrambles on selected trained models sharply reduced sampled novel-class accuracy, especially context-block, branch-alignment, and projection scrambles.

## Claims

### Expressivity

Supported: the first-order rooted tree-sum basis remains the scoped expressivity theory, and normal-fan active-tree/tree-count variables weakly predict best-seed ICL under exact controls. Not supported: repaired gamma_no_bias as a large-sweep selector or any universal scalar law.

### Trainability

Supported: mean-seed ICL follows the same weak normal-fan/tree-count signal. Not supported: a clean seed-variance or optimizer-reliability metric; seed-std LOO models remain negative or small.

### Mechanism

Supported: trained models that work are sensitive to branch/projection organization. Statistic-preserving scrambles collapse selected trained models, and post-training diagnostics expose branch/root/tree alignment and ablation-sensitive edges. This is mechanism evidence after training, not pre-training causal proof of a single overlap metric.

### Causal Evidence

The first prospective tree-difference exact-control test weakens the strong interpretation of the older fixed-m20 reanalysis. Tree-difference overlap remains a useful mask-design covariate, but it did not produce the expected positive causal effect in the balanced-load prospective contrast.

### Thermodynamic Physics

No thermodynamic Fmax claim was tested. That remains untested until a reversible-support Markov parameterization is built and validated.

## Next Metric

Use normal-fan active-tree/tree-count variables plus grouped controls for the next larger exact-control experiment. Keep normalized tree-difference overlap as a controlled mask-design variable, and compute repaired gamma_no_bias as a diagnostic until it proves predictive beyond existing metrics.
"""
    write_text(OUT / "post_gamma_repair_exact_control_synthesis.md", text)
    write_json(OUT / "post_gamma_repair_exact_control_synthesis.json", payload)


def write_track3_aliases() -> None:
    src_md = OUT / "exact_degree_exact_drel_exact_multiplicity_normal_fan_training_report.md"
    src_json = OUT / "exact_degree_exact_drel_exact_multiplicity_normal_fan_training_report.json"
    dst_md = OUT / "exact_degree_exact_drel_exact_multiplicity_training_report.md"
    dst_json = OUT / "exact_degree_exact_drel_exact_multiplicity_training_report.json"
    if src_md.exists():
        shutil.copyfile(src_md, dst_md)
    if src_json.exists():
        shutil.copyfile(src_json, dst_json)


def main() -> None:
    gamma_report = load_json(OUT / "repaired_gamma_existing_data_reanalysis.json")
    prospective_report = load_json(OUT / "prospective_tree_diff_multiplicity_causal_report.json")
    normal_report = load_json(OUT / "exact_degree_exact_drel_exact_multiplicity_normal_fan_training_report.json")
    mechanism_rows = load_csv(OUT / "prospective_tree_diff_multiplicity_mechanism_results.csv")
    selected_ablation_rows = load_csv(
        OUT / "prospective_tree_diff_multiplicity_selected_ablation_mechanism_results.csv"
    )
    intervention_rows = load_csv(OUT / "prospective_tree_diff_multiplicity_causal_interventions.csv")
    intervention_summary = load_json(OUT / "prospective_tree_diff_multiplicity_causal_interventions.json")

    write_track3_aliases()

    track4 = make_track4(gamma_report, prospective_report, normal_report)
    write_track4(track4)

    track5 = make_track5(mechanism_rows, selected_ablation_rows, intervention_rows, intervention_summary)
    write_track5(track5)

    final = make_final_synthesis(gamma_report, prospective_report, normal_report, track4, track5)
    write_final(final)

    print(f"Wrote final reports under {OUT}")


if __name__ == "__main__":
    main()
