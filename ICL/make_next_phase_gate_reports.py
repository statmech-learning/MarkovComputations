"""Write gate-stopped Phase 4-6 reports and final synthesis.

The next-phase goal requires Steps 1-3 before expensive new training.  When the
gamma toy gate fails, later experimental reports should exist as explicit
blocked artifacts rather than as silent omissions or as broad sweeps run out of
order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "ICL" / "results" / "next_phase_stats"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def model_lookup(tree_report: Mapping[str, Any], dataset: str, outcome: str) -> dict[str, Any]:
    for analysis in tree_report.get("analyses", []):
        if analysis.get("name") != dataset:
            continue
        return {
            row.get("model"): row
            for row in analysis.get("models", [])
            if row.get("outcome") == outcome
        }
    return {}


def write_md(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines.extend([f"## {heading}", "", body.strip(), ""])
    path.write_text("\n".join(lines))


def build_reports() -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reconciliation = read_json(OUT_DIR / "predictor_name_reconciliation.json")
    tree = read_json(OUT_DIR / "tree_level_multiplicity_reanalysis.json")
    gamma = read_json(OUT_DIR / "gamma_toy_validation_report.json")
    phase3_pass = bool(gamma.get("phase_3_gate_passed"))

    fixed_mean = model_lookup(tree, "fixed_m20_masks_cluster_topology", "mean_novel_icl")
    hard_margin = model_lookup(tree, "hard_full_mask_local", "trained_branch_margin")
    hard_failure = model_lookup(tree, "hard_full_mask_local", "branch_failure_percent")

    written: list[Path] = []

    phase4 = {
        "schema": "input_multiplicity_causal_control_report.v1",
        "status": "not_launched_blocked_by_phase_3_gate" if not phase3_pass else "ready_for_targeted_training",
        "gate_dependency": "gamma_toy_validation_report.phase_3_gate_passed",
        "phase_3_gate_passed": phase3_pass,
        "no_new_training_launched": True,
        "primary_question": "Does comparison-coordinate tree/difference overlap causally predict novel-class ICL at fixed count and d_rel?",
        "existing_noncausal_screen": {
            "fixed_m20_mean_novel_icl": fixed_mean,
            "hard_full_mask_trained_branch_margin": hard_margin,
            "hard_full_mask_branch_failures": hard_failure,
        },
        "blocked_next_action": [
            "repair gamma*_ICL toy validation or explicitly decouple Phase 4 from gamma gating",
            "construct matched masks on one physical G with fixed input count, d_rel, and M_mean",
            "train grouped seeds only after the gate is cleared",
        ],
        "source_artifacts": [
            "ICL/results/next_phase_stats/tree_level_multiplicity_reanalysis.json",
            "ICL/results/next_phase_stats/gamma_toy_validation_report.json",
            "ICL/results/next_phase_stats/input_multiplicity_control_report.json",
        ],
    }
    p = OUT_DIR / "input_multiplicity_causal_control_report.json"
    write_json(p, phase4)
    written.append(p)
    p = OUT_DIR / "input_multiplicity_causal_control_report.md"
    write_md(
        p,
        "Input Multiplicity Causal Control Report",
        [
            ("Status", f"`{phase4['status']}`. No new training was launched."),
            (
                "Reason",
                "The Phase 3 gamma toy validation gate did not pass, and the goal explicitly forbids broad downstream experiments before Steps 1-3 pass.",
            ),
            (
                "Existing Evidence",
                "The Phase 2 fixed-m20 screen is noncausal: edge-level multiplicity had LOO R2 "
                f"`{fixed_mean.get('edge_level_multiplicity', {}).get('loo_r2')}`, tree-level had "
                f"`{fixed_mean.get('tree_level_multiplicity', {}).get('loo_r2')}`, and tree-difference had "
                f"`{fixed_mean.get('tree_difference_multiplicity', {}).get('loo_r2')}` for mean novel-class ICL.",
            ),
            (
                "Required Next Action",
                "After the gate is cleared, construct matched masks on an exact physical graph with fixed input count, d_rel, and M_mean, then vary tree/difference overlap and load imbalance with grouped seeds.",
            ),
        ],
    )
    written.append(p)

    phase5 = {
        "schema": "exact_degree_exact_drel_exact_multiplicity_normal_fan_report.v1",
        "status": "not_launched_blocked_by_phase_3_gate" if not phase3_pass else "ready_for_exact_control_scaleup",
        "phase_3_gate_passed": phase3_pass,
        "no_new_training_launched": True,
        "primary_question": "After controlling count, degree sequence, d_rel, and multiplicity, does tree-polytope branch geometry still matter?",
        "existing_related_pilot": "ICL/results/next_phase_stats/exact_degree_multiplicity_normal_fan_report.json",
        "missing_controls_before_claim": [
            "exact d_rel matching across all trained groups",
            "exact multiplicity distribution matching",
            "held-out-family or held-out-base-graph checks",
            "gamma*_ICL lower-tail validation",
        ],
    }
    p = OUT_DIR / "exact_degree_exact_drel_exact_multiplicity_normal_fan_report.json"
    write_json(p, phase5)
    written.append(p)
    p = OUT_DIR / "exact_degree_exact_drel_exact_multiplicity_normal_fan_report.md"
    write_md(
        p,
        "Exact-Degree Exact-drel Exact-Multiplicity Normal-Fan Report",
        [
            ("Status", f"`{phase5['status']}`. No new scale-up sweep was launched."),
            (
                "Reason",
                "The existing normal-fan artifact is a related pilot, but this requested exact-degree / exact-d_rel / exact-multiplicity falsification test remains gated.",
            ),
            (
                "Required Controls",
                "The next run must fix in-degree sequence, out-degree sequence, input count, d_rel, and multiplicity distribution, then vary active-tree count, branch-tree NMI, tree-polytope support geometry, branch sharpness, and tree-level multiplicity.",
            ),
        ],
    )
    written.append(p)

    phase6 = {
        "schema": "thermodynamic_fmax_experiment_report.v1",
        "status": "not_launched_blocked_by_phase_3_gate_and_missing_thermodynamic_parameterization",
        "phase_3_gate_passed": phase3_pass,
        "no_new_training_launched": True,
        "existing_related_audit": "ICL/results/next_phase_stats/thermodynamic_force_budget_report.json",
        "required_before_sweep": [
            "bidirected reversible support",
            "W_ij = exp(E_j - B_ij + F_ij/2 + input_drive)",
            "B_ij = B_ji and F_ij = -F_ji",
            "|F_ij| <= F_max",
            "detailed-balance behavior at F_max=0",
            "stable steady-state solve",
        ],
        "thermodynamic_claim_supported": False,
    }
    p = OUT_DIR / "thermodynamic_fmax_experiment_report.json"
    write_json(p, phase6)
    written.append(p)
    p = OUT_DIR / "thermodynamic_fmax_experiment_report.md"
    write_md(
        p,
        "Thermodynamic Fmax Experiment Report",
        [
            ("Status", f"`{phase6['status']}`. No Fmax sweep was launched."),
            (
                "Reason",
                "Existing first-order CRN runs use arbitrary directed exponential rates and are not valid evidence for thermodynamic force-budget claims.",
            ),
            (
                "Required Model",
                "`W_ij = exp(E_j - B_ij + F_ij/2 + input_drive)` with reversible support, symmetric barriers, antisymmetric forces, and an explicit `F_max` bound.",
            ),
        ],
    )
    written.append(p)

    synthesis = {
        "schema": "final_markov_icl_next_phase_synthesis.v1",
        "status": "gated_after_phase_3",
        "phase_1_supported": bool(reconciliation.get("discrepancies")),
        "phase_2_supported": bool(tree.get("analyses")),
        "phase_3_passed": phase3_pass,
        "claims": {
            "expressivity": {
                "supported": [
                    "The ambiguous fixed-m20 tree_geometry numbers are different regressions and now have separate names.",
                    "Tree-level and tree-difference multiplicity metrics are implemented in the rooted tree-sum basis.",
                    "Existing fixed-m20 data show tree/difference multiplicity can predict mean novel-class ICL better than edge-level multiplicity in grouped LOO screens.",
                ],
                "not_supported": [
                    "A validated universal scalar gamma*_ICL law.",
                    "Large gamma*-based topology selection.",
                ],
            },
            "trainability": {
                "supported": [
                    "Mean, best-seed, and seed-standard-deviation outcomes are separated in the reports.",
                    "Hard full-mask data include trained branch-margin and branch-failure outcomes for grouped screens.",
                ],
                "not_supported": [
                    "A claim that gamma*_ICL predicts best-seed ICL; toy validation failed first.",
                ],
            },
            "mechanism": {
                "supported": [
                    "Mechanism metrics are treated as post-training outcomes/descriptors, not pre-training predictors.",
                ],
                "not_supported": [
                    "Motif uniqueness or a universal mechanism law.",
                ],
            },
            "causal_interventions": {
                "supported": [
                    "A targeted matched-mask causal design is specified.",
                ],
                "not_supported": [
                    "The requested causal control has not been launched because the Phase 3 gate failed.",
                ],
            },
            "thermodynamics": {
                "supported": [
                    "Existing arbitrary directed-rate runs are explicitly excluded from thermodynamic force-budget claims.",
                ],
                "not_supported": [
                    "Any F_max sweep result or entropy-production claim.",
                ],
            },
        },
        "deliverables": {
            "phase_1": [
                "ICL/results/next_phase_stats/predictor_name_reconciliation.md",
                "ICL/results/next_phase_stats/predictor_name_reconciliation.json",
            ],
            "phase_2": [
                "ICL/tree_level_multiplicity_metrics.py",
                "ICL/results/next_phase_stats/tree_level_multiplicity_reanalysis.md",
                "ICL/results/next_phase_stats/tree_level_multiplicity_reanalysis.json",
            ],
            "phase_3": [
                "ICL/results/next_phase_stats/gamma_toy_validation_report.md",
                "ICL/results/next_phase_stats/gamma_toy_validation_report.json",
            ],
            "gated_phase_4_to_6": [
                "ICL/results/next_phase_stats/input_multiplicity_causal_control_report.md",
                "ICL/results/next_phase_stats/input_multiplicity_causal_control_report.json",
                "ICL/results/next_phase_stats/exact_degree_exact_drel_exact_multiplicity_normal_fan_report.md",
                "ICL/results/next_phase_stats/exact_degree_exact_drel_exact_multiplicity_normal_fan_report.json",
                "ICL/results/next_phase_stats/thermodynamic_fmax_experiment_report.md",
                "ICL/results/next_phase_stats/thermodynamic_fmax_experiment_report.json",
            ],
        },
    }
    p = OUT_DIR / "final_markov_icl_next_phase_synthesis.json"
    write_json(p, synthesis)
    written.append(p)
    p = OUT_DIR / "final_markov_icl_next_phase_synthesis.md"
    write_md(
        p,
        "Final Markov ICL Next-Phase Synthesis",
        [
            ("Status", "`gated_after_phase_3`. The required later sweeps were not launched."),
            (
                "Expressivity",
                "Supported: predictor names are reconciled, and tree/difference multiplicity is implemented in the tree-sum basis. Not supported: a universal scalar law or gamma*-based topology selection.",
            ),
            (
                "Trainability",
                "Supported: mean, best-seed, and seed-std outcomes remain separated. Not supported: gamma*_ICL as a best-seed predictor.",
            ),
            (
                "Mechanism",
                "Supported: post-training branch margin and branch failure are treated as outcomes/descriptors. Not supported: motif uniqueness.",
            ),
            (
                "Causal Interventions",
                "Supported: the matched-mask design is specified and Phase 2 provides noncausal screening. Not supported: a completed causal training control.",
            ),
            (
                "Thermodynamics",
                "Supported: arbitrary directed-rate runs are excluded from thermodynamic claims. Not supported: any valid Fmax result.",
            ),
        ],
    )
    written.append(p)
    return written


def main() -> None:
    for path in build_reports():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
