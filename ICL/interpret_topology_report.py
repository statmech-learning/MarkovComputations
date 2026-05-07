"""Interpret completed topology-ICL reports as H0/H1 evidence.

The report builders produce the regression and mechanism tables. This script
turns those machine-readable JSON reports into a compact, reproducible decision
aid:

* H0: raw degree/count controls are sufficient.
* H1: topology-derived tree geometry or trained functional topology explains
  residual novel-class ICL variation.

The interpretation is deliberately conservative. Leave-one-out R2 is preferred
over in-sample R2. When LOO R2 is unavailable, the script labels the fallback
explicitly and avoids strong claims for small fitted sample counts.
"""

import argparse
import json
import math
import os


DEFAULT_MIN_N = 6
DEFAULT_DELTA = 0.05


INPUT_MASK_COMPARISONS = [
    ("run", "run_regressions", "raw_counts", ["physical_backbone", "mask_family", "masked_geometry", "mechanism"]),
    ("mask_mean", "aggregate_target_mean", "raw_counts", ["physical_backbone", "mask_family", "masked_geometry", "mechanism"]),
    ("mask_best", "aggregate_target_max", "raw_counts", ["physical_backbone", "mask_family", "masked_geometry", "mechanism"]),
    ("mask_seed_std", "aggregate_target_std", "raw_counts", ["masked_geometry", "mechanism"]),
]

RESEARCH_COMPARISONS = [
    ("run", "run_level", "edge_count", ["edge_plus_drel", "input_plus_masked_geometry", "edge_plus_mechanism", "edge_plus_projection"]),
    ("topology_mean", "aggregate_target_mean", "edge_count", ["edge_plus_drel", "input_plus_masked_geometry", "edge_plus_mechanism", "edge_plus_projection"]),
    ("topology_best", "aggregate_target_max", "edge_count", ["edge_plus_drel", "input_plus_masked_geometry", "edge_plus_mechanism", "edge_plus_projection"]),
    ("retrain_mean", "retrain_target_mean", "layout_type", ["layout_plus_input_plus_drel", "layout_plus_input_plus_masked_geometry"]),
    ("retrain_best", "retrain_target_max", "layout_type", ["layout_plus_input_plus_drel", "layout_plus_input_plus_masked_geometry"]),
]


def parse_float(value):
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def load_json(path):
    with open(path) as f:
        return json.load(f)


def detect_kind(report):
    pooled = report.get("pooled", {})
    if "run_regressions" in pooled or "aggregate_target_std" in pooled:
        return "input_mask"
    if "run_level" in pooled or "retrain_target_mean" in pooled:
        return "research"
    return "unknown"


def fit_score(fit):
    if not isinstance(fit, dict):
        return {"score": None, "metric": "missing", "n": 0, "r2": None, "loo_r2": None}
    loo = parse_float(fit.get("leave_one_out_r2"))
    r2 = parse_float(fit.get("r2"))
    if loo is not None:
        score = loo
        metric = "LOO_R2"
    else:
        score = r2
        metric = "R2"
    return {
        "score": score,
        "metric": metric if score is not None else "missing",
        "n": int(parse_float(fit.get("n")) or 0),
        "r2": r2,
        "loo_r2": loo,
    }


def compare_model_sets(pooled, comparisons, min_n, delta):
    rows = []
    for scope, section, baseline_name, candidate_names in comparisons:
        models = pooled.get(section, {})
        baseline = fit_score(models.get(baseline_name))
        for candidate_name in candidate_names:
            candidate = fit_score(models.get(candidate_name))
            if baseline["score"] is None or candidate["score"] is None:
                change = None
            else:
                change = candidate["score"] - baseline["score"]
            supported = (
                change is not None
                and candidate["n"] >= min_n
                and candidate["metric"] == "LOO_R2"
                and change >= delta
            )
            rows.append(
                {
                    "scope": scope,
                    "section": section,
                    "baseline": baseline_name,
                    "candidate": candidate_name,
                    "baseline_score": baseline["score"],
                    "candidate_score": candidate["score"],
                    "score_metric": candidate["metric"],
                    "delta": change,
                    "n": candidate["n"],
                    "supported": supported,
                }
            )
    return rows


def input_mask_count_control(report):
    summary = report.get("pooled", {}).get("run_summary", {})
    edge_values = summary.get("n_edges_values") or []
    input_values = summary.get("input_coupled_parameter_count_values") or []
    return {
        "fixed_n_edges": len(edge_values) == 1,
        "fixed_input_coupled_parameter_count": len(input_values) == 1,
        "n_edges_values": edge_values,
        "input_coupled_parameter_count_values": input_values,
    }


def essential_retention(report, kind):
    rows = []
    if kind == "input_mask":
        for experiment in report.get("experiments", []):
            comparison = experiment.get("essential_inputmask50", {}).get("comparison", {})
            rows.append(
                {
                    "experiment": experiment.get("name"),
                    "layout": "input mask",
                    "n_joined": comparison.get("n_joined"),
                    "retention_mean_mean": parse_float(comparison.get("retention_mean_mean")),
                    "retention_max_mean": parse_float(comparison.get("retention_max_mean")),
                    "retrain_mean_mean": parse_float(comparison.get("retrain_mean_mean")),
                    "retrain_max_best": parse_float(comparison.get("retrain_max_best")),
                }
            )
    elif kind == "research":
        for experiment in report.get("experiments", []):
            for layout in experiment.get("essential_input50", {}).get("layouts", []):
                comparison = layout.get("comparison", {})
                rows.append(
                    {
                        "experiment": experiment.get("name"),
                        "layout": layout.get("label"),
                        "n_joined": comparison.get("n_joined"),
                        "retention_mean_mean": parse_float(comparison.get("retention_mean_mean")),
                        "retention_max_mean": parse_float(comparison.get("retention_max_mean")),
                        "retrain_mean_mean": parse_float(comparison.get("retrain_mean_mean")),
                        "retrain_max_best": parse_float(comparison.get("retrain_max_best")),
                    }
                )
    return rows


def summarize_support(comparison_rows):
    structural_names = {
        "physical_backbone",
        "mask_family",
        "masked_geometry",
        "edge_plus_drel",
        "input_plus_masked_geometry",
        "layout_plus_input_plus_drel",
        "layout_plus_input_plus_masked_geometry",
    }
    mechanism_names = {"mechanism", "edge_plus_mechanism", "edge_plus_projection"}
    structural = [row for row in comparison_rows if row["candidate"] in structural_names and row["supported"]]
    mechanism = [row for row in comparison_rows if row["candidate"] in mechanism_names and row["supported"]]
    usable = [row for row in comparison_rows if row["delta"] is not None]
    best = sorted(usable, key=lambda row: row["delta"], reverse=True)[:5]
    if structural and mechanism:
        verdict = "strong_positive"
        interpretation = "Topology-derived predictors and trained functional diagnostics both improve over count baselines."
    elif structural:
        verdict = "structural_positive"
        interpretation = "Topology-derived predictors improve over count baselines; mechanism evidence is weaker or unavailable."
    elif mechanism:
        verdict = "functional_positive"
        interpretation = "Post-training functional diagnostics improve over count baselines; pre-training structural evidence is weaker."
    elif usable:
        verdict = "weak_or_negative"
        interpretation = "Available fitted comparisons do not show robust topology improvements over count baselines."
    else:
        verdict = "insufficient"
        interpretation = "The report does not contain enough fitted comparisons to assess H0 versus H1."
    return {
        "verdict": verdict,
        "interpretation": interpretation,
        "n_supported_structural": len(structural),
        "n_supported_mechanism": len(mechanism),
        "best_deltas": best,
    }


def build_interpretation(report, kind, min_n, delta):
    if kind == "auto":
        kind = detect_kind(report)
    if kind not in {"input_mask", "research"}:
        raise ValueError(f"Cannot interpret report kind {kind!r}")
    comparisons = INPUT_MASK_COMPARISONS if kind == "input_mask" else RESEARCH_COMPARISONS
    comparison_rows = compare_model_sets(report.get("pooled", {}), comparisons, min_n, delta)
    support = summarize_support(comparison_rows)
    payload = {
        "report_kind": kind,
        "target": report.get("target"),
        "min_n": min_n,
        "support_delta_threshold": delta,
        "count_control": input_mask_count_control(report) if kind == "input_mask" else None,
        "model_comparisons": comparison_rows,
        "support_summary": support,
        "essential_retention": essential_retention(report, kind),
    }
    return payload


def fmt(value, digits=3):
    value = parse_float(value)
    return "NA" if value is None else f"{value:.{digits}f}"


def markdown_table(rows):
    lines = [
        "| scope | candidate | baseline | metric | n | baseline | candidate | delta | supported |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scope"]),
                    str(row["candidate"]),
                    str(row["baseline"]),
                    str(row["score_metric"]),
                    str(row["n"]),
                    fmt(row["baseline_score"]),
                    fmt(row["candidate_score"]),
                    fmt(row["delta"]),
                    "yes" if row["supported"] else "no",
                ]
            )
            + " |"
        )
    return lines


def retention_table(rows):
    lines = [
        "| experiment | layout | joined | retention mean | retention max | retrain mean | retrain best |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("experiment") or "NA"),
                    str(row.get("layout") or "NA"),
                    str(row.get("n_joined") or "NA"),
                    fmt(row.get("retention_mean_mean")),
                    fmt(row.get("retention_max_mean")),
                    fmt(row.get("retrain_mean_mean")),
                    fmt(row.get("retrain_max_best")),
                ]
            )
            + " |"
        )
    return lines


def build_markdown(payload):
    support = payload["support_summary"]
    lines = [
        "# Topology-ICL Interpretation",
        "",
        f"Report kind: `{payload['report_kind']}`. Target: `{payload.get('target')}`.",
        "",
        "## Verdict",
        "",
        f"`{support['verdict']}`: {support['interpretation']}",
        "",
        f"Support threshold: candidate minus baseline >= `{payload['support_delta_threshold']}` using LOO R2 with `n >= {payload['min_n']}`.",
        "",
    ]
    count_control = payload.get("count_control")
    if count_control:
        lines.extend(
            [
                "## Count Control",
                "",
                f"Fixed physical edge count: `{count_control['fixed_n_edges']}` with values `{count_control['n_edges_values']}`.",
                f"Fixed input-coupled parameter count: `{count_control['fixed_input_coupled_parameter_count']}` with values `{count_control['input_coupled_parameter_count_values']}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Model Comparisons",
            "",
            *markdown_table(payload["model_comparisons"]),
            "",
            "## Essential Retrain Retention",
            "",
            *retention_table(payload["essential_retention"]),
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report_json", type=str, required=True)
    parser.add_argument("--report_kind", choices=["auto", "input_mask", "research"], default="auto")
    parser.add_argument("--min_n", type=int, default=DEFAULT_MIN_N)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--output_md", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    report = load_json(args.report_json)
    payload = build_interpretation(report, args.report_kind, args.min_n, args.delta)
    markdown = build_markdown(payload)
    if args.output_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_md)), exist_ok=True)
        with open(args.output_md, "w") as f:
            f.write(markdown)
            f.write("\n")
        print(f"Wrote {args.output_md}")
    else:
        print(markdown)
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
