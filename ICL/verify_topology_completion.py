"""Final non-mutating completion gate for topology-ICL cluster artifacts.

This script is intentionally stricter than the individual collectors. It is
meant to run after guarded recovery/finalization has produced the final
Markdown/JSON report. It checks both:

1. the strict artifact audit for the supplied experiments; and
2. the final report files and JSON structure that the scientific writeup will
   actually use.
"""

import argparse
import json
import os
import subprocess
import sys


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TARGET = "test_novel_classes"


def parse_experiment(raw):
    if "=" in raw:
        name, root = raw.split("=", 1)
    else:
        root = raw
        name = os.path.basename(os.path.abspath(root.rstrip(os.sep)))
    return name, os.path.abspath(root)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def require(condition, message, failures):
    if not condition:
        failures.append(message)


def positive_count(payload, path):
    current = payload
    for key in path:
        if not isinstance(current, dict):
            return 0
        current = current.get(key)
    try:
        return int(current)
    except (TypeError, ValueError):
        return 0


def source_counts_have_no_unknown(report, key):
    counts = report.get("pooled", {}).get(key, {})
    return isinstance(counts, dict) and counts and not counts.get("unknown")


def check_common_report_fields(report, experiments, target, failures):
    require(report.get("target") == target, f"report target is {report.get('target')!r}, expected {target!r}", failures)
    expected_names = [name for name, _ in experiments]
    report_experiments = report.get("experiments", [])
    report_names = [item.get("name") for item in report_experiments if isinstance(item, dict)]
    require(report_names == expected_names, f"report experiments {report_names!r} do not match {expected_names!r}", failures)
    return report_experiments


def check_provenance_counts(report, keys, failures, allow_unknown_provenance=False):
    for key in keys:
        if allow_unknown_provenance:
            counts = report.get("pooled", {}).get(key)
            require(isinstance(counts, dict) and bool(counts), f"missing pooled {key}", failures)
        else:
            require(source_counts_have_no_unknown(report, key), f"missing or unknown provenance in pooled {key}", failures)


def verify_input_mask_report(report, markdown, experiments, target, allow_unknown_provenance=False):
    failures = []
    for section in [
        "Input-Mask Topology-ICL Report",
        "Pooled Fixed-Input-Count Regressions",
        "Essential Input-Mask Retraining",
        "Common branch-rank source counts",
        "Input-overlap source counts",
    ]:
        require(section in markdown, f"report Markdown missing section/text: {section}", failures)

    report_experiments = check_common_report_fields(report, experiments, target, failures)

    pooled = report.get("pooled", {})
    require(positive_count(pooled, ["run_summary", "n"]) > 0, "pooled report has no run rows", failures)
    require(
        positive_count(pooled, ["aggregate_summary", "n"]) > 0,
        "pooled report has no aggregate groups",
        failures,
    )
    check_provenance_counts(
        report,
        [
            "run_common_branch_source_counts",
            "aggregate_common_branch_source_counts",
            "run_input_overlap_source_counts",
            "aggregate_input_overlap_source_counts",
        ],
        failures,
        allow_unknown_provenance=allow_unknown_provenance,
    )

    for item in report_experiments:
        name = item.get("name")
        require(positive_count(item, ["run_summary", "n"]) > 0, f"{name}: no run rows", failures)
        require(positive_count(item, ["aggregate_summary", "n"]) > 0, f"{name}: no aggregate groups", failures)
        essential = item.get("essential_inputmask50", {})
        selected = positive_count(essential, ["selected_summary", "n_selected"])
        joined = positive_count(essential, ["comparison", "n_joined"])
        retrained = positive_count(essential, ["retrain_aggregate", "n"])
        require(selected > 0, f"{name}: no selected essential input masks", failures)
        require(joined == selected, f"{name}: retrain comparison joined {joined}/{selected}", failures)
        require(retrained == selected, f"{name}: retrain aggregate groups {retrained}/{selected}", failures)

    return failures


def verify_research_report(report, markdown, experiments, target, allow_unknown_provenance=False):
    failures = []
    for section in [
        "Topology-ICL Progress Report",
        "Pooled Fixed-Edge Regime Analysis",
        "Common branch-rank source counts",
        "Input-overlap source counts",
        "Essential Motif Retraining",
    ]:
        require(section in markdown, f"report Markdown missing section/text: {section}", failures)

    report_experiments = check_common_report_fields(report, experiments, target, failures)
    pooled = report.get("pooled", {})
    require(positive_count(pooled, ["run_rows"]) > 0, "pooled report has no run rows", failures)
    require(positive_count(pooled, ["aggregate_groups"]) > 0, "pooled report has no aggregate groups", failures)
    require(positive_count(pooled, ["retrain_groups"]) > 0, "pooled report has no retrain groups", failures)
    check_provenance_counts(
        report,
        [
            "run_common_branch_source_counts",
            "aggregate_common_branch_source_counts",
            "run_input_overlap_source_counts",
            "aggregate_input_overlap_source_counts",
        ],
        failures,
        allow_unknown_provenance=allow_unknown_provenance,
    )

    for item in report_experiments:
        name = item.get("name")
        require(positive_count(item, ["run_summary", "n_runs"]) > 0, f"{name}: no run rows", failures)
        require(
            positive_count(item, ["aggregate_summary", "n_topology_groups"]) > 0,
            f"{name}: no aggregate groups",
            failures,
        )
        layouts = item.get("essential_input50", {}).get("layouts", [])
        require(layouts, f"{name}: no essential retrain layouts", failures)
        for layout in layouts:
            label = layout.get("label") or layout.get("source_dir") or "layout"
            selected = positive_count(layout, ["comparison", "n_joined"])
            retrained = positive_count(layout, ["retrain_aggregate", "n_topology_groups"])
            require(selected > 0, f"{name} {label}: no joined retrained motifs", failures)
            require(retrained == selected, f"{name} {label}: retrain aggregate groups {retrained}/{selected}", failures)

    return failures


def verify_report(
    report_md,
    report_json,
    experiments,
    target,
    report_kind="input_mask",
    allow_unknown_provenance=False,
):
    failures = []
    require(os.path.exists(report_md), f"missing report Markdown: {report_md}", failures)
    require(os.path.exists(report_json), f"missing report JSON: {report_json}", failures)
    if not os.path.exists(report_md) or not os.path.exists(report_json):
        return failures

    require(os.path.getsize(report_md) > 0, f"empty report Markdown: {report_md}", failures)
    require(os.path.getsize(report_json) > 0, f"empty report JSON: {report_json}", failures)
    with open(report_md) as f:
        markdown = f.read()
    report = load_json(report_json)
    if report_kind == "input_mask":
        failures.extend(
            verify_input_mask_report(
                report,
                markdown,
                experiments,
                target,
                allow_unknown_provenance=allow_unknown_provenance,
            )
        )
    elif report_kind == "research":
        failures.extend(
            verify_research_report(
                report,
                markdown,
                experiments,
                target,
                allow_unknown_provenance=allow_unknown_provenance,
            )
        )
    else:
        failures.append(f"unknown report kind: {report_kind}")
    return failures


def strict_audit_command(experiments, seeds, essential_kind="inputmask"):
    parts = [
        sys.executable,
        "audit_topology_artifacts.py",
    ]
    for name, root in experiments:
        parts.extend(["--experiment", f"{name}={root}"])
    if essential_kind == "physical":
        parts.extend(
            [
                "--essential_directory",
                "essential_input50",
                "--retrain_directory",
                "essential_input50_retrain",
                "--essential_kind",
                "physical",
            ]
        )
    parts.extend(
        [
            "--seeds",
            seeds,
            "--require_source_results",
            "--require_mechanisms",
            "--require_essential",
            "--require_essential_retrains",
            "--strict",
        ]
    )
    return parts


def run_strict_audit(experiments, seeds, essential_kind="inputmask"):
    parts = strict_audit_command(experiments, seeds, essential_kind=essential_kind)
    print(" ".join(parts))
    return subprocess.run(parts, cwd=THIS_DIR, text=True, capture_output=True, check=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        help="Experiment root as NAME=PATH or PATH. May be repeated.",
    )
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--report_md", type=str, required=True)
    parser.add_argument("--report_json", type=str, required=True)
    parser.add_argument(
        "--report_kind",
        choices=["input_mask", "research"],
        default="input_mask",
        help="Expected report schema to verify.",
    )
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
    parser.add_argument("--allow_unknown_provenance", action="store_true")
    parser.add_argument(
        "--skip_audit",
        action="store_true",
        help="Only validate report files. Intended for unit tests and diagnostics.",
    )
    args = parser.parse_args()

    experiments = [parse_experiment(raw) for raw in args.experiment]
    failures = []
    if not args.skip_audit:
        audit_kinds = ["inputmask"]
        if args.report_kind == "research":
            audit_kinds.append("physical")
        for kind in audit_kinds:
            result = run_strict_audit(experiments, args.seeds, essential_kind=kind)
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr)
            if result.returncode != 0:
                failures.append(f"strict topology artifact audit failed for {kind}")

    failures.extend(
        verify_report(
            args.report_md,
            args.report_json,
            experiments,
            args.target,
            report_kind=args.report_kind,
            allow_unknown_provenance=args.allow_unknown_provenance,
        )
    )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit("Topology completion verification failed")

    print("Topology completion verification passed")


if __name__ == "__main__":
    main()
