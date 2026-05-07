"""Audit topology-sweep artifacts without mutating experiment outputs.

This script is deliberately read-only. It is meant for long-running cluster
sweeps where training, mechanism analysis, essential-mask extraction, and
retraining may finish at different times. The audit reports concrete artifact
counts so recovery can use ``--missing_only`` or guarded finalizers instead of
guessing which stage completed.
"""

import argparse
import csv
import json
import os
import sys


SOURCE_FILES = [
    "topology_results.csv",
    "topology_regression.json",
    "topology_seed_aggregates.csv",
    "topology_seed_aggregates.json",
    "mechanism_results.csv",
    "mechanism_summary.json",
]

RETRAIN_FILES = [
    "topology_results.csv",
    "topology_regression.json",
    "topology_seed_aggregates.csv",
    "topology_seed_aggregates.json",
]
REQUIRED_RETRAIN_RUN_FILES = ["results.pkl", "topology_metrics.json", "config.json"]


def parse_experiment(raw):
    if "=" in raw:
        name, root = raw.split("=", 1)
    else:
        root = raw
        name = os.path.basename(os.path.abspath(root.rstrip(os.sep)))
    return name, os.path.abspath(root)


def parse_seeds(raw):
    return [int(seed) for seed in raw.split(",") if seed.strip()]


def exists(path):
    return os.path.exists(path)


def count_files(root, filename, exclude_roots=None):
    total = 0
    if not os.path.isdir(root):
        return total
    exclude_roots = [
        os.path.abspath(path)
        for path in (exclude_roots or [])
    ]
    for current_dir, _, files in os.walk(root):
        current = os.path.abspath(current_dir)
        if any(current == excluded or current.startswith(excluded + os.sep) for excluded in exclude_roots):
            continue
        if filename in files:
            total += 1
    return total


def count_csv_rows(path, selected_only=False):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if selected_only:
        rows = [
            row
            for row in rows
            if str(row.get("selected", "1")) in {"1", "True", "true"}
        ]
    return len(rows)


def read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def count_lines(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def file_status(root, filenames):
    return {name: exists(os.path.join(root, name)) for name in filenames}


def selected_rows_only(rows):
    return [
        row
        for row in rows
        if str(row.get("selected", "1")) in {"1", "True", "true"}
    ]


def selected_topology_ids(rows):
    ids = []
    missing = []
    for idx, row in enumerate(rows):
        topology_id = row.get("topology_id")
        if topology_id:
            ids.append(topology_id)
        else:
            missing.append(str(idx))
    return ids, missing


def find_files(root, filename):
    paths = []
    if not os.path.isdir(root):
        return paths
    for current, _, files in os.walk(root):
        if filename in files:
            paths.append(os.path.join(current, filename))
    return sorted(paths)


def source_status(root, exclude_roots=None):
    return {
        "results_pkl": count_files(root, "results.pkl", exclude_roots=exclude_roots),
        "topology_metrics_json": count_files(root, "topology_metrics.json", exclude_roots=exclude_roots),
        "mechanism_metrics_json": count_files(root, "mechanism_metrics.json", exclude_roots=exclude_roots),
        "files": file_status(root, SOURCE_FILES),
    }


def validate_edge_json(path):
    payload = read_json(path)
    if not isinstance(payload, dict):
        return "not a JSON object"
    try:
        n_nodes = int(payload["n_nodes"])
    except (KeyError, TypeError, ValueError):
        return "missing integer n_nodes"
    edges = payload.get("edges")
    if not isinstance(edges, list) or not edges:
        return "missing non-empty edges list"
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2:
            return "edge entries must be length-2 lists"
        try:
            source, target = int(edge[0]), int(edge[1])
        except (TypeError, ValueError):
            return "edge endpoints must be integers"
        if not 0 <= source < n_nodes or not 0 <= target < n_nodes:
            return "edge endpoint outside n_nodes"
    return None


def validate_input_mask_json(path):
    payload = read_json(path)
    matrix = payload.get("input_mask") if isinstance(payload, dict) else payload
    if not isinstance(matrix, list) or not matrix:
        return "missing non-empty input_mask matrix"
    width = None
    for row in matrix:
        if not isinstance(row, list) or not row:
            return "input_mask rows must be non-empty lists"
        if width is None:
            width = len(row)
        elif len(row) != width:
            return "input_mask rows must be rectangular"
        for value in row:
            if value not in (0, 1, False, True):
                return "input_mask values must be binary"
    return None


def input_mask_matrix(payload):
    return payload.get("input_mask") if isinstance(payload, dict) else payload


def edge_mask_pair_status(rows):
    invalid = []
    for row in rows:
        edge_path = row.get("edge_json")
        mask_path = row.get("input_mask_json")
        if not edge_path or not mask_path:
            continue
        if not os.path.exists(edge_path) or not os.path.exists(mask_path):
            continue
        try:
            edge_payload = read_json(edge_path)
            mask_payload = read_json(mask_path)
            edges = edge_payload.get("edges") if isinstance(edge_payload, dict) else None
            matrix = input_mask_matrix(mask_payload)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            invalid.append(f"{edge_path} + {mask_path}: {exc}")
            continue
        if not isinstance(edges, list) or not isinstance(matrix, list):
            continue
        if len(matrix) != len(edges):
            invalid.append(
                f"{mask_path}: input_mask rows {len(matrix)} do not match edge count {len(edges)}"
            )
            continue
        expected_p = None
        if isinstance(mask_payload, dict) and mask_payload.get("p") not in (None, ""):
            expected_p = mask_payload.get("p")
        elif row.get("p") not in (None, ""):
            expected_p = row.get("p")
        if expected_p is not None:
            try:
                expected_p = int(expected_p)
            except (TypeError, ValueError):
                invalid.append(f"{mask_path}: invalid p={expected_p!r}")
                continue
            width = len(matrix[0]) if matrix else 0
            if width != expected_p:
                invalid.append(
                    f"{mask_path}: input_mask width {width} does not match p={expected_p}"
                )
                continue
        mask_edges = mask_payload.get("edges") if isinstance(mask_payload, dict) else None
        if mask_edges is not None and mask_edges != edges:
            invalid.append(f"{mask_path}: edge order does not match {edge_path}")
    return {
        "invalid": len(invalid),
        "invalid_examples": invalid[:5],
    }


def referenced_file_status(rows, field, validator=None):
    missing = []
    invalid = []
    present = 0
    for row in rows:
        path = row.get(field)
        if not path:
            missing.append("")
        elif not os.path.exists(path):
            missing.append(path)
        else:
            present += 1
            if validator is not None:
                try:
                    error = validator(path)
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                    error = str(exc)
                if error:
                    invalid.append(f"{path}: {error}")
    return {
        "present": present,
        "missing": len(missing),
        "missing_examples": missing[:5],
        "invalid": len(invalid),
        "invalid_examples": invalid[:5],
    }


def retrain_exact_status(retrain_root, topology_ids, seeds):
    if not topology_ids:
        return {
            "exact_path_check": False,
            "completed_exact": None,
            "missing_exact": None,
            "missing_required_run_files": None,
            "unexpected_results": None,
            "missing_examples": [],
            "missing_required_examples": [],
            "unexpected_examples": [],
        }
    expected_dirs = [
        os.path.join(retrain_root, f"{topology_id}_trainseed{seed}")
        for topology_id in topology_ids
        for seed in seeds
    ]
    expected_paths = [os.path.join(path, "results.pkl") for path in expected_dirs]
    existing = set(find_files(retrain_root, "results.pkl"))
    expected_set = set(expected_paths)
    missing = sorted(expected_set - existing)
    missing_required = sorted(
        os.path.join(run_dir, filename)
        for run_dir in expected_dirs
        for filename in REQUIRED_RETRAIN_RUN_FILES
        if not os.path.exists(os.path.join(run_dir, filename))
    )
    unexpected = sorted(existing - expected_set)
    return {
        "exact_path_check": True,
        "completed_exact": len(expected_paths) - len(missing),
        "missing_exact": len(missing),
        "missing_required_run_files": len(missing_required),
        "unexpected_results": len(unexpected),
        "missing_examples": missing[:5],
        "missing_required_examples": missing_required[:5],
        "unexpected_examples": unexpected[:5],
    }


def manifest_status(path):
    rows = read_csv_rows(path)
    if not rows:
        return {
            "exists": os.path.exists(path),
            "rows": 0,
            "completed_at_manifest_time": 0,
            "missing_at_manifest_time": 0,
        }
    completed = sum(1 for row in rows if str(row.get("completed")) == "True")
    return {
        "exists": True,
        "rows": len(rows),
        "completed_at_manifest_time": completed,
        "missing_at_manifest_time": len(rows) - completed,
    }


def essential_inputmask_status(root, directory):
    essential_root = os.path.join(root, directory)
    library_csv = os.path.join(essential_root, "library.csv")
    selected_csv = os.path.join(essential_root, "selected.csv")
    summary_json = os.path.join(essential_root, "summary.json")
    selected_rows = selected_rows_only(read_csv_rows(selected_csv))
    topology_ids, missing_topology_id = selected_topology_ids(selected_rows)
    return {
        "root": essential_root,
        "library_csv": exists(library_csv),
        "selected_csv": exists(selected_csv),
        "summary_json": exists(summary_json),
        "library_rows": count_csv_rows(library_csv),
        "selected_rows": len(selected_rows) if os.path.exists(selected_csv) else None,
        "selected_topology_ids": topology_ids,
        "missing_topology_id": len(missing_topology_id),
        "missing_topology_id_examples": missing_topology_id[:5],
        "summary": read_json(summary_json),
        "edge_json": referenced_file_status(selected_rows, "edge_json", validate_edge_json),
        "input_mask_json": referenced_file_status(
            selected_rows,
            "input_mask_json",
            validate_input_mask_json,
        ),
        "edge_mask_pair": edge_mask_pair_status(selected_rows),
    }


def retrain_status(root, directory, selected_rows, topology_ids, seeds):
    retrain_root = os.path.join(root, directory)
    expected = None if selected_rows is None else selected_rows * len(seeds)
    completed = count_files(retrain_root, "results.pkl")
    manifest = os.path.join(retrain_root, "task_manifest.csv")
    commands = os.path.join(retrain_root, "_array_meta", "commands.txt")
    status = {
        "root": retrain_root,
        "exists": os.path.isdir(retrain_root),
        "expected_results": expected,
        "completed_results": completed,
        "missing_results": None if expected is None else max(expected - completed, 0),
        "files": file_status(retrain_root, RETRAIN_FILES),
        "manifest": manifest_status(manifest),
        "array_commands": count_lines(commands),
    }
    status.update(retrain_exact_status(retrain_root, topology_ids, seeds))
    return status


def comparison_status(root, essential_directory):
    essential_root = os.path.join(root, essential_directory)
    comparison_csv = os.path.join(essential_root, "retrain_comparison.csv")
    comparison_json = os.path.join(essential_root, "retrain_comparison.json")
    return {
        "comparison_csv": exists(comparison_csv),
        "comparison_json": exists(comparison_json),
        "comparison_rows": count_csv_rows(comparison_csv),
        "summary": read_json(comparison_json),
    }


def audit_experiment(name, root, args):
    source = source_status(
        root,
        exclude_roots=[
            os.path.join(root, args.essential_directory),
            os.path.join(root, args.retrain_directory),
        ],
    )
    essential = essential_inputmask_status(root, args.essential_directory)
    retrain = retrain_status(
        root,
        args.retrain_directory,
        essential["selected_rows"],
        essential["selected_topology_ids"],
        args.seeds,
    )
    comparison = comparison_status(root, args.essential_directory)
    failures = []

    if args.require_source_results and source["results_pkl"] == 0:
        failures.append("source has no results.pkl files")
    if args.require_mechanisms:
        if source["mechanism_metrics_json"] == 0:
            failures.append("source has no mechanism_metrics.json files")
        elif source["mechanism_metrics_json"] < source["results_pkl"]:
            failures.append("source mechanism_metrics.json count is lower than results.pkl count")
    if args.require_essential_inputmask:
        if not essential["library_csv"]:
            failures.append(f"missing {args.essential_directory}/library.csv")
        if not essential["selected_csv"]:
            failures.append(f"missing {args.essential_directory}/selected.csv")
        if not essential["summary_json"]:
            failures.append(f"missing {args.essential_directory}/summary.json")
        if not essential["selected_rows"]:
            failures.append("no selected essential input masks")
        if essential["missing_topology_id"]:
            failures.append("selected essential masks missing topology_id")
        if essential["edge_json"]["missing"]:
            failures.append("selected essential masks reference missing edge_json files")
        if essential["input_mask_json"]["missing"]:
            failures.append("selected essential masks reference missing input_mask_json files")
        if essential["edge_json"]["invalid"]:
            failures.append("selected essential masks reference invalid edge_json files")
        if essential["input_mask_json"]["invalid"]:
            failures.append("selected essential masks reference invalid input_mask_json files")
        if essential["edge_mask_pair"]["invalid"]:
            failures.append("selected essential masks have edge/input-mask mismatches")
    if args.require_essential_retrains:
        if retrain["expected_results"] is None:
            failures.append("cannot infer expected retrain count without selected.csv")
        elif retrain["exact_path_check"]:
            if retrain["completed_exact"] != retrain["expected_results"]:
                failures.append(
                    "essential retrains incomplete: "
                    f"{retrain['completed_exact']}/{retrain['expected_results']}"
                )
            if retrain["unexpected_results"]:
                failures.append(
                    "essential retrains have unexpected results: "
                    f"{retrain['unexpected_results']} extra results.pkl files"
                )
            if retrain["missing_required_run_files"]:
                failures.append(
                    "essential retrains missing required run sidecars: "
                    f"{retrain['missing_required_run_files']} files"
                )
        elif retrain["completed_results"] != retrain["expected_results"]:
            failures.append(
                "essential retrains incomplete: "
                f"{retrain['completed_results']}/{retrain['expected_results']}"
            )
        missing_retrain_files = [
            filename
            for filename, present in retrain["files"].items()
            if not present
        ]
        if missing_retrain_files:
            failures.append("missing retrain summary files: " + ", ".join(missing_retrain_files))
        if not comparison["comparison_csv"] or not comparison["comparison_json"]:
            failures.append("missing essential retrain comparison files")
        elif essential["selected_rows"] is not None:
            if comparison["comparison_rows"] != essential["selected_rows"]:
                failures.append(
                    "essential retrain comparison row count mismatch: "
                    f"{comparison['comparison_rows']}/{essential['selected_rows']}"
                )
            joined = (comparison["summary"] or {}).get("n_joined")
            if joined is not None and joined != essential["selected_rows"]:
                failures.append(
                    "essential retrain comparison n_joined mismatch: "
                    f"{joined}/{essential['selected_rows']}"
                )

    return {
        "name": name,
        "root": root,
        "source": source,
        "essential_inputmask": essential,
        "essential_inputmask_retrain": retrain,
        "comparison": comparison,
        "failures": failures,
    }


def format_bool(value):
    return "yes" if value else "no"


def print_experiment(report):
    source = report["source"]
    essential = report["essential_inputmask"]
    retrain = report["essential_inputmask_retrain"]
    comparison = report["comparison"]
    print(f"{report['name']}: {report['root']}")
    print(
        "  source: "
        f"results={source['results_pkl']} "
        f"metrics={source['topology_metrics_json']} "
        f"mechanisms={source['mechanism_metrics_json']} "
        f"aggregates={format_bool(source['files']['topology_seed_aggregates.csv'])}"
    )
    print(
        "  essential_inputmask: "
        f"library={essential['library_rows']} "
        f"selected={essential['selected_rows']} "
        f"missing_topology_id={essential['missing_topology_id']} "
        f"edge_json_missing={essential['edge_json']['missing']} "
        f"edge_json_invalid={essential['edge_json']['invalid']} "
        f"input_mask_json_missing={essential['input_mask_json']['missing']} "
        f"input_mask_json_invalid={essential['input_mask_json']['invalid']} "
        f"edge_mask_pair_invalid={essential['edge_mask_pair']['invalid']}"
    )
    completed = (
        retrain["completed_exact"]
        if retrain.get("exact_path_check")
        else retrain["completed_results"]
    )
    missing = (
        retrain["missing_exact"]
        if retrain.get("exact_path_check")
        else retrain["missing_results"]
    )
    print(
        "  essential_retrain: "
        f"completed={completed} "
        f"expected={retrain['expected_results']} "
        f"missing={missing} "
        f"missing_run_files={retrain.get('missing_required_run_files')} "
        f"unexpected={retrain['unexpected_results']} "
        f"manifest_rows={retrain['manifest']['rows']} "
        f"array_commands={retrain['array_commands']}"
    )
    print(
        "  comparison: "
        f"csv={format_bool(comparison['comparison_csv'])} "
        f"json={format_bool(comparison['comparison_json'])} "
        f"rows={comparison['comparison_rows']}"
    )
    if report["failures"]:
        for failure in report["failures"]:
            print(f"  FAIL: {failure}")
    else:
        print("  status: ok")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        help="Experiment root as NAME=PATH or PATH. May be repeated.",
    )
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--essential_directory", type=str, default="essential_inputmask50")
    parser.add_argument("--retrain_directory", type=str, default="essential_inputmask50_retrain")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--require_source_results", action="store_true")
    parser.add_argument("--require_mechanisms", action="store_true")
    parser.add_argument("--require_essential_inputmask", action="store_true")
    parser.add_argument("--require_essential_retrains", action="store_true")
    args = parser.parse_args()
    args.seeds = parse_seeds(args.seeds)

    reports = [
        audit_experiment(name, root, args)
        for name, root in [parse_experiment(raw) for raw in args.experiment]
    ]
    for report in reports:
        print_experiment(report)

    payload = {"experiments": reports}
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {os.path.abspath(args.output_json)}")

    failed = any(report["failures"] for report in reports)
    if failed and args.strict:
        raise SystemExit("Topology artifact audit failed")


if __name__ == "__main__":
    main()
