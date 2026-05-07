"""Collect causal topology intervention reports into a flat CSV table."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Iterable, List


FIELDS = [
    "run_dir",
    "label",
    "topology_name",
    "n_samples",
    "n_repeats",
    "intervention",
    "repeat",
    "seed",
    "baseline_target_accuracy",
    "target_accuracy",
    "target_accuracy_delta",
    "baseline_target_logprob_margin_mean",
    "target_logprob_margin_mean",
    "target_logprob_margin_mean_delta",
    "baseline_branch_active_tree_mi",
    "branch_active_tree_mi",
    "branch_active_tree_mi_delta",
    "baseline_branch_active_root_mi",
    "branch_active_root_mi",
    "branch_active_root_mi_delta",
    "tree_entropy_mean",
    "tree_entropy_mean_delta",
    "root_entropy_mean",
    "root_entropy_mean_delta",
    "active_tree_matched_comparison_gap_mean",
    "active_tree_matched_comparison_gap_mean_delta",
    "posterior_matched_comparison_gap_mean",
    "posterior_matched_comparison_gap_mean_delta",
]


def finite_or_empty(value):
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def iter_reports(root: str):
    if os.path.isfile(root):
        yield root
        return
    for current, _, files in os.walk(root):
        if "causal_interventions.json" in files:
            yield os.path.join(current, "causal_interventions.json")


def load_report(path: str) -> List[dict]:
    with open(path) as handle:
        payload = json.load(handle)
    baseline = payload.get("baseline", {})
    run_dir = payload.get("run_dir") or os.path.dirname(path)
    label = os.path.basename(os.path.abspath(run_dir.rstrip(os.sep)))
    rows = []
    for item in payload.get("interventions", []):
        row = {
            "run_dir": run_dir,
            "label": label,
            "topology_name": payload.get("topology_name", ""),
            "n_samples": payload.get("n_samples", ""),
            "n_repeats": payload.get("n_repeats", ""),
            "intervention": item.get("intervention", ""),
            "repeat": item.get("repeat", ""),
            "seed": item.get("seed", ""),
        }
        for key in [
            "target_accuracy",
            "target_logprob_margin_mean",
            "branch_active_tree_mi",
            "branch_active_root_mi",
        ]:
            row[f"baseline_{key}"] = baseline.get(key)
            row[key] = item.get(key)
            row[f"{key}_delta"] = item.get(f"{key}_delta")
        for key in [
            "tree_entropy_mean",
            "root_entropy_mean",
            "active_tree_matched_comparison_gap_mean",
            "posterior_matched_comparison_gap_mean",
        ]:
            row[key] = item.get(key)
            row[f"{key}_delta"] = item.get(f"{key}_delta")
        rows.append(row)
    return rows


def summarize(rows: List[dict]) -> dict:
    by_intervention = {}
    for row in rows:
        intervention = row.get("intervention", "")
        by_intervention.setdefault(intervention, []).append(row)
    summary = {
        "n_rows": len(rows),
        "n_runs": len({row.get("run_dir") for row in rows}),
        "interventions": {},
    }
    for intervention, items in sorted(by_intervention.items()):
        deltas = []
        for item in items:
            value = item.get("target_accuracy_delta")
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                deltas.append(value)
        summary["interventions"][intervention] = {
            "n": len(items),
            "target_accuracy_delta_mean": sum(deltas) / len(deltas) if deltas else None,
            "target_accuracy_delta_min": min(deltas) if deltas else None,
            "target_accuracy_delta_max": max(deltas) if deltas else None,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    rows = []
    for path in sorted(iter_reports(args.input_root)):
        try:
            rows.extend(load_report(path))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"Skipping {path}: {exc}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: finite_or_empty(row.get(field)) for field in FIELDS} for row in rows)

    summary = summarize(rows)
    if args.output_json:
        with open(args.output_json, "w") as handle:
            json.dump(summary, handle, indent=2)
    print(f"Wrote {len(rows)} intervention rows to {args.output_csv}")
    if args.output_json:
        print(f"Wrote intervention summary to {args.output_json}")


if __name__ == "__main__":
    main()
