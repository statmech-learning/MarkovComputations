"""Refresh selected sections of an existing next-phase evidence report.

``make_next_phase_evidence_report.py`` builds a report from all source
artifacts.  During long cluster follow-up work, some older intermediate JSONs
may not be present in the active worktree, while newly completed hard-pilot
mechanism, causal, capacity, or clustered artifacts need to be folded into the
already tracked report.  This script updates only the labeled sections supplied
on the command line, preserves the rest of the report JSON, and re-renders the
Markdown with the canonical next-phase report builder.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Callable, Iterable, List

from make_next_phase_evidence_report import (
    build_markdown,
    capacity_summary,
    causal_summary,
    clustered_summary,
    expanded_status,
    load_json,
    matched_motif_summary,
    parse_labeled_path,
)


def replace_by_label(existing: List[dict], replacements: Iterable[dict]) -> List[dict]:
    """Replace matching labeled entries and append new labels at the end."""

    replacement_by_label = {
        item.get("label"): item
        for item in replacements
        if isinstance(item, dict) and item.get("label")
    }
    seen = set()
    out = []
    for item in existing:
        label = item.get("label") if isinstance(item, dict) else None
        if label in replacement_by_label:
            out.append(replacement_by_label[label])
            seen.add(label)
        else:
            out.append(item)
    for label, item in replacement_by_label.items():
        if label not in seen:
            out.append(item)
    return out


def load_labeled_update(raw: str, summarizer: Callable[[dict], dict]) -> dict:
    label, path = parse_labeled_path(raw)
    return summarizer({"label": label, "path": path, "payload": load_json(path)})


def labeled_updates(items: List[str], summarizer: Callable[[dict], dict]) -> List[dict]:
    return [load_labeled_update(item, summarizer) for item in items]


def apply_updates(report: dict, args) -> dict:
    report = dict(report)
    if args.clustered_json:
        report["clustered_inference"] = replace_by_label(
            report.get("clustered_inference", []),
            labeled_updates(args.clustered_json, clustered_summary),
        )
    if args.causal_json:
        report["causal_interventions"] = replace_by_label(
            report.get("causal_interventions", []),
            labeled_updates(args.causal_json, causal_summary),
        )
    if args.branch_capacity_json:
        report["branch_margin_capacity"] = replace_by_label(
            report.get("branch_margin_capacity", []),
            labeled_updates(args.branch_capacity_json, capacity_summary),
        )
    if args.matched_motif_json:
        report["matched_motif_controls"] = replace_by_label(
            report.get("matched_motif_controls", []),
            labeled_updates(args.matched_motif_json, matched_motif_summary),
        )
    if args.expanded_root:
        expanded_updates = [
            item
            for item in expanded_status(args.expanded_root)
            if (
                item.get("results_pkl_count", 0) > 0
                or item.get("mechanism_count", 0) > 0
                or item.get("causal_count", 0) > 0
                or args.allow_zero_expanded_updates
            )
        ]
        report["expanded_pilot_status"] = replace_by_label(
            report.get("expanded_pilot_status", []),
            expanded_updates,
        )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report_json", required=True, help="Existing next-phase report JSON to update.")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--clustered_json", action="append", default=[])
    parser.add_argument("--causal_json", action="append", default=[])
    parser.add_argument("--branch_capacity_json", action="append", default=[])
    parser.add_argument("--matched_motif_json", action="append", default=[])
    parser.add_argument("--expanded_root", action="append", default=[])
    parser.add_argument(
        "--allow_zero_expanded_updates",
        action="store_true",
        help=(
            "Overwrite expanded pilot status rows even when the supplied root "
            "has no per-run result, mechanism, or causal files. By default, "
            "all-zero rows are ignored so a source-light checkout does not "
            "downgrade an existing report."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    report = apply_updates(load_json(args.report_json), args)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_md)), exist_ok=True)
    with open(args.output_json, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(args.output_md, "w") as handle:
        handle.write(build_markdown(report))
    print(f"Refreshed next-phase report JSON: {args.output_json}")
    print(f"Refreshed next-phase report Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
