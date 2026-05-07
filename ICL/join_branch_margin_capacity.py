"""Join branch-margin capacity probes onto topology result rows.

``collect_branch_margin_capacity.py`` writes one pre-training capacity row per
selected topology or input mask.  Training sweeps write one row per train seed.
This utility left-joins the capacity rows onto each training row so the same
regression and clustered-inference tooling can test branch-margin capacity
proxies against novel-class ICL.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, Iterable, List, Sequence


DEFAULT_SKIP_COLUMNS = {
    "topology_id",
    "topology_name",
    "family",
    "physical_topology_name",
    "mask_name",
    "input_mask_name",
    "edge_json",
    "input_mask_json",
}
NONINFORMATIVE_KEYS = {"", "full", "none", "None", "NA", "nan"}


def load_csv(path: str) -> List[dict]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


def candidate_values(row: dict, keys: Sequence[str]) -> List[str]:
    values = []
    for key in keys:
        value = row.get(key)
        if value in (None, ""):
            continue
        value = str(value)
        if value in NONINFORMATIVE_KEYS:
            continue
        if value not in values:
            values.append(value)
    return values


def prefixed_capacity_fields(
    capacity_rows: Sequence[dict],
    prefix: str,
    skip: Iterable[str],
) -> List[str]:
    skip_set = set(skip)
    fields = []
    seen = set()
    for row in capacity_rows:
        for key in row:
            if key in skip_set:
                continue
            field = f"{prefix}{key}"
            if field not in seen:
                seen.add(field)
                fields.append(field)
    return fields


def capacity_index(
    capacity_rows: Sequence[dict],
    capacity_keys: Sequence[str],
) -> Dict[str, dict]:
    index = {}
    duplicates = set()
    for row in capacity_rows:
        for key in candidate_values(row, capacity_keys):
            if key in duplicates:
                continue
            if key in index and index[key] is not row:
                duplicates.add(key)
                index.pop(key, None)
                continue
            index[key] = row
    index["_duplicate_key_count"] = len(duplicates)
    return index


def join_rows(
    topology_rows: Sequence[dict],
    capacity_rows: Sequence[dict],
    topology_keys: Sequence[str],
    capacity_keys: Sequence[str],
    prefix: str = "capacity_",
    skip_columns: Iterable[str] = DEFAULT_SKIP_COLUMNS,
) -> tuple[List[dict], List[str], dict]:
    """Return topology rows enriched with prefixed capacity columns."""

    index = capacity_index(capacity_rows, capacity_keys)
    duplicate_key_count = index.pop("_duplicate_key_count", 0)
    capacity_fields = prefixed_capacity_fields(capacity_rows, prefix, skip_columns)
    skip_set = set(skip_columns)

    output_rows = []
    matched = 0
    missing = 0
    for row in topology_rows:
        out = dict(row)
        keys = candidate_values(row, topology_keys)
        capacity = None
        for key in keys:
            capacity = index.get(key)
            if capacity is not None:
                break
        if capacity is None:
            missing += 1
            for field in capacity_fields:
                out.setdefault(field, "")
        else:
            matched += 1
            for column, value in capacity.items():
                if column in skip_set:
                    continue
                out[f"{prefix}{column}"] = value
            for field in capacity_fields:
                out.setdefault(field, "")
        output_rows.append(out)

    fieldnames = list(topology_rows[0].keys()) if topology_rows else []
    for field in capacity_fields:
        if field not in fieldnames:
            fieldnames.append(field)
    report = {
        "n_topology_rows": len(topology_rows),
        "n_capacity_rows": len(capacity_rows),
        "n_matched_rows": matched,
        "n_missing_rows": missing,
        "capacity_keys_indexed": len(index),
        "capacity_duplicate_keys_ignored": duplicate_key_count,
    }
    return output_rows, fieldnames, report


def parse_keys(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology_csv", required=True)
    parser.add_argument("--capacity_csv", required=True, nargs="+")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument(
        "--topology_keys",
        default="physical_topology_name,topology_name,input_mask_name,mask_name,label",
    )
    parser.add_argument(
        "--capacity_keys",
        default="physical_topology_name,topology_name,mask_name,input_mask_name,topology_id",
    )
    parser.add_argument("--prefix", default="capacity_")
    args = parser.parse_args()

    topology_rows = load_csv(args.topology_csv)
    capacity_rows = []
    for path in args.capacity_csv:
        capacity_rows.extend(load_csv(path))

    rows, fieldnames, report = join_rows(
        topology_rows,
        capacity_rows,
        topology_keys=parse_keys(args.topology_keys),
        capacity_keys=parse_keys(args.capacity_keys),
        prefix=args.prefix,
    )
    write_csv(args.output_csv, rows, fieldnames)
    print(
        "Joined branch-margin capacity: "
        f"{report['n_matched_rows']}/{report['n_topology_rows']} training rows matched "
        f"from {report['capacity_keys_indexed']} capacity keys"
    )
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
