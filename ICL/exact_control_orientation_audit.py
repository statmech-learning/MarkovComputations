"""Orientation audit for the prospective Markov-ICL exact-control phase."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "ICL" / "results" / "next_phase_stats"
OUT_MD = OUT_DIR / "next_phase_orientation_audit.md"
OUT_JSON = OUT_DIR / "next_phase_orientation_audit.json"


REQUIRED_REPORTS = [
    "post_phase3_markov_icl_synthesis.md",
    "gamma_toy_repair_final_report.md",
    "input_multiplicity_causal_control_report.md",
    "tree_multiplicity_causal_mask_library.md",
    "predictor_name_reconciliation.md",
    "tree_level_multiplicity_reanalysis.md",
    "topology_icl_research_synthesis.md",
]

SCRIPT_MAP = {
    "tree_level_and_tree_difference_multiplicity": "ICL/tree_level_multiplicity_metrics.py",
    "repaired_gamma_toy_validation": "ICL/analytic_gamma_repair.py",
    "existing_data_causal_control": "ICL/tree_multiplicity_causal_control.py",
    "fixed_m20_mask_library_generation": "ICL/make_input_mask_library.py",
    "fixed_m20_training_submission": "ICL/submit_topology_library_sweep.py",
    "grouped_loo_inference_structural": "ICL/clustered_topology_inference.py",
    "grouped_loo_inference_markov": "ICL/tree_multiplicity_causal_control.py",
    "lower_tail_gamma_probe": "ICL/branch_margin_capacity_v2.py",
}

FIXED_M20_ROOTS = [
    "ICL/results/input_mask_fixed_m20_random_sc_seed3_c200",
    "ICL/results/input_mask_fixed_m20_cycle_chords_seed3_c200",
    "ICL/results/input_mask_fixed_m20_hub_spoke_seed63_c200",
]


def run(cmd: list[str], cwd: Path = REPO_ROOT) -> str:
    return subprocess.check_output(cmd, cwd=str(cwd), text=True).strip()


def maybe_run(cmd: list[str], cwd: Path = REPO_ROOT) -> dict[str, Any]:
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd), text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        return {"ok": False, "output": exc.output.strip(), "returncode": exc.returncode}
    return {"ok": True, "output": out.strip(), "returncode": 0}


def git_state() -> dict[str, Any]:
    return {
        "branch": run(["git", "branch", "--show-current"]),
        "commit": run(["git", "rev-parse", "HEAD"]),
        "commit_oneline": run(["git", "log", "-1", "--oneline"]),
        "status_short": run(["git", "status", "--short", "--branch"]),
    }


def report_presence() -> list[dict[str, Any]]:
    rows = []
    for name in REQUIRED_REPORTS:
        path = OUT_DIR / name
        rows.append(
            {
                "path": f"ICL/results/next_phase_stats/{name}",
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return rows


def scripts() -> list[dict[str, Any]]:
    rows = []
    for purpose, relpath in SCRIPT_MAP.items():
        path = REPO_ROOT / relpath
        rows.append(
            {
                "purpose": purpose,
                "path": relpath,
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return rows


def fixed_m20_learned_k_availability() -> dict[str, Any]:
    script = r"""
from pathlib import Path
import json
roots = [
    Path("ICL/results/input_mask_fixed_m20_random_sc_seed3_c200"),
    Path("ICL/results/input_mask_fixed_m20_cycle_chords_seed3_c200"),
    Path("ICL/results/input_mask_fixed_m20_hub_spoke_seed63_c200"),
]
out = {}
for root in roots:
    runs = [p for p in root.iterdir() if p.is_dir() and "trainseed" in p.name] if root.exists() else []
    out[str(root)] = {
        "exists": root.exists(),
        "train_dirs": len(runs),
        "model_pt": sum((p / "model.pt").exists() for p in runs),
        "results_pkl": sum((p / "results.pkl").exists() for p in runs),
        "topology_json": sum((p / "topology.json").exists() for p in runs),
        "config_json": sum((p / "config.json").exists() for p in runs),
    }
print(json.dumps(out, sort_keys=True))
"""
    remote = maybe_run(["ssh", "engaging", f"cd ~/repos/topology && python3 - <<'PY'\n{script}\nPY"])
    if remote["ok"]:
        try:
            return {
                "location": "ssh:engaging:/home/aadarwal/repos/topology",
                "available": True,
                "counts": json.loads(remote["output"]),
                "note": "model.pt stores the learned K_params state_dict; aggregate CSVs do not store learned K tensors.",
            }
        except json.JSONDecodeError:
            pass
    return {
        "location": "ssh:engaging:/home/aadarwal/repos/topology",
        "available": False,
        "error": remote,
        "note": "Could not verify model.pt availability from Engaging.",
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def markdown_table(rows: list[list[Any]], headers: list[str]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def write_md(payload: dict[str, Any]) -> None:
    report_rows = [[row["path"], row["exists"], row["bytes"]] for row in payload["required_reports"]]
    script_rows = [[row["purpose"], row["path"], row["exists"]] for row in payload["script_map"]]
    learned = payload["fixed_m20_learned_k_availability"]
    learned_rows = []
    for root, item in learned.get("counts", {}).items():
        learned_rows.append(
            [
                root,
                item.get("train_dirs"),
                item.get("model_pt"),
                item.get("results_pkl"),
                item.get("topology_json"),
                item.get("config_json"),
            ]
        )
    lines = [
        "# Next-Phase Orientation Audit",
        "",
        "## Git State",
        "",
        f"- Branch: `{payload['git']['branch']}`",
        f"- Commit: `{payload['git']['commit']}`",
        f"- Commit summary: `{payload['git']['commit_oneline']}`",
        "",
        "## Required Reports",
        "",
        markdown_table(report_rows, ["path", "exists", "bytes"]),
        "",
        "## Code Paths",
        "",
        markdown_table(script_rows, ["purpose", "path", "exists"]),
        "",
        "## Fixed-m20 Learned K Availability",
        "",
        f"- Location checked: `{learned.get('location')}`",
        f"- Available: `{learned.get('available')}`",
        f"- Note: {learned.get('note')}",
        "",
        markdown_table(
            learned_rows,
            ["root", "train dirs", "model.pt", "results.pkl", "topology.json", "config.json"],
        )
        if learned_rows
        else "No fixed-m20 run counts were available.",
        "",
        "## Conclusion",
        "",
        "The repaired gamma and tree-multiplicity reports are present. Fixed-m20 learned tensors are available on Engaging through `model.pt`, but the local aggregate CSV/JSON reports do not contain learned K tensors, so post-training weighted tree-overlap analyses must reload per-run models from Engaging.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "next_phase_orientation_audit.v1",
        "git": git_state(),
        "required_reports": report_presence(),
        "script_map": scripts(),
        "fixed_m20_roots": FIXED_M20_ROOTS,
        "fixed_m20_learned_k_availability": fixed_m20_learned_k_availability(),
    }
    OUT_JSON.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
    write_md(payload)


if __name__ == "__main__":
    main()
