"""
Integration runner for the WTA-ICL topology analysis.

Discovers all checkpoints, builds the shared traces once via core, runs every
analysis module (m1..m6), and writes a combined summary.json. Re-run any time
new checkpoints land — it picks them up automatically.

    python run_all.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

import m1_connectivity
import m2_utilization
import m3_routing
import m4_geometry
import m5_parameters
import m6_comparison

MODULES = [
    ("m1_connectivity", m1_connectivity),
    ("m2_utilization", m2_utilization),
    ("m3_routing", m3_routing),
    ("m4_geometry", m4_geometry),
    ("m5_parameters", m5_parameters),
    ("m6_comparison", m6_comparison),
]


def main():
    print("=" * 70)
    print("WTA-ICL TOPOLOGY ANALYSIS — run_all")
    print("=" * 70)
    checkpoints, traces = core.load_all()
    print(f"Discovered {len(checkpoints)} checkpoint(s): "
          f"{', '.join(c.label for c in checkpoints)}\n")

    summary = {
        "n_checkpoints": len(checkpoints),
        "checkpoints": [
            {"label": c.label, "n_nodes": c.n_nodes, "seed": c.seed,
             "origin": c.origin, "stored": c.results}
            for c in checkpoints
        ],
        "gauge_note": (
            "The model has an exact per-species gauge symmetry "
            "(K_r,beta_r,B_r)->(lam*K_r,beta_r/lam,B_r/lam). Quantities derived "
            "from Y_r alone (Y_frac, dom_species, peak_share, eff_node_count) "
            "are gauge-dependent — they describe the trained network's actual "
            "concentrations but are not function-invariant. m2/m3 also report "
            "gauge-invariant versions built on the contribution Y_r*||B_r||."
        ),
        "modules": {},
    }

    for name, mod in MODULES:
        outdir = core.module_outdir(name)
        print(f"--- {name} ---")
        try:
            res = mod.run(checkpoints, traces, outdir)
            summary["modules"][name] = res
            print(f"    {name}: OK\n")
        except Exception as e:
            import traceback
            summary["modules"][name] = {"error": repr(e),
                                        "traceback": traceback.format_exc()}
            print(f"    {name}: FAILED — {e!r}\n")

    out = core.OUTDIR / "summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("=" * 70)
    ok = sum(1 for m in summary["modules"].values() if "error" not in m)
    print(f"{ok}/{len(MODULES)} modules OK   →   summary: {out}")
    print("=" * 70)
    return 0 if ok == len(MODULES) else 1


if __name__ == "__main__":
    sys.exit(main())
