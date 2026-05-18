"""
Markov-ICL capacity sweep — does ICL collapse below a coverage threshold?

Why
---
Result 6 shows accuracy tracks projection coverage across 16 masks, but all 16
sit at d_rel = 200, above the prior program's required dimension
n_req = 2*N*(N+1)*D = 160. The model is never seen to fail. To turn "capacity
is coverage" from a 12-point correlation into a capacity law we must sweep the
coverage axis DOWN through failure.

Knob: input-mask density. Each edge couples to round(20*density) of the 20
input coordinates; lower density -> lower masked relative dimension d_rel ->
less of the input space is reachable. At density 0 the rates are input-blind
and ICL is impossible (chance); at high density d_rel saturates. The shape of
the curve between -- sharp threshold near d_rel = n_req, or gradual -- is the
experiment, and it directly tests the prior program's untested n_req formula.

Modes
-----
    python capacity_sweep.py precheck   # zero-cost: d_rel vs density curve
    python capacity_sweep.py train      # train the chosen sweep to convergence

precheck confirms the sweep crosses n_req and picks the density levels;
train writes checkpoints to capacity_grid/ for degeneracy_test / analysis.
"""

import sys
import os
import io
import json
import time
import glob
import pickle
import contextlib

import numpy as np
import torch
import torch.nn as nn

TOPO_ICL = "/Users/aadarwal/code/statmech/topology/ICL"
GRID = os.path.join(TOPO_ICL, "results", "prospective_tree_diff_multiplicity_training")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "capacity_grid")

sys.path.insert(0, TOPO_ICL)
sys.path.insert(0, HERE)
from topology_metrics import masked_relative_svd_metrics  # noqa: E402
from models.topology_markov_icl import TopologyMatrixTreeMarkovICL  # noqa: E402
from data_generation import GaussianMixtureModel, generate_icl_gmm_data  # noqa: E402
from evaluation import test_icl  # noqa: E402
from make_decorrelated_masks import build_D  # noqa: E402

P = 20
N_EDGES = 20
# task-intrinsic required dimension, from the prior program: 2*N*(N+1)*D
N_REQ = 2 * 4 * 5 * 4  # = 160

# training schedule (matches retrain.py: converged, cosine LR decay)
EPOCHS = 500
LR0 = 0.0025
BATCH = 50
TRAIN_SAMPLES = 10000
VAL_SAMPLES = 2000
TEST_SAMPLES = 1000
SEEDS = [1, 2, 3, 4, 5]


def density_mask(density, rng):
    """20x20 mask; each edge couples to round(20*density) random coordinates."""
    k = max(1, min(P, round(P * density)))
    m = np.zeros((N_EDGES, P), dtype=int)
    for r in range(N_EDGES):
        m[r, rng.choice(P, size=k, replace=False)] = 1
    return m


# --------------------------------------------------------------------------
# pre-check: d_rel vs density (zero training cost)
# --------------------------------------------------------------------------
def precheck():
    D, topo = build_D()
    rng = np.random.default_rng(42)
    densities = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                 0.50, 0.60, 0.75, 1.00]
    print(f"n_req = 2*N*(N+1)*D = {N_REQ}\n")
    print(f"{'density':>8} {'edges/coord':>12} {'d_rel':>14} "
          f"{'eff_rank':>14} {'d_rel/n_req':>12}")
    rows = []
    for d in densities:
        dr, er = [], []
        for _ in range(24):
            m = density_mask(d, rng)
            s = masked_relative_svd_metrics(D, m.astype(float), P)
            dr.append(s["rank"]); er.append(s["effective_rank"])
        dr, er = np.array(dr), np.array(er)
        k = max(1, round(P * d))
        rows.append({"density": d, "k": k, "d_rel_mean": float(dr.mean()),
                     "d_rel_std": float(dr.std()),
                     "eff_rank_mean": float(er.mean())})
        print(f"{d:8.2f} {k:12d} {dr.mean():9.1f}+-{dr.std():<4.1f} "
              f"{er.mean():14.2f} {dr.mean()/N_REQ:12.2f}")

    below = [r for r in rows if r["d_rel_mean"] < N_REQ]
    above = [r for r in rows if r["d_rel_mean"] >= N_REQ]
    print()
    if below and above:
        print(f"sweep crosses n_req: d_rel ranges "
              f"{rows[0]['d_rel_mean']:.0f} (density {rows[0]['density']}) "
              f"to {rows[-1]['d_rel_mean']:.0f} (density {rows[-1]['density']}); "
              f"{len(below)} levels below n_req, {len(above)} above. "
              f"WELL-POSED.")
    else:
        print("sweep does NOT bracket n_req — adjust density range.")

    json.dump({"n_req": N_REQ, "rows": rows},
              open(os.path.join(HERE, "capacity_precheck.json"), "w"), indent=2)
    try:
        _precheck_plot(rows)
    except Exception as e:
        print(f"(plot skipped: {e!r})")
    print(f"\n-> capacity_precheck.json")
    return 0


def _precheck_plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = [r["density"] for r in rows]
    dr = [r["d_rel_mean"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(d, dr, "o-", color="#1f5066", lw=2, label="d_rel (reachable dimension)")
    ax.axhline(N_REQ, color="crimson", ls="--", lw=1.6,
               label=f"n_req = {N_REQ}  (task requirement)")
    ax.fill_between(d, 0, N_REQ, color="crimson", alpha=0.06)
    ax.set_xlabel("input-mask density  (fraction of K entries coupled)")
    ax.set_ylabel("d_rel  —  masked relative dimension")
    ax.set_title("Capacity sweep is well-posed: d_rel crosses the task requirement")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "capacity_precheck.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# training (a converged sweep over capacity levels)
# --------------------------------------------------------------------------
def _quiet_test(*a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return test_icl(*a, **kw)


def _stack(data):
    return (torch.stack([d[0] for d in data]),
            torch.stack([d[1] for d in data]),
            torch.tensor([float(d[2]) for d in data]))


def train_one(topo_edges, n_nodes, mask, cfg, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TopologyMatrixTreeMarkovICL(
        n_nodes=n_nodes, z_dim=cfg["D"], L=cfg["L"], N=cfg["N"],
        edges=[tuple(e) for e in topo_edges], input_mask=mask.tolist(),
        transform_func=cfg["transform_func"], learn_base_rates=cfg["learn_base_rates"],
        use_label_mod=cfg["use_label_mod"], print_creation=False)
    gmm = GaussianMixtureModel(cfg["K"], cfg["D"], L=cfg["L"],
                               epsilon=cfg["epsilon"], seed=seed, offset=0.0)
    z, lab, tgt = _stack(generate_icl_gmm_data(
        gmm, TRAIN_SAMPLES, cfg["N"], novel_classes=False,
        exact_copy=cfg["exact_copy"], B=cfg["B"], L=cfg["L"], shuffle_context=True))
    va = _stack(generate_icl_gmm_data(
        gmm, VAL_SAMPLES, cfg["N"], novel_classes=False,
        exact_copy=cfg["exact_copy"], B=cfg["B"], L=cfg["L"], shuffle_context=True))
    opt = torch.optim.Adam(model.parameters(), lr=LR0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.NLLLoss()
    method, temp = cfg["method"], cfg["temperature"]
    n = z.shape[0]
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            opt.zero_grad()
            loss = crit(model(z[idx], lab[idx], method, temp), tgt[idx].long() - 1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    final = _quiet_test(model, gmm, cfg["N"], "cpu", n_samples=TEST_SAMPLES,
                        exact_copy=cfg["exact_copy"], B=cfg["B"], method=method,
                        L=cfg["L"], temperature=temp, shuffle_context=True)
    return model, final


def train():
    os.makedirs(OUT, exist_ok=True)
    pre = json.load(open(os.path.join(HERE, "capacity_precheck.json")))
    # one capacity level per density in the pre-check (skip the redundant tail)
    levels = [r for r in pre["rows"] if r["density"] <= 0.75]
    D, topo = build_D()
    base_cfg = json.load(open(os.path.join(
        glob.glob(os.path.join(GRID, "*_trainseed1"))[0], "config.json")))
    rng = np.random.default_rng(2026)
    print(f"Capacity sweep: {len(levels)} density levels x {len(SEEDS)} seeds, "
          f"{EPOCHS} epochs each.\n")
    t0 = time.time()
    for li, r in enumerate(levels):
        dens = r["density"]
        mask = density_mask(dens, rng)
        s = masked_relative_svd_metrics(D, mask.astype(float), P)
        print(f"[{li+1}/{len(levels)}] density {dens:.2f}  "
              f"d_rel {s['rank']}  (n_req {N_REQ})")
        for seed in SEEDS:
            rd = os.path.join(OUT, f"density{int(dens*100):03d}_trainseed{seed}")
            if os.path.exists(os.path.join(rd, "results.pkl")):
                print(f"   seed{seed}: done, skip"); continue
            cfg = dict(base_cfg)
            cfg.update(seed=seed, epochs=EPOCHS, lr=LR0,
                       train_samples=TRAIN_SAMPLES, lr_schedule="cosine",
                       mask_density=dens)
            ts = time.time()
            model, final = train_one(topo["edges"], topo["n_nodes"], mask, cfg, seed)
            dt = time.time() - ts
            os.makedirs(rd, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(rd, "model.pt"))
            json.dump({"n_nodes": topo["n_nodes"], "edges": topo["edges"],
                       "input_mask": mask.tolist()},
                      open(os.path.join(rd, "topology.json"), "w"))
            json.dump(cfg, open(os.path.join(rd, "config.json"), "w"), indent=2)
            json.dump({"density": dens, "d_rel": int(s["rank"]),
                       "effective_rank": s["effective_rank"], "n_req": N_REQ},
                      open(os.path.join(rd, "capacity.json"), "w"), indent=2)
            pickle.dump({"results": final, "params": cfg, "execution_time": dt},
                        open(os.path.join(rd, "results.pkl"), "wb"))
            print(f"   seed{seed}: ICL {final['novel_classes']:5.1f}%  ({dt:.0f}s)")
        print()
    print(f"Done in {(time.time()-t0)/60:.1f} min  ->  {OUT}")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "precheck"
    sys.exit(precheck() if mode == "precheck" else train())
