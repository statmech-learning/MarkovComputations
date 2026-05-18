"""
Retrain Markov-ICL topologies to convergence, for the degeneracy test.

Why
---
The 80 prior checkpoints were trained for only 100 epochs at a constant
learning rate; their ICL-accuracy curves are still rising and oscillating at
the final epoch (train loss 0.3-0.8). The cross-seed degeneracy verdict
(DIVERGENT) is therefore confounded: we cannot tell whether the seeds diverge
because the topology admits a non-identifiable solution family, or simply
because training stopped mid-descent.

This script removes the confound. For a few topologies it retrains 5 seeds
each to convergence -- many more epochs, cosine LR decay to settle the
end-of-run oscillation, larger training set -- and writes checkpoints in the
same on-disk layout the prior grid used, so degeneracy_test.py can be pointed
straight at them:

    python retrain.py            # train the grid (writes retrained_grid/)
    python degeneracy_test.py --grid <...>/retrained_grid --tag converged

The topology repo is imported read-only: this driver reuses its model, data
generator and evaluator but runs its own train loop (the repo's train_model
has no LR scheduler).
"""

import sys
import os
import io
import json
import time
import shutil
import pickle
import contextlib

import numpy as np
import torch
import torch.nn as nn

TOPO_ICL = "/Users/aadarwal/code/statmech/topology/ICL"
GRID = os.path.join(TOPO_ICL, "results", "prospective_tree_diff_multiplicity_training")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "retrained_grid")

sys.path.insert(0, TOPO_ICL)
from models.topology_markov_icl import TopologyMatrixTreeMarkovICL  # noqa: E402
from data_generation import GaussianMixtureModel, generate_icl_gmm_data  # noqa: E402
from evaluation import test_icl  # noqa: E402

# --- training schedule (the fix) -----------------------------------------
EPOCHS = 500
LR0 = 0.0025
BATCH = 50
TRAIN_SAMPLES = 10000
VAL_SAMPLES = 2000
EVAL_FREQ = 20
TEST_SAMPLES = 1000
SEEDS = [1, 2, 3, 4, 5]

import glob


def discover_topologies():
    """All (graph+mask) groups in the prior grid; the _trainseedN suffix varies."""
    keys = set()
    for d in glob.glob(os.path.join(GRID, "*_trainseed*")):
        if os.path.isdir(d):
            keys.add(os.path.basename(d).rsplit("_trainseed", 1)[0])
    return sorted(keys)


# all 16 masks of the prospective grid (one fixed graph, 16 input masks)
TOPOLOGIES = discover_topologies()


def quiet_test_icl(*a, **kw):
    """test_icl prints a verbose block; silence it."""
    with contextlib.redirect_stdout(io.StringIO()):
        return test_icl(*a, **kw)


def stack_data(data):
    z = torch.stack([d[0] for d in data])
    lab = torch.stack([d[1] for d in data])
    tgt = torch.tensor([float(d[2]) for d in data])
    return z, lab, tgt


def train_one(topo, cfg, seed):
    """Train one seed of one topology to convergence; return model + history."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TopologyMatrixTreeMarkovICL(
        n_nodes=topo["n_nodes"], z_dim=cfg["D"], L=cfg["L"], N=cfg["N"],
        edges=[tuple(e) for e in topo["edges"]], input_mask=topo["input_mask"],
        transform_func=cfg["transform_func"], learn_base_rates=cfg["learn_base_rates"],
        use_label_mod=cfg["use_label_mod"], print_creation=False)

    gmm = GaussianMixtureModel(cfg["K"], cfg["D"], L=cfg["L"],
                               epsilon=cfg["epsilon"], seed=seed, offset=0.0)
    tr = stack_data(generate_icl_gmm_data(
        gmm, TRAIN_SAMPLES, cfg["N"], novel_classes=False,
        exact_copy=cfg["exact_copy"], B=cfg["B"], L=cfg["L"], shuffle_context=True))
    va = stack_data(generate_icl_gmm_data(
        gmm, VAL_SAMPLES, cfg["N"], novel_classes=False,
        exact_copy=cfg["exact_copy"], B=cfg["B"], L=cfg["L"], shuffle_context=True))

    opt = torch.optim.Adam(model.parameters(), lr=LR0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.NLLLoss()
    method, temp = cfg["method"], cfg["temperature"]

    z_tr, lab_tr, tgt_tr = tr
    n = z_tr.shape[0]
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
            "icl_acc": [], "lr": []}

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        losses, correct = [], 0
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            opt.zero_grad()
            logits = model(z_tr[idx], lab_tr[idx], method, temp)
            tgt = tgt_tr[idx].long() - 1
            loss = crit(logits, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            correct += (logits.argmax(1) == tgt).sum().item()
        sched.step()

        model.eval()
        with torch.no_grad():
            vlog = model(va[0], va[1], method, temp)
            vtgt = va[2].long() - 1
            vloss = crit(vlog, vtgt).item()
            vacc = 100.0 * (vlog.argmax(1) == vtgt).float().mean().item()
        hist["train_loss"].append(float(np.mean(losses)))
        hist["val_loss"].append(vloss)
        hist["train_acc"].append(100.0 * correct / n)
        hist["val_acc"].append(vacc)
        hist["lr"].append(opt.param_groups[0]["lr"])

        if (epoch + 1) % EVAL_FREQ == 0:
            r = quiet_test_icl(model, gmm, cfg["N"], "cpu", n_samples=300,
                         exact_copy=cfg["exact_copy"], B=cfg["B"], method=method,
                         L=cfg["L"], temperature=temp, shuffle_context=True)
            hist["icl_acc"].append((epoch + 1, r["novel_classes"]))
        else:
            hist["icl_acc"].append(None)

    model.eval()
    final = quiet_test_icl(model, gmm, cfg["N"], "cpu", n_samples=TEST_SAMPLES,
                     exact_copy=cfg["exact_copy"], B=cfg["B"], method=method,
                     L=cfg["L"], temperature=temp, shuffle_context=True)
    return model, hist, final


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"Retraining {len(TOPOLOGIES)} topologies x {len(SEEDS)} seeds "
          f"-> {EPOCHS} epochs, cosine LR decay.\n")
    t0 = time.time()

    for ti, tname in enumerate(TOPOLOGIES):
        src = os.path.join(GRID, f"{tname}_trainseed1")
        topo = json.load(open(os.path.join(src, "topology.json")))
        base_cfg = json.load(open(os.path.join(src, "config.json")))
        print(f"[{ti+1}/{len(TOPOLOGIES)}] {tname.split('__')[-1]}")

        for seed in SEEDS:
            rd = os.path.join(OUT, f"{tname}_trainseed{seed}")
            if os.path.exists(os.path.join(rd, "model.pt")):
                print(f"   seed{seed}: already done, skipping")
                continue

            cfg = dict(base_cfg)
            cfg["seed"] = seed
            cfg["epochs"] = EPOCHS
            cfg["lr"] = LR0
            cfg["train_samples"] = TRAIN_SAMPLES
            cfg["lr_schedule"] = "cosine"

            ts = time.time()
            model, hist, final = train_one(topo, cfg, seed)
            dt = time.time() - ts

            rd = os.path.join(OUT, f"{tname}_trainseed{seed}")
            os.makedirs(rd, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(rd, "model.pt"))
            json.dump(topo, open(os.path.join(rd, "topology.json"), "w"), indent=2)
            json.dump(cfg, open(os.path.join(rd, "config.json"), "w"), indent=2)
            # carry topology_metrics over for the later rank-vs-shape analysis
            tm = os.path.join(src, "topology_metrics.json")
            if os.path.exists(tm):
                shutil.copy(tm, os.path.join(rd, "topology_metrics.json"))
            pickle.dump({"results": final, "history": hist, "params": cfg,
                         "execution_time": dt},
                        open(os.path.join(rd, "results.pkl"), "wb"))

            icl_pts = [v for v in hist["icl_acc"] if v is not None]
            tail = [v for _, v in icl_pts[-5:]]
            print(f"   seed{seed}: ICL {final['novel_classes']:5.1f}%  "
                  f"in_dist {final['in_dist']:5.1f}%  "
                  f"train_loss {hist['train_loss'][-1]:.3f}  "
                  f"last5-evals {['%.0f'%v for v in tail]}  ({dt:.0f}s)")
        print()

    print(f"Done in {(time.time()-t0)/60:.1f} min  ->  {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
