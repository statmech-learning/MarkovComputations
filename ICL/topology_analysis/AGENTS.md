# WTA-ICL Topology Analysis — Agent Goals

Six agents, one analysis module each, all built on the frozen `core.py`.
This file is the brief: read it fully, then build your assigned module.

---

## 0. The model — how it actually computes (read this first)

For an input `X` = flattened `[ctx0, ctx1, ctx2, ctx3, query]` ∈ ℝ²⁰, each
species `j` (of `n_nodes`) does:

1. **Linear feature** `Wⱼ·X` (bias `w0` is hard-zeroed, not learned) → `softplus`.
2. **Rate** `fⱼ = softplus(Wⱼ·X) / Kⱼ`.
3. **Selection score** `ratioⱼ = βⱼ / fⱼ`.
4. **Soft winner-take-all**: `Y = softmin(ratios; τ) ⊙ Y_potential`, where
   `Y_potentialⱼ = Kⱼ · softplus(R0·fⱼ/βⱼ − 1)`.
5. **Routing** `q = Y @ B` (B is `n × 4`) → `attention = softmax(q/temp)` over
   the 4 context positions → predict the attended position's label.

**⚠️ The WTA is SOFT — this is verified empirically, do not assume otherwise.**
`Y` spreads over **~2.5–3 species** (the top species holds only ~56–68% of
total `Y`). `Y` is a point on a **simplex**, not a one-hot vector; `q` is a
**graded mixture** of `B`-rows. Treat the discrete "winner" as a *summary
statistic* and always *measure* how concentrated `Y` is — never assume it.

Two winner notions exist and disagree ~24% of the time:
- `winner` = `argmin βⱼ/fⱼ` — the WTA *selection rule*.
- `dom_species` = `argmax Yⱼ` — the species that actually *dominates* `Y`.
Use `dom_species` + `Y_frac` for "which species is active"; `winner` for the
literal selection rule.

**Topology** = (a) connectivity `W`, (b) the (soft, graded) species usage,
(c) how the `Y`-mixture maps to attention/answers.

---

## 1. The contract — `core.py` (FROZEN, do not modify)

Every module imports **only** `core`. Standard header:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
```

Key API (see `core.py` docstring for full detail):
- `core.discover_checkpoints()` → `list[Checkpoint]`
- `core.load_all(n_eval=2000)` → `(checkpoints, traces)`
- `core.instrument(model, eval_set, temperature)` → `Trace` (build your own
  probe inputs and call this — m4 needs it)
- `Checkpoint`: `.label .origin .n_nodes .seed .model .params .results .history`
- `Trace`: per-example arrays `z_flat, f, ratios, winner, dom_species,
  softmin_w, Y, Y_frac, q, attention, pred_pos, true_pos, pred_label, target,
  correct, K, beta`; props `.n .accuracy .peak_share`.
- `core.setup_style()`, `core.module_outdir(name)`, `core.save_fig(fig, outdir, name)`

---

## 2. Rules for every agent

- **Build exactly one file**: `mX_<name>.py` in this directory. Do **not**
  touch `core.py` or another agent's file.
- **Module entry point** — every module exposes:
  ```python
  def run(checkpoints, traces, outdir):
      """checkpoints: list[Checkpoint]; traces: {label: {'in_dist':Trace,'novel':Trace}};
         outdir: Path for this module's figures. Returns a dict of scalar findings."""
      ...
      return {...}
  ```
  Also support `python mX_<name>.py` standalone (call `core.load_all()` then `run`).
- **Generality**: handle *any* number of checkpoints and *any* `n_nodes`. The
  training grid adds ~11 more checkpoints; your code must pick them up with no
  changes (iterate `checkpoints`, never hard-code labels or `n=8/12`).
- **Develop now** against the 4 checkpoints already present
  (`ICL/paper_checkpoints/`, `ICL/results/wta_n_nodes_rhoall_seed/`). Grid
  checkpoints appear later and are picked up automatically.
- **Do NOT git-commit or push.** Shared tree. Create only your file; the lead
  agent (W2) commits everything once all modules are done.
- Figures go to `outdir` (provided). Use the `novel` split as primary (that is
  true ICL); use `in_dist` only where a contrast is informative.
- Return JSON-serializable scalars from `run()` (floats/ints/lists/dicts).

---

## 3. Agent assignments (tmux windows 2–7)

Each agent has an Engaging SSH pane beside it. Honest note: only **W2** needs
it heavily (training grid). W3–W7: use the pane only to `git pull` when grid
checkpoints land — analysis itself is local and runs in seconds.

### W2 — LEAD: training grid + M6 comparison + integration
File: `m6_comparison.py`, plus `run_all.py`.

**First, on the Engaging pane**, submit the training grid (SLURM runs all 11
in parallel, ~40 min):
```bash
cd /orcd/home/002/aadarwal/MarkovComputations && git pull
WTA_JOBS="2 1.0 30;4 1.0 30;6 1.0 30;10 1.0 30;12 1.0 30;8 1.0 31;8 1.0 32;8 1.0 33;12 1.0 31;12 1.0 32;12 1.0 33" \
  bash ICL/engaging/run_on_engaging.sh
```
Monitor `squeue -u aadarwal`; as runs finish, `verify_checkpoints.py` each,
then `git add ICL/results/wta_n_nodes_rhoall_seed && git commit && git push`.

**M6 — is the learned topology canonical?**
- For each group of same-`n_nodes` checkpoints: align species pairwise via
  Hungarian matching — *parametric* (cosine similarity of `W` rows) and
  *functional* (co-occurrence of `dom_species` on the shared eval set).
  Report agreement %. High → canonical; low → degenerate solution space.
- 8-node vs 12-node: compare effective node count + accuracy.
- Capacity-sweep curve: accuracy (and effective node count) vs `n_nodes`,
  using all checkpoints' stored `results` + traces.
- Outputs: alignment matrices, agreement bars, sweep curves.

**Integration**: write `run_all.py` — discover checkpoints, `core.load_all()`,
call `m1..m6` `run()` with their `core.module_outdir(...)`, merge the returned
dicts into `ICL/results/topology_analysis/summary.json`. Run it, commit
everything (modules + figures + summary).

### W3 — M1: connectivity (`m1_connectivity.py`)
Question: *what linear feature does each species detect?*
- Reshape each `Wⱼ` to (5 positions × 4 dims); heatmap grid per checkpoint.
- SVD of `W` → scree plot (effective rank).
- Column-norm grouped by position → does the model weight the query slot vs
  the 4 context slots differently?
- Return: effective rank, query-vs-context norm ratio, per checkpoint.

### W4 — M2: node utilization (`m2_utilization.py`)
Question: *how many species does the model actually use?*
- Histogram of `dom_species`; bar of mean `Y_frac` per species (graded use).
- **Effective node count** = participation ratio `(Σpᵢ)²/Σpᵢ²` of mean
  `Y_frac`. Plot it vs `n_nodes` across the capacity sweep.
- WTA hardness: distribution of `peak_share`, mean count of species with
  `Y_frac>5%`.
- Return: effective_node_count, mean_peak_share, per checkpoint.

### W5 — M3: the routing algorithm (`m3_routing.py`)
Question: *how does species activity map to the answer?*
- Per species: `softmax(B[j,:]/temp)` — its attention pattern over 4 positions.
- `dom_species` × `true_pos` confusion matrix — does the dominant species
  predict the right context position?
- **Because the WTA is soft**: quantify how much the mixture matters — compare
  actual `pred_pos` to a "dominant-species-only" prediction (`q` from that one
  `B`-row). Report agreement; low agreement ⇒ the graded mixture is essential.
- Return: routing accuracy, dom-only agreement, per checkpoint.

### W6 — M4: decision geometry (`m4_geometry.py`)
Question: *how is the input space partitioned?*
- PCA (2D) of `z_flat`, colored by `dom_species` and by `true_pos`.
- Simplex/ternary plot of `Y_frac` (top-3 species) — visualize the soft mix.
- 1-D **query-interpolation sweep**: fix 4 context means, slide the query
  along a line between two of them; build inputs, `core.instrument`, plot
  `dom_species`, `peak_share`, `attention` vs the interpolation parameter.
- Return: qualitative (figure paths); any scalars optional.

### W7 — M5: reaction parameters (`m5_parameters.py`)
Question: *what do `K` and `β` do?*
- Per species: `Kⱼ`, `βⱼ`, gain `gⱼ = 1/(Kⱼ·βⱼ)`. Bar plots.
- Note `winner = argmax softplus(Wⱼ·X)/(Kⱼ·βⱼ)` — selection depends only on
  the **product** `Kⱼ·βⱼ`. Confirm empirically.
- Scatter gain `gⱼ` vs species utilization (mean `Y_frac`) — do high-gain
  species win more?
- Return: correlation(gain, utilization), per checkpoint.

---

## 4. Definition of done

All six `mX_*.py` files exist and run standalone without error on the current
4 checkpoints; `run_all.py` produces figures under
`ICL/results/topology_analysis/` and a `summary.json`; the lead agent has
committed everything and confirmed the grid checkpoints are also analyzed.
