# WTA-ICL Topology Analysis — Agent Goals

Six agents, one analysis module each, all built on the frozen `core.py`.
This file is the brief: read it fully, then build your assigned module.

---

## 0. The model — how it actually computes (read this first)

The paper's claim: chemical reaction networks do ICL **without transformer-style
pairwise attention**. There is no `z_i·z_q` dot product anywhere. Instead the
mechanism is **global subspace projection → soft winner-take-all → learned
decoding**. For input `X` = flattened `[ctx0, ctx1, ctx2, ctx3, query]` ∈ ℝ²⁰:

1. **Encoder — global projection.** Each species `j` projects the *whole*
   input: `Wⱼ·X` (bias `w0` is hard-zeroed). The species are meant to detect
   the **comparison subspaces** `Mᵢ = {X : zᵢ ≈ z_query}` — i.e. "the query
   matches context item `i`". A clean comparator has `Wⱼ` equal-and-opposite
   across context-block `i` and the query block.
2. **Rate** `fⱼ = softplus(Wⱼ·X) / Kⱼ`; **selection score** `ratioⱼ = βⱼ/fⱼ`.
3. **Soft winner-take-all** `Y = softmin(ratios; τ) ⊙ Y_potential`,
   `Y_potentialⱼ = Kⱼ·softplus(R0·fⱼ/βⱼ − 1)`.
4. **Decoder** `q = Y @ B` → `attention = softmax(q/temp)` over 4 context
   positions → predict the attended position's label.

**⚠️ Three facts that must shape every analysis (all verified empirically):**

- **The WTA is SOFT.** `Y` spreads over **~2.5–3 species** (top species holds
  only ~56–68% of total `Y`). `Y` is a point on a **simplex**, not one-hot;
  `q` is a **graded mixture** of `B`-rows. Treat the discrete "winner" as a
  *summary statistic* and always *measure* `Y`'s concentration — never assume.
  Two winner notions (disagree ~24%): `winner`=`argmin βⱼ/fⱼ` (selection rule),
  `dom_species`=`argmax Yⱼ` (dominant species). Use `dom_species`/`Y_frac` for
  "which species is active"; `winner` for the literal selection rule.

- **GAUGE SYMMETRY.** `(Kᵣ,βᵣ,Bᵣ) → (λKᵣ, βᵣ/λ, Bᵣ/λ)` leaves the model's
  function *identically* unchanged. So **raw `K` and `β` are unphysical**. The
  identifiable quantities are `W`, the products `Kᵣβᵣ`, and the effective
  decoder `Kᵣ·Bᵣ`. Use `core.physical_params()`; never compare raw `K`, `β`,
  or raw `B` across runs.

- **Default/threshold species may exist.** A species with `Wⱼ ≈ 0` has a
  near-constant score and wins "by default" when no real detector fires. Flag
  near-zero-`W` species — they are part of the mechanism, not dead weight.

**Topology** = (a) connectivity `W` and which comparison subspaces it covers,
(b) the soft, graded species usage, (c) how the `Y`-mixture decodes to answers.

**Hypothesis to TEST (do not assume):** the paper's degree-counting suggests
~`2·N_c` detector species are needed — roughly **2 detectors per context
position** (so ≈8 for the n=8 model, ≈3-per-position for n=12), because each
comparison subspace may need covering in more than one "branch". Whether
species actually group ~2-per-B-target, and whether such branch structure is
real, is something M2/M3 must *measure*, not presuppose.

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
- `core.physical_params(model)` → gauge-invariant `W / Kbeta / eff_decoder`
- `core.comparison_scores(W, N, D)` → `(n, N)` subspace-projection diagnostic
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
  training grid adds ~11 more checkpoints; iterate `checkpoints`, never
  hard-code labels or `n=8/12`.
- **Develop now** against the 4 checkpoints already present
  (`ICL/paper_checkpoints/`, `ICL/results/wta_n_nodes_rhoall_seed/`). Grid
  checkpoints appear later and are picked up automatically.
- **Do NOT git-commit or push.** Shared tree. Create only your file; the lead
  agent (W2) commits everything once all modules are done.
- Figures → `outdir`. Use the `novel` split as primary (true ICL); use
  `in_dist` only for an informative contrast.
- Return JSON-serializable scalars from `run()`.

---

## 3. Agent assignments (tmux windows 2–7)

**Cluster access:** only **W2** touches the Engaging cluster — to submit and
monitor the training grid. **W3–W7 need no cluster access at all**: they run
locally (the repo's `.venv`) on checkpoints in the local repo, and receive the
grid's checkpoints via `git pull` from GitHub once W2 has committed them
(`git pull` ≠ Engaging). An agent cannot clear Duo 2FA on its own, so W2's
Engaging session must be opened by the user; W2 then drives that live pane via
tmux. `git pull` here works anonymously (public repo).

### W2 — LEAD: training grid + M6 comparison + integration
File: `m6_comparison.py`, plus `run_all.py`.

**You are the W2 agent, on tmux window 2.** The pane directly beside you holds
a *pre-authenticated, live* SSH session into the MIT Engaging cluster, opened
for you by the user. **That pane is your ONLY way to reach the cluster** —
drive it with `tmux send-keys` / `tmux capture-pane`, and treat it as a
persistent shell already logged in at `/orcd/home/002/aadarwal/...`. Do NOT run
a fresh `ssh`: Engaging requires Duo 2FA (a push to a human's phone) and a new
login will just hang. The cluster work is yours alone — no other agent (W3–W7)
has or needs cluster access; they work purely locally.

**Step 1 — submit the training grid** through that pane (SLURM runs all 11 in
parallel, ~40 min):
```bash
cd /orcd/home/002/aadarwal/MarkovComputations && git pull
WTA_JOBS="2 1.0 30;4 1.0 30;6 1.0 30;10 1.0 30;12 1.0 30;8 1.0 31;8 1.0 32;8 1.0 33;12 1.0 31;12 1.0 32;12 1.0 33" \
  bash ICL/engaging/run_on_engaging.sh
```
Monitor `squeue` (via the pane); as runs finish, `verify_checkpoints.py` each,
then `git add ICL/results/wta_n_nodes_rhoall_seed && git commit && git push`.

**M6 — is the learned topology canonical?** Align species across same-`n_nodes`
checkpoints. Two comparison regimes — keep them SEPARATE:
- **Same-seed pair** (paper vs engaging, both seed 30 / 20): they differ only
  by CPU/BLAS numerical noise. Agreement here measures **training stability**,
  NOT canonicality. Report it as such.
- **Cross-seed** (grid seeds 31/32/33 vs 30): genuinely independent inits.
  Agreement here is the **canonical-vs-degenerate** result.
Align using **gauge-invariant** quantities only: `core.comparison_scores(W)`
and `W`-row cosine (parametric), and `dom_species` co-occurrence on the shared
eval set (functional). Do **not** align on raw `B` or raw `K`/`β`. Then 8 vs 12
(effective node count, accuracy) and the capacity-sweep curves.

**Integration**: write `run_all.py` — `core.load_all()`, call `m1..m6` `run()`
with `core.module_outdir(...)`, merge returned dicts into
`ICL/results/topology_analysis/summary.json`. Run it, commit everything.

### W3 — M1: connectivity & subspace projection (`m1_connectivity.py`)
Question: *what does each species' projection `Wⱼ` detect?*
- **Headline**: heatmap of `core.comparison_scores(W, N, D)` `(n × N_c)` per
  checkpoint — which species are clean "position i vs query" comparators.
- Reshape each `Wⱼ` to (5 positions × 4 dims); heatmap grid.
- SVD of `W` → effective rank; flag near-zero-norm **default species**.
- Return: per-checkpoint effective rank, count of comparator vs default
  species, which positions are covered.

### W4 — M2: node utilization (`m2_utilization.py`)
Question: *how many species does the model actually use?*
- `dom_species` histogram; mean `Y_frac` per species (graded use).
- **Effective node count** = participation ratio `(Σpᵢ)²/Σpᵢ²` of mean
  `Y_frac`. Plot vs `n_nodes` across the capacity sweep.
- WTA hardness: `peak_share` distribution; mean count of species with
  `Y_frac>5%`.
- Interpretation guide: ~`2·N_c` active ⇒ branch-covering solution;
  ~`N_c` ⇒ compressed/default-threshold solution; `<N_c` ⇒ suspect.
- Return: effective_node_count, mean_peak_share, n_active, per checkpoint.

### W5 — M3: the routing algorithm (`m3_routing.py`)
Question: *how does species activity decode to the answer?*
- Per species: `softmax(B[j,:]/temp)` attention pattern; group species by
  `argmax B[j,:]` (decoded position) — **test** whether ~2–3 species share
  each of the `N_c` targets (the branch hypothesis). If species share a
  target, probe whether their `Wⱼ` rows differ systematically — define any
  "branch" structure *empirically*, do not impose a sign label.
- `dom_species` × `true_pos` confusion matrix.
- **Soft-mixture check**: compare actual `pred_pos` to a dominant-species-only
  prediction (`q` from that one `B`-row). Low agreement ⇒ the graded mixture
  is essential to the computation.
- Return: routing accuracy, dom-only agreement, species-per-target grouping.

### W6 — M4: decision geometry (`m4_geometry.py`)
Question: *how is the input space partitioned?*
- Embed in **score space**, not raw `z_flat`: PCA of the per-species score
  matrix `{Wⱼ·X}` (or `f`, or `ratios`), colored by `dom_species` and
  `true_pos`. (Raw `z_flat` PCA is visually weak — use it only as a contrast.)
- Simplex/ternary plot of `Y_frac` (top-3 species).
- 1-D **query-interpolation sweep**: fix 4 context means, slide the query
  between two of them; build inputs, `core.instrument`, plot `dom_species`,
  `peak_share`, `attention` vs the interpolation parameter. Note boundaries
  are piecewise-smooth (curved near softplus knees), not exact hyperplanes.
- Return: figure paths; scalars optional.

### W7 — M5: physical reaction parameters (`m5_parameters.py`)
Question: *what do the (gauge-invariant) parameters do?*
- Use `core.physical_params(model)`. Plot the **gauge-invariant** quantities:
  `W`-row norms, the products `Kᵣβᵣ`, and the effective decoder `Kᵣ·Bᵣ`.
  Raw `K`/`β` are gauge — if you plot them, label them as such.
- Confirm empirically: the winner ranking / softmin depends only on `Kᵣβᵣ`.
- Scatter `Kᵣβᵣ` (and `W`-norm) vs species utilization (mean `Y_frac`) — do
  low-score / high-`W` species win more? Identify default species.
- Return: correlation(score, utilization), per checkpoint.

---

## 4. Definition of done

All six `mX_*.py` files run standalone without error on the current 4
checkpoints; `run_all.py` produces figures under
`ICL/results/topology_analysis/` and a `summary.json`; the lead agent has
committed everything and confirmed the grid checkpoints are also analyzed.
