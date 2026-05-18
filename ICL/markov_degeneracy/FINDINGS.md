# Markov-ICL degeneracy test — findings

## Bottom line

The first-order Markov-ICL model is **parametrically degenerate and
functionally divergent**, and this **survives training to convergence**. Five
seeds of one fixed topology reach similar accuracy through parametrically
unrelated networks (gauge-quotiented weight cosine ≈ 0, identical to the
random-init floor) whose internal representations are *more different from
each other than from an untrained network*. The functions they compute are
only weakly shared, and the sharing grows as the topology's accuracy ceiling
approaches 100%.

## The two runs

`degeneracy_test.py` was run on the prior 100-epoch checkpoints and on
checkpoints retrained to convergence (`retrain.py`: 500 epochs, cosine LR
decay, 10k samples). Aggregate, mean over groups:

| metric | undertrained (100 ep) | converged (500 ep) | random-init floor |
|---|---|---|---|
| ICL accuracy | 65–83% | 77–95% | — |
| **weight cosine** (gauge-quotiented) | −0.02 | **+0.00** | +0.01 |
| **steady-state agreement** (1−TV of p(z)) | 0.35 | **0.35** | 0.46 |
| error-slot agreement (decisive, accuracy-stripped) | +0.23 | +0.23 | 0 |
| slot kappa | +0.47 | +0.70 | 0 |
| attention agreement (1−TV) | 0.63 | 0.75 | 0.36 |
| prediction agreement | 0.61 | 0.78 | 0.27 |

## How to read it

1. **The undertraining confound is removed.** Convergence raised accuracy
   15–20 points (balanced topologies now ~95%), yet the three decisive
   metrics — weight cosine, steady-state agreement, error-slot agreement —
   are *unchanged*. The divergence is not an artifact of stopping early.

2. **The parameterization is non-identifiable, period.** Gauge-quotiented
   weight cosine between two converged seeds equals the random-init floor.
   No two seeds share any parametric structure, at any accuracy.

3. **The internal representation is divergent.** Steady-state agreement
   (0.35) is *below* the trained-vs-random value (0.46): training pushes each
   seed's internal distribution p(z) onto a distinct region of the simplex.
   This uses all 600 queries and all parameters — no sample-starvation, no
   accuracy inflation.

4. **The function partially converges toward the performance ceiling.** As
   accuracy rises, kappa / attention / prediction agreement rise — but the
   accuracy-stripped error-slot probe stays at +0.23. Balanced topologies
   (~95%, kappa 0.80) are more functionally converged than imbalanced (~80%,
   kappa 0.60). Extrapolating: a topology that reached 100% would be
   functionally unique; any topology that caps below 100% leaves a residual
   manifold of equally-accurate-but-different functions, and seeds scatter
   across it.

5. **The landscape is multi-basin.** Even balanced topologies have one seed
   stuck below ceiling (load_0066 seed4 at 74%, load_0114 seed1 at 82%) with
   flat eval curves and higher final loss — genuinely worse optima, not
   undertraining.

6. **A real topology effect is now visible.** Balanced-load input masks
   converge to ~95%, imbalanced to ~80%. Converging the training and using 5
   seeds exposed the topology signal that the prior single-seed, 100-epoch
   runs buried under noise.

## Unified picture (with WTA-ICL)

| model / regime | accuracy ceiling | function across seeds | parameters across seeds |
|---|---|---|---|
| WTA-ICL, above capacity threshold | ~100% | unique | degenerate |
| Markov, balanced topology | ~95% | nearly converged | divergent |
| Markov, imbalanced topology | ~80% | divergent | divergent |

Projection-bank coverage (topology + input mask) sets the accuracy ceiling;
the ceiling sets how functionally converged the seeds are; the
parameterization is **always** degenerate. Functional uniqueness is the
special case of reaching 100% — degeneracy is the rule.

## Consequence for the prior topology program

Single-seed topology→performance regressions were fitting two kinds of noise
at once: cross-seed basin scatter and undertraining. The corrected signal is
that topology acts only through the **accuracy ceiling** it permits (a
coverage/rank quantity), and even that is measurable only with multi-seed,
converged runs.

## Files

- `degeneracy_test.py` — the test (`--grid`, `--tag` select the checkpoint set)
- `retrain.py` — retrains topologies to convergence
- `degeneracy_summary.json` / `_report.md` / `_scatter.png` — 100-epoch run
- `degeneracy_summary_converged.json` / `_report_converged.md` /
  `_scatter_converged.png` — converged run
- `retrained_grid/` — 4 topologies × 5 seeds, converged checkpoints
