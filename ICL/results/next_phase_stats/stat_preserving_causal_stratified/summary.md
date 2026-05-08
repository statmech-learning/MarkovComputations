# Stratified Statistic-Preserving Causal Scramble Pilot

Regime: `n5_m12_N3_D2`. Selection: two low, two mid, and two high novel-class runs by `test_novel_classes`. Each run used 120 sampled novel-class inputs, two repeats per intervention, and CPU evaluation.

| bucket | label | reported novel | sampled baseline | branch-align delta mean | projection delta mean |
|---|---|---:|---:|---:|---:|
| low | `g0267_hub_spoke_seed76_trainseed4` | 65.8 | 68.33 | -50.83 | -39.58 |
| low | `g0289_two_module_seed2_trainseed5` | 67.4 | 73.33 | -68.33 | -30.00 |
| mid | `g0040_cycle_chords_seed41_trainseed3` | 85.4 | 88.33 | -76.67 | -55.42 |
| mid | `g0187_random_sc_seed92_trainseed1` | 85.4 | 90.00 | -76.67 | -53.33 |
| high | `g0187_random_sc_seed92_trainseed2` | 98.0 | 98.33 | -95.83 | -67.92 |
| high | `g0354_degree_balanced_seed9_trainseed4` | 98.6 | 98.33 | -96.67 | -64.17 |

Interpretation guardrail: this is still a pilot, but it tests whether statistic-preserving scrambles damage models across the observed accuracy spectrum rather than only one cherry-picked high-ICL run.
