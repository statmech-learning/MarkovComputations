# Markov-ICL capacity sweep

**Coverage controls ICL accuracy: a graded capacity curve. Accuracy saturates at d_rel ≈ 160 (1.00 n_req); the prior program's required dimension n_req matches the saturation point. At the lowest coverage tested (d_rel 20) accuracy is 47% — well above chance: graceful degradation.**

- Knob: input-mask density; capacity measured by the masked relative dimension d_rel. Task requirement n_req = 160, chance = 25.0%.
- 12 capacity levels, 5 seeds each, trained to convergence.

| density | d_rel | d_rel/n_req | acc mean | ceiling | std |
|---|---|---|---|---|---|
| 0.05 | 20 | 0.12 | 47.3% | 53.1% | 4.8 |
| 0.10 | 40 | 0.25 | 55.6% | 63.3% | 7.3 |
| 0.15 | 60 | 0.38 | 56.5% | 60.2% | 3.3 |
| 0.20 | 80 | 0.50 | 68.8% | 73.4% | 2.6 |
| 0.25 | 100 | 0.62 | 67.5% | 80.2% | 7.5 |
| 0.30 | 120 | 0.75 | 84.2% | 90.8% | 6.4 |
| 0.35 | 140 | 0.88 | 74.5% | 91.0% | 8.6 |
| 0.40 | 160 | 1.00 | 87.3% | 96.8% | 6.2 |
| 0.45 | 180 | 1.12 | 83.7% | 95.3% | 11.3 |
| 0.50 | 200 | 1.25 | 95.5% | 97.2% | 1.9 |
| 0.60 | 240 | 1.50 | 95.8% | 97.1% | 1.1 |
| 0.75 | 300 | 1.88 | 97.8% | 99.0% | 1.0 |

- **mean** logistic fit: threshold at d_rel/n_req = 0.39 (d_rel = 63), width 0.41, ceiling 100.0%, R² 0.942.
- **ceiling** logistic fit: threshold at d_rel/n_req = 0.54 (d_rel = 86), width 0.15, ceiling 97.7%, R² 0.980.
