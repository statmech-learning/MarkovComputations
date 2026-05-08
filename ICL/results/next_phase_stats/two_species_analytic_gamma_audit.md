# Two-Species Analytic Gamma Audit

Toy A expected failure observed: `True`.
Toy B no-bias pass: `True`.

## Toy A: Both Branches

| branch | n | accuracy | ordering | mean margin | p10 margin | LCVaR margin | failure |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1< | 256 | 0.000 | 0.000 | -7.360 | -11.023 | -11.565 | 1.000 |
| M1> | 256 | 1.000 | 1.000 | 7.488 | 3.779 | 3.366 | 0.000 |
| M2< | 256 | 0.000 | 0.000 | -7.493 | -10.828 | -11.377 | 1.000 |
| M2> | 256 | 1.000 | 1.000 | 7.866 | 4.161 | 3.543 | 0.000 |

## Toy B: One Branch max(z1,z2)

| branch | n | accuracy | ordering | mean margin | p10 margin | LCVaR margin | failure |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1> | 256 | 1.000 | 1.000 | 7.521 | 4.081 | 3.467 | 0.000 |
| M2> | 256 | 1.000 | 1.000 | 7.615 | 3.913 | 3.566 | 0.000 |

## Delta Sweep

| delta | Toy B accuracy | Toy B LCVaR | Toy C accuracy | Toy C LCVaR |
| --- | --- | --- | --- | --- |
| 0.100 | 1.000 | 1.682 | 0.979 | 0.466 |
| 0.250 | 1.000 | 3.402 | 1.000 | 1.565 |
| 0.500 | 1.000 | 6.268 | 1.000 | 3.400 |
