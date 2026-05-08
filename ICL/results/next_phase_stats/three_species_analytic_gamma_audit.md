# Three-Species Analytic Gamma Audit

Toy C no-bias pass: `True`.

## Toy C: Both Branches

| branch | n | accuracy | ordering | mean margin | p10 margin | LCVaR margin | failure |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1< | 256 | 1.000 | 1.000 | 4.943 | 2.568 | 2.359 | 0.000 |
| M1> | 256 | 1.000 | 1.000 | 5.078 | 2.776 | 2.445 | 0.000 |
| M2< | 256 | 1.000 | 1.000 | 3.886 | 1.635 | 1.252 | 0.000 |
| M2> | 256 | 1.000 | 1.000 | 3.929 | 1.456 | 1.211 | 0.000 |

## Delta Sweep

| delta | Toy B accuracy | Toy B LCVaR | Toy C accuracy | Toy C LCVaR |
| --- | --- | --- | --- | --- |
| 0.100 | 1.000 | 1.682 | 0.979 | 0.466 |
| 0.250 | 1.000 | 3.402 | 1.000 | 1.565 |
| 0.500 | 1.000 | 6.268 | 1.000 | 3.400 |
