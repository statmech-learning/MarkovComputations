# Branch Dataset Audit

## Finding

The previous Toy B used active_branches=[0], which encodes one class/one branch. The paper's one-branch condition is z_q=max(z_1,z_2), containing M1> and M2>.

- Toy B contains both `M1>` and `M2>` now: `True`.
- Toy A contains four sign branches: `True`.
- Toy C uses the same branch set as Toy A: `True`.

## toy_A_two_species_both_branches

| branch | label | kind | n | min delta separation | mean delta separation |
| --- | --- | --- | --- | --- | --- |
| M1< | 0 | min | 256 | 0.255 | 0.613 |
| M1> | 0 | max | 256 | 0.250 | 0.624 |
| M2< | 1 | min | 256 | 0.258 | 0.624 |
| M2> | 1 | max | 256 | 0.252 | 0.656 |

## toy_B_two_species_one_branch_max

| branch | label | kind | n | min delta separation | mean delta separation |
| --- | --- | --- | --- | --- | --- |
| M1> | 0 | max | 256 | 0.250 | 0.624 |
| M2> | 1 | max | 256 | 0.255 | 0.613 |

## toy_C_three_species_both_branches

| branch | label | kind | n | min delta separation | mean delta separation |
| --- | --- | --- | --- | --- | --- |
| M1< | 0 | min | 256 | 0.255 | 0.613 |
| M1> | 0 | max | 256 | 0.250 | 0.624 |
| M2< | 1 | min | 256 | 0.258 | 0.624 |
| M2> | 1 | max | 256 | 0.252 | 0.656 |
