# Thermodynamic Force-Budget Report

Status: `no_valid_Fmax_sweep_available`.

Existing models use arbitrary directed exponential rates. They may break detailed balance, but they were not parameterized as reversible-edge thermodynamic Markov processes with antisymmetric force budget F_max.

## Reversible-Support Audit

| groups | mean | min | max |
| --- | --- | --- | --- |
| 36 | 0.512 | 0.000 | 1.000 |

## Required Next Implementation

- construct bidirected physical support
- parameterize W_ij = exp(E_j - B_ij + F_ij/2 + input_drive)
- enforce B_ij = B_ji and F_ij = -F_ji
- sweep max absolute antisymmetric force F_max
- report novel-class ICL and lower-tail branch margins by F_max
