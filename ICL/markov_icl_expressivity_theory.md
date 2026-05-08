# Markov-ICL Expressivity Theory

## Scope

This theory is for first-order CRNs / Markov jump processes with exponential input-dependent rates. It should not be transferred to autocatalytic or WTA models without a separate derivation.

Keep the physical reaction graph G separate from the input mask Omega. G controls which rooted spanning trees exist. Omega controls which input coordinates can move which edge rates. Deleting input coupling is therefore not the same operation as deleting a physical edge.

Novel-class ICL accuracy is the primary behavioral target. Training accuracy and ordinary validation accuracy can diagnose optimization, but they do not establish in-context generalization.

## Exact First-Order Representation

For edge e, k_e(z) = exp(b_e + K_e^T z). The matrix-tree theorem gives rooted tree numerators whose exponential projections are tree sums, Theta_T = sum_{e in T} K_e. Therefore the computational basis is the rooted tree-sum basis, not isolated edge projections.

For root r, the steady-state coordinate is a normalized sum over T in T_r(G), with numerator exp(beta_T + Theta_T^T z). All branch-separation, sharpness, and coefficient-control questions should be asked in this tree-sum feature space.

## Input Multiplicity

For a CRN-ICL input mask Omega, define M_alpha = sum_e Omega[e, alpha]. Use the Markov expressivity paper's input multiplicity as a structural measure and hypothesis generator, not as a direct proof of the (2M+1)^D bound for learned continuous K.

Important ICL multiplicities are context/query paired: for branch i and feature dimension d, compare M_{i,d}, M_{q,d}, their overlap, and their imbalance.

Useful pre-training summaries are min M_alpha, mean M_alpha, variance, Gini, zero-coordinate fraction, monoRisk fraction for M_alpha <= 1, and sum_alpha log(2M_alpha + 1). For branch decisions, context/query overlap and imbalance matter more directly than the global average.

## Monotonicity Risk

Very low multiplicity can limit two-sided branch responses. Pre-training risk metrics are M_alpha <= 1, zero-coordinate fraction, context/query imbalance, and low overlap on comparison coordinates. Post-training effective multiplicity can be computed from learned K using participation-style ratios.

For learned weights, M_eff_alpha = (sum_e |K_ealpha|)^2 / sum_e K_ealpha^2 is a participation-style diagnostic. It is not a replacement for Omega because optimization can ignore available coordinates.

## Branch Sharpness

For branch direction u_b, tree-drive range R_{r,b} = max_T Theta_T^T u_b - min_T Theta_T^T u_b measures whether rooted tree polytopes can create sharp branch margins. Coverage and sharpness are distinct.

A topology may cover every branch while still producing weak lower-tail margins if the accessible tree-sum directions are poorly separated or badly conditioned.

## Coefficient Controllability

Tree-polynomial coefficients are constrained by overlapping spanning trees. Rank can overestimate useful capacity, so coefficient-map effective rank, condition number, entropy, extremal-tree accessibility, and branch-specific concentration should be tracked.

This is the main reason d_rel should remain a baseline, not a final theory. Two topologies can have the same relative rank while differing in tree-count distribution, normal-fan coverage, extremal-tree accessibility, or coefficient conditioning.

## Capacity Target

The correct gamma-style target is lower-tail or worst-branch margin, max_theta min_b LCVaR_alpha[m_theta(z) | z in branch b], under constraints on K, b, decoder B, and the input mask. Average branch accuracy is not the primary objective.

The three useful finite-sample probes are exact log-sum-exp tree features, tropical max-over-tree features, and hard-root structural compatibility. They should report branch-wise margins and failures, not only one scalar score.

## Expressivity Versus Trainability

Best seed is an expressivity-envelope proxy; mean seed is a trainability/reliability proxy; seed variance is an optimization-instability proxy. A valid expressivity metric can predict best-seed behavior without explaining mean-seed reliability.

## Thermodynamics

Arbitrary directed exponential rates are not enough for thermodynamic claims. A thermodynamic CRN-ICL variant must use reversible support and a parameterization such as W_ij = exp(E_j - B_ij + F_ij/2 + input drive), with B_ij = B_ji, F_ij = -F_ji, and |F_ij| <= F_max.

Existing arbitrary directed models can be audited for reversible support, but entropy-production or force-budget claims require the controlled reversible-edge parameterization and an explicit F_max sweep.
