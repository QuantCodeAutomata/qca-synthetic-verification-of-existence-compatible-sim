# Synthetic Verification of Existence-Compatible Simulation and PSD Preservation for Matrix Affine Volterra Covariance Processes

A fully synthetic numerical pipeline that simulates **matrix-valued affine Volterra processes** on the cone of positive semidefinite (PSD) matrices and validates the paper's existence-compatible claims and exponential-affine transform theory numerically.

## Overview

This repository implements six self-contained experiments targeting the theoretical claims of the paper on matrix affine Volterra processes:

| Experiment | Description |
|---|---|
| **Exp 1** | PSD preservation and symmetry of simulated paths for `X_t` and `V_t` under Markovian and fractional kernels |
| **Exp 2** | Exponential-affine transform via Riccati-Volterra equations — unconditional and conditional MC validation |
| **Exp 3** | Path-dependent resolvent representation — compare direct and resolvent-based `Y_t` constructions |
| **Exp 4** | Brownian-functional extension of the affine transform, including `z = i` (imaginary) cases |
| **Exp 5** | Commodity-model Laplace transform for the Volterra-Wishart covariance process |
| **Exp 6** | Full Fourier-Laplace transform in the Volterra-Wishart Gibson-Schwartz commodity model |

## Repository Structure

```
.
├── data/                        # Directory for any saved synthetic data
├── src/                         # Core implementation modules
│   ├── kernels.py               # Markovian & shifted-fractional kernels, admissibility
│   ├── psd.py                   # PSD operations: symmetrize, sqrt (eigh), project
│   ├── volterra_sim.py          # Euler-Volterra simulation of X_t and V_t
│   ├── riccati_volterra.py      # Deterministic Riccati-Volterra solver (backward)
│   ├── resolvent.py             # Resolvent of first kind L(ds) and identity check
│   ├── transforms.py            # Y_t computation, payoff evaluation, Y_0 formula
│   ├── brownian_functional.py   # N_t functional and modified Riccati solver (z, H)
│   ├── commodity_model.py       # V_t, delta_t, log S_t simulation
│   ├── commodity_transform.py   # Full Fourier-Laplace transform objects and solver
│   ├── diagnostics.py           # PSD diagnostics, Monte Carlo error summaries, plots
│   └── nested_mc.py             # Nested conditional Monte Carlo estimator
├── exp/                         # Experiment scripts (call src/)
│   ├── exp_1_psd_simulation.py
│   ├── exp_2_affine_transform.py
│   ├── exp_3_resolvent.py
│   ├── exp_4_brownian_functional.py
│   ├── exp_5_commodity_laplace.py
│   └── exp_6_commodity_fourier_laplace.py
├── tests/                       # pytest test suite
│   ├── test_kernels.py
│   ├── test_psd.py
│   ├── test_volterra_sim.py
│   ├── test_riccati_volterra.py
│   ├── test_resolvent.py
│   ├── test_transforms.py
│   ├── test_brownian_functional.py
│   ├── test_commodity_model.py
│   └── test_commodity_transform.py
├── results/                     # Saved figures and CSV metrics
│   └── RESULTS.md               # Experiment summary and findings
├── requirements.txt
└── README.md
```

## Mathematical Background

### Matrix Affine Volterra Process

The general process on `S_d^+` is:

```
X_t = X_0 + ∫₀ᵗ K(t-s) b(X_s) ds + ∫₀ᵗ K(t-s)(√X_s dW_s Q + Q^T dW_s^T √X_s)
```

with `K(t) = k(t) I_d`, `b(x) = b₀ + B(x)`, `B(x) = Mx + xM^T`.

### Kernels Tested

| Kernel | Formula | Notes |
|---|---|---|
| Markovian | `k(t) = 1` | Reduces to standard Wishart process |
| Shifted fractional | `k_{a,ε}(t) = (t+ε)^(a-1) / Γ(a)` | `a ∈ (½,1)`, `ε > 0` |

### Admissibility Condition

For `d=2`: `b₀ ≥ k(0) · α` (in the PSD sense), where `α = Q^T Q`.

### Riccati-Volterra Equation

```
ψ(t) = u K(T-t) + ∫ₜᵀ [f(s) + B^T(ψ(s)) - 2ψ(s)αψ(s)] K(s-t) ds
```

Solved backward in time by time-reversal and a forward recursive scheme.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10+ is recommended.

## Running Tests

```bash
pytest tests/ -v
```

All 99 tests should pass.

## Running Experiments

```bash
python exp/exp_1_psd_simulation.py
python exp/exp_2_affine_transform.py
python exp/exp_3_resolvent.py
python exp/exp_4_brownian_functional.py
python exp/exp_5_commodity_laplace.py
python exp/exp_6_commodity_fourier_laplace.py
```

Each script saves figures (`.png`) and metrics (`.csv`) to `results/`.

## Key Design Decisions

- **No market data**: all inputs are synthetically generated under the paper's admissibility restrictions.
- **Dual output streams**: raw (unprojected) and PSD-projected paths are tracked separately so projection does not mask the unprojected approximation's convergence.
- **Symmetry restoration**: after each Euler step, `A ← (A + A^T)/2` is applied for numerical housekeeping; raw asymmetry is recorded before the correction.
- **Square roots via `eigh`**: `scipy.linalg.eigh` is used throughout for reliable PSD square roots; tiny negative eigenvalues from round-off are optionally clipped and recorded.
- **O(N²) convolution**: the Volterra history sum is computed directly; kernel weights are precomputed per grid for efficiency.
- **Custom implementations**: matrix Riccati-Volterra solvers, resolvent solvers, and Brownian-functional transforms are custom — Context7 confirmed no off-the-shelf package covers these exact equations.

## Results Summary

See [`results/RESULTS.md`](results/RESULTS.md) for a detailed summary of findings.
