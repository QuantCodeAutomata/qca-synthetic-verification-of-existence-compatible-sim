# AGENTS.md — Repository Knowledge for Matrix Affine Volterra Covariance Processes

## Project Purpose
Synthetic numerical verification of matrix affine Volterra processes on S_d^+ (cone of positive-semidefinite d×d matrices). No market data — all inputs are synthetically generated under paper admissibility conditions.

## Repository Layout
```
src/          Core math modules (kernels, psd, volterra_sim, riccati_volterra,
              resolvent, transforms, brownian_functional, commodity_model,
              commodity_transform, diagnostics, nested_mc)
exp/          Six standalone experiment scripts (exp_1 … exp_6)
tests/        99-test pytest suite (all passing)
results/      PNG figures + CSV metrics + RESULTS.md
data/         Placeholder; no raw data committed
```

## Key Design Patterns

### Kernel API
All kernels are plain callables `k(t: float) -> float`:
- `src/kernels.k_markov(t)` → always 1
- `src/kernels.k_shifted_fractional(t, a, eps)` → `(t+eps)^(a-1) / Gamma(a)`, a∈(½,1), ε>0
- Admissibility validation: `validate_admissibility(k0, alpha, b)` checks `min_eig(b − k(0)*alpha) ≥ 0`

### PSD Operations (`src/psd.py`)
- `symmetrize(A)` → `(A + A.T) / 2`  
- `psd_sqrt_eigh(A)` → eigendecomposition square root (clips negatives, records them)
- `min_eig(A)` → scalar minimum eigenvalue  
- `project_to_psd(A)` → clips negative eigenvalues to 0 (optional; raw diagnostics kept separate)

### Volterra Simulation (`src/volterra_sim.py`)
Left-point Euler-Volterra: O(N²) per path. Kernel weights precomputed per grid.
- `simulate_X(grid, X0, Q, b0, M, alpha, kernel_fn, n_paths, rng)` → study A
- `simulate_V(grid, V0, Q, b, M, alpha, kernel_fn, n_paths, rng)` → study B (Wishart)
- `simulate_dW(n_steps, d, n_paths, rng, dt)` → returns shape `(n_steps, d, d, n_paths)` Gaussian increments

### Riccati-Volterra Solver (`src/riccati_volterra.py`)
Backward-in-time via time reversal to a forward Volterra integral equation.
- `solve_riccati_volterra(grid, u, f, alpha, M, kernel_fn)` → psi shape `(N, d, d)`
- Symmetry enforced at each step: `psi <- (psi + psi.T) / 2`
- `compute_residual(psi, grid, u, f, alpha, M, kernel_fn)` → max/mean residual norms

### Resolvent (`src/resolvent.py`)
For scalar kernel k with k(0)>0 and k∈C²([0,T]):
- ell₀ = 1/k(0)
- ell(t) solved by forward Volterra: `ell(t) = −k'(t)/k(0)² − (1/k(0)) ∫₀ᵗ k'(t−s) ell(s) ds`
- `verify_resolvent_identity(k_grid, L_mass, L_density, grid)` → checks k∗L ≈ 1

### Brownian Functional (`src/brownian_functional.py`)
- `compute_N_t(paths_sqrt, dW, H_fn, grid)` → cumulative `Tr(H(t)^T sqrt(X_t) dW_t)` along paths
- `operator_C_H(H, Q, psi)` → `H Q psi + psi Q^T H^T` (extra Riccati term for z≠0)
- Modified Riccati: `solve_riccati_brownian_functional(grid, u, f, H_fn, z, alpha, M, kernel_fn)`

### Commodity Model (`src/commodity_model.py`)
- Parameters: `kappa, mu, theta, S0, delta0`; covariance `V0, Q, M, b, alpha`; kernel `k`
- `simulate_commodity(grid, params, kernel_fn, rho, n_paths, rng)` → (V_paths, delta_paths, logS_paths, dW, dW_tilde)
- Z = W ρ + √(1−ρᵀρ) W̃ (row-wise loading of matrix BM into 2D factor BM)

### Commodity Transform (`src/commodity_transform.py`)
- `compute_deterministic_objects(grid, T, params, u1, u2)` → (C_T, chi_fn, H_fn, f_fn, M_chi_fn)
- `solve_commodity_riccati(grid, u3, f_fn, M_chi_fn, alpha, kernel_fn)` → psi shape (N,2,2)
- `compute_Phi_t(...)` → stochastic exponential along paths

## Admissibility Conditions (paper-exact)
- Study A (general process): `b₀ ≥ k(0) α` in PSD sense (d=2 so `b₀ ≥ k(0) α`)
- Study B / commodity V: `b ≥ k(0) α`
- Kernel: bounded, nonneg, continuous, nonincreasing, `k(0) > 0`
- Shifted fractional: `a ∈ (½,1)`, `ε > 0`; shift makes kernel smooth on `[0,T]`

## Numerical Notes
- Discretization bias of ~7-25% is expected and reported in RESULTS.md; not a bug
- Markovian (k≡1) is always the easiest/most stable benchmark; exact Riccati residuals = 0
- Fractional kernel residuals at machine epsilon (3-7e-18) after symmetrization
- For complex-symmetric matrices (Fourier-Laplace), do NOT conjugate — use `.T` only
- PSD sqrt: clip negatives to 0 silently; record in diagnostics but don't raise

## Testing
```bash
pytest tests/ -v   # 99 tests, ~0.8s
```
Tests cover: kernel admissibility, PSD ops, Volterra sim shape/symmetry, Riccati solver residuals, resolvent identity, transform terminal identity, brownian functional z=0 reduction, commodity path shapes, Fourier-Laplace PSD diagnostics.

## Running Experiments
```bash
python exp/exp_1_psd_simulation.py      # ~20-30s
python exp/exp_2_affine_transform.py    # ~30-60s
python exp/exp_3_resolvent.py           # ~20-30s
python exp/exp_4_brownian_functional.py # ~30-60s
python exp/exp_5_commodity_laplace.py   # ~30-60s
python exp/exp_6_commodity_fourier_laplace.py  # ~60-120s
```
All results (PNG, CSV) saved to `results/`.

## Key Invariants to Never Break
1. `symmetrize()` called after every matrix update (both in simulation and Riccati solver)
2. Raw (unprojected) diagnostics always recorded separately from projected paths
3. Full path history is cached for nested MC (non-Markov model — X_t alone insufficient)
4. Complex symmetric matrices: use `.T`, never `.conj().T`
5. No market data in the repo — all synthetic
