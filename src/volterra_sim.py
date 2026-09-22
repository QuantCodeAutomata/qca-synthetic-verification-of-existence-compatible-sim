"""Simulation of matrix affine Volterra processes.

Implements left-point Euler-Volterra scheme for:
  - General affine Volterra process X_t on S_d^+
  - Volterra-Wishart covariance process V_t

Both processes use the convolution discretization:
  X_n = X_0
    + sum_{j=0}^{n-1} k(t_n - t_j) * (b0 + B(X_j)) * dt_j      [drift]
    + sum_{j=0}^{n-1} k(t_n - t_j) * (sqX_j dW_j Q + Q^T dW_j^T sqX_j)  [diffusion]
with B(x) = Mx + xM^T.
"""
import numpy as np
from typing import Callable, Optional, Dict, Any

from .psd import (
    symmetrize, psd_sqrt_eigh, psd_sqrt_eigh_batch, min_eig,
    symmetry_error, project_to_psd, project_to_psd_batch,
)


def _drift_B(x: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Linear drift map B(x) = Mx + xM^T (preserves symmetry)."""
    return M @ x + x @ M.T


def simulate_X(
    X0: np.ndarray,
    grid: np.ndarray,
    Q: np.ndarray,
    b0: np.ndarray,
    M: np.ndarray,
    kernel_fn: Callable[[float], float],
    n_paths: int = 100,
    seed: Optional[int] = 42,
    project: bool = False,
) -> Dict[str, Any]:
    """Simulate the general affine Volterra process X_t on S_d^+.

    Equation (left-point Euler-Volterra):
        X_n = X_0
              + sum_{j<n} k(t_n-t_j) * (b0 + B(X_j)) * dt_j
              + sum_{j<n} k(t_n-t_j) * (sqrt(X_j) dW_j Q + Q^T dW_j^T sqrt(X_j))
    with B(x) = Mx + xM^T.

    Parameters
    ----------
    X0 : ndarray, shape (d, d)
        Initial condition in S_d^+.
    grid : ndarray, shape (N,)
        Time grid.
    Q : ndarray, shape (d, d)
        Diffusion loading matrix.
    b0 : ndarray, shape (d, d)
        Constant drift (must satisfy b0 >= k(0)*alpha admissibility).
    M : ndarray, shape (d, d)
        Linear drift coefficient.
    kernel_fn : callable
        Scalar kernel k: [0,T] -> R.
    n_paths : int
        Number of Monte Carlo paths.
    seed : int or None
        RNG seed for reproducibility.
    project : bool
        If True, project each step onto PSD cone.

    Returns
    -------
    dict with keys:
        paths: (n_paths, N, d, d) symmetrized paths
        paths_projected: (n_paths, N, d, d) PSD-projected paths
        min_eigs: (n_paths, N) minimum eigenvalues
        sym_errors: (n_paths, N) asymmetry norms (pre-symmetrization)
        psd_violation_raw: (n_paths, N) bool flags for min_eig < -1e-10
        grid: (N,) time grid
        dW: (n_paths, N-1, d, d) Brownian increments
        kernel_weights: (N, N) precomputed kernel values
    """
    rng = np.random.default_rng(seed)
    N = len(grid)
    dt = np.diff(grid)
    d = X0.shape[0]

    paths = np.zeros((n_paths, N, d, d))
    paths_projected = np.zeros_like(paths)
    paths[:, 0] = X0[None]
    paths_projected[:, 0] = project_to_psd(X0)[None]

    min_eigs = np.zeros((n_paths, N))
    sym_errors = np.zeros((n_paths, N))
    psd_violation_raw = np.zeros((n_paths, N), dtype=bool)

    min_eigs[:, 0] = min_eig(X0)
    sym_errors[:, 0] = symmetry_error(X0)

    # Precompute kernel weights: kweights[n, j] = k(t_n - t_j) for j < n
    kweights = np.zeros((N, N))
    for n in range(1, N):
        for j in range(n):
            kweights[n, j] = kernel_fn(grid[n] - grid[j])

    # Brownian increments: dW[p, j] ~ N(0, dt[j] * I_{d x d})
    dW = rng.standard_normal((n_paths, N - 1, d, d)) * np.sqrt(dt)[None, :, None, None]

    for n in range(1, N):
        # Vectorized over all paths simultaneously
        drift_sum = np.zeros((n_paths, d, d))
        noise_sum = np.zeros((n_paths, d, d))

        for j in range(n):
            Xj = paths[:, j]  # (n_paths, d, d)
            # Batched sqrt: psd_sqrt_eigh_batch handles all paths at once
            sqXj = psd_sqrt_eigh_batch(Xj, clip_neg=0.0)

            kw = kweights[n, j]
            # Drift: k*(b0 + B(X_j))*dt for all paths
            B_Xj = M @ Xj + Xj @ M.T  # (n_paths, d, d) via broadcasting
            drift_sum += kw * (b0 + B_Xj) * dt[j]
            # Diffusion: k*(sqX*dW*Q + Q^T*dW^T*sqX)
            # Since sqX is symmetric: Q^T*dW^T*sqX = (sqX*dW*Q)^T
            DW = dW[:, j]  # (n_paths, d, d)
            term1 = np.einsum("pij,pjk,kl->pil", sqXj, DW, Q)
            noise_sum += kw * (term1 + term1.transpose(0, 2, 1))

        Xn_raw = X0 + drift_sum + noise_sum
        # Record raw asymmetry before symmetrization
        sym_errors[:, n] = np.linalg.norm(Xn_raw - Xn_raw.transpose(0, 2, 1), axis=(1, 2))
        Xn_sym = (Xn_raw + Xn_raw.transpose(0, 2, 1)) / 2.0

        # Batched min eigenvalue computation
        eig_mins = np.linalg.eigvalsh(Xn_sym).min(axis=-1)  # (n_paths,)
        min_eigs[:, n] = eig_mins
        psd_violation_raw[:, n] = eig_mins < -1e-10

        Xn_proj = project_to_psd_batch(Xn_sym)
        paths[:, n] = Xn_proj if project else Xn_sym
        paths_projected[:, n] = Xn_proj

    return {
        "paths": paths,
        "paths_projected": paths_projected,
        "min_eigs": min_eigs,
        "sym_errors": sym_errors,
        "psd_violation_raw": psd_violation_raw,
        "grid": grid,
        "dW": dW,
        "kernel_weights": kweights,
    }


def simulate_V(
    V0: np.ndarray,
    grid: np.ndarray,
    Q: np.ndarray,
    b: np.ndarray,
    M: np.ndarray,
    kernel_fn: Callable[[float], float],
    n_paths: int = 100,
    seed: Optional[int] = 42,
    project: bool = False,
) -> Dict[str, Any]:
    """Simulate the Volterra-Wishart covariance process V_t.

    Equation:
        V_t = V_0
              + int_0^t K(t-s)(b + MV_s + V_sM^T) ds
              + int_0^t K(t-s)(sqrt(V_s) dW_s Q + Q^T dW_s^T sqrt(V_s))

    Parameters
    ----------
    V0 : ndarray, shape (d, d)
        Initial covariance matrix in S_d^+.
    b : ndarray, shape (d, d)
        Constant drift (must satisfy b >= k(0)*alpha admissibility).
    (other parameters same as simulate_X)

    Returns
    -------
    dict with same keys as simulate_X (using V instead of X naming).
    """
    rng = np.random.default_rng(seed)
    N = len(grid)
    dt = np.diff(grid)
    d = V0.shape[0]

    paths = np.zeros((n_paths, N, d, d))
    paths[:, 0] = V0[None]

    min_eigs = np.zeros((n_paths, N))
    sym_errors = np.zeros((n_paths, N))
    psd_violations = np.zeros((n_paths, N), dtype=bool)
    min_eigs[:, 0] = min_eig(V0)

    kweights = np.zeros((N, N))
    for n in range(1, N):
        for j in range(n):
            kweights[n, j] = kernel_fn(grid[n] - grid[j])

    dW = rng.standard_normal((n_paths, N - 1, d, d)) * np.sqrt(dt)[None, :, None, None]

    for n in range(1, N):
        drift_sum = np.zeros((n_paths, d, d))
        noise_sum = np.zeros((n_paths, d, d))

        for j in range(n):
            Vj = paths[:, j]  # (n_paths, d, d)
            sqVj = psd_sqrt_eigh_batch(Vj, clip_neg=0.0)

            kw = kweights[n, j]
            B_Vj = M @ Vj + Vj @ M.T  # (n_paths, d, d)
            drift_sum += kw * (b + B_Vj) * dt[j]
            DW = dW[:, j]
            term1 = np.einsum("pij,pjk,kl->pil", sqVj, DW, Q)
            noise_sum += kw * (term1 + term1.transpose(0, 2, 1))

        Vn_raw = V0 + drift_sum + noise_sum
        sym_errors[:, n] = np.linalg.norm(Vn_raw - Vn_raw.transpose(0, 2, 1), axis=(1, 2))
        Vn_sym = (Vn_raw + Vn_raw.transpose(0, 2, 1)) / 2.0

        eig_mins = np.linalg.eigvalsh(Vn_sym).min(axis=-1)
        min_eigs[:, n] = eig_mins
        psd_violations[:, n] = eig_mins < -1e-10
        paths[:, n] = project_to_psd_batch(Vn_sym) if project else Vn_sym

    return {
        "paths": paths,
        "min_eigs": min_eigs,
        "sym_errors": sym_errors,
        "psd_violations": psd_violations,
        "grid": grid,
        "dW": dW,
        "kernel_weights": kweights,
    }
