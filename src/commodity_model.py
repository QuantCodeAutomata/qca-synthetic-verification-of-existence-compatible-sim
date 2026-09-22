"""Commodity model simulation: Volterra-Wishart Gibson-Schwartz model.

Implements the full commodity model:
    V_t: 2x2 Volterra-Wishart covariance process
    delta_t: convenience yield (Ornstein-Uhlenbeck driven by V_t)
    log S_t: log commodity spot price

The correlated Brownian driver:
    Z_t = W_t rho + sqrt(1 - rho^T rho) W_tilde_t
where W_t is the 2x2 matrix Brownian and W_tilde_t is an independent 2D vector BM.
"""
import numpy as np
from typing import Callable, Optional, Dict, Any

from .psd import symmetrize, psd_sqrt_eigh, min_eig
from .volterra_sim import simulate_V


def simulate_commodity(
    V0: np.ndarray,
    delta0: float,
    logS0: float,
    grid: np.ndarray,
    Q: np.ndarray,
    b: np.ndarray,
    M: np.ndarray,
    kernel_fn: Callable[[float], float],
    kappa: float,
    mu: float,
    theta: float,
    rho: np.ndarray,
    n_paths: int = 100,
    seed: Optional[int] = 42,
    project_V: bool = True,
) -> Dict[str, Any]:
    """Simulate the full Volterra-Wishart Gibson-Schwartz commodity model.

    Dynamics:
        d(log S_t) = (mu - delta_t - V_{11,t}/2) dt + e_1^T sqrt(V_t) dZ_t
        d(delta_t) = kappa*(theta - delta_t) dt + e_2^T sqrt(V_t) dZ_t
        Z_t = W_t rho + sqrt(1-rho^T rho) W_tilde_t   [correlated 2D driver]

    where e_1 = [1,0], e_2 = [0,1], W_t is the 2x2 matrix BM driving V_t,
    and W_tilde_t is an independent standard 2D BM.

    Parameters
    ----------
    V0 : ndarray, shape (2, 2)
        Initial covariance matrix (PSD).
    delta0 : float
        Initial convenience yield.
    logS0 : float
        Initial log spot price.
    grid : ndarray, shape (N,)
        Time grid.
    Q : ndarray, shape (2, 2)
        Diffusion loading for V process.
    b : ndarray, shape (2, 2)
        Constant drift for V (must satisfy b >= k(0)*Q^TQ).
    M : ndarray, shape (2, 2)
        Mean-reversion matrix for V.
    kernel_fn : callable
        Scalar kernel k for V process.
    kappa : float
        Mean-reversion speed for convenience yield.
    mu : float
        Drift of log spot price.
    theta : float
        Long-run mean of convenience yield.
    rho : ndarray, shape (2,)
        Correlation vector, |rho| <= 1.
    n_paths : int
        Number of Monte Carlo paths.
    seed : int or None
        RNG seed.
    project_V : bool
        If True, project V onto PSD cone at each step.

    Returns
    -------
    dict with:
        V_paths : (n_paths, N, 2, 2) - covariance paths
        delta_paths : (n_paths, N) - convenience yield paths
        logS_paths : (n_paths, N) - log spot price paths
        dW : (n_paths, N-1, 2, 2) - matrix BM increments for V
        dW_tilde : (n_paths, N-1, 2) - independent 2D BM increments
        grid : (N,) time grid
    """
    rng = np.random.default_rng(seed)
    N = len(grid)
    dt = np.diff(grid)
    d = 2
    rho_norm_sq = float(np.dot(rho, rho))
    if rho_norm_sq > 1.0 + 1e-10:
        raise ValueError(f"rho^T rho = {rho_norm_sq:.4f} > 1, invalid correlation.")
    sqrt_1_minus_rho = float(np.sqrt(max(0.0, 1.0 - rho_norm_sq)))

    # Simulate V_t (Volterra-Wishart)
    V_sim = simulate_V(
        V0=V0, grid=grid, Q=Q, b=b, M=M,
        kernel_fn=kernel_fn, n_paths=n_paths, seed=seed, project=project_V,
    )
    V_paths = V_sim["paths"]  # (n_paths, N, d, d)
    dW = V_sim["dW"]  # (n_paths, N-1, d, d)

    # Independent 2D BM for orthogonal component
    dW_tilde = rng.standard_normal((n_paths, N - 1, d)) * np.sqrt(dt)[None, :, None]

    # Simulate delta_t and log S_t using Euler scheme
    delta_paths = np.zeros((n_paths, N))
    logS_paths = np.zeros((n_paths, N))
    delta_paths[:, 0] = delta0
    logS_paths[:, 0] = logS0

    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])

    for j in range(N - 1):
        Vj = V_paths[:, j]  # (n_paths, d, d)
        delta_j = delta_paths[:, j]
        logS_j = logS_paths[:, j]

        sqVj = np.zeros_like(Vj)
        for p in range(n_paths):
            sqVj[p], _ = psd_sqrt_eigh(symmetrize(Vj[p]), clip_neg=0.0)

        # Construct dZ_j (n_paths, 2) - correlated 2D BM increment
        # dZ = dW * rho (matrix times vector: row-wise projection) + sqrt(1-rho^2) * dW_tilde
        # dW[:, j] is (n_paths, d, d), rho is (d,)
        # dW * rho: contract over second (column) index: (n_paths, d)
        dW_j = dW[:, j]  # (n_paths, 2, 2)
        dW_tilde_j = dW_tilde[:, j]  # (n_paths, 2)
        dZ_j = dW_j @ rho + sqrt_1_minus_rho * dW_tilde_j  # (n_paths, 2)

        # sqrt(V_j) @ dZ_j: (n_paths, 2)
        sqV_dZ = np.einsum("pij,pj->pi", sqVj, dZ_j)  # (n_paths, 2)

        # V_{11,j} for log S drift
        V11_j = Vj[:, 0, 0]  # (n_paths,)

        # d(log S_t) = (mu - delta_t - V11_t/2) dt + e1^T sqrt(V_t) dZ_t
        dlogS = (mu - delta_j - V11_j / 2.0) * dt[j] + sqV_dZ[:, 0]  # e1^T = [1,0]

        # d(delta_t) = kappa*(theta - delta_t) dt + e2^T sqrt(V_t) dZ_t
        ddelta = kappa * (theta - delta_j) * dt[j] + sqV_dZ[:, 1]  # e2^T = [0,1]

        logS_paths[:, j + 1] = logS_j + dlogS
        delta_paths[:, j + 1] = delta_j + ddelta

    return {
        "V_paths": V_paths,
        "delta_paths": delta_paths,
        "logS_paths": logS_paths,
        "dW": dW,
        "dW_tilde": dW_tilde,
        "grid": grid,
        "V_min_eigs": V_sim["min_eigs"],
        "V_psd_violations": V_sim["psd_violations"],
    }
