"""Resolvent of the first kind for scalar Volterra kernels.

The resolvent L of k is the measure satisfying (k * L)(t) = 1 for t > 0,
written as L(ds) = ell_0 * delta_0(ds) + ell(s) ds, with:
    ell_0 = 1 / k(0)
    ell(t) = -k'(t)/k(0)^2 - (1/k(0)) * int_0^t k'(t-s) * ell(s) ds

This module provides:
- compute_resolvent: discrete resolvent via forward Volterra recursion
- verify_resolvent_identity: check (k*L)(t) ≈ 1 numerically
- compute_Y_t_resolvent_based: alternative path-dependent Y_t representation
"""
import numpy as np
from typing import Callable, Optional, Dict, Any


def _finite_diff_dk(
    kernel_fn: Callable[[float], float], h: float = 1e-6
) -> Callable[[float], float]:
    """Return a finite-difference approximation of k'(t)."""

    def dk(t: float) -> float:
        if t > h:
            return (kernel_fn(t + h) - kernel_fn(t - h)) / (2.0 * h)
        else:
            return (kernel_fn(t + h) - kernel_fn(t)) / h

    return dk


def compute_resolvent(
    grid: np.ndarray,
    kernel_fn: Callable[[float], float],
    dk_fn: Optional[Callable[[float], float]] = None,
) -> Dict[str, Any]:
    """Compute the resolvent of the first kind on a time grid.

    Solves the forward Volterra equation for ell:
        ell(t) = -k'(t)/k(0)^2 - (1/k(0)) * int_0^t k'(t-s)*ell(s) ds

    Parameters
    ----------
    grid : ndarray, shape (N,)
        Monotone time grid 0 = t_0 < ... < t_{N-1} = T.
    kernel_fn : callable
        Scalar kernel k: [0,T] -> R. Must satisfy k(0) > 0.
    dk_fn : callable or None
        Derivative k': [0,T] -> R. If None, computed via finite differences.

    Returns
    -------
    dict with:
        ell_0 : float - point mass at 0: ell_0 = 1/k(0)
        ell : ndarray, shape (N,) - density part of resolvent
        k0 : float - k(0) value
    """
    k0 = kernel_fn(0.0)
    if k0 <= 0:
        raise ValueError(f"k(0) = {k0} must be > 0 for resolvent construction.")
    ell_0 = 1.0 / k0

    if dk_fn is None:
        dk_fn = _finite_diff_dk(kernel_fn)

    N = len(grid)
    dt = np.diff(grid)
    ell = np.zeros(N)

    # Left-point forward Volterra recursion:
    # ell(t_n) = -dk(t_n)/k0^2 - (1/k0) * sum_{j=0}^{n-1} dk(t_n - t_j) * ell(t_j) * dt_j
    for n in range(N):
        t_n = grid[n]
        rhs = -dk_fn(t_n) / (k0 ** 2)
        conv = 0.0
        for j in range(n):
            conv += dk_fn(t_n - grid[j]) * ell[j] * dt[j]
        ell[n] = rhs - conv / k0

    return {"ell_0": float(ell_0), "ell": ell, "k0": float(k0)}


def verify_resolvent_identity(
    grid: np.ndarray,
    kernel_fn: Callable[[float], float],
    res_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Verify the resolvent identity (k * L)(t) ≈ 1.

    (k * L)(t) = k(t) * ell_0 + int_0^t k(t-s) * ell(s) ds

    Parameters
    ----------
    grid : ndarray, shape (N,)
    kernel_fn : callable
    res_data : dict from compute_resolvent

    Returns
    -------
    dict with:
        residual : ndarray, shape (N,) - |(k*L)(t_n) - 1| for each n
        max_error : float
        mean_error : float
    """
    ell_0 = res_data["ell_0"]
    ell = res_data["ell"]
    N = len(grid)
    dt = np.diff(grid)
    residual = np.zeros(N)
    for n in range(N):
        val = kernel_fn(grid[n]) * ell_0
        for j in range(n):
            val += kernel_fn(grid[n] - grid[j]) * ell[j] * dt[j]
        residual[n] = abs(val - 1.0)
    return {
        "residual": residual,
        "max_error": float(np.max(residual)),
        "mean_error": float(np.mean(residual)),
    }


def compute_Y_t_resolvent_based(
    paths: np.ndarray,
    dW: np.ndarray,
    psi: np.ndarray,
    res_data: Dict[str, Any],
    grid: np.ndarray,
    Q: np.ndarray,
    M: np.ndarray,
    alpha: np.ndarray,
    b0: np.ndarray,
    f_running: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    """Compute Y_t via the resolvent-based path-dependent representation.

    For the resolvent L = ell_0 * delta_0 + ell(s) ds of k, the path-dependent
    representation of Y_t uses the past history {X_s: s <= t}:

        Y^{path}(t) = ell_0 * Tr(psi(t) @ X_t)
                      + int_0^t ell(t-s) * Tr(psi(s) @ X_s) ds

    This is an internally consistent representation derived from the resolvent
    identity k * L = 1 and the Riccati-Volterra structure. It provides an
    independent check on Y_t without repeating the stochastic integral.

    For k(t)=1 (Markovian): ell=0 and ell_0=1, so Y^{path}(t) = Tr(psi(t) @ X_t).

    Parameters
    ----------
    paths : ndarray, shape (n_paths, N, d, d)
    psi : ndarray, shape (N, d, d) - Riccati-Volterra solution
    res_data : dict from compute_resolvent (ell_0 and ell)
    (other params match compute_Y_t_direct)

    Returns
    -------
    Y_t_resolvent : ndarray, shape (n_paths, N) real
    """
    n_paths, N, d, _ = paths.shape
    dt = np.diff(grid)
    ell_0 = res_data["ell_0"]
    ell = res_data["ell"]
    Y_res = np.zeros((n_paths, N))

    for p in range(n_paths):
        for n in range(N):
            # Leading term: ell_0 * Tr(psi(t_n) @ X_{t_n})
            val = ell_0 * float(np.real(np.trace(psi[n] @ paths[p, n])))
            # Past-path convolution: int_0^{t_n} ell(t_n - s) Tr(psi(s) X_s) ds
            for j in range(n):
                idx = n - j
                ell_val = ell[idx] if idx < len(ell) else 0.0
                val += ell_val * float(np.real(np.trace(psi[j] @ paths[p, j]))) * dt[j]
            Y_res[p, n] = val

    return Y_res

