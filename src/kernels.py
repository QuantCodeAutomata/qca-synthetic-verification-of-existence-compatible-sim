"""Kernel functions for matrix affine Volterra processes.

Provides Markovian and shifted fractional kernels along with
admissibility validation routines following the paper's Assumption 2.10.
"""
import numpy as np
from scipy.special import gamma
from typing import Callable, Dict, Any


def k_markov(t: float) -> float:
    """Markovian benchmark kernel k(t) = 1."""
    return 1.0


def k_shifted_fractional(t: float, a: float = 0.7, eps: float = 0.1) -> float:
    """Shifted fractional kernel k_{a,eps}(t) = ((t + eps)^(a-1)) / Gamma(a).

    Parameters
    ----------
    t : float
        Time argument t >= 0.
    a : float
        Hurst-like parameter, must satisfy a in (1/2, 1).
    eps : float
        Shift parameter, must be > 0. Ensures k(0) = eps^(a-1)/Gamma(a) > 0.
    """
    if not (0.5 < a < 1.0):
        raise ValueError(f"a must be in (1/2, 1), got {a}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    return (t + eps) ** (a - 1) / gamma(a)


def make_kernel_weights(grid: np.ndarray, kernel_fn: Callable[[float], float]) -> np.ndarray:
    """Precompute kernel weight matrix k(t_n - t_j) for Volterra convolution.

    Parameters
    ----------
    grid : ndarray, shape (N,)
        Monotone time grid 0 = t_0 < ... < t_{N-1} = T.
    kernel_fn : callable
        Scalar kernel function k: [0, T] -> R.

    Returns
    -------
    weights : ndarray, shape (N, N)
        weights[n, j] = k(grid[n] - grid[j]) for j < n, else 0.
    """
    N = len(grid)
    weights = np.zeros((N, N))
    for n in range(1, N):
        for j in range(n):
            weights[n, j] = kernel_fn(grid[n] - grid[j])
    return weights


def validate_kernel_admissibility(
    grid: np.ndarray,
    kernel_fn: Callable[[float], float],
    alpha: np.ndarray,
    b0: np.ndarray,
) -> Dict[str, Any]:
    """Validate kernel and drift admissibility conditions.

    Checks:
    - k(0) > 0
    - k nonnegative on grid
    - k nonincreasing on grid (paper requires this)
    - b0 - k(0)*alpha is PSD (admissibility b0 >= k(0)*alpha for d=2)

    Parameters
    ----------
    grid : ndarray, shape (N,)
        Time grid.
    kernel_fn : callable
        Scalar kernel function.
    alpha : ndarray, shape (d, d)
        Diffusion matrix alpha = Q^T Q.
    b0 : ndarray, shape (d, d)
        Constant drift matrix.

    Returns
    -------
    dict with admissibility diagnostics.
    """
    vals = np.array([kernel_fn(t) for t in grid])
    k0 = kernel_fn(0.0)
    k0_positive = bool(k0 > 0)
    nonneg_ok = bool(np.all(vals >= -1e-12))
    diffs = np.diff(vals)
    noninc_ok = bool(np.all(diffs <= 1e-10))
    margin = b0 - k0 * alpha
    eigs_margin = np.linalg.eigvalsh(margin)
    admissible = bool(np.all(eigs_margin >= -1e-10))
    return {
        "k0": float(k0),
        "k0_positive": k0_positive,
        "nonnegative": nonneg_ok,
        "nonincreasing": noninc_ok,
        "admissibility_margin_eigenvalues": eigs_margin,
        "admissible": admissible,
        "admissibility_gap": float(eigs_margin.min()),
    }
