"""Transform utilities for the exponential-affine Volterra transform.

Provides functions to:
- Compute Laplace-type payoffs exp(-Tr(uX_T) - int_0^T Tr(f(s)X_s)ds)
- Estimate theoretical Y_0 from the Riccati solution
- Compute running integral int_0^T Tr(f(s) X_s) ds
- Compute Y_t from the affine transform representation
"""
import numpy as np
from typing import Callable, Optional, Dict, Any

from .psd import symmetrize, psd_sqrt_eigh


def compute_running_integral(
    paths: np.ndarray,
    grid: np.ndarray,
    f_running: np.ndarray,
) -> np.ndarray:
    """Compute the running integral int_0^T Tr(f(s) X_s) ds for each path.

    Uses left-point Riemann quadrature: sum_j Tr(f(t_j) @ X_j) * dt_j.

    Parameters
    ----------
    paths : ndarray, shape (n_paths, N, d, d)
        Simulated process paths.
    grid : ndarray, shape (N,)
        Time grid.
    f_running : ndarray, shape (N, d, d)
        Running loading f(t) evaluated at each grid point.

    Returns
    -------
    integral : ndarray, shape (n_paths,)
        int_0^T Tr(f(s) X_s) ds for each path.
    """
    dt = np.diff(grid)
    n_paths = paths.shape[0]
    N = len(grid)
    integral = np.zeros(n_paths)
    for j in range(N - 1):
        # Tr(f(t_j) @ X_j) for all paths simultaneously
        # paths[:, j] shape (n_paths, d, d), f_running[j] shape (d, d)
        traces = np.einsum("ij,pji->p", f_running[j].real, paths[:, j].real)
        integral += traces * dt[j]
    return integral


def compute_payoff_laplace(
    paths: np.ndarray,
    grid: np.ndarray,
    u: np.ndarray,
    f_running: np.ndarray,
) -> np.ndarray:
    """Compute Laplace-type payoff exp(-Tr(u @ X_T) - int_0^T Tr(f(s)@X_s)ds).

    Parameters
    ----------
    paths : ndarray, shape (n_paths, N, d, d)
        Simulated process paths.
    u : ndarray, shape (d, d)
        Terminal loading matrix (real PSD for Laplace case).
    f_running : ndarray, shape (N, d, d)
        Running loading f(t) evaluated at each grid point.

    Returns
    -------
    payoffs : ndarray, shape (n_paths,)
        Payoff values for each path.
    """
    n_paths = paths.shape[0]
    # Terminal term: Tr(u @ X_T) for each path
    X_T = paths[:, -1]  # (n_paths, d, d)
    terminal = np.einsum("ij,pji->p", u.real, X_T.real)
    # Running integral
    running = compute_running_integral(paths, grid, f_running)
    exponent = terminal + running
    payoffs = np.exp(-exponent)
    return payoffs


def compute_Y0_theoretical(
    X0: np.ndarray,
    psi: np.ndarray,
    grid: np.ndarray,
    b0: np.ndarray,
    M: np.ndarray,
    alpha: np.ndarray,
    f_running: np.ndarray,
    kernel_fn: Callable = None,
) -> complex:
    """Compute the theoretical Y_0 from the affine Volterra transform structure.

    The affine representation states:
        E[exp(-Tr(uX_T + int_0^T f(s)X_s ds))] = exp(-Y_0)
    where:
        Y_0 = Tr(psi(0) @ X_0) + phi(0)
        phi(0) = int_0^T k(s) * Tr(psi(s) @ b_0) ds   (scalar integral over constant drift)

    The k(s) weighting arises because the constant drift b_0 enters the process via
    the Volterra kernel K(t-s) = k(t-s) I, so phi(0) = int_0^T k(s) Tr(psi(s) b_0) ds.
    For the Markovian case k≡1, this reduces to int_0^T Tr(psi(s) b_0) ds.

    Parameters
    ----------
    X0 : ndarray, shape (d, d)
        Initial state.
    psi : ndarray, shape (N, d, d)
        Riccati-Volterra solution on the time grid.
    grid : ndarray, shape (N,)
        Time grid.
    b0 : ndarray, shape (d, d)
        Constant drift matrix.
    M, alpha : drift matrices (unused directly; kept for API consistency).
    f_running : ndarray, shape (N, d, d)
        Running loading (unused; already encoded in psi).
    kernel_fn : callable, optional
        Scalar kernel k(t). If None, uses k≡1 (Markovian).

    Returns
    -------
    Y0 : complex scalar
    """
    if kernel_fn is None:
        kernel_fn = lambda t: 1.0

    dt = np.diff(grid)
    N = len(grid)

    # Linear-in-X_0 term: Tr(psi(0) @ X_0)
    Y0 = np.trace(psi[0] @ X0)

    # Scalar phi(0) = int_0^T k(s) Tr(psi(s) @ b_0) ds
    # Use right-point Riemann sum consistent with the Riccati Volterra discretization
    for j in range(1, N):
        k_tj = kernel_fn(grid[j])
        Y0 += np.trace(psi[j] @ b0) * k_tj * dt[j - 1]

    return complex(Y0)


def compute_Y_t_direct(
    paths: np.ndarray,
    dW: np.ndarray,
    psi: np.ndarray,
    grid: np.ndarray,
    Q: np.ndarray,
    M: np.ndarray,
    alpha: np.ndarray,
    b0: np.ndarray,
    f_running: np.ndarray,
    kernel_fn: Callable = None,
) -> np.ndarray:
    """Compute Y_t directly along each path via stochastic integral representation.

    Y_t evolves according to (discrete Euler approximation):
        dY_t = Tr(psi(t)(b_0 + B(X_t)) - 2*X_t*psi(t)*alpha*psi(t) + f(t)*X_t) dt
               + Tr(psi(t)*(sqrt(X_t)*dW_t*Q + Q^T*dW_t^T*sqrt(X_t)))

    with Y_0 given by compute_Y0_theoretical.

    Parameters
    ----------
    paths : ndarray, shape (n_paths, N, d, d)
        Simulated X paths.
    dW : ndarray, shape (n_paths, N-1, d, d)
        Brownian increments used in simulation.
    psi : ndarray, shape (N, d, d)
        Riccati-Volterra solution.
    Q : ndarray, shape (d, d)
        Diffusion loading.
    M, alpha, b0 : drift parameters.
    f_running : ndarray, shape (N, d, d)

    Returns
    -------
    Y_t : ndarray, shape (n_paths, N) complex
    """
    n_paths, N, d, _ = paths.shape
    dt = np.diff(grid)
    Y_t = np.zeros((n_paths, N), dtype=complex)

    X0 = paths[0, 0]
    Y0 = compute_Y0_theoretical(X0, psi, grid, b0, M, alpha, f_running, kernel_fn=kernel_fn)
    Y_t[:, 0] = Y0

    for j in range(N - 1):
        Xj = paths[:, j]          # (n_paths, d, d)
        psi_j = psi[j]            # (d, d)
        f_j = f_running[j]        # (d, d)
        sqXj = np.zeros_like(Xj)
        for p in range(n_paths):
            sqXj[p], _ = psd_sqrt_eigh(Xj[p], clip_neg=0.0)

        # Drift: Tr(psi_j @ (b0 + B(X_j))) - 2 Tr(X_j @ psi_j @ alpha @ psi_j) + Tr(f_j @ X_j)
        B_Xj = M @ Xj + Xj @ M.T  # (n_paths, d, d) via broadcasting
        b_Xj = b0 + B_Xj           # (n_paths, d, d)
        # Tr(psi_j @ b_Xj): einsum over d,d matrices
        term1 = np.einsum("ij,pji->p", psi_j, b_Xj)
        # Tr(X_j @ psi_j @ alpha @ psi_j): contract all indices
        psi_alpha_psi = psi_j @ alpha @ psi_j  # (d, d)
        term2 = 2.0 * np.einsum("pij,ji->p", Xj, psi_alpha_psi)
        # Tr(f_j @ X_j)
        term3 = np.einsum("ij,pji->p", f_j, Xj)
        drift = term1 - term2 + term3

        # Stochastic: Tr(psi_j @ (sqX_j dW_j Q + Q^T dW_j^T sqX_j))
        stoch = np.zeros(n_paths, dtype=complex)
        for p in range(n_paths):
            diff_mat = sqXj[p] @ dW[p, j] @ Q + Q.T @ dW[p, j].T @ sqXj[p]
            stoch[p] = np.trace(psi_j @ diff_mat)

        Y_t[:, j + 1] = Y_t[:, j] + drift * dt[j] + stoch

    return Y_t


def check_terminal_identity(
    paths: np.ndarray,
    Y_t: np.ndarray,
    grid: np.ndarray,
    u: np.ndarray,
    f_running: np.ndarray,
) -> Dict[str, float]:
    """Check the terminal identity Y_T ≈ Tr(uX_T) + int_0^T Tr(f(s)X_s)ds.

    Parameters
    ----------
    paths : ndarray, shape (n_paths, N, d, d)
    Y_t : ndarray, shape (n_paths, N)
    u, f_running : loadings

    Returns
    -------
    dict with error statistics.
    """
    n_paths = paths.shape[0]
    X_T = paths[:, -1]
    terminal = np.einsum("ij,pji->p", u.real, X_T.real)
    running = compute_running_integral(paths, grid, f_running)
    payoff_exponent = terminal + running
    Y_T_real = np.real(Y_t[:, -1])
    errors = np.abs(Y_T_real - payoff_exponent)
    return {
        "mean_abs_error": float(np.mean(errors)),
        "max_abs_error": float(np.max(errors)),
        "median_abs_error": float(np.median(errors)),
        "std_error": float(np.std(errors)),
    }
