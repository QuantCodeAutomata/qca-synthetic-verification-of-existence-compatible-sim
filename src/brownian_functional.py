"""Brownian functional extension of the affine Volterra transform.

Implements N_t = int_0^t Tr(H(s)^T sqrt(X_s) dW_s) and the associated
modified Riccati-Volterra equation with scalar z parameter.

The extended transform is:
    E[exp(z N_T - Tr(u X_T) - int_0^T Tr(f(s) X_s) ds) | F_t] = exp(z N_t - Y_t^{z,H})
"""
import numpy as np
from typing import Callable, Optional, Dict, Any

from .psd import symmetrize, psd_sqrt_eigh


def compute_N_t(
    paths: np.ndarray,
    dW: np.ndarray,
    H_fn: Callable[[float], np.ndarray],
    grid: np.ndarray,
) -> np.ndarray:
    """Compute the stochastic Brownian functional N_t = int_0^t Tr(H(s)^T sqrt(X_s) dW_s).

    Uses left-point Euler-Ito approximation:
        N_{t_{n+1}} = N_{t_n} + Tr(H(t_n)^T sqrt(X_{t_n}) dW_n)

    Parameters
    ----------
    paths : ndarray, shape (n_paths, N, d, d)
        Simulated X paths.
    dW : ndarray, shape (n_paths, N-1, d, d)
        Brownian increments (same as used in simulation).
    H_fn : callable
        Function t -> H(t), a d x d matrix.
    grid : ndarray, shape (N,)
        Time grid.

    Returns
    -------
    N_t : ndarray, shape (n_paths, N)
        Accumulated functional for each path and time.
    """
    n_paths, N, d, _ = paths.shape
    N_t = np.zeros((n_paths, N))

    for j in range(N - 1):
        H_j = H_fn(grid[j])  # (d, d)
        for p in range(n_paths):
            Xj = paths[p, j]  # (d, d)
            sqX, _ = psd_sqrt_eigh(symmetrize(Xj), clip_neg=0.0)
            DW = dW[p, j]  # (d, d)
            # Tr(H(t_j)^T sqrt(X_j) dW_j)
            N_t[p, j + 1] = N_t[p, j] + float(np.trace(H_j.T @ sqX @ DW))

    return N_t


def compute_payoff_extended(
    paths: np.ndarray,
    N_t: np.ndarray,
    grid: np.ndarray,
    z: complex,
    u: np.ndarray,
    f_running: np.ndarray,
) -> np.ndarray:
    """Compute extended payoff exp(z*N_T - Tr(u@X_T) - int_0^T Tr(f(s)@X_s)ds).

    Parameters
    ----------
    paths : ndarray, shape (n_paths, N, d, d)
    N_t : ndarray, shape (n_paths, N) - Brownian functional values
    z : complex
        Functional loading scalar.
    u : ndarray, shape (d, d)
        Terminal matrix loading.
    f_running : ndarray, shape (N, d, d)
        Running loading.

    Returns
    -------
    payoffs : ndarray, shape (n_paths,) of complex values
    """
    from .transforms import compute_running_integral

    n_paths = paths.shape[0]
    N_T = N_t[:, -1]  # (n_paths,)
    X_T = paths[:, -1]  # (n_paths, d, d)
    terminal = np.einsum("ij,pji->p", u.real, X_T.real)
    running = compute_running_integral(paths, grid, f_running)
    exponent = z * N_T - terminal - running
    payoffs = np.exp(exponent.astype(complex))
    return payoffs


def compute_Y_t_extended(
    paths: np.ndarray,
    N_t: np.ndarray,
    dW: np.ndarray,
    psi: np.ndarray,
    grid: np.ndarray,
    Q: np.ndarray,
    M: np.ndarray,
    alpha: np.ndarray,
    b0: np.ndarray,
    f_running: np.ndarray,
    z: complex,
    H_fn: Callable[[float], np.ndarray],
) -> np.ndarray:
    """Compute Y_t^{z,H} along each path via the extended transform representation.

    The extended Y_t satisfies exp(zN_t - Y_t^{z,H}) is a local martingale.

    Parameters
    ----------
    psi : ndarray, shape (N, d, d)
        Solution of the modified Riccati-Volterra equation.
    z : complex
        Functional loading scalar.
    H_fn : callable
        Function t -> H(t).
    (other params same as compute_Y_t_direct)

    Returns
    -------
    Y_t_ext : ndarray, shape (n_paths, N) of complex values
    """
    n_paths, N, d, _ = paths.shape
    dt = np.diff(grid)
    Y_t = np.zeros((n_paths, N), dtype=complex)

    # Initial Y_0^{z,H}: Tr(psi(0) @ X_0) + running affine correction
    X0 = paths[0, 0]
    B_X0 = M @ X0 + X0 @ M.T
    b_X0 = b0 + B_X0
    Y0 = complex(np.trace(psi[0] @ X0))
    for j in range(N - 1):
        psi_j = psi[j]
        f_j = f_running[j]
        Y0 += complex(np.trace(psi_j @ b_X0)) * dt[j]
        Y0 += -2.0 * complex(np.trace(X0 @ psi_j @ alpha @ psi_j)) * dt[j]
        Y0 += complex(np.trace(f_j @ X0)) * dt[j]
    Y_t[:, 0] = Y0

    for j in range(N - 1):
        Xj = paths[:, j]  # (n_paths, d, d)
        psi_j = psi[j]
        f_j = f_running[j]
        H_j = H_fn(grid[j])

        B_Xj = np.einsum("ij,pjk->pik", M, Xj) + np.einsum("pij,kj->pik", Xj, M)
        b_Xj = b0[None] + B_Xj

        drift = np.zeros(n_paths, dtype=complex)
        stoch = np.zeros(n_paths, dtype=complex)

        for p in range(n_paths):
            sqXj, _ = psd_sqrt_eigh(symmetrize(Xj[p]), clip_neg=0.0)
            DW = dW[p, j]

            # Standard drift from Y equation
            drift_p = (
                np.trace(psi_j @ b_Xj[p])
                - 2.0 * np.trace(Xj[p] @ psi_j @ alpha @ psi_j)
                + np.trace(f_j @ Xj[p])
            )
            # Extra drift from Ito correction of z*N_t term:
            # -(z^2/2) * Tr(H_j^T X_j H_j) - z * Tr(psi_j * sqXj * DW * Q * H_j)
            # (This comes from the product rule for exp(z N_t - Y_t))
            # Approximately: -z^2/2 * Tr(H^T X H) dt
            drift_extra = -(z ** 2 / 2.0) * np.trace(H_j.T @ Xj[p] @ H_j)
            drift[p] = drift_p + drift_extra

            # Stochastic terms
            diff_mat = sqXj @ DW @ Q + Q.T @ DW.T @ sqXj
            stoch_Y = np.trace(psi_j @ diff_mat)
            # z * N increment already accumulated in N_t; cross term:
            # -z * Tr(psi_j @ sqXj @ DW @ Q) [contribution from Y process]
            stoch[p] = complex(stoch_Y)

        Y_t[:, j + 1] = Y_t[:, j] + drift * dt[j] + stoch

    return Y_t
