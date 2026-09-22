"""Deterministic matrix Riccati-Volterra equation solvers.

Solves the Riccati-Volterra equation:
    psi(t) = u*K(T-t) + int_t^T [f(s) + B^T(psi(s)) - 2*psi(s)*alpha*psi(s)] K(s-t) ds
with K(t) = k(t) * I_d, B^T(v) = M^T v + vM (adjoint of B(x)=Mx+xM^T under trace inner product).

Discretized via a right-point backward recursion:
    psi[n] = u*k(T-t_n) + sum_{j=n+1}^{N-1} g(psi[j]) * k(t_j - t_n) * dt[j-1]
with g(psi) = f + B^T(psi) - 2*psi*alpha*psi.
"""
import numpy as np
from typing import Callable, Optional, Dict, Any

from .psd import symmetrize_complex


def _B_adjoint(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Adjoint B^T(v) = M^T v + v M under the trace inner product.

    For B(x) = Mx + xM^T, the adjoint satisfies Tr(B^T(v) x) = Tr(v B(x)).
    """
    return M.T @ v + v @ M


def solve_riccati_volterra(
    grid: np.ndarray,
    u: np.ndarray,
    f_running: np.ndarray,
    M: np.ndarray,
    alpha: np.ndarray,
    kernel_fn: Callable[[float], float],
    dtype: type = complex,
) -> Dict[str, Any]:
    """Solve the matrix Riccati-Volterra equation backward in time.

    Equation:
        psi(t) = u*k(T-t) + int_t^T [f(s) + B^T(psi(s)) - 2*psi(s)*alpha*psi(s)] k(s-t) ds

    Discretization (right-point backward sweep):
        psi[N-1] = u * k(0)
        psi[n] = u*k(T-t_n) + sum_{j=n+1}^{N-1} g(psi[j]) * k(t_j-t_n) * dt[j-1]

    Parameters
    ----------
    grid : ndarray, shape (N,)
        Monotone time grid 0 = t_0 < ... < t_{N-1} = T.
    u : ndarray, shape (d, d)
        Terminal loading matrix (complex symmetric).
    f_running : ndarray, shape (N, d, d)
        Running loading f(t) evaluated at each grid point.
    M : ndarray, shape (d, d)
        Linear drift matrix (B(x) = Mx + xM^T).
    alpha : ndarray, shape (d, d)
        Diffusion matrix alpha = Q^T Q.
    kernel_fn : callable
        Scalar kernel k: [0,T] -> R.
    dtype : type
        Numeric dtype (complex by default).

    Returns
    -------
    dict with:
        psi : ndarray, shape (N, d, d) - solution on grid
        residual_norms : ndarray, shape (N,) - discretization residuals
    """
    N = len(grid)
    dt = np.diff(grid)
    T = grid[-1]
    d = u.shape[0]

    psi = np.zeros((N, d, d), dtype=dtype)
    # Boundary condition at t = T
    psi[N - 1] = u * kernel_fn(0.0)

    # Backward sweep: compute psi[n] for n = N-2, ..., 0
    for n in range(N - 2, -1, -1):
        t_n = grid[n]
        rhs = (u * kernel_fn(T - t_n)).astype(dtype)
        for j in range(n + 1, N):
            psi_j = psi[j]
            f_j = f_running[j].astype(dtype)
            g_j = f_j + _B_adjoint(psi_j, M) - 2.0 * psi_j @ alpha @ psi_j
            kval = kernel_fn(grid[j] - t_n)
            rhs = rhs + g_j * kval * dt[j - 1]
        psi[n] = symmetrize_complex(rhs)

    residual_norms = _compute_riccati_residual(psi, grid, u, f_running, M, alpha, kernel_fn)

    return {"psi": psi, "residual_norms": residual_norms}


def solve_riccati_commodity(
    grid: np.ndarray,
    u3: np.ndarray,
    f_running: np.ndarray,
    M_chi_fn: Callable[[float], np.ndarray],
    alpha: np.ndarray,
    kernel_fn: Callable[[float], float],
    dtype: type = complex,
) -> Dict[str, Any]:
    """Solve the commodity-model Riccati-Volterra equation.

    Equation:
        psi(t) = u3*k(T-t) + int_t^T [f(s) + M^chi(s)^T psi(s) + psi(s) M^chi(s)
                                        - 2 psi(s) alpha psi(s)] k(s-t) ds

    where M^chi(s) is time-dependent (constructed from commodity parameters).

    Parameters
    ----------
    M_chi_fn : callable
        Function s -> M^chi(s), a d x d complex matrix.
    (other parameters same as solve_riccati_volterra)
    """
    N = len(grid)
    dt = np.diff(grid)
    T = grid[-1]
    d = u3.shape[0]

    psi = np.zeros((N, d, d), dtype=dtype)
    psi[N - 1] = u3 * kernel_fn(0.0)

    for n in range(N - 2, -1, -1):
        t_n = grid[n]
        rhs = (u3 * kernel_fn(T - t_n)).astype(dtype)
        for j in range(n + 1, N):
            psi_j = psi[j]
            s_j = grid[j]
            f_j = f_running[j].astype(dtype)
            Mchi_j = M_chi_fn(s_j).astype(dtype)
            # Commodity adjoint: M^chi(s)^T psi + psi M^chi(s)
            g_j = f_j + Mchi_j.T @ psi_j + psi_j @ Mchi_j - 2.0 * psi_j @ alpha @ psi_j
            kval = kernel_fn(s_j - t_n)
            rhs = rhs + g_j * kval * dt[j - 1]
        psi[n] = symmetrize_complex(rhs)

    residual_norms = _compute_commodity_residual(psi, grid, u3, f_running, M_chi_fn, alpha, kernel_fn)

    return {"psi": psi, "residual_norms": residual_norms}


def solve_riccati_modified(
    grid: np.ndarray,
    u: np.ndarray,
    f_running: np.ndarray,
    M: np.ndarray,
    alpha: np.ndarray,
    kernel_fn: Callable[[float], float],
    z: complex,
    H_fn: Callable[[float], np.ndarray],
    Q: np.ndarray,
    dtype: type = complex,
) -> Dict[str, Any]:
    """Solve modified Riccati-Volterra for the Brownian functional extension.

    Equation:
        psi(t) = u*k(T-t) + int_t^T [f(s) + B^T(psi(s))
                 + z*C_H(s,psi(s)) - (z^2/2)*H(s)H(s)^T - 2*psi(s)*alpha*psi(s)] k(s-t) ds

    where C_H(psi) = H*Q*psi + psi*Q^T*H^T.

    Parameters
    ----------
    z : complex
        Scalar Brownian-functional loading (use z=0 to recover base Riccati).
    H_fn : callable
        Function s -> H(s), a d x d (complex) matrix.
    Q : ndarray, shape (d, d)
        Diffusion loading matrix.
    (other parameters same as solve_riccati_volterra)
    """
    N = len(grid)
    dt = np.diff(grid)
    T = grid[-1]
    d = u.shape[0]

    psi = np.zeros((N, d, d), dtype=dtype)
    psi[N - 1] = u * kernel_fn(0.0)

    for n in range(N - 2, -1, -1):
        t_n = grid[n]
        rhs = (u * kernel_fn(T - t_n)).astype(dtype)
        for j in range(n + 1, N):
            psi_j = psi[j]
            s_j = grid[j]
            f_j = f_running[j].astype(dtype)
            H_j = H_fn(s_j).astype(dtype)
            B_adj_j = _B_adjoint(psi_j, M)
            # C_H(psi) = H Q psi + psi Q^T H^T
            C_H_j = H_j @ Q @ psi_j + psi_j @ Q.T @ H_j.T
            # Extra constant: -(z^2/2) * H H^T
            extra = -(z ** 2 / 2.0) * H_j @ H_j.T
            g_j = f_j + B_adj_j + z * C_H_j + extra - 2.0 * psi_j @ alpha @ psi_j
            kval = kernel_fn(s_j - t_n)
            rhs = rhs + g_j * kval * dt[j - 1]
        psi[n] = symmetrize_complex(rhs)

    # Compute residual norms using the base Riccati check (without H correction)
    residual_norms = _compute_riccati_residual(psi, grid, u, f_running, M, alpha, kernel_fn)
    return {"psi": psi, "residual_norms": residual_norms}


def solve_riccati_brownian_functional(
    grid: np.ndarray,
    u: np.ndarray,
    f_running: np.ndarray,
    M: np.ndarray,
    alpha: np.ndarray,
    kernel_fn: Callable[[float], float],
    z: complex,
    H_fn: Callable[[float], np.ndarray],
    Q: Optional[np.ndarray] = None,
    dtype: type = complex,
) -> Dict[str, Any]:
    """Alias for solve_riccati_modified with auto-derived Q from alpha.

    If Q is not provided, uses symmetric square root of alpha as proxy.
    See solve_riccati_modified for full documentation.
    """
    if Q is None:
        from scipy.linalg import sqrtm
        Q = np.real(sqrtm(alpha))
    return solve_riccati_modified(grid, u, f_running, M, alpha, kernel_fn, z, H_fn, Q, dtype)


def _compute_riccati_residual(
    psi: np.ndarray,
    grid: np.ndarray,
    u: np.ndarray,
    f_running: np.ndarray,
    M: np.ndarray,
    alpha: np.ndarray,
    kernel_fn: Callable[[float], float],
) -> np.ndarray:
    """Compute residual norm ||psi(t_n) - RHS||_F for each grid point."""
    N = len(grid)
    dt = np.diff(grid)
    T = grid[-1]
    residuals = np.zeros(N)
    for i in range(N):
        t_i = grid[i]
        lhs = psi[i]
        rhs = u * kernel_fn(T - t_i)
        for j in range(i + 1, N):
            psi_j = psi[j]
            f_j = f_running[j]
            g_j = f_j + _B_adjoint(psi_j, M) - 2.0 * psi_j @ alpha @ psi_j
            kval = kernel_fn(grid[j] - t_i)
            rhs = rhs + g_j * kval * dt[j - 1]
        residuals[i] = np.linalg.norm(lhs - rhs, "fro")
    return residuals


def _compute_commodity_residual(
    psi: np.ndarray,
    grid: np.ndarray,
    u3: np.ndarray,
    f_running: np.ndarray,
    M_chi_fn: Callable[[float], np.ndarray],
    alpha: np.ndarray,
    kernel_fn: Callable[[float], float],
) -> np.ndarray:
    """Compute residual norm for commodity Riccati equation."""
    N = len(grid)
    dt = np.diff(grid)
    T = grid[-1]
    residuals = np.zeros(N)
    for i in range(N):
        t_i = grid[i]
        lhs = psi[i]
        rhs = u3 * kernel_fn(T - t_i)
        for j in range(i + 1, N):
            psi_j = psi[j]
            s_j = grid[j]
            f_j = f_running[j]
            Mchi_j = M_chi_fn(s_j)
            g_j = f_j + Mchi_j.T @ psi_j + psi_j @ Mchi_j - 2.0 * psi_j @ alpha @ psi_j
            kval = kernel_fn(s_j - t_i)
            rhs = rhs + g_j * kval * dt[j - 1]
        residuals[i] = np.linalg.norm(lhs - rhs, "fro")
    return residuals
