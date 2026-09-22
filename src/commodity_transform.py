"""Fourier-Laplace transform for the Volterra-Wishart commodity model.

Constructs the deterministic objects and solves the Riccati-Volterra equation
for the full transform:
    Phi_t(u1, u2, u3) = E[exp(u1*logS_T + u2*delta_T - Tr(u3*V_T)) | F_t]

Key deterministic objects (from the paper's commodity model theorem):
    C_T = u1*(logS_0 + mu*T - ...) + u2*(theta + exp(-kappa*T)*(delta_0-theta))
    chi(s) = [u1, u2*exp(-kappa*(T-s)) - (u1/kappa)*(1-exp(-kappa*(T-s)))]
    f(s) = (u1/2)*e1*e1^T - (1/2)*chi(s)^T chi(s)   [2x2 matrix]
    M^chi(s) = M + Q^T * rho * chi(s)^T               [2x2 matrix]
"""
import numpy as np
from typing import Callable, Dict, Any, Tuple

from .psd import symmetrize_complex, psd_sqrt_eigh, min_eig
from .riccati_volterra import solve_riccati_commodity


def build_deterministic_objects(
    grid: np.ndarray,
    u1: complex,
    u2: complex,
    u3: np.ndarray,
    kappa: float,
    mu: float,
    theta: float,
    delta0: float,
    logS0: float,
    rho: np.ndarray,
    Q: np.ndarray,
    M: np.ndarray,
) -> Dict[str, Any]:
    """Build all deterministic objects required by the commodity transform theorem.

    Parameters
    ----------
    grid : ndarray, shape (N,)
        Time grid, with T = grid[-1].
    u1, u2 : complex
        Fourier-Laplace arguments for logS_T and delta_T.
        Pure imaginary u1 = i*xi, u2 = i*eta for Fourier transform.
    u3 : ndarray, shape (2, 2) complex
        Matrix loading for V_T (PSD real part in Laplace case).
    kappa, mu, theta : float
        Commodity model parameters.
    delta0, logS0 : float
        Initial state values.
    rho : ndarray, shape (2,)
        Correlation vector.
    Q : ndarray, shape (2, 2)
        Diffusion loading for V process.
    M : ndarray, shape (2, 2)
        Mean-reversion matrix for V process.

    Returns
    -------
    dict with:
        C_T : complex scalar
        chi : ndarray, shape (N, 2) complex
        f_running : ndarray, shape (N, 2, 2) complex - f(s) matrix loading
        M_chi : ndarray, shape (N, 2, 2) complex - M^chi(s) matrix
        H_arr : ndarray, shape (N, 2, 2) - H(s) = chi(s)^T rho^T (as 2x2 matrix)
    """
    T = grid[-1]
    N = len(grid)
    d = 2

    # C_T: scalar constant
    exp_kT = np.exp(-kappa * T)
    one_minus_exp = 1.0 - exp_kT
    C_T = (
        u1 * (logS0 + mu * T - (one_minus_exp / kappa) * delta0
              - theta * (T - one_minus_exp / kappa))
        + u2 * (theta + exp_kT * (delta0 - theta))
    )

    # chi(s) for each grid point s
    # chi(s) = [u1, u2*exp(-kappa*(T-s)) - (u1/kappa)*(1-exp(-kappa*(T-s)))]
    chi = np.zeros((N, 2), dtype=complex)
    for i, s in enumerate(grid):
        exp_k_Ts = np.exp(-kappa * (T - s))
        one_minus_exp_Ts = 1.0 - exp_k_Ts
        chi[i, 0] = u1
        chi[i, 1] = u2 * exp_k_Ts - (u1 / kappa) * one_minus_exp_Ts

    # f(s) = (1/2) * (u1 * e1 e1^T - chi(s) outer chi(s))
    # where e1 = [1, 0]
    e1 = np.array([1.0, 0.0])
    f_running = np.zeros((N, d, d), dtype=complex)
    for i in range(N):
        chi_i = chi[i]
        f_running[i] = 0.5 * (u1 * np.outer(e1, e1) - np.outer(chi_i, chi_i))

    # M^chi(s) = M + Q^T @ rho @ chi(s)^T
    # Q^T rho chi^T: outer product giving (d x d) matrix
    M_chi = np.zeros((N, d, d), dtype=complex)
    Qrho = Q.T @ rho  # (d,) vector
    for i in range(N):
        chi_i = chi[i]
        # M^chi(s) = M + Q^T rho (chi(s))^T
        M_chi[i] = M + np.outer(Qrho, chi_i)

    # H(s): for the Brownian functional interpretation,
    # H(s) is constructed from chi(s) and rho (paper's notation)
    # eta(s) = Im(chi(s)) [imaginary part], H(s) = eta(s)^T rho^T as 2x2 matrix
    H_arr = np.zeros((N, d, d), dtype=complex)
    for i in range(N):
        eta_i = np.imag(chi[i])  # shape (2,)
        # H(s) as d x d matrix: outer product eta rho^T
        H_arr[i] = np.outer(eta_i, rho)

    return {
        "C_T": complex(C_T),
        "chi": chi,
        "f_running": f_running,
        "M_chi": M_chi,
        "H_arr": H_arr,
    }


def solve_commodity_riccati(
    grid: np.ndarray,
    u3: np.ndarray,
    det_objs: Dict[str, Any],
    alpha: np.ndarray,
    kernel_fn: Callable[[float], float],
) -> Dict[str, Any]:
    """Solve the commodity-model Riccati-Volterra equation.

    psi(t) = u3*K(T-t) + int_t^T [f(s) + M^chi(s)^T psi(s) + psi(s) M^chi(s)
                                    - 2 psi(s) alpha psi(s)] K(s-t) ds

    Parameters
    ----------
    det_objs : dict from build_deterministic_objects
    alpha : ndarray, shape (2, 2) - diffusion matrix Q^T Q

    Returns
    -------
    dict with psi (N, 2, 2) and residual_norms (N,)
    """
    N = len(grid)
    f_running = det_objs["f_running"]
    M_chi_arr = det_objs["M_chi"]

    def M_chi_fn(s: float) -> np.ndarray:
        idx = np.searchsorted(grid, s, side="right") - 1
        idx = max(0, min(idx, N - 1))
        return M_chi_arr[idx]

    result = solve_riccati_commodity(
        grid=grid,
        u3=u3.astype(complex),
        f_running=f_running,
        M_chi_fn=M_chi_fn,
        alpha=alpha,
        kernel_fn=kernel_fn,
        dtype=complex,
    )
    return result


def compute_stochastic_integral_chi(
    V_paths: np.ndarray,
    dW: np.ndarray,
    dW_tilde: np.ndarray,
    chi: np.ndarray,
    rho: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Compute the stochastic integral int_0^t chi(s) sqrt(V_s) dZ_s.

    Uses left-point Euler:
        I_{n+1} = I_n + chi(t_n) [sqrt(V_n) dZ_n]
    where dZ_n = dW_n rho + sqrt(1-rho^Trho) dW_tilde_n.

    Parameters
    ----------
    chi : ndarray, shape (N, 2) complex - chi(s) vector at each grid point
    dW : ndarray, shape (n_paths, N-1, 2, 2) - matrix BM increments
    dW_tilde : ndarray, shape (n_paths, N-1, 2) - 2D BM increments

    Returns
    -------
    stoch_int : ndarray, shape (n_paths, N) complex
    """
    n_paths, N, d, _ = V_paths.shape
    sqrt_1_rho = float(np.sqrt(max(0.0, 1.0 - float(np.dot(rho, rho)))))
    stoch_int = np.zeros((n_paths, N), dtype=complex)

    for j in range(N - 1):
        Vj = V_paths[:, j]
        chi_j = chi[j]  # (2,) complex
        dW_j = dW[:, j]  # (n_paths, 2, 2)
        dW_tilde_j = dW_tilde[:, j]  # (n_paths, 2)

        for p in range(n_paths):
            sqVj, _ = psd_sqrt_eigh(Vj[p], clip_neg=0.0)
            # dZ_j = dW_j rho + sqrt(1-rho^2) dW_tilde_j
            dZ_j = dW_j[p] @ rho + sqrt_1_rho * dW_tilde_j[p]  # (2,)
            # chi(t_j)^T [sqrt(V_j) dZ_j]
            sqV_dZ = sqVj @ dZ_j  # (2,)
            increment = complex(np.dot(chi_j, sqV_dZ))
            stoch_int[p, j + 1] = stoch_int[p, j] + increment

    return stoch_int


def compute_quadratic_integral_chi(
    V_paths: np.ndarray,
    chi: np.ndarray,
    grid: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    """Compute (1-rho^Trho)/2 * int_0^t chi(s) V_s chi(s)^T ds.

    Parameters
    ----------
    chi : ndarray, shape (N, 2) complex
    rho_sq : float - rho^T rho

    Returns
    -------
    quad_int : ndarray, shape (n_paths, N) complex
    """
    n_paths = V_paths.shape[0]
    N = len(grid)
    dt = np.diff(grid)
    rho_sq = float(np.dot(rho, rho))
    factor = (1.0 - rho_sq) / 2.0
    quad_int = np.zeros((n_paths, N), dtype=complex)

    for j in range(N - 1):
        Vj = V_paths[:, j]  # (n_paths, 2, 2)
        chi_j = chi[j]  # (2,) complex
        # chi V chi^T = chi^T @ V @ chi (scalar)
        chi_V_chi = np.einsum("i,pij,j->p", chi_j, Vj.real, chi_j.real).astype(complex)
        quad_int[:, j + 1] = quad_int[:, j] + factor * chi_V_chi * dt[j]

    return quad_int


def compute_Y_t_commodity(
    V_paths: np.ndarray,
    dW: np.ndarray,
    psi: np.ndarray,
    grid: np.ndarray,
    Q: np.ndarray,
    M: np.ndarray,
    alpha: np.ndarray,
    b: np.ndarray,
    f_running: np.ndarray,
    M_chi_arr: np.ndarray,
) -> np.ndarray:
    """Compute Y_t^{i,H} for the commodity model along each path.

    Parameters
    ----------
    psi : ndarray, shape (N, 2, 2) - Riccati solution
    M_chi_arr : ndarray, shape (N, 2, 2) - M^chi(s) at each grid point

    Returns
    -------
    Y_t : ndarray, shape (n_paths, N) complex
    """
    n_paths, N, d, _ = V_paths.shape
    dt = np.diff(grid)
    Y_t = np.zeros((n_paths, N), dtype=complex)

    # Initial Y_0
    V0 = V_paths[0, 0]
    B_V0 = M @ V0 + V0 @ M.T
    b_V0 = b + B_V0
    Y0 = complex(np.trace(psi[0] @ V0))
    for j in range(N - 1):
        psi_j = psi[j]
        f_j = f_running[j]
        Y0 += complex(np.trace(psi_j @ b_V0)) * dt[j]
        Y0 += -2.0 * complex(np.trace(V0 @ psi_j @ alpha @ psi_j)) * dt[j]
        Y0 += complex(np.trace(f_j @ V0)) * dt[j]
    Y_t[:, 0] = Y0

    for j in range(N - 1):
        Vj = V_paths[:, j]
        psi_j = psi[j]
        f_j = f_running[j]
        Mchi_j = M_chi_arr[j]

        drift = np.zeros(n_paths, dtype=complex)
        stoch = np.zeros(n_paths, dtype=complex)

        for p in range(n_paths):
            sqVj, _ = psd_sqrt_eigh(symmetrize_complex(Vj[p].astype(complex)).real, clip_neg=0.0)
            DW = dW[p, j]

            b_Vj = b + M @ Vj[p] + Vj[p] @ M.T
            drift_p = (
                complex(np.trace(psi_j @ b_Vj))
                - 2.0 * complex(np.trace(Vj[p] @ psi_j @ alpha @ psi_j))
                + complex(np.trace(f_j @ Vj[p]))
            )
            drift[p] = drift_p

            diff_mat = sqVj @ DW @ Q + Q.T @ DW.T @ sqVj
            stoch[p] = complex(np.trace(psi_j @ diff_mat))

        Y_t[:, j + 1] = Y_t[:, j] + drift * dt[j] + stoch

    return Y_t


def compute_full_transform(
    commodity_sim: Dict[str, Any],
    det_objs: Dict[str, Any],
    psi: np.ndarray,
    grid: np.ndarray,
    u1: complex,
    u2: complex,
    u3: np.ndarray,
    rho: np.ndarray,
    Q: np.ndarray,
    M: np.ndarray,
    alpha: np.ndarray,
    b: np.ndarray,
    f_running: np.ndarray,
    M_chi_arr: np.ndarray,
    kappa: float,
    theta: float,
) -> Dict[str, Any]:
    """Compute the full Fourier-Laplace transform Phi_t(u1, u2, u3).

    The transform representation:
        Phi_t = exp(C_T + int_0^t chi(s) sqrt(V_s) dZ_s
                    - (1-rho^Trho)/2 * int_0^t chi(s) V_s chi(s)^T ds
                    - Y_t^{i,H})

    Parameters
    ----------
    commodity_sim : output of simulate_commodity
    det_objs : output of build_deterministic_objects
    psi : Riccati solution

    Returns
    -------
    dict with:
        Phi_t : (n_paths, N) complex - transform values
        payoffs : (n_paths,) - terminal payoff exp(u1*logS_T + u2*delta_T - Tr(u3*V_T))
        mc_mean : complex - Monte Carlo mean of terminal payoff
        M_t : (n_paths, N) - local martingale values (should be ~1 in absolute value)
    """
    V_paths = commodity_sim["V_paths"]
    delta_paths = commodity_sim["delta_paths"]
    logS_paths = commodity_sim["logS_paths"]
    dW = commodity_sim["dW"]
    dW_tilde = commodity_sim["dW_tilde"]

    n_paths, N, d, _ = V_paths.shape
    C_T = det_objs["C_T"]
    chi = det_objs["chi"]

    # Stochastic integral int_0^t chi sqrt(V) dZ
    stoch_int = compute_stochastic_integral_chi(V_paths, dW, dW_tilde, chi, rho, grid)

    # Quadratic term (1-rho^2)/2 * int_0^t chi V chi^T ds
    quad_int = compute_quadratic_integral_chi(V_paths, chi, grid, rho)

    # Y_t^{i,H} for commodity model
    Y_t = compute_Y_t_commodity(V_paths, dW, psi, grid, Q, M, alpha, b, f_running, M_chi_arr)

    # Transform Phi_t = exp(C_T + stoch - quad - Y_t)
    Phi_t = np.exp(C_T + stoch_int - quad_int - Y_t)

    # Terminal payoff: exp(u1*logS_T + u2*delta_T - Tr(u3*V_T))
    logS_T = logS_paths[:, -1]
    delta_T = delta_paths[:, -1]
    V_T = V_paths[:, -1]
    terminal = (
        u1 * logS_T.astype(complex)
        + u2 * delta_T.astype(complex)
        - np.array([np.trace(u3 @ V_T[p]) for p in range(n_paths)], dtype=complex)
    )
    payoffs = np.exp(terminal)
    mc_mean = complex(np.mean(payoffs))

    # M_t = Phi_t / Phi_0 (normalized martingale)
    Phi_0 = Phi_t[:, 0:1]
    M_t = Phi_t / (Phi_0 + 1e-300)

    return {
        "Phi_t": Phi_t,
        "payoffs": payoffs,
        "mc_mean": mc_mean,
        "M_t": M_t,
        "stoch_int": stoch_int,
        "quad_int": quad_int,
        "Y_t": Y_t,
    }
