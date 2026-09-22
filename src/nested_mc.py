"""Nested Monte Carlo for conditional transform validation.

For the non-Markovian Volterra process, the conditional expectation
E[payoff | F_t] depends on the FULL path history {X_s: s <= t},
not just the current state X_t.

This module provides functions to:
- Continue a Volterra path forward from a conditioning time
- Estimate E[payoff | F_t] via nested MC
- Compare to the theoretical transform value exp(-Y_t)
"""
import numpy as np
from typing import Callable, Dict, Any, Optional, Tuple

from .psd import symmetrize, psd_sqrt_eigh


def continue_volterra_path(
    past_X: np.ndarray,
    past_dW: np.ndarray,
    past_grid: np.ndarray,
    future_grid: np.ndarray,
    Q: np.ndarray,
    b0: np.ndarray,
    M: np.ndarray,
    kernel_fn: Callable[[float], float],
    seed: Optional[int] = None,
    n_inner: int = 200,
) -> np.ndarray:
    """Continue a Volterra process path forward from conditioning time.

    The Volterra process at future time t_n involves the full history including
    past path values via the convolution integral.

    Parameters
    ----------
    past_X : ndarray, shape (n_past, d, d)
        Realized path values X_{t_0}, ..., X_{t_{n_cond}}.
    past_dW : ndarray, shape (n_past-1, d, d)
        Past Brownian increments.
    past_grid : ndarray, shape (n_past,)
        Grid points 0, ..., t_cond.
    future_grid : ndarray, shape (n_future,)
        Grid points t_cond, t_{cond+1}, ..., T.
        The first element must equal past_grid[-1].
    Q, b0, M : process parameters
    kernel_fn : callable - scalar kernel
    n_inner : int - number of inner paths
    seed : int or None - RNG seed

    Returns
    -------
    future_paths : ndarray, shape (n_inner, n_future, d, d)
        Future path continuations (first time point = X_t_cond).
    """
    rng = np.random.default_rng(seed)
    n_past = len(past_grid)
    n_future = len(future_grid)
    d = Q.shape[0]
    dt_future = np.diff(future_grid)

    future_paths = np.zeros((n_inner, n_future, d, d))
    future_paths[:, 0] = past_X[-1]  # X at conditioning time

    # Future Brownian increments
    dW_fut = rng.standard_normal((n_inner, n_future - 1, d, d)) * np.sqrt(dt_future)[None, :, None, None]

    X0 = past_X[0]  # initial condition for the integral formula

    for fi in range(1, n_future):
        t_n = future_grid[fi]
        for p in range(n_inner):
            drift_sum = np.zeros((d, d))
            noise_sum = np.zeros((d, d))

            # Past contribution (j in past grid, excluding last point = conditioning time)
            for j in range(n_past - 1):
                Xj = past_X[j]
                Xj_sym = symmetrize(Xj)
                sqX, _ = psd_sqrt_eigh(Xj_sym, clip_neg=0.0)
                kw = kernel_fn(t_n - past_grid[j])
                dt_j = past_grid[j + 1] - past_grid[j]
                b_Xj = b0 + M @ Xj_sym + Xj_sym @ M.T
                drift_sum += kw * b_Xj * dt_j
                DW_j = past_dW[j]
                noise_sum += kw * (sqX @ DW_j @ Q + Q.T @ DW_j.T @ sqX)

            # Future contribution (j in future grid up to fi-1)
            for fj in range(fi):
                if fj == 0:
                    Xj = past_X[-1]  # X at conditioning time (shared with future[0])
                else:
                    Xj = future_paths[p, fj]
                Xj_sym = symmetrize(Xj)
                sqX, _ = psd_sqrt_eigh(Xj_sym, clip_neg=0.0)
                kw = kernel_fn(t_n - future_grid[fj])
                dt_j = dt_future[fj] if fj < len(dt_future) else 0.0
                b_Xj = b0 + M @ Xj_sym + Xj_sym @ M.T
                drift_sum += kw * b_Xj * dt_j
                if fj < fi - 1:
                    DW_j = dW_fut[p, fj]
                    noise_sum += kw * (sqX @ DW_j @ Q + Q.T @ DW_j.T @ sqX)

            Xn_raw = X0 + drift_sum + noise_sum
            future_paths[p, fi] = symmetrize(Xn_raw)

    return future_paths, dW_fut


def run_nested_mc_conditional(
    conditioning_time_idx: int,
    outer_path_X: np.ndarray,
    outer_dW: np.ndarray,
    grid: np.ndarray,
    Q: np.ndarray,
    b0: np.ndarray,
    M: np.ndarray,
    kernel_fn: Callable[[float], float],
    u: np.ndarray,
    f_running: np.ndarray,
    n_inner: int = 200,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Estimate E[exp(-Tr(uX_T) - int_t^T Tr(f(s)X_s)ds) | F_t] via nested MC.

    Uses the full path history up to t (non-Markovian conditioning).

    Parameters
    ----------
    conditioning_time_idx : int
        Index into grid at which we condition (t = grid[conditioning_time_idx]).
    outer_path_X : ndarray, shape (N, d, d)
        Full outer path realization X_{t_0}, ..., X_{t_{N-1}}.
    outer_dW : ndarray, shape (N-1, d, d)
        Outer path Brownian increments.
    u : ndarray, shape (d, d) - terminal loading
    f_running : ndarray, shape (N, d, d) - running loading
    n_inner : int - number of inner continuation paths
    seed : int or None

    Returns
    -------
    dict with:
        estimate : float - MC estimate of conditional expectation
        std_error : float - standard error
        ci_95 : (float, float) - 95% confidence interval
        n_inner : int
    """
    from .transforms import compute_running_integral

    N = len(grid)
    t_cond_idx = conditioning_time_idx

    past_X = outer_path_X[: t_cond_idx + 1]
    past_dW = outer_dW[:t_cond_idx]
    past_grid = grid[: t_cond_idx + 1]
    future_grid = grid[t_cond_idx:]
    n_future = len(future_grid)

    # Generate future continuations
    future_paths, dW_fut = continue_volterra_path(
        past_X=past_X, past_dW=past_dW,
        past_grid=past_grid, future_grid=future_grid,
        Q=Q, b0=b0, M=M, kernel_fn=kernel_fn,
        seed=seed, n_inner=n_inner,
    )
    # future_paths: (n_inner, n_future, d, d)

    # Compute payoffs from conditioning time to T
    f_future = f_running[t_cond_idx:]  # (n_future, d, d)
    terminal = np.array([
        np.trace(u @ future_paths[p, -1]) for p in range(n_inner)
    ])
    running = compute_running_integral(future_paths, future_grid, f_future)
    payoffs = np.exp(-(terminal + running))

    estimate = float(np.mean(payoffs))
    std_error = float(np.std(payoffs) / np.sqrt(n_inner))
    ci_95 = (estimate - 1.96 * std_error, estimate + 1.96 * std_error)

    return {
        "estimate": estimate,
        "std_error": std_error,
        "ci_95": ci_95,
        "n_inner": n_inner,
        "payoffs": payoffs,
    }
