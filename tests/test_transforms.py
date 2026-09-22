"""Tests for src/transforms.py — Y_t direct representation and payoff."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.kernels import k_markov
from src.volterra_sim import simulate_X
from src.riccati_volterra import solve_riccati_volterra
from src.transforms import (
    compute_Y0_theoretical,
    compute_Y_t_direct,
    compute_payoff_laplace,
)

# ---------------------------------------------------------------------------
# Common setup
# ---------------------------------------------------------------------------
Q = np.eye(2) * 0.3
alpha = Q.T @ Q
b0 = np.diag([0.5, 0.5])
M = np.array([[-0.5, 0.1], [0.0, -0.5]])
X0 = np.array([[1.0, 0.1], [0.1, 1.0]])
grid = np.linspace(0, 1.0, 15)
N = len(grid)
N_PATHS = 30
SEED = 99
u = np.diag([0.2, 0.1]).astype(complex)
f_running = np.zeros((N, 2, 2), dtype=complex)

sim = simulate_X(X0, grid, Q, b0, M, k_markov, n_paths=N_PATHS, seed=SEED)
paths, dW = sim["paths"], sim["dW"]
riccati = solve_riccati_volterra(grid, u, f_running, M, alpha, k_markov)
psi = riccati["psi"]


# ---------------------------------------------------------------------------
# compute_Y0_theoretical
# ---------------------------------------------------------------------------

def test_Y0_theoretical_is_scalar():
    y0 = compute_Y0_theoretical(X0, psi, grid, b0, M, alpha, f_running, k_markov)
    assert np.isscalar(y0) or y0.ndim == 0, f"Y0 should be scalar, got {y0}"


def test_Y0_theoretical_real_for_real_psd_u():
    """With real PSD u and f=0, k=1, Y0 should be real and positive."""
    y0 = compute_Y0_theoretical(X0, psi, grid, b0, M, alpha, f_running, k_markov)
    assert abs(np.imag(y0)) < 1e-10, f"Y0 should be real, imag={np.imag(y0)}"
    assert np.real(y0) > 0, f"Y0 should be positive, got {np.real(y0)}"


def test_Y0_theoretical_finite():
    y0 = compute_Y0_theoretical(X0, psi, grid, b0, M, alpha, f_running, k_markov)
    assert np.isfinite(y0), f"Y0 is not finite: {y0}"


# ---------------------------------------------------------------------------
# compute_terminal_payoff
# ---------------------------------------------------------------------------

def test_terminal_payoff_shape():
    payoffs = compute_payoff_laplace(paths, grid, u, f_running)
    assert payoffs.shape == (N_PATHS,), \
        f"Payoffs shape {payoffs.shape} != ({N_PATHS},)"


def test_terminal_payoff_real_values():
    """With real PSD u, f=0, payoffs are in (0, 1]."""
    payoffs = compute_payoff_laplace(paths, grid, u, f_running)
    real_part = np.real(payoffs)
    assert np.all(real_part > 0), "exp(-...) should be positive"
    assert np.all(real_part <= 1.0 + 1e-9), "exp(-...) should be <= 1 for nonneg u"


def test_terminal_payoff_finite():
    payoffs = compute_payoff_laplace(paths, grid, u, f_running)
    assert np.all(np.isfinite(np.real(payoffs))), "Payoffs should be finite"


# ---------------------------------------------------------------------------
# compute_Y_t_direct
# ---------------------------------------------------------------------------

def test_Y_t_direct_shape():
    Y_t = compute_Y_t_direct(paths, dW, psi, grid, Q, alpha, b0, M, f_running, k_markov)
    assert Y_t.shape == (N_PATHS, N), \
        f"Y_t shape {Y_t.shape} != ({N_PATHS},{N})"


def test_Y_t_direct_initial_approx_Y0():
    """Y_t at t=0 should be close to the theoretical Y_0 for all paths."""
    Y_t = compute_Y_t_direct(paths, dW, psi, grid, Q, alpha, b0, M, f_running, k_markov)
    y0_theo = compute_Y0_theoretical(X0, psi, grid, b0, M, alpha, f_running, k_markov)
    # Each path's Y_t[:, 0] should equal the same Y_0 (deterministic start)
    max_dev = float(np.max(np.abs(np.real(Y_t[:, 0]) - np.real(y0_theo))))
    assert max_dev < 1.0, f"Y_0 deviation from theoretical Y_0: {max_dev}"


def test_Y_t_direct_finite():
    Y_t = compute_Y_t_direct(paths, dW, psi, grid, Q, alpha, b0, M, f_running, k_markov)
    assert np.all(np.isfinite(np.real(Y_t))), "Y_t contains non-finite values"


# ---------------------------------------------------------------------------
# Consistency: unconditional transform
# ---------------------------------------------------------------------------

def test_unconditional_transform_consistency():
    """E[exp(-payoff)] from MC should roughly match exp(-Y0)."""
    payoffs = compute_payoff_laplace(paths, grid, u, f_running)
    mc_mean = float(np.mean(np.real(payoffs)))
    y0 = compute_Y0_theoretical(X0, psi, grid, b0, M, alpha, f_running, k_markov)
    theo_val = float(np.real(np.exp(-y0)))
    # Relative error should be reasonable with 30 paths / coarse grid
    rel_err = abs(mc_mean - theo_val) / (abs(theo_val) + 1e-8)
    assert rel_err < 2.0, f"Unconditional transform relative error too large: {rel_err:.3f}"
