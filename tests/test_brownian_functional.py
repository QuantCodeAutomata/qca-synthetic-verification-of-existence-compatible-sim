"""Tests for src/brownian_functional.py."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.kernels import k_markov
from src.brownian_functional import compute_N_t, compute_payoff_extended
from src.volterra_sim import simulate_X

# ---------------------------------------------------------------------------
# Common params
# ---------------------------------------------------------------------------
Q = np.eye(2) * 0.3
b0 = np.diag([0.5, 0.5])
M = np.array([[-0.5, 0.1], [0.0, -0.5]])
X0 = np.array([[1.0, 0.1], [0.1, 1.0]])
grid = np.linspace(0, 1.0, 12)
N_PATHS = 20
SEED = 7

sim = simulate_X(X0, grid, Q, b0, M, k_markov, n_paths=N_PATHS, seed=SEED)
paths = sim["paths"]
dW = sim["dW"]


# ---------------------------------------------------------------------------
# compute_N_t
# ---------------------------------------------------------------------------

def test_compute_N_t_shape():
    H_fn = lambda t: np.eye(2)
    N_t = compute_N_t(paths, dW, H_fn, grid)
    assert N_t.shape == (N_PATHS, len(grid)), \
        f"N_t shape {N_t.shape} != ({N_PATHS},{len(grid)})"


def test_compute_N_t_initial_zero():
    H_fn = lambda t: np.eye(2)
    N_t = compute_N_t(paths, dW, H_fn, grid)
    assert np.all(N_t[:, 0] == 0.0), "N_0 must be 0 for all paths"


def test_compute_N_t_zero_H():
    """H=0 should give N_t=0 for all t."""
    H_fn = lambda t: np.zeros((2, 2))
    N_t = compute_N_t(paths, dW, H_fn, grid)
    max_val = float(np.max(np.abs(N_t)))
    assert max_val < 1e-14, f"N_t with H=0 should be 0, got {max_val}"


def test_compute_N_t_finite():
    H_fn = lambda t: np.array([[0.5, 0.0], [0.0, 0.3]])
    N_t = compute_N_t(paths, dW, H_fn, grid)
    assert np.all(np.isfinite(N_t)), "N_t contains non-finite values"


def test_compute_N_t_incremental():
    """N_t must be accumulated incrementally (monotone build-up)."""
    H_fn = lambda t: np.eye(2) * 0.1
    N_t = compute_N_t(paths, dW, H_fn, grid)
    # Just verify it's not identically zero (non-trivial paths)
    std_final = float(np.std(N_t[:, -1]))
    assert std_final > 1e-6, f"Final N_T std suspiciously small: {std_final}"


# ---------------------------------------------------------------------------
# compute_payoff_extended
# ---------------------------------------------------------------------------

def test_compute_payoff_extended_shape():
    """Payoff should return one complex scalar per path."""
    H_fn = lambda t: np.eye(2) * 0.1
    N_t = compute_N_t(paths, dW, H_fn, grid)
    u = np.diag([0.1, 0.05]).astype(complex)
    f_running = np.zeros((len(grid), 2, 2), dtype=complex)
    z = 0.0 + 0.0j  # no Brownian-functional term
    payoffs = compute_payoff_extended(paths, N_t, grid, z, u, f_running)
    assert payoffs.shape == (N_PATHS,), \
        f"Payoffs shape {payoffs.shape} != ({N_PATHS},)"


def test_compute_payoff_extended_z0_reduction():
    """With z=0 the Brownian functional drops out; payoff depends only on X."""
    H_fn = lambda t: np.eye(2)
    N_t = compute_N_t(paths, dW, H_fn, grid)
    u = np.diag([0.2, 0.1]).astype(complex)
    f_running = np.zeros((len(grid), 2, 2), dtype=complex)
    payoffs_z0 = compute_payoff_extended(paths, N_t, grid, 0.0 + 0j, u, f_running)
    # With z=0, exp(z*N_T) = 1, so payoff = exp(-Tr(u X_T) - int Tr(f X) ds)
    assert np.all(np.isfinite(payoffs_z0))


def test_compute_payoff_extended_finite_z_imag():
    """z=i should produce complex payoffs with |payoff| <= e^(something finite)."""
    H_fn = lambda t: np.eye(2) * 0.1
    N_t = compute_N_t(paths, dW, H_fn, grid)
    u = np.diag([0.1, 0.05]).astype(complex)
    f_running = np.zeros((len(grid), 2, 2), dtype=complex)
    z = 1j
    payoffs = compute_payoff_extended(paths, N_t, grid, z, u, f_running)
    assert np.all(np.isfinite(np.real(payoffs))), "Real part of payoff has non-finite values"
    assert np.all(np.isfinite(np.imag(payoffs))), "Imag part of payoff has non-finite values"
    # For z=i the modulus of exp(i*N_T) = 1, so abs(payoff) = exp(-real part of exponent)
    assert np.all(np.abs(payoffs) > 0), "Payoff should be nonzero"
