"""Tests for src/commodity_model.py — Volterra-Wishart Gibson-Schwartz model."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.kernels import k_markov
from src.commodity_model import simulate_commodity

# ---------------------------------------------------------------------------
# Common params
# ---------------------------------------------------------------------------
Q = np.eye(2) * 0.25
b0 = np.diag([0.5, 0.5])
M = np.array([[-0.4, 0.0], [0.0, -0.4]])
V0 = np.array([[1.0, 0.2], [0.2, 0.8]])
kappa = 0.5
mu = 0.03
theta = 0.02
delta0 = 0.01
logS0 = 0.0
rho = np.array([0.1, 0.1])
grid = np.linspace(0, 1.0, 16)
N_PATHS = 20
SEED = 123


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

def test_simulate_commodity_output_keys():
    res = simulate_commodity(
        V0, delta0, logS0, grid, Q, b0, M, k_markov,
        kappa, mu, theta, rho,
        n_paths=N_PATHS, seed=SEED,
    )
    for key in ("V_paths", "delta_paths", "logS_paths", "dW", "grid"):
        assert key in res, f"Missing key '{key}'"


def test_simulate_commodity_shapes():
    res = simulate_commodity(
        V0, delta0, logS0, grid, Q, b0, M, k_markov,
        kappa, mu, theta, rho,
        n_paths=N_PATHS, seed=SEED,
    )
    N = len(grid)
    assert res["V_paths"].shape == (N_PATHS, N, 2, 2), \
        f"V_paths shape {res['V_paths'].shape}"
    assert res["delta_paths"].shape == (N_PATHS, N), \
        f"delta_paths shape {res['delta_paths'].shape}"
    assert res["logS_paths"].shape == (N_PATHS, N), \
        f"logS_paths shape {res['logS_paths'].shape}"


def test_simulate_commodity_initial_conditions():
    res = simulate_commodity(
        V0, delta0, logS0, grid, Q, b0, M, k_markov,
        kappa, mu, theta, rho,
        n_paths=N_PATHS, seed=SEED,
    )
    for p in range(N_PATHS):
        assert np.allclose(res["V_paths"][p, 0], V0, atol=1e-12), \
            f"V0 mismatch at path {p}"
        assert abs(res["delta_paths"][p, 0] - delta0) < 1e-12, \
            f"delta0 mismatch at path {p}"
        assert abs(res["logS_paths"][p, 0] - logS0) < 1e-12, \
            f"logS0 mismatch at path {p}"


def test_simulate_commodity_finite():
    res = simulate_commodity(
        V0, delta0, logS0, grid, Q, b0, M, k_markov,
        kappa, mu, theta, rho,
        n_paths=N_PATHS, seed=SEED,
    )
    assert np.all(np.isfinite(res["V_paths"])), "V_paths contains non-finite values"
    assert np.all(np.isfinite(res["delta_paths"])), "delta_paths non-finite"
    assert np.all(np.isfinite(res["logS_paths"])), "logS_paths non-finite"


def test_simulate_commodity_reproducibility():
    r1 = simulate_commodity(
        V0, delta0, logS0, grid, Q, b0, M, k_markov,
        kappa, mu, theta, rho,
        n_paths=N_PATHS, seed=SEED,
    )
    r2 = simulate_commodity(
        V0, delta0, logS0, grid, Q, b0, M, k_markov,
        kappa, mu, theta, rho,
        n_paths=N_PATHS, seed=SEED,
    )
    assert np.allclose(r1["V_paths"], r2["V_paths"], atol=1e-14)
    assert np.allclose(r1["logS_paths"], r2["logS_paths"], atol=1e-14)


def test_simulate_commodity_rho_zero():
    """Zero correlation should still work."""
    rho_zero = np.zeros(2)
    res = simulate_commodity(
        V0, delta0, logS0, grid, Q, b0, M, k_markov,
        kappa, mu, theta, rho_zero,
        n_paths=N_PATHS, seed=SEED,
    )
    assert np.all(np.isfinite(res["logS_paths"]))


def test_simulate_commodity_rho_full():
    """Full correlation rho^T rho = 1 (edge case) should not crash."""
    rho_full = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])
    res = simulate_commodity(
        V0, delta0, logS0, grid, Q, b0, M, k_markov,
        kappa, mu, theta, rho_full,
        n_paths=N_PATHS, seed=SEED,
    )
    assert np.all(np.isfinite(res["logS_paths"]))
