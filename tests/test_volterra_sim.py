"""Tests for src/volterra_sim.py — Euler-Volterra path simulation."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.kernels import k_markov, k_shifted_fractional
from src.volterra_sim import simulate_X, simulate_V

# ---------------------------------------------------------------------------
# Common test parameters (small grid, few paths for speed)
# ---------------------------------------------------------------------------
Q = np.eye(2) * 0.3       # alpha = 0.09 * I
b0 = np.diag([0.5, 0.5])  # admissible: b0 >> k(0)*alpha = 0.09*I
M = np.array([[-0.5, 0.1], [0.0, -0.5]])
X0 = np.array([[1.0, 0.1], [0.1, 1.0]])
V0 = np.array([[1.0, 0.2], [0.2, 0.8]])
grid = np.linspace(0, 1.0, 16)
N_PATHS = 30
SEED = 42


# ---------------------------------------------------------------------------
# simulate_X
# ---------------------------------------------------------------------------

def test_simulate_X_output_keys():
    res = simulate_X(X0, grid, Q, b0, M, k_markov, n_paths=N_PATHS, seed=SEED)
    for key in ("paths", "paths_projected", "min_eigs", "sym_errors", "dW", "grid"):
        assert key in res, f"Key '{key}' missing from simulate_X output"


def test_simulate_X_initial_condition():
    res = simulate_X(X0, grid, Q, b0, M, k_markov, n_paths=N_PATHS, seed=SEED)
    paths = res["paths"]
    for p in range(N_PATHS):
        assert np.allclose(paths[p, 0], X0, atol=1e-12), \
            f"Path {p} initial condition mismatch"


def test_simulate_X_shape():
    res = simulate_X(X0, grid, Q, b0, M, k_markov, n_paths=N_PATHS, seed=SEED)
    d = 2
    assert res["paths"].shape == (N_PATHS, len(grid), d, d), \
        f"Expected ({N_PATHS},{len(grid)},2,2), got {res['paths'].shape}"


def test_simulate_X_symmetry():
    res = simulate_X(X0, grid, Q, b0, M, k_markov, n_paths=N_PATHS, seed=SEED)
    sym_errs = res["sym_errors"]
    mean_sym_err = float(np.mean(sym_errs[:, 1:]))  # skip t=0
    assert mean_sym_err < 1e-12, f"Mean symmetry error too large: {mean_sym_err}"


def test_simulate_X_psd_violation_rate_low():
    """With conservative params, PSD violations should be rare (< 20%)."""
    res = simulate_X(X0, grid, Q, b0, M, k_markov, n_paths=N_PATHS, seed=SEED)
    violation_rate = float(np.mean(res["psd_violation_raw"][:, 1:]))
    assert violation_rate < 0.20, f"PSD violation rate too high: {violation_rate:.3f}"


def test_simulate_X_dW_shape():
    res = simulate_X(X0, grid, Q, b0, M, k_markov, n_paths=N_PATHS, seed=SEED)
    expected = (N_PATHS, len(grid) - 1, 2, 2)
    assert res["dW"].shape == expected, f"dW shape {res['dW'].shape} != {expected}"


def test_simulate_X_reproducibility():
    r1 = simulate_X(X0, grid, Q, b0, M, k_markov, n_paths=N_PATHS, seed=SEED)
    r2 = simulate_X(X0, grid, Q, b0, M, k_markov, n_paths=N_PATHS, seed=SEED)
    assert np.allclose(r1["paths"], r2["paths"], atol=1e-14), "Same seed should give same paths"


# ---------------------------------------------------------------------------
# simulate_V
# ---------------------------------------------------------------------------

def test_simulate_V_output_keys():
    res = simulate_V(V0, grid, Q, b0, M, k_markov, n_paths=N_PATHS, seed=SEED)
    for key in ("paths", "min_eigs", "sym_errors", "dW", "grid"):
        assert key in res, f"Key '{key}' missing from simulate_V output"


def test_simulate_V_initial_condition():
    res = simulate_V(V0, grid, Q, b0, M, k_markov, n_paths=N_PATHS, seed=SEED)
    for p in range(N_PATHS):
        assert np.allclose(res["paths"][p, 0], V0, atol=1e-12), \
            f"V path {p} initial condition mismatch"


def test_simulate_V_shape():
    res = simulate_V(V0, grid, Q, b0, M, k_markov, n_paths=N_PATHS, seed=SEED)
    assert res["paths"].shape == (N_PATHS, len(grid), 2, 2)


def test_simulate_V_reproducibility():
    r1 = simulate_V(V0, grid, Q, b0, M, k_markov, n_paths=N_PATHS, seed=SEED)
    r2 = simulate_V(V0, grid, Q, b0, M, k_markov, n_paths=N_PATHS, seed=SEED)
    assert np.allclose(r1["paths"], r2["paths"], atol=1e-14)


def test_simulate_V_psd_violation_rate_low():
    res = simulate_V(V0, grid, Q, b0, M, k_markov, n_paths=N_PATHS, seed=SEED)
    viol_rate = float(np.mean(res["psd_violations"][:, 1:]))
    assert viol_rate < 0.20, f"V PSD violation rate: {viol_rate:.3f}"
