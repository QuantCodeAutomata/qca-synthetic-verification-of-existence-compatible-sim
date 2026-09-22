"""Tests for src/commodity_transform.py — Fourier-Laplace transform."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.kernels import k_markov
from src.commodity_transform import build_deterministic_objects, solve_commodity_riccati

# ---------------------------------------------------------------------------
# Common params
# ---------------------------------------------------------------------------
Q = np.eye(2) * 0.25
alpha = Q.T @ Q
M = np.array([[-0.4, 0.0], [0.0, -0.4]])
kappa = 0.5
mu = 0.03
theta = 0.02
delta0 = 0.01
logS0 = 0.0
rho = np.array([0.1, 0.1])
grid = np.linspace(0, 1.0, 15)
N = len(grid)
T = grid[-1]

u1 = 0.0 + 0.0j   # Laplace (covariance only) reduction
u2 = 0.0 + 0.0j
u3 = np.diag([0.1, 0.08]).astype(complex)


# ---------------------------------------------------------------------------
# build_deterministic_objects
# ---------------------------------------------------------------------------

def test_build_deterministic_objects_keys():
    obj = build_deterministic_objects(
        grid, u1, u2, u3, kappa, mu, theta, delta0, logS0, rho, Q, M,
    )
    for key in ("C_T", "chi", "f_running", "M_chi"):
        assert key in obj, f"Missing key '{key}'"


def test_build_deterministic_objects_chi_shape():
    obj = build_deterministic_objects(
        grid, u1, u2, u3, kappa, mu, theta, delta0, logS0, rho, Q, M,
    )
    assert obj["chi"].shape == (N, 2), \
        f"chi shape {obj['chi'].shape} != ({N}, 2)"


def test_build_deterministic_objects_f_running_shape():
    obj = build_deterministic_objects(
        grid, u1, u2, u3, kappa, mu, theta, delta0, logS0, rho, Q, M,
    )
    assert obj["f_running"].shape == (N, 2, 2), \
        f"f_running shape {obj['f_running'].shape} != ({N}, 2, 2)"


def test_build_deterministic_objects_C_T_zero_u1_u2():
    """With u1=u2=0, C_T should be zero."""
    obj = build_deterministic_objects(
        grid, 0.0 + 0j, 0.0 + 0j, u3, kappa, mu, theta, delta0, logS0, rho, Q, M,
    )
    assert abs(obj["C_T"]) < 1e-12, f"C_T should be 0 for u1=u2=0, got {obj['C_T']}"


def test_build_deterministic_objects_chi_zero_u1_u2():
    """With u1=u2=0, chi(s) should be zero vector for all s."""
    obj = build_deterministic_objects(
        grid, 0.0 + 0j, 0.0 + 0j, u3, kappa, mu, theta, delta0, logS0, rho, Q, M,
    )
    max_chi = float(np.max(np.abs(obj["chi"])))
    assert max_chi < 1e-12, f"chi should be 0 for u1=u2=0, max={max_chi}"


def test_build_M_chi_shape():
    """M_chi should be (N, 2, 2) array."""
    obj = build_deterministic_objects(
        grid, u1, u2, u3, kappa, mu, theta, delta0, logS0, rho, Q, M,
    )
    assert obj["M_chi"].shape == (N, 2, 2), \
        f"M_chi shape {obj['M_chi'].shape} != ({N}, 2, 2)"


# ---------------------------------------------------------------------------
# solve_commodity_riccati
# ---------------------------------------------------------------------------

def test_solve_commodity_riccati_shape():
    """Commodity Riccati should return psi with shape (N, 2, 2)."""
    obj = build_deterministic_objects(
        grid, u1, u2, u3, kappa, mu, theta, delta0, logS0, rho, Q, M,
    )
    res = solve_commodity_riccati(grid, u3, obj, alpha, k_markov)
    assert "psi" in res, "Missing key 'psi'"
    assert res["psi"].shape == (N, 2, 2), \
        f"psi shape {res['psi'].shape} != ({N}, 2, 2)"


def test_solve_commodity_riccati_terminal():
    """psi(T) should equal u3 * k(0) = u3 for Markovian kernel."""
    obj = build_deterministic_objects(
        grid, u1, u2, u3, kappa, mu, theta, delta0, logS0, rho, Q, M,
    )
    res = solve_commodity_riccati(grid, u3, obj, alpha, k_markov)
    psi_T = res["psi"][-1]
    expected = u3 * k_markov(0.0)
    err = np.linalg.norm(psi_T - expected, "fro")
    assert err < 1e-12, f"Terminal condition error: {err}"


def test_solve_commodity_riccati_symmetry():
    """psi(t) should be symmetric for all t."""
    obj = build_deterministic_objects(
        grid, u1, u2, u3, kappa, mu, theta, delta0, logS0, rho, Q, M,
    )
    res = solve_commodity_riccati(grid, u3, obj, alpha, k_markov)
    psi = res["psi"]
    max_asym = max(np.linalg.norm(psi[n] - psi[n].T, "fro") for n in range(N))
    assert max_asym < 1e-12, f"psi not symmetric, max error={max_asym}"


def test_solve_commodity_riccati_residuals_small():
    """Riccati residuals should be reasonably small."""
    obj = build_deterministic_objects(
        grid, u1, u2, u3, kappa, mu, theta, delta0, logS0, rho, Q, M,
    )
    res = solve_commodity_riccati(grid, u3, obj, alpha, k_markov)
    mean_res = float(np.mean(res["residual_norms"]))
    assert mean_res < 0.1, f"Mean commodity Riccati residual {mean_res} too large"


def test_solve_commodity_riccati_zero_u3():
    """u3=0 with u1=u2=0 and f=0 → psi=0."""
    u3_zero = np.zeros((2, 2), dtype=complex)
    obj = build_deterministic_objects(
        grid, 0.0 + 0j, 0.0 + 0j, u3_zero, kappa, mu, theta, delta0, logS0,
        np.zeros(2), Q, M,
    )
    res = solve_commodity_riccati(grid, u3_zero, obj, alpha, k_markov)
    max_norm = float(np.max(np.linalg.norm(res["psi"], axis=(1, 2))))
    assert max_norm < 1e-12, f"psi should be 0 for u3=0, got max norm {max_norm}"
