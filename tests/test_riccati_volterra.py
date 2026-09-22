"""Tests for src/riccati_volterra.py."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.kernels import k_markov
from src.riccati_volterra import (
    _B_adjoint,
    solve_riccati_volterra,
    _compute_riccati_residual,
    solve_riccati_commodity,
)

# ---------------------------------------------------------------------------
# Common params
# ---------------------------------------------------------------------------
Q = np.eye(2) * 0.3
alpha = Q.T @ Q          # 0.09 * I
M = np.array([[-0.3, 0.05], [0.0, -0.3]])
u = np.diag([0.2, 0.15]).astype(complex)
grid = np.linspace(0, 1.0, 20)
N = len(grid)
f_running = np.zeros((N, 2, 2), dtype=complex)


# ---------------------------------------------------------------------------
# _B_adjoint
# ---------------------------------------------------------------------------

def test_B_adjoint_symmetry():
    """B^T(v) = M^T v + v M should preserve symmetric v."""
    v = np.array([[1.0, 0.3], [0.3, 2.0]])
    res = _B_adjoint(v, M)
    err = np.linalg.norm(res - res.T)
    assert err < 1e-14, f"B^T(v) not symmetric, err={err}"


def test_B_adjoint_linear():
    v1 = np.eye(2)
    v2 = np.array([[2.0, 0.5], [0.5, 1.0]])
    result1 = _B_adjoint(v1 + v2, M)
    result2 = _B_adjoint(v1, M) + _B_adjoint(v2, M)
    assert np.allclose(result1, result2), "B^T should be linear"


def test_B_adjoint_trace_pairing():
    """Verify Tr(B^T(v) x) = Tr(v B(x)) for B(x)=Mx+xM^T."""
    v = np.array([[1.0, 0.2], [0.2, 3.0]])
    x = np.array([[0.5, 0.1], [0.1, 0.8]])
    BT_v_x = np.trace(_B_adjoint(v, M) @ x)
    v_Bx = np.trace(v @ (M @ x + x @ M.T))
    assert abs(BT_v_x - v_Bx) < 1e-12, "Trace pairing identity failed"


# ---------------------------------------------------------------------------
# solve_riccati_volterra
# ---------------------------------------------------------------------------

def test_solve_riccati_returns_keys():
    res = solve_riccati_volterra(grid, u, f_running, M, alpha, k_markov)
    assert "psi" in res
    assert "residual_norms" in res


def test_solve_riccati_shape():
    res = solve_riccati_volterra(grid, u, f_running, M, alpha, k_markov)
    assert res["psi"].shape == (N, 2, 2), f"psi shape {res['psi'].shape} != ({N}, 2, 2)"


def test_solve_riccati_terminal_condition():
    """psi(T) should equal u * k(0)."""
    res = solve_riccati_volterra(grid, u, f_running, M, alpha, k_markov)
    psi_T = res["psi"][-1]
    expected = u * k_markov(0.0)
    err = np.linalg.norm(psi_T - expected, "fro")
    assert err < 1e-12, f"Terminal condition error: {err}"


def test_solve_riccati_symmetry():
    """psi(t) should be (complex) symmetric."""
    res = solve_riccati_volterra(grid, u, f_running, M, alpha, k_markov)
    psi = res["psi"]
    max_asym = max(np.linalg.norm(psi[n] - psi[n].T, "fro") for n in range(N))
    assert max_asym < 1e-12, f"psi not symmetric, max error={max_asym}"


def test_solve_riccati_residuals_small():
    res = solve_riccati_volterra(grid, u, f_running, M, alpha, k_markov)
    mean_res = float(np.mean(res["residual_norms"]))
    assert mean_res < 0.05, f"Mean Riccati residual {mean_res} too large"


def test_solve_riccati_real_psd_terminal_loading():
    """Real PSD u with real kernel should give real PSD psi at terminal time."""
    u_real = np.diag([0.2, 0.1]).astype(complex)
    f_zero = np.zeros((N, 2, 2), dtype=complex)
    res = solve_riccati_volterra(grid, u_real, f_zero, M, alpha, k_markov)
    psi_T = np.real(res["psi"][-1])
    eigs = np.linalg.eigvalsh(psi_T)
    assert eigs.min() >= -1e-10, f"psi(T) not PSD, min eig={eigs.min()}"


# ---------------------------------------------------------------------------
# solve_riccati_commodity
# ---------------------------------------------------------------------------

def test_solve_riccati_commodity_shape():
    """Smoke test: commodity Riccati should return an (N, 2, 2) psi."""
    kappa, T = 0.4, 1.0
    u3 = np.diag([0.1, 0.08]).astype(complex)
    u1, u2 = 0.0 + 0.0j, 0.0 + 0.0j   # Laplace case
    rho = np.array([0.0, 0.0])
    chi = np.zeros((N, 2), dtype=complex)
    f_com = np.zeros((N, 2, 2), dtype=complex)
    # M_chi_fn = M (constant, no correlation)
    M_chi_vals = np.stack([M.astype(complex)] * N, axis=0)

    def M_chi_fn(t):
        idx = int(np.searchsorted(grid, t, side="right")) - 1
        idx = max(0, min(idx, N - 1))
        return M_chi_vals[idx]

    res = solve_riccati_commodity(
        grid, u3, f_com, M_chi_fn, alpha, k_markov
    )
    assert "psi" in res
    assert res["psi"].shape == (N, 2, 2)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_solve_riccati_zero_u():
    """u=0 with f=0 should give psi=0."""
    u_zero = np.zeros((2, 2), dtype=complex)
    f_zero = np.zeros((N, 2, 2), dtype=complex)
    res = solve_riccati_volterra(grid, u_zero, f_zero, M, alpha, k_markov)
    max_norm = float(np.max(np.linalg.norm(res["psi"], axis=(1, 2))))
    assert max_norm < 1e-12, f"psi should be 0 for u=0, f=0, got max norm {max_norm}"


def test_solve_riccati_single_step():
    """Two-point grid should not raise."""
    g2 = np.array([0.0, 1.0])
    f2 = np.zeros((2, 2, 2), dtype=complex)
    res = solve_riccati_volterra(g2, u, f2, M, alpha, k_markov)
    assert res["psi"].shape == (2, 2, 2)
