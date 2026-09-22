"""Tests for src/resolvent.py — resolvent of the first kind."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.kernels import k_markov, k_shifted_fractional
from src.resolvent import compute_resolvent, verify_resolvent_identity


GRID = np.linspace(0, 1.0, 30)


# ---------------------------------------------------------------------------
# compute_resolvent
# ---------------------------------------------------------------------------

def test_compute_resolvent_markov_ell0():
    """For k(t)=1, ell_0 = 1/k(0) = 1."""
    res = compute_resolvent(GRID, k_markov)
    assert abs(res["ell_0"] - 1.0) < 1e-12, f"ell_0={res['ell_0']} != 1"


def test_compute_resolvent_returns_shape():
    res = compute_resolvent(GRID, k_markov)
    assert res["ell"].shape == (len(GRID),)


def test_compute_resolvent_markov_ell_zero():
    """For k(t)=1, k'(t)=0 so ell(t)=0 for all t."""
    res = compute_resolvent(GRID, k_markov)
    # For k=1 (constant), dk=0, so ell should be identically 0
    max_ell = float(np.max(np.abs(res["ell"])))
    assert max_ell < 1e-10, f"Markovian resolvent ell should be ~0, got {max_ell}"


def test_compute_resolvent_fractional_shape():
    kernel_fn = lambda t: k_shifted_fractional(t, a=0.7, eps=0.1)
    res = compute_resolvent(GRID, kernel_fn)
    assert res["ell"].shape == (len(GRID),)
    assert res["k0"] > 0


def test_compute_resolvent_ell0_fractional():
    a, eps = 0.7, 0.1
    k0_expected = (0.0 + eps) ** (a - 1.0) / float(__import__("math").gamma(a))
    kernel_fn = lambda t: k_shifted_fractional(t, a=a, eps=eps)
    res = compute_resolvent(GRID, kernel_fn)
    assert abs(res["ell_0"] - 1.0 / k0_expected) < 1e-10


# ---------------------------------------------------------------------------
# verify_resolvent_identity
# ---------------------------------------------------------------------------

def test_verify_resolvent_identity_markov():
    """For k=1, resolvent identity k*L = 1 should hold exactly."""
    res = compute_resolvent(GRID, k_markov)
    check = verify_resolvent_identity(GRID, k_markov, res)
    assert "max_error" in check
    max_err = float(check["max_error"])
    assert max_err < 0.05, f"Markov resolvent identity error: {max_err}"


def test_verify_resolvent_identity_fractional():
    """For fractional kernel, error should be reasonably small."""
    a, eps = 0.7, 0.1
    kernel_fn = lambda t: k_shifted_fractional(t, a=a, eps=eps)
    res = compute_resolvent(GRID, kernel_fn)
    check = verify_resolvent_identity(GRID, kernel_fn, res)
    residual = check["residual"]
    max_err = float(np.max(np.abs(residual[1:])))   # skip t=0
    assert max_err < 0.15, f"Fractional resolvent identity error: {max_err}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_resolvent_k0_zero_raises():
    """k with k(0)=0 should raise ValueError."""
    bad_kernel = lambda t: t  # k(0) = 0
    with pytest.raises((ValueError, ZeroDivisionError)):
        compute_resolvent(GRID, bad_kernel)


def test_resolvent_two_point_grid():
    """Resolvent should work for a minimal 2-point grid."""
    g2 = np.array([0.0, 1.0])
    res = compute_resolvent(g2, k_markov)
    assert res["ell"].shape == (2,)
