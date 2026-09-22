"""Tests for src/kernels.py — kernel functions and admissibility validation."""
import os
import sys

import numpy as np
import pytest
from scipy.special import gamma

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.kernels import (
    k_markov,
    k_shifted_fractional,
    make_kernel_weights,
    validate_kernel_admissibility,
)


# ---------------------------------------------------------------------------
# k_markov
# ---------------------------------------------------------------------------

def test_k_markov_returns_one():
    for t in [0.0, 0.5, 1.0, 5.0, 100.0]:
        assert k_markov(t) == 1.0, f"k_markov({t}) should be 1.0"


# ---------------------------------------------------------------------------
# k_shifted_fractional
# ---------------------------------------------------------------------------

def test_k_shifted_fractional_at_zero():
    a, eps = 0.7, 0.1
    expected = eps ** (a - 1) / gamma(a)
    got = k_shifted_fractional(0.0, a=a, eps=eps)
    assert abs(got - expected) < 1e-12, f"k_frac(0) mismatch: {got} vs {expected}"


def test_k_shifted_fractional_positive():
    a, eps = 0.75, 0.05
    for t in [0.0, 0.1, 0.5, 1.0, 2.0]:
        assert k_shifted_fractional(t, a=a, eps=eps) > 0.0


def test_k_shifted_fractional_decreasing():
    a, eps = 0.7, 0.1
    vals = [k_shifted_fractional(t, a=a, eps=eps) for t in [0.0, 0.3, 0.7, 1.5]]
    for i in range(len(vals) - 1):
        assert vals[i] >= vals[i + 1], "k_frac should be nonincreasing"


def test_k_shifted_fractional_invalid_a_low():
    with pytest.raises((ValueError, AssertionError)):
        k_shifted_fractional(0.1, a=0.3, eps=0.1)


def test_k_shifted_fractional_invalid_a_high():
    with pytest.raises((ValueError, AssertionError)):
        k_shifted_fractional(0.1, a=1.0, eps=0.1)


def test_k_shifted_fractional_invalid_eps_zero():
    with pytest.raises((ValueError, AssertionError)):
        k_shifted_fractional(0.1, a=0.7, eps=0.0)


def test_k_shifted_fractional_invalid_eps_negative():
    with pytest.raises((ValueError, AssertionError)):
        k_shifted_fractional(0.1, a=0.7, eps=-0.1)


# ---------------------------------------------------------------------------
# make_kernel_weights
# ---------------------------------------------------------------------------

def test_make_kernel_weights_shape():
    grid = np.linspace(0, 1.0, 10)
    W = make_kernel_weights(grid, k_markov)
    assert W.shape == (10, 10), f"Expected (10,10), got {W.shape}"


def test_make_kernel_weights_lower_triangular():
    grid = np.linspace(0, 1.0, 8)
    W = make_kernel_weights(grid, k_markov)
    for n in range(8):
        for j in range(n, 8):
            assert W[n, j] == 0.0, f"W[{n},{j}] should be 0 (upper triangle)"


def test_make_kernel_weights_markov_values():
    """For k(t)=1, W[n,j] = k(t_n - t_j) = 1.0 for j < n."""
    grid = np.linspace(0, 1.0, 6)
    W = make_kernel_weights(grid, k_markov)
    for n in range(1, 6):
        for j in range(n):
            assert abs(W[n, j] - 1.0) < 1e-14, f"W[{n},{j}] should be 1.0"


# ---------------------------------------------------------------------------
# validate_kernel_admissibility
# ---------------------------------------------------------------------------

def test_validate_kernel_admissibility_admissible():
    """Conservative b0 >= k(0)*alpha should pass."""
    Q = np.eye(2) * 0.3
    alpha = Q.T @ Q          # 0.09 * I
    b0 = np.diag([0.5, 0.5])  # >> 0.09 * I
    grid = np.linspace(0, 1.0, 20)
    result = validate_kernel_admissibility(grid, k_markov, alpha, b0)
    assert result["k0_positive"], "k(0) must be positive"
    assert result["admissible"], "b0 should satisfy admissibility"


def test_validate_kernel_admissibility_inadmissible():
    """b0 below k(0)*alpha should fail the admissibility check."""
    Q = np.eye(2) * 0.3
    alpha = Q.T @ Q           # 0.09 * I
    b0 = np.diag([-0.1, -0.1])  # negative: clearly inadmissible
    grid = np.linspace(0, 1.0, 20)
    result = validate_kernel_admissibility(grid, k_markov, alpha, b0)
    assert not result["admissible"], "Should fail for b0 < k(0)*alpha"
