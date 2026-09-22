"""Tests for src/psd.py — PSD matrix operations."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.psd import (
    symmetrize,
    psd_sqrt_eigh,
    min_eig,
    project_to_psd,
    check_psd,
    symmetry_error,
)


# ---------------------------------------------------------------------------
# symmetrize
# ---------------------------------------------------------------------------

def test_symmetrize_symmetric():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((2, 2))
    S = symmetrize(A)
    assert np.allclose(S, S.T, atol=1e-14), "symmetrize must return symmetric matrix"


def test_symmetrize_complex_no_conjugate():
    """symmetrize uses (A + A.T)/2, NOT (A + A.conj().T)/2."""
    A = np.array([[1 + 2j, 3j], [4, 5 - 1j]])
    S = symmetrize(A)
    expected = (A + A.T) / 2
    assert np.allclose(S, expected, atol=1e-14), "symmetrize must be (A+A.T)/2"


def test_symmetrize_idempotent():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((3, 3))
    S = symmetrize(A)
    assert np.allclose(symmetrize(S), S, atol=1e-14)


# ---------------------------------------------------------------------------
# psd_sqrt_eigh
# ---------------------------------------------------------------------------

def test_psd_sqrt_eigh_identity():
    sqI, evals = psd_sqrt_eigh(np.eye(2))
    assert np.allclose(sqI, np.eye(2), atol=1e-12), "sqrt(I) == I"


def test_psd_sqrt_eigh_known_diagonal():
    A = np.diag([4.0, 9.0])
    sqA, _ = psd_sqrt_eigh(A)
    expected = np.diag([2.0, 3.0])
    assert np.allclose(sqA, expected, atol=1e-10), f"sqrt(diag(4,9)) = {sqA}"


def test_psd_sqrt_eigh_reconstructs():
    rng = np.random.default_rng(2)
    L = rng.standard_normal((2, 2))
    A = L @ L.T + 0.1 * np.eye(2)
    sqA, _ = psd_sqrt_eigh(A)
    assert np.allclose(sqA @ sqA.T, A, atol=1e-10), "sqrt(A)@sqrt(A).T should recover A"


def test_psd_sqrt_eigh_clips_negative():
    """Tiny negative eigenvalues from roundoff should be clipped before sqrt."""
    A = np.array([[1.0, 0.0], [0.0, -1e-15]])  # nearly singular
    sqA, evals = psd_sqrt_eigh(A, clip_neg=0.0)
    # Raw eigenvalues are returned as-is; the negative one is clipped only for sqrt
    assert np.all(np.isfinite(sqA)), "sqrt result must be finite (no NaN from negative eig)"
    # Resulting sqA should be PSD (clip maps -1e-15 -> 0, so sqA@sqA recovers diag(1,0))
    assert np.all(np.linalg.eigvalsh(sqA @ sqA.T) >= -1e-12), "sqA@sqA.T must be PSD"


def test_psd_sqrt_eigh_batch_shape():
    rng = np.random.default_rng(3)
    A = np.eye(2) + rng.standard_normal((2, 2)) * 0.1
    A = symmetrize(A)
    sqA, _ = psd_sqrt_eigh(A)
    assert sqA.shape == (2, 2), f"Shape should be (2,2), got {sqA.shape}"


# ---------------------------------------------------------------------------
# min_eig
# ---------------------------------------------------------------------------

def test_min_eig_positive():
    A = np.array([[2.0, 0.5], [0.5, 3.0]])
    assert min_eig(A) > 0, "PD matrix min eigenvalue must be > 0"


def test_min_eig_zero():
    A = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank-deficient
    assert abs(min_eig(A)) < 1e-12, f"Rank-1 PSD matrix: min_eig = {min_eig(A)}"


def test_min_eig_negative():
    A = np.array([[1.0, 2.0], [2.0, 1.0]])  # indefinite
    assert min_eig(A) < 0


# ---------------------------------------------------------------------------
# project_to_psd
# ---------------------------------------------------------------------------

def test_project_to_psd_nonneg_eigs():
    A = np.array([[1.0, 2.0], [2.0, 1.0]])  # indefinite
    P = project_to_psd(A)
    evals = np.linalg.eigvalsh(P)
    assert np.all(evals >= -1e-14), f"Projected eigs should be >= 0: {evals}"


def test_project_to_psd_pd_unchanged():
    A = np.array([[3.0, 0.5], [0.5, 2.0]])
    P = project_to_psd(A)
    assert np.allclose(P, A, atol=1e-12), "PD matrix should not be changed"


# ---------------------------------------------------------------------------
# check_psd
# ---------------------------------------------------------------------------

def test_check_psd_true():
    A = np.array([[2.0, 0.0], [0.0, 1.0]])
    assert check_psd(A), "Diagonal PD matrix should be PSD"


def test_check_psd_false():
    A = np.array([[1.0, 2.0], [2.0, 1.0]])  # indefinite
    assert not check_psd(A), "Indefinite matrix should fail PSD check"


# ---------------------------------------------------------------------------
# symmetry_error
# ---------------------------------------------------------------------------

def test_symmetry_error_zero():
    A = np.array([[1.0, 0.5], [0.5, 2.0]])
    assert symmetry_error(A) < 1e-14, "Symmetric matrix has zero symmetry error"


def test_symmetry_error_positive():
    A = np.array([[1.0, 0.0], [1.0, 2.0]])  # asymmetric
    assert symmetry_error(A) > 0.0, "Asymmetric matrix has positive symmetry error"
