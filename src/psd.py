"""Positive-semidefinite (PSD) matrix operations.

Utilities for symmetrization, eigendecomposition-based square roots,
projection onto the PSD cone, and diagnostics.
"""
import numpy as np
from scipy.linalg import eigh
from typing import Tuple


def symmetrize(A: np.ndarray) -> np.ndarray:
    """Symmetrize a real matrix: returns (A + A.T) / 2."""
    return (A + A.T) / 2.0


def symmetrize_complex(A: np.ndarray) -> np.ndarray:
    """Symmetric complex version: (A + A.T) / 2 (no conjugation).

    The paper works with complex symmetric matrices (A = A.T),
    NOT Hermitian matrices (A = A^*). Hence we do NOT conjugate.
    """
    return (A + A.T) / 2.0


def psd_sqrt_eigh(
    A: np.ndarray, clip_neg: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute PSD matrix square root via eigendecomposition.

    Parameters
    ----------
    A : ndarray, shape (d, d)
        Symmetric (or nearly symmetric) matrix.
    clip_neg : float
        Negative eigenvalues are clipped to this value before taking sqrt.
        Default 0.0 gives the PSD-projected square root.

    Returns
    -------
    sqrt_A : ndarray, shape (d, d)
        Matrix square root of the (clipped) matrix.
    eigvals : ndarray, shape (d,)
        Raw (unclipped) eigenvalues from eigh.
    """
    A_sym = symmetrize(A)
    eigvals, eigvecs = eigh(A_sym)
    eigvals_clipped = np.maximum(eigvals, clip_neg)
    sqrt_A = eigvecs @ np.diag(np.sqrt(eigvals_clipped)) @ eigvecs.T
    return sqrt_A, eigvals


def psd_sqrt_eigh_batch(A: np.ndarray, clip_neg: float = 0.0) -> np.ndarray:
    """Batched PSD matrix square root for arrays of shape (..., d, d).

    Uses numpy's batched eigh for efficiency over many paths simultaneously.

    Parameters
    ----------
    A : ndarray, shape (..., d, d)
        Batch of symmetric matrices.
    clip_neg : float
        Clip eigenvalues below this threshold before sqrt.

    Returns
    -------
    sqrt_A : ndarray, shape (..., d, d)
    """
    A_sym = (A + A.swapaxes(-2, -1)) / 2.0
    eigvals, eigvecs = np.linalg.eigh(A_sym)  # batched
    eigvals_c = np.maximum(eigvals, clip_neg)
    # Reconstruct: eigvecs @ diag(sqrt(eigvals_c)) @ eigvecs.T
    sq = np.sqrt(eigvals_c)  # (..., d)
    return np.einsum("...ij,...j,...kj->...ik", eigvecs, sq, eigvecs)


def min_eig(A: np.ndarray) -> float:
    """Return the minimum eigenvalue of a symmetric matrix."""
    A_sym = symmetrize(A)
    return float(np.linalg.eigvalsh(A_sym).min())


def project_to_psd(A: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone by clipping negative eigenvalues."""
    A_sym = symmetrize(A)
    eigvals, eigvecs = eigh(A_sym)
    eigvals_clipped = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T


def project_to_psd_batch(A: np.ndarray) -> np.ndarray:
    """Batched PSD projection for arrays of shape (..., d, d)."""
    A_sym = (A + A.swapaxes(-2, -1)) / 2.0
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    eigvals_c = np.maximum(eigvals, 0.0)
    return np.einsum("...ij,...j,...kj->...ik", eigvecs, eigvals_c, eigvecs)


def check_psd(A: np.ndarray, tol: float = -1e-8) -> bool:
    """Return True if minimum eigenvalue >= tol (approximately PSD)."""
    return min_eig(A) >= tol


def symmetry_error(A: np.ndarray) -> float:
    """Return Frobenius norm of A - A.T as a measure of asymmetry."""
    return float(np.linalg.norm(A - A.T, "fro"))
