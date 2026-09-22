"""Diagnostic utilities for PSD simulation experiments.

Provides functions to summarize PSD diagnostics, compare kernels,
assess grid refinement convergence, and generate plots.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List


def summarize_psd_diagnostics(
    min_eigs: np.ndarray,
    grid: np.ndarray,
) -> Dict[str, float]:
    """Compute PSD violation statistics over all paths and time points.

    Parameters
    ----------
    min_eigs : ndarray, shape (n_paths, N)
        Minimum eigenvalues at each path/time.
    grid : ndarray, shape (N,)
        Time grid (used for context, not computation).

    Returns
    -------
    dict with:
        violation_rate: fraction of (path, time) pairs where min_eig < -1e-10
        mean_min_eig: mean of minimum eigenvalue over all paths/times
        p5_min_eig: 5th percentile of minimum eigenvalue
        p1_min_eig: 1st percentile
        min_of_min: global minimum eigenvalue across all paths/times
        n_paths: number of paths
        N: number of time steps
    """
    n_paths, N = min_eigs.shape
    flat = min_eigs.ravel()
    violation_rate = float(np.mean(flat < -1e-10))
    return {
        "violation_rate": violation_rate,
        "mean_min_eig": float(np.mean(flat)),
        "p5_min_eig": float(np.percentile(flat, 5)),
        "p1_min_eig": float(np.percentile(flat, 1)),
        "min_of_min": float(np.min(flat)),
        "n_paths": n_paths,
        "N": N,
    }


def compare_kernel_diagnostics(results_dict: Dict) -> Dict[str, Any]:
    """Compare PSD diagnostics across different kernel scenarios.

    Parameters
    ----------
    results_dict : dict
        Keys are scenario identifiers (tuples or strings).
        Values are dicts with 'diag' key containing summarize_psd_diagnostics output.

    Returns
    -------
    comparison : dict mapping scenario keys to diagnostic summaries.
    """
    comparison = {}
    for key, val in results_dict.items():
        diag = val.get("diag", {})
        comparison[str(key)] = {
            "violation_rate": diag.get("violation_rate", float("nan")),
            "mean_min_eig": diag.get("mean_min_eig", float("nan")),
            "p5_min_eig": diag.get("p5_min_eig", float("nan")),
        }
    return comparison


def compute_grid_refinement_comparison(
    coarse_results: Dict[str, Any],
    fine_results: Dict[str, Any],
) -> Dict[str, float]:
    """Compare PSD violation rates between coarse and fine grids.

    Parameters
    ----------
    coarse_results, fine_results : dicts with 'diag' key.

    Returns
    -------
    dict with rate_coarse, rate_fine, and improvement ratio.
    """
    rate_c = coarse_results["diag"]["violation_rate"]
    rate_f = fine_results["diag"]["violation_rate"]
    improvement = rate_c - rate_f if rate_c > 0 else 0.0
    return {
        "rate_coarse": rate_c,
        "rate_fine": rate_f,
        "improvement": improvement,
        "converging": bool(rate_f <= rate_c),
    }


def plot_sample_paths(
    paths: np.ndarray,
    grid: np.ndarray,
    title: str = "Sample Paths",
    save_path: Optional[str] = None,
    n_show: int = 5,
) -> None:
    """Plot sample paths of matrix entries over time.

    Parameters
    ----------
    paths : ndarray, shape (n_paths, N, d, d)
        Simulated paths.
    grid : ndarray, shape (N,)
        Time grid.
    title : str
        Plot title.
    save_path : str or None
        If given, save the figure to this path.
    n_show : int
        Number of sample paths to plot.
    """
    d = paths.shape[2]
    fig, axes = plt.subplots(d, d, figsize=(5 * d, 4 * d), squeeze=False)
    colors = plt.cm.tab10.colors
    for i in range(d):
        for j_idx in range(d):
            ax = axes[i][j_idx]
            for p in range(min(n_show, paths.shape[0])):
                ax.plot(grid, paths[p, :, i, j_idx], alpha=0.6, color=colors[p % len(colors)])
            ax.set_title(f"$X_{{{i+1},{j_idx+1}}}$")
            ax.set_xlabel("$t$")
            ax.grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_min_eigenvalue(
    min_eigs: np.ndarray,
    grid: np.ndarray,
    title: str = "Minimum Eigenvalue",
    save_path: Optional[str] = None,
    n_show: int = 20,
) -> None:
    """Plot minimum eigenvalue trajectories over time.

    Parameters
    ----------
    min_eigs : ndarray, shape (n_paths, N)
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    n_show = min(n_show, min_eigs.shape[0])
    for p in range(n_show):
        ax.plot(grid, min_eigs[p], alpha=0.3, linewidth=0.8, color="steelblue")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="PSD boundary (λ=0)")
    ax.plot(grid, np.mean(min_eigs, axis=0), color="navy", linewidth=2.0, label="Mean min eig")
    ax.plot(grid, np.percentile(min_eigs, 5, axis=0), color="orange", linewidth=1.5, label="5th percentile")
    ax.set_xlabel("$t$")
    ax.set_ylabel("Min eigenvalue")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_kernel_comparison(
    grid: np.ndarray,
    kernel_fns: Dict[str, Any],
    save_path: Optional[str] = None,
) -> None:
    """Plot multiple kernel functions on the same axes.

    Parameters
    ----------
    kernel_fns : dict mapping label -> callable kernel function
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, kfn in kernel_fns.items():
        vals = np.array([kfn(t) for t in grid])
        ax.plot(grid, vals, label=label, linewidth=2)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$k(t)$")
    ax.set_title("Kernel Functions")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_mc_validation(
    mc_estimates: List[float],
    mc_stds: List[float],
    theoretical_values: List[float],
    labels: List[str],
    title: str = "MC vs Theoretical",
    save_path: Optional[str] = None,
) -> None:
    """Bar chart comparing MC estimates to theoretical values with CIs."""
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, 2 * len(labels)), 5))
    ax.bar(x - 0.2, mc_estimates, 0.35, yerr=[2 * s for s in mc_stds],
           label="Monte Carlo", capsize=5, alpha=0.8)
    ax.bar(x + 0.2, theoretical_values, 0.35, label="Theoretical", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Transform value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
