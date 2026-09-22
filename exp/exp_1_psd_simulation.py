"""Experiment 1: Synthetic verification of PSD preservation for matrix affine Volterra processes.

Tests whether admissible synthetic parameterizations produce approximately positive-semidefinite
paths under time discretization for both the general X_t process and the Volterra-Wishart V_t.

Studies:
    A - General affine Volterra process on S_2^+
    B - Volterra-Wishart covariance process V_t
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial

from src.kernels import k_markov, k_shifted_fractional, validate_kernel_admissibility
from src.psd import min_eig, project_to_psd
from src.volterra_sim import simulate_X, simulate_V
from src.diagnostics import (
    summarize_psd_diagnostics,
    plot_sample_paths,
    plot_min_eigenvalue,
    plot_kernel_comparison,
)

os.makedirs("results", exist_ok=True)


def run_study_A() -> dict:
    """Study A: General affine Volterra process X_t on S_2^+."""
    print("\n" + "=" * 60)
    print("STUDY A: General Affine Volterra Process on S_2^+")
    print("=" * 60)

    T = 1.0
    n_paths = 200
    Q = np.array([[0.3, 0.05], [0.05, 0.2]])
    alpha = Q.T @ Q
    M = np.array([[-0.5, 0.1], [0.1, -0.4]])
    X0 = np.array([[1.0, 0.2], [0.2, 0.8]])

    kernels = {
        "markov": k_markov,
        "frac_a07_eps01": partial(k_shifted_fractional, a=0.7, eps=0.1),
        "frac_a07_eps001": partial(k_shifted_fractional, a=0.7, eps=0.01),
    }
    grid_sizes = [20, 50, 100]
    scenarios = {"conservative": 3.0, "boundary": 1.05}

    results = {}
    param_rows = []
    for grid_size in grid_sizes:
        grid = np.linspace(0, T, grid_size + 1)
        for kname, kfn in kernels.items():
            k0 = kfn(0.0)
            for sc_name, b_factor in scenarios.items():
                b0 = b_factor * k0 * alpha
                adm = validate_kernel_admissibility(grid, kfn, alpha, b0)
                sim = simulate_X(X0, grid, Q, b0, M, kfn, n_paths=n_paths, seed=42)
                diag = summarize_psd_diagnostics(sim["min_eigs"], grid)
                key = (grid_size, kname, sc_name)
                results[key] = {"diag": diag, "sim": sim, "admissibility": adm}
                param_rows.append({
                    "Study": "A", "GridSize": grid_size, "Kernel": kname,
                    "Scenario": sc_name,
                    "k0": f"{adm['k0']:.4f}",
                    "admissible": adm["admissible"],
                    "adm_gap": f"{adm['admissibility_gap']:.4f}",
                    "viol_rate": f"{diag['violation_rate']:.4f}",
                    "mean_min_eig": f"{diag['mean_min_eig']:.4f}",
                    "p5_min_eig": f"{diag['p5_min_eig']:.4f}",
                })
                print(f"  Grid={grid_size:3d} | {kname:20s} | {sc_name:12s} | "
                      f"ViolRate={diag['violation_rate']:.4f} | "
                      f"MeanMinEig={diag['mean_min_eig']:.4f} | "
                      f"k0={adm['k0']:.4f}")
    return results, pd.DataFrame(param_rows)


def run_study_B() -> dict:
    """Study B: Volterra-Wishart covariance process V_t."""
    print("\n" + "=" * 60)
    print("STUDY B: Volterra-Wishart Covariance Process V_t")
    print("=" * 60)

    T = 1.0
    n_paths = 200
    Q = np.array([[0.3, 0.05], [0.05, 0.2]])
    alpha = Q.T @ Q
    M = np.array([[-0.6, 0.05], [0.05, -0.5]])
    V0 = np.array([[1.0, 0.15], [0.15, 0.7]])

    kernels = {
        "markov": k_markov,
        "frac_a07_eps01": partial(k_shifted_fractional, a=0.7, eps=0.1),
        "frac_a07_eps001": partial(k_shifted_fractional, a=0.7, eps=0.01),
        "frac_a06_eps01": partial(k_shifted_fractional, a=0.6, eps=0.1),
    }
    grid_sizes = [20, 50, 100]
    scenarios = {"conservative": 3.0, "boundary": 1.1}

    results = {}
    param_rows = []
    for grid_size in grid_sizes:
        grid = np.linspace(0, T, grid_size + 1)
        for kname, kfn in kernels.items():
            k0 = kfn(0.0)
            for sc_name, b_factor in scenarios.items():
                b = b_factor * k0 * alpha
                adm = validate_kernel_admissibility(grid, kfn, alpha, b)
                sim = simulate_V(V0, grid, Q, b, M, kfn, n_paths=n_paths, seed=42)
                diag = summarize_psd_diagnostics(sim["min_eigs"], grid)
                key = (grid_size, kname, sc_name)
                results[key] = {"diag": diag, "admissibility": adm, "sim": sim}
                param_rows.append({
                    "Study": "B", "GridSize": grid_size, "Kernel": kname,
                    "Scenario": sc_name,
                    "k0": f"{adm['k0']:.4f}",
                    "admissible": adm["admissible"],
                    "adm_gap": f"{adm['admissibility_gap']:.4f}",
                    "viol_rate": f"{diag['violation_rate']:.4f}",
                    "mean_min_eig": f"{diag['mean_min_eig']:.4f}",
                    "p5_min_eig": f"{diag['p5_min_eig']:.4f}",
                })
                print(f"  Grid={grid_size:3d} | {kname:22s} | {sc_name:12s} | "
                      f"ViolRate={diag['violation_rate']:.4f} | "
                      f"MeanMinEig={diag['mean_min_eig']:.4f}")
    return results, pd.DataFrame(param_rows)


def run_kernel_sensitivity() -> pd.DataFrame:
    """Test sensitivity of k(0) to eps parameter in shifted fractional kernel."""
    print("\n" + "=" * 60)
    print("KERNEL SENSITIVITY: Effect of eps on k(0) and admissibility")
    print("=" * 60)
    T = 1.0
    grid = np.linspace(0, T, 51)
    Q = np.array([[0.3, 0.05], [0.05, 0.2]])
    alpha = Q.T @ Q

    rows = []
    for a in [0.6, 0.7, 0.8]:
        for eps in [0.001, 0.01, 0.05, 0.1, 0.2]:
            kfn = partial(k_shifted_fractional, a=a, eps=eps)
            k0 = kfn(0.0)
            b0_min = k0 * alpha  # minimum admissible b0
            adm = validate_kernel_admissibility(grid, kfn, alpha, 1.1 * b0_min)
            rows.append({
                "a": a, "eps": eps, "k0": k0,
                "k0_min_b0_eigenvalue": float(np.linalg.eigvalsh(k0 * alpha).min()),
                "admissible": adm["admissible"],
            })
            print(f"  a={a:.1f}, eps={eps:.3f} -> k(0)={k0:.4f}")
    return pd.DataFrame(rows)


def make_plots(results_A: dict, results_B: dict) -> None:
    """Generate diagnostic plots for Experiment 1."""
    # Plot sample paths for best scenario (conservative markov, fine grid)
    key_A = (100, "markov", "conservative")
    if key_A in results_A:
        sim = results_A[key_A]["sim"]
        grid = sim["grid"]
        plot_sample_paths(
            sim["paths"], grid,
            title="Study A: Volterra Process X_t Sample Paths (Markov, Conservative)",
            save_path="results/exp1_study_A_sample_paths.png",
            n_show=10,
        )
        plot_min_eigenvalue(
            sim["min_eigs"], grid,
            title="Study A: Minimum Eigenvalue (Markov, Conservative, N=100)",
            save_path="results/exp1_study_A_min_eig.png",
            n_show=50,
        )

    # Plot min eigenvalue comparison: fractional vs markov
    key_frac = (100, "frac_a07_eps01", "conservative")
    if key_frac in results_A:
        sim = results_A[key_frac]["sim"]
        grid = sim["grid"]
        plot_min_eigenvalue(
            sim["min_eigs"], grid,
            title="Study A: Min Eigenvalue (Fractional a=0.7, eps=0.1, N=100)",
            save_path="results/exp1_study_A_min_eig_frac.png",
        )

    # Study B: V_t paths
    key_B = (100, "markov", "conservative")
    if key_B in results_B:
        sim = results_B[key_B]["sim"]
        grid = sim["grid"]
        plot_sample_paths(
            sim["paths"], grid,
            title="Study B: Volterra-Wishart V_t Sample Paths (Markov, Conservative)",
            save_path="results/exp1_study_B_sample_paths.png",
        )
        plot_min_eigenvalue(
            sim["min_eigs"], grid,
            title="Study B: Minimum Eigenvalue V_t (Markov, Conservative)",
            save_path="results/exp1_study_B_min_eig.png",
        )

    # Plot kernel functions
    T_plot = 1.0
    grid_plot = np.linspace(0, T_plot, 101)
    from functools import partial
    plot_kernel_comparison(
        grid_plot,
        {
            "k(t)=1 (Markov)": k_markov,
            "Shifted frac (a=0.7, eps=0.1)": partial(k_shifted_fractional, a=0.7, eps=0.1),
            "Shifted frac (a=0.7, eps=0.01)": partial(k_shifted_fractional, a=0.7, eps=0.01),
            "Shifted frac (a=0.6, eps=0.1)": partial(k_shifted_fractional, a=0.6, eps=0.1),
        },
        save_path="results/exp1_kernel_comparison.png",
    )
    print("Saved experiment 1 plots.")


def plot_grid_refinement(results_A: dict) -> None:
    """Plot PSD violation rate vs grid size."""
    kernels = ["markov", "frac_a07_eps01"]
    scenarios = ["conservative", "boundary"]
    grid_sizes = [20, 50, 100]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, sc in zip(axes, scenarios):
        for kname in kernels:
            rates = []
            for g in grid_sizes:
                key = (g, kname, sc)
                if key in results_A:
                    rates.append(results_A[key]["diag"]["violation_rate"])
                else:
                    rates.append(np.nan)
            ax.plot(grid_sizes, rates, marker="o", label=kname)
        ax.set_xlabel("Grid size N")
        ax.set_ylabel("PSD violation rate")
        ax.set_title(f"Grid refinement: {sc} scenario")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle("Study A: PSD Violation Rate vs Grid Size")
    plt.tight_layout()
    plt.savefig("results/exp1_grid_refinement.png", dpi=100)
    plt.close()
    print("Saved results/exp1_grid_refinement.png")


if __name__ == "__main__":
    print("Running Experiment 1: PSD Simulation Verification")

    results_A, df_A = run_study_A()
    results_B, df_B = run_study_B()
    df_sensitivity = run_kernel_sensitivity()

    # Combine tables
    df_all = pd.concat([df_A, df_B], ignore_index=True)
    df_all.to_csv("results/exp1_psd_diagnostics.csv", index=False)
    df_sensitivity.to_csv("results/exp1_kernel_sensitivity.csv", index=False)
    print("\nSaved results/exp1_psd_diagnostics.csv")
    print("Saved results/exp1_kernel_sensitivity.csv")
    print("\nFull diagnostics table:")
    print(df_all.to_string(index=False))

    make_plots(results_A, results_B)
    plot_grid_refinement(results_A)

    print("\nExperiment 1 complete.")
