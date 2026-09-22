"""Experiment 2: Verify exponential-affine transform via Riccati-Volterra equations.

Validates:
  E[exp(-Tr(uX_T + int_0^T f(s)X_s ds)) | F_t] = exp(-Y_t)

Checks performed:
  1. Riccati-Volterra residuals (small and decreasing under refinement)
  2. Unconditional MC validation: E[payoff] ≈ exp(-Y_0)
  3. Conditional nested MC validation at selected times
  4. Comparison between Markovian and fractional kernels
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

from src.kernels import k_markov, k_shifted_fractional
from src.volterra_sim import simulate_X
from src.riccati_volterra import solve_riccati_volterra
from src.transforms import (
    compute_payoff_laplace,
    compute_Y0_theoretical,
    compute_running_integral,
)
from src.nested_mc import run_nested_mc_conditional
from src.diagnostics import plot_min_eigenvalue

os.makedirs("results", exist_ok=True)


def setup_parameters():
    """Common parameter setup for Exp 2."""
    T = 1.0
    d = 2
    Q = np.array([[0.3, 0.05], [0.05, 0.25]])
    alpha = Q.T @ Q
    M = np.array([[-0.5, 0.1], [0.1, -0.4]])
    X0 = np.array([[1.0, 0.2], [0.2, 0.8]])
    # Real PSD Laplace loading
    u = np.array([[0.3, 0.05], [0.05, 0.2]])
    return T, d, Q, alpha, M, X0, u


def run_riccati_validation(T, d, Q, alpha, M, X0, u, grid_sizes, kernels, n_paths=500):
    """Check 1 & 2: Riccati residuals and unconditional MC validation."""
    results = {}
    rows = []

    for grid_size in grid_sizes:
        grid = np.linspace(0, T, grid_size + 1)
        N = len(grid)

        for kname, kfn in kernels.items():
            k0 = kfn(0.0)
            b0 = 3.0 * k0 * alpha  # conservative admissibility

            # Running loading f(s): constant nonzero PSD matrix
            f_running = np.zeros((N, d, d))
            f_const = np.array([[0.05, 0.01], [0.01, 0.04]])
            f_running[:] = f_const

            # Solve Riccati-Volterra
            rv = solve_riccati_volterra(
                grid, u.astype(complex), f_running.astype(complex),
                M, alpha, kfn
            )
            psi = rv["psi"]
            residuals = rv["residual_norms"]

            # Theoretical Y_0: Tr(psi(0) @ X_0) + phi(0) where phi(0) = int_0^T k(s) Tr(psi(s) b_0) ds
            Y0_theo = compute_Y0_theoretical(X0, psi, grid, b0, M, alpha, f_running, kernel_fn=kfn)

            # Simulate X paths
            sim = simulate_X(X0, grid, Q, b0, M, kfn, n_paths=n_paths, seed=123)
            paths = sim["paths"]

            # MC estimate
            payoffs = compute_payoff_laplace(paths, grid, u, f_running)
            mc_mean = float(np.mean(payoffs))
            mc_std = float(np.std(payoffs) / np.sqrt(n_paths))
            theo_val = float(np.exp(-np.real(Y0_theo)))
            rel_error = abs(mc_mean - theo_val) / (abs(theo_val) + 1e-10)
            min_eig_psi0 = float(np.linalg.eigvalsh(np.real(psi[0])).min())

            key = (grid_size, kname)
            results[key] = {
                "psi": psi, "residuals": residuals, "sim": sim,
                "payoffs": payoffs, "mc_mean": mc_mean, "mc_std": mc_std,
                "theo_val": theo_val, "Y0_theo": Y0_theo,
            }
            rows.append({
                "GridSize": grid_size, "Kernel": kname,
                "MC_mean": f"{mc_mean:.5f}", "MC_2std": f"{2*mc_std:.5f}",
                "Theoretical": f"{theo_val:.5f}",
                "RelError": f"{rel_error:.4f}",
                "MaxResidual": f"{np.max(residuals):.2e}",
                "MeanResidual": f"{np.mean(residuals):.2e}",
                "psi0_mineig": f"{min_eig_psi0:.4f}",
            })
            print(f"  Grid={grid_size:3d} | {kname:20s} | MC={mc_mean:.4f}±{2*mc_std:.4f} | "
                  f"Theo={theo_val:.4f} | RelErr={rel_error:.3f} | MaxRes={np.max(residuals):.2e}")
    return results, pd.DataFrame(rows)


def run_conditional_validation(results, T, d, Q, alpha, M, X0, u, kernels):
    """Check 3: Nested MC conditional validation at selected times."""
    rows = []
    print("\n--- Conditional Transform Validation (Nested MC) ---")

    # Use the finest grid and first kernel for conditional test
    grid_size = max(k[0] for k in results.keys())
    kname = list(kernels.keys())[0]
    key = (grid_size, kname)

    if key not in results:
        print("  Skipping conditional test (key not found).")
        return pd.DataFrame()

    res = results[key]
    sim = res["sim"]
    psi = res["psi"]
    grid = sim["grid"]
    kfn = kernels[kname]
    k0 = kfn(0.0)
    b0 = 3.0 * k0 * alpha

    N = len(grid)
    f_running = np.zeros((N, d, d))
    f_running[:] = np.array([[0.05, 0.01], [0.01, 0.04]])

    # Conditioning times: 25%, 50%, 75% of T
    cond_indices = [N // 4, N // 2, 3 * N // 4]

    for ci in cond_indices:
        t_cond = grid[ci]
        outer_path = sim["paths"][0]  # Use first outer path
        outer_dW = sim["dW"][0]

        nested = run_nested_mc_conditional(
            conditioning_time_idx=ci,
            outer_path_X=outer_path,
            outer_dW=outer_dW,
            grid=grid, Q=Q, b0=b0, M=M, kernel_fn=kfn,
            u=u, f_running=f_running,
            n_inner=100, seed=999,
        )
        # Theoretical: exp(-Y_t) at conditioning time
        Y_t_theo = float(np.real(np.trace(psi[ci] @ outer_path[ci])))
        theo_cond = float(np.exp(-Y_t_theo))

        rows.append({
            "Kernel": kname, "GridSize": grid_size,
            "CondTime": f"{t_cond:.3f}",
            "NestedMC_estimate": f"{nested['estimate']:.5f}",
            "NestedMC_2se": f"{2*nested['std_error']:.5f}",
            "Theoretical_approx": f"{theo_cond:.5f}",
            "CI_lower": f"{nested['ci_95'][0]:.5f}",
            "CI_upper": f"{nested['ci_95'][1]:.5f}",
        })
        print(f"  t={t_cond:.3f}: Nested MC={nested['estimate']:.4f}±{2*nested['std_error']:.4f} | "
              f"Theo≈{theo_cond:.4f}")

    return pd.DataFrame(rows)


def plot_riccati_residuals(results: dict) -> None:
    """Plot Riccati-Volterra residual norms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: residuals for one grid size
    ax = axes[0]
    grid_size = max(k[0] for k in results.keys())
    for kname in ["markov", "frac"]:
        key = (grid_size, kname)
        if key in results:
            res = results[key]
            psi = res["psi"]
            residuals = res["residuals"]
            grid = res["sim"]["grid"]
            ax.semilogy(grid, residuals + 1e-16, label=kname)
    ax.set_xlabel("$t$")
    ax.set_ylabel("Residual norm")
    ax.set_title(f"Riccati-Volterra Residuals (N={grid_size})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: max residual vs grid size
    ax2 = axes[1]
    for kname in ["markov", "frac"]:
        grid_sizes = sorted(set(k[0] for k in results.keys()))
        max_res = []
        for g in grid_sizes:
            key = (g, kname)
            if key in results:
                max_res.append(float(np.max(results[key]["residuals"])))
            else:
                max_res.append(np.nan)
        ax2.semilogy(grid_sizes, [r + 1e-16 for r in max_res], marker="o", label=kname)
    ax2.set_xlabel("Grid size N")
    ax2.set_ylabel("Max residual norm")
    ax2.set_title("Max Riccati Residual vs Grid Size")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Experiment 2: Riccati-Volterra Diagnostics")
    plt.tight_layout()
    plt.savefig("results/exp2_riccati_residuals.png", dpi=100)
    plt.close()
    print("Saved results/exp2_riccati_residuals.png")


def plot_psi_solution(results: dict) -> None:
    """Plot psi(t) solution entries."""
    key = list(results.keys())[-1]  # finest grid
    res = results[key]
    psi = np.real(res["psi"])
    grid = res["sim"]["grid"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(grid, psi[:, 0, 0], label="psi_11")
    axes[0].set_title("psi_11(t)")
    axes[1].plot(grid, psi[:, 1, 1], label="psi_22", color="orange")
    axes[1].set_title("psi_22(t)")
    axes[2].plot(grid, psi[:, 0, 1], label="psi_12", color="green")
    axes[2].set_title("psi_12(t)")
    for ax in axes:
        ax.set_xlabel("$t$")
        ax.grid(True, alpha=0.3)
    plt.suptitle("Experiment 2: Riccati-Volterra Solution psi(t)")
    plt.tight_layout()
    plt.savefig("results/exp2_psi_solution.png", dpi=100)
    plt.close()
    print("Saved results/exp2_psi_solution.png")


if __name__ == "__main__":
    print("Running Experiment 2: Affine Transform via Riccati-Volterra")

    T, d, Q, alpha, M, X0, u = setup_parameters()
    kernels = {
        "markov": k_markov,
        "frac": partial(k_shifted_fractional, a=0.7, eps=0.1),
    }
    grid_sizes = [30, 60]

    print("\n--- Riccati Validation + Unconditional MC ---")
    results, df_main = run_riccati_validation(T, d, Q, alpha, M, X0, u, grid_sizes, kernels, n_paths=500)

    df_conditional = run_conditional_validation(results, T, d, Q, alpha, M, X0, u, kernels)

    # Save tables
    df_main.to_csv("results/exp2_transform_validation.csv", index=False)
    if not df_conditional.empty:
        df_conditional.to_csv("results/exp2_conditional_validation.csv", index=False)
    print("\nSaved results/exp2_transform_validation.csv")
    print("\nMain results:")
    print(df_main.to_string(index=False))
    if not df_conditional.empty:
        print("\nConditional results:")
        print(df_conditional.to_string(index=False))

    plot_riccati_residuals(results)
    plot_psi_solution(results)

    print("\nExperiment 2 complete.")
