"""Experiment 3: Synthetic verification of path-dependent resolvent representation.

Validates the resolvent-based and direct-transform constructions of Y_t agree
pathwise for matrix affine Volterra processes.

Stages:
    1. Simulate X_t paths
    2. Compute resolvent of first kind L and verify k * L ≈ 1
    3. Solve Riccati-Volterra for psi(t)
    4. Compute Y_t via direct formula and resolvent-based formula
    5. Compare pathwise differences
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
from src.resolvent import (
    compute_resolvent,
    verify_resolvent_identity,
    compute_Y_t_resolvent_based,
)
from src.transforms import compute_Y_t_direct, compute_payoff_laplace

os.makedirs("results", exist_ok=True)


def setup_parameters():
    T = 1.0
    d = 2
    Q = np.array([[0.3, 0.05], [0.05, 0.25]])
    alpha = Q.T @ Q
    M = np.array([[-0.5, 0.1], [0.1, -0.4]])
    X0 = np.array([[1.0, 0.2], [0.2, 0.8]])
    u = np.array([[0.2, 0.04], [0.04, 0.15]])
    return T, d, Q, alpha, M, X0, u


def run_resolvent_verification(T, grid, kfn, kname, alpha, u):
    """Stage 2: Compute resolvent and verify convolution identity."""
    res_data = compute_resolvent(grid, kfn)
    identity_check = verify_resolvent_identity(grid, kfn, res_data)
    print(f"  [{kname}] Resolvent check: max|kL-1| = {identity_check['max_error']:.4e}, "
          f"mean_err = {identity_check['mean_error']:.4e}")
    return res_data, identity_check


def run_Y_comparison(T, d, Q, alpha, M, X0, u, grid_sizes, kernels, n_paths=50):
    """Compare direct and resolvent-based Y_t constructions."""
    results = {}
    rows = []

    for grid_size in grid_sizes:
        grid = np.linspace(0, T, grid_size + 1)
        N = len(grid)

        for kname, kfn in kernels.items():
            k0 = kfn(0.0)
            b0 = 3.0 * k0 * alpha
            f_running = np.zeros((N, d, d))
            f_running[:] = np.array([[0.04, 0.01], [0.01, 0.03]])

            # Stage 2: Resolvent
            res_data, identity_check = run_resolvent_verification(T, grid, kfn, kname, alpha, u)

            # Stage 3: Riccati-Volterra
            rv = solve_riccati_volterra(
                grid, u.astype(complex), f_running.astype(complex), M, alpha, kfn
            )
            psi = rv["psi"]

            # Simulate X paths
            sim = simulate_X(X0, grid, Q, b0, M, kfn, n_paths=n_paths, seed=42)
            paths = sim["paths"]
            dW = sim["dW"]

            # Stage 4a: Y_t via direct formula
            Y_direct = compute_Y_t_direct(paths, dW, psi, grid, Q, M, alpha, b0, f_running)

            # Stage 4b: Y_t via resolvent-based formula
            Y_resolvent = compute_Y_t_resolvent_based(
                paths, dW, psi, res_data, grid, Q, M, alpha, b0, f_running, u
            )

            # Stage 5: Compare pathwise at final time
            diff_T = np.abs(Y_direct[:, -1] - Y_resolvent[:, -1])
            mean_diff = float(np.mean(diff_T))
            max_diff = float(np.max(diff_T))
            std_diff = float(np.std(diff_T))

            key = (grid_size, kname)
            results[key] = {
                "Y_direct": Y_direct, "Y_resolvent": Y_resolvent,
                "diff_T": diff_T, "psi": psi, "sim": sim,
                "identity_check": identity_check, "grid": grid,
            }
            rows.append({
                "GridSize": grid_size, "Kernel": kname,
                "Mean_Y_diff": f"{mean_diff:.4e}",
                "Max_Y_diff": f"{max_diff:.4e}",
                "Std_Y_diff": f"{std_diff:.4e}",
                "Resolvent_identity_err": f"{identity_check['max_error']:.4e}",
            })
            print(f"  Grid={grid_size:3d} | {kname:20s} | "
                  f"Mean|Y_direct-Y_resolv|={mean_diff:.4e} | Max={max_diff:.4e}")

    return results, pd.DataFrame(rows)


def plot_Y_comparison(results: dict) -> None:
    """Plot direct vs resolvent Y_t for selected paths and kernels."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    row_idx = 0
    for (grid_size, kname), res in results.items():
        if row_idx >= 2:
            break
        Y_d = np.real(res["Y_direct"])
        Y_r = np.real(res["Y_resolvent"])
        grid = res["grid"]
        n_show = min(5, Y_d.shape[0])

        ax = axes[row_idx, 0]
        for p in range(n_show):
            ax.plot(grid, Y_d[p], alpha=0.5, linestyle="-", label="Direct" if p == 0 else "")
            ax.plot(grid, Y_r[p], alpha=0.5, linestyle="--", label="Resolvent" if p == 0 else "")
        ax.set_title(f"Y_t comparison ({kname}, N={grid_size})")
        ax.set_xlabel("$t$")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax2 = axes[row_idx, 1]
        diff_paths = np.abs(Y_d - Y_r)
        ax2.plot(grid, np.mean(diff_paths, axis=0), label="Mean |diff|")
        ax2.plot(grid, np.percentile(diff_paths, 95, axis=0), linestyle="--", label="95th pct")
        ax2.set_title(f"|Y_direct - Y_resolvent| ({kname})")
        ax2.set_xlabel("$t$")
        ax2.set_yscale("log")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        row_idx += 1

    plt.suptitle("Experiment 3: Direct vs Resolvent Y_t Comparison")
    plt.tight_layout()
    plt.savefig("results/exp3_Y_comparison.png", dpi=100)
    plt.close()
    print("Saved results/exp3_Y_comparison.png")


def plot_resolvent(results: dict) -> None:
    """Plot resolvent density ell(t)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, ((grid_size, kname), res) in zip(axes, list(results.items())[:2]):
        grid = res["grid"]
        ic = res["identity_check"]
        residual = ic["residual"]
        ax.semilogy(grid, np.abs(residual) + 1e-16, label="|kL-1|")
        ax.axhline(1e-10, color="red", linestyle="--", linewidth=0.8, label="1e-10 threshold")
        ax.set_title(f"Resolvent identity error ({kname}, N={grid_size})")
        ax.set_xlabel("$t$")
        ax.set_ylabel("|k * L - 1|")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle("Experiment 3: Resolvent Convolution Identity Verification")
    plt.tight_layout()
    plt.savefig("results/exp3_resolvent_identity.png", dpi=100)
    plt.close()
    print("Saved results/exp3_resolvent_identity.png")


if __name__ == "__main__":
    print("Running Experiment 3: Resolvent Representation Verification")

    T, d, Q, alpha, M, X0, u = setup_parameters()
    kernels = {
        "markov": k_markov,
        "frac_a07_eps01": partial(k_shifted_fractional, a=0.7, eps=0.1),
    }
    grid_sizes = [30, 60]

    results, df_main = run_Y_comparison(T, d, Q, alpha, M, X0, u, grid_sizes, kernels, n_paths=50)

    df_main.to_csv("results/exp3_resolvent_comparison.csv", index=False)
    print("\nSaved results/exp3_resolvent_comparison.csv")
    print("\nResults table:")
    print(df_main.to_string(index=False))

    plot_Y_comparison(results)
    plot_resolvent(results)

    print("\nExperiment 3 complete.")
