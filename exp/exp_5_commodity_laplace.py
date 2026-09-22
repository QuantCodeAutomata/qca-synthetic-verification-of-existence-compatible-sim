"""Experiment 5: Validate commodity-model Laplace transform for Volterra-Wishart covariance.

Validates:
    E[exp(-Tr(u3*V_T) - int_0^T Tr(f_L(s)*V_s)ds) | F_t] = exp(-Y_t^L)

Test scenarios:
    - f_L = 0 (simpler case)
    - f_L = constant PSD (nontrivial running loading)
    - Markovian k(t)=1 vs shifted fractional kernels
    - Coarse and fine grids for convergence
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
from src.volterra_sim import simulate_V
from src.riccati_volterra import solve_riccati_volterra
from src.transforms import compute_running_integral, compute_Y0_theoretical
from src.diagnostics import summarize_psd_diagnostics, plot_min_eigenvalue

os.makedirs("results", exist_ok=True)


def setup_parameters():
    T = 1.0
    d = 2
    Q = np.array([[0.3, 0.05], [0.05, 0.25]])
    alpha = Q.T @ Q
    M = np.array([[-0.6, 0.05], [0.05, -0.5]])
    V0 = np.array([[1.0, 0.15], [0.15, 0.7]])
    u3 = np.array([[0.3, 0.04], [0.04, 0.2]])  # PSD terminal loading
    return T, d, Q, alpha, M, V0, u3


def compute_payoff_commodity_laplace(V_paths, grid, u3, f_running):
    """Compute exp(-Tr(u3*V_T) - int_0^T Tr(f_L(s)*V_s)ds)."""
    n_paths = V_paths.shape[0]
    V_T = V_paths[:, -1]
    terminal = np.array([np.trace(u3 @ V_T[p]) for p in range(n_paths)])
    running = compute_running_integral(V_paths, grid, f_running)
    return np.exp(-(terminal + running))


def run_laplace_scenario(V0, grid, kfn, kname, Q, alpha, M, b, u3, f_running,
                         n_paths, d, scenario_name):
    """Run one Laplace transform scenario."""
    N = len(grid)
    # Solve commodity Riccati-Volterra for Laplace case
    rv = solve_riccati_volterra(
        grid, u3.astype(complex), f_running.astype(complex), M, alpha, kfn
    )
    psi = rv["psi"]
    residuals = rv["residual_norms"]

    # Theoretical Y_0^L: Tr(psi(0) @ V_0) + phi(0) where phi(0) = int_0^T k(s) Tr(psi(s) b) ds
    Y0_L = compute_Y0_theoretical(V0, psi, grid, b, M, alpha, f_running, kernel_fn=kfn)
    theo_val = float(np.exp(-np.real(Y0_L)))

    # Simulate V paths
    sim = simulate_V(V0, grid, Q, b, M, kfn, n_paths=n_paths, seed=42)
    V_paths = sim["paths"]
    diag = summarize_psd_diagnostics(sim["min_eigs"], grid)

    # MC estimate
    payoffs = compute_payoff_commodity_laplace(V_paths, grid, u3, f_running)
    mc_mean = float(np.mean(payoffs))
    mc_std = float(np.std(payoffs) / np.sqrt(n_paths))
    rel_error = abs(mc_mean - theo_val) / (abs(theo_val) + 1e-10)
    min_eig_psi = float(np.linalg.eigvalsh(np.real(psi[0])).min())

    return {
        "psi": psi, "residuals": residuals, "Y0_L": Y0_L,
        "mc_mean": mc_mean, "mc_std": mc_std,
        "theo_val": theo_val, "rel_error": rel_error,
        "diag": diag, "sim": sim, "payoffs": payoffs,
        "min_eig_psi_0": min_eig_psi,
    }


def run_all_scenarios(T, d, Q, alpha, M, V0, u3, kernels, grid_sizes, n_paths=400):
    """Run all Laplace transform scenarios."""
    rows = []
    results = {}

    f_loading_cases = {
        "f_L=0": "zero",
        "f_L=const": "nonzero",
    }

    for grid_size in grid_sizes:
        grid = np.linspace(0, T, grid_size + 1)
        N = len(grid)

        for kname, kfn in kernels.items():
            k0 = kfn(0.0)
            b = 3.0 * k0 * alpha  # conservative admissibility
            adm = validate_kernel_admissibility(grid, kfn, alpha, b)

            for fl_name, fl_type in f_loading_cases.items():
                f_running = np.zeros((N, d, d))
                if fl_type == "nonzero":
                    f_running[:] = np.array([[0.05, 0.01], [0.01, 0.04]])

                scenario_name = f"{kname}_{fl_name}_N{grid_size}"
                res = run_laplace_scenario(
                    V0, grid, kfn, kname, Q, alpha, M, b, u3, f_running,
                    n_paths, d, scenario_name
                )
                key = (grid_size, kname, fl_name)
                results[key] = res

                rows.append({
                    "GridSize": grid_size, "Kernel": kname, "f_loading": fl_name,
                    "MC_mean": f"{res['mc_mean']:.5f}",
                    "MC_2std": f"{2*res['mc_std']:.5f}",
                    "Theoretical": f"{res['theo_val']:.5f}",
                    "RelError": f"{res['rel_error']:.4f}",
                    "MaxResidual": f"{np.max(res['residuals']):.2e}",
                    "ViolRate_V": f"{res['diag']['violation_rate']:.4f}",
                    "MinEig_psi0": f"{res['min_eig_psi_0']:.4f}",
                })
                print(f"  Grid={grid_size:3d} | {kname:20s} | {fl_name:10s} | "
                      f"MC={res['mc_mean']:.4f}±{2*res['mc_std']:.4f} | "
                      f"Theo={res['theo_val']:.4f} | RelErr={res['rel_error']:.3f}")

    return results, pd.DataFrame(rows)


def plot_psi_L(results: dict) -> None:
    """Plot psi_L(t) solution for Laplace case."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    row_map = {"markov": 0, "frac": 1}
    for (grid_size, kname, fl_name), res in results.items():
        if grid_size != max(k[0] for k in results.keys()):
            continue
        if fl_name != "f_L=const":
            continue
        row = 0 if "markov" in kname else 1
        if row >= 2:
            continue
        psi = np.real(res["psi"])
        grid = res["sim"]["grid"]
        for j, (i_row, j_col) in enumerate([(0, 0), (1, 1), (0, 1)]):
            axes[row][j].plot(grid, psi[:, i_row, j_col])
            axes[row][j].set_title(f"psi_L_{i_row+1}{j_col+1} ({kname})")
            axes[row][j].set_xlabel("$t$")
            axes[row][j].grid(True, alpha=0.3)

    plt.suptitle("Experiment 5: Riccati-Volterra Solution psi_L(t)")
    plt.tight_layout()
    plt.savefig("results/exp5_psi_L.png", dpi=100)
    plt.close()
    print("Saved results/exp5_psi_L.png")


def plot_V_paths(results: dict) -> None:
    """Plot V_t sample paths and min eigenvalue."""
    key = max(results.keys(), key=lambda k: k[0])
    res = results[key]
    sim = res["sim"]
    grid = sim["grid"]

    plot_min_eigenvalue(
        sim["min_eigs"], grid,
        title=f"Exp 5: Min eigenvalue of V_t ({key[1]}, {key[2]})",
        save_path="results/exp5_V_min_eig.png",
        n_show=30,
    )

    # V entries
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    V_paths = sim["paths"]
    n_show = min(10, V_paths.shape[0])
    for p in range(n_show):
        axes[0].plot(grid, V_paths[p, :, 0, 0], alpha=0.4)
        axes[1].plot(grid, V_paths[p, :, 1, 1], alpha=0.4)
        axes[2].plot(grid, V_paths[p, :, 0, 1], alpha=0.4)
    for ax, label in zip(axes, ["V_11", "V_22", "V_12"]):
        ax.set_title(f"${label}$ paths ({key[1]})")
        ax.set_xlabel("$t$")
        ax.grid(True, alpha=0.3)
    plt.suptitle("Experiment 5: Volterra-Wishart V_t Sample Paths")
    plt.tight_layout()
    plt.savefig("results/exp5_V_paths.png", dpi=100)
    plt.close()
    print("Saved results/exp5_V_paths.png")


if __name__ == "__main__":
    print("Running Experiment 5: Commodity Laplace Transform Verification")

    T, d, Q, alpha, M, V0, u3 = setup_parameters()
    kernels = {
        "markov": k_markov,
        "frac_a07_eps01": partial(k_shifted_fractional, a=0.7, eps=0.1),
    }
    grid_sizes = [30, 60]

    results, df_main = run_all_scenarios(T, d, Q, alpha, M, V0, u3, kernels, grid_sizes, n_paths=400)

    df_main.to_csv("results/exp5_commodity_laplace.csv", index=False)
    print("\nSaved results/exp5_commodity_laplace.csv")
    print("\nResults:")
    print(df_main.to_string(index=False))

    plot_psi_L(results)
    plot_V_paths(results)

    print("\nExperiment 5 complete.")
