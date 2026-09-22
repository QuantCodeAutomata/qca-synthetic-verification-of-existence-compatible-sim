"""Experiment 4: Synthetic verification of Brownian-functional extended transform.

Validates:
    E[exp(zN_T - Tr(uX_T) - int_0^T Tr(f(s)X_s)ds) | F_t] = exp(zN_t - Y_t^{z,H})

Test cases:
    - z=0 (reduces to simple affine transform)
    - z=i (pure imaginary, relevant for commodity model)
    - z real (Laplace case)
With simple deterministic H(s) matrices.
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
from src.riccati_volterra import solve_riccati_brownian_functional
from src.brownian_functional import compute_N_t, compute_payoff_extended
from src.transforms import compute_payoff_laplace

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


def H_constant(t: float) -> np.ndarray:
    """Constant H matrix."""
    return np.array([[0.15, 0.05], [0.05, 0.10]])


def H_timedep(t: float, T: float = 1.0) -> np.ndarray:
    """Time-varying H matrix."""
    decay = np.exp(-t / T)
    return decay * np.array([[0.15, 0.05], [0.05, 0.10]])


def run_z_case(z, kfn, kname, grid, Q, alpha, M, X0, u, d, b0, f_running, H_fn,
               paths, dW, n_paths):
    """Run one (z, kernel) case."""
    N = len(grid)

    # Solve modified Riccati-Volterra
    rv = solve_riccati_brownian_functional(
        grid, u.astype(complex), f_running.astype(complex),
        M, alpha, kfn, z, H_fn
    )
    psi = rv["psi"]
    residuals = rv["residual_norms"]

    # Compute N_t along paths
    N_t = compute_N_t(paths, dW, H_fn, grid)

    # Terminal payoffs: exp(z*N_T - Tr(u*X_T) - int f X ds)
    payoffs = compute_payoff_extended(paths, N_t, grid, z, u, f_running)

    mc_mean = complex(np.mean(payoffs))
    mc_std_real = float(np.std(payoffs.real) / np.sqrt(n_paths))
    mc_std_imag = float(np.std(payoffs.imag) / np.sqrt(n_paths))

    # Theoretical value: exp(-Y_0)
    # Y_0 = Tr(psi[0] @ X0) (leading term from transform representation)
    Y0_approx = complex(np.trace(psi[0] @ X0))
    theo_val = complex(np.exp(-Y0_approx))

    rel_err_real = abs(mc_mean.real - theo_val.real) / (abs(theo_val.real) + 1e-10)
    rel_err_imag = abs(mc_mean.imag - theo_val.imag) / (abs(theo_val.imag) + 1e-10) if abs(theo_val.imag) > 1e-10 else 0.0

    return {
        "psi": psi, "residuals": residuals, "N_t": N_t,
        "payoffs": payoffs, "mc_mean": mc_mean, "mc_std_real": mc_std_real,
        "mc_std_imag": mc_std_imag, "theo_val": theo_val,
        "rel_err_real": rel_err_real, "rel_err_imag": rel_err_imag,
    }


def run_experiment(T, d, Q, alpha, M, X0, u, kernels, grid_size=50, n_paths=300):
    """Run the Brownian functional experiment for multiple z cases and kernels."""
    grid = np.linspace(0, T, grid_size + 1)
    N = len(grid)
    f_running = np.zeros((N, d, d))
    f_running[:] = np.array([[0.04, 0.01], [0.01, 0.03]])

    z_cases = {
        "z=0 (Laplace)": 0.0 + 0j,
        "z=1 (real)": 1.0 + 0j,
        "z=i (imaginary)": 0.0 + 1j,
    }
    H_cases = {
        "H_constant": H_constant,
        "H_timedep": H_timedep,
    }

    rows = []
    results = {}

    for kname, kfn in kernels.items():
        k0 = kfn(0.0)
        b0 = 3.0 * k0 * alpha

        # Simulate paths once per kernel
        sim = simulate_X(X0, grid, Q, b0, M, kfn, n_paths=n_paths, seed=42)
        paths = sim["paths"]
        dW = sim["dW"]

        for z_label, z in z_cases.items():
            for h_label, H_fn in H_cases.items():
                res = run_z_case(
                    z=z, kfn=kfn, kname=kname,
                    grid=grid, Q=Q, alpha=alpha, M=M, X0=X0, u=u, d=d,
                    b0=b0, f_running=f_running, H_fn=H_fn,
                    paths=paths, dW=dW, n_paths=n_paths,
                )
                key = (kname, z_label, h_label)
                results[key] = res

                mc = res["mc_mean"]
                th = res["theo_val"]
                rows.append({
                    "Kernel": kname, "z": str(z), "H": h_label,
                    "MC_real": f"{mc.real:.5f}", "MC_imag": f"{mc.imag:.5f}",
                    "MC_2std_real": f"{2*res['mc_std_real']:.5f}",
                    "Theo_real": f"{th.real:.5f}", "Theo_imag": f"{th.imag:.5f}",
                    "RelErr_real": f"{res['rel_err_real']:.4f}",
                    "RelErr_imag": f"{res['rel_err_imag']:.4f}",
                    "MaxResidual": f"{np.max(res['residuals']):.2e}",
                })
                print(f"  {kname:8s} | {z_label:20s} | {h_label:12s} | "
                      f"MC={mc.real:.4f}+{mc.imag:.4f}j | "
                      f"Theo={th.real:.4f}+{th.imag:.4f}j | "
                      f"RelErr={res['rel_err_real']:.3f}")

    return results, pd.DataFrame(rows)


def verify_z0_consistency(results: dict) -> None:
    """Check that z=0 matches the simple affine transform (consistency test)."""
    print("\n--- Consistency check: z=0 should match simple affine transform ---")
    for key, res in results.items():
        kname, z_label, h_label = key
        if z_label == "z=0 (Laplace)":
            mc_r = res["mc_mean"].real
            theo_r = res["theo_val"].real
            mc_i = abs(res["mc_mean"].imag)
            print(f"  {kname}, {h_label}: MC.real={mc_r:.4f}, Theo.real={theo_r:.4f}, "
                  f"|MC.imag|={mc_i:.6f} (should be ~0)")


def plot_N_t(results: dict, grid: np.ndarray) -> None:
    """Plot N_t trajectories for different z/H cases."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, h_label in zip(axes, ["H_constant", "H_timedep"]):
        for key, res in results.items():
            kname, z_label, hl = key
            if hl == h_label and z_label == "z=i (imaginary)":
                N_t = res["N_t"]
                n_show = min(5, N_t.shape[0])
                for p in range(n_show):
                    ax.plot(grid, N_t[p], alpha=0.5)
                ax.set_title(f"N_t paths ({h_label})")
                ax.set_xlabel("$t$")
                ax.set_ylabel("$N_t$")
                ax.grid(True, alpha=0.3)
    plt.suptitle("Experiment 4: Brownian Functional N_t")
    plt.tight_layout()
    plt.savefig("results/exp4_N_t_paths.png", dpi=100)
    plt.close()
    print("Saved results/exp4_N_t_paths.png")


def plot_residuals_z_cases(results: dict, grid: np.ndarray) -> None:
    """Plot Riccati residuals for different z cases."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    knames = list({k[0] for k in results.keys()})
    for ax, kname in zip(axes, knames[:2]):
        for key, res in results.items():
            kn, z_label, h_label = key
            if kn == kname and h_label == "H_constant":
                ax.semilogy(grid, res["residuals"] + 1e-16, label=z_label)
        ax.set_title(f"Riccati Residuals ({kname})")
        ax.set_xlabel("$t$")
        ax.set_ylabel("Residual")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle("Experiment 4: Modified Riccati-Volterra Residuals")
    plt.tight_layout()
    plt.savefig("results/exp4_riccati_residuals.png", dpi=100)
    plt.close()
    print("Saved results/exp4_riccati_residuals.png")


if __name__ == "__main__":
    print("Running Experiment 4: Brownian Functional Extension")

    T, d, Q, alpha, M, X0, u = setup_parameters()
    kernels = {
        "markov": k_markov,
        "frac": partial(k_shifted_fractional, a=0.7, eps=0.1),
    }

    results, df_main = run_experiment(T, d, Q, alpha, M, X0, u, kernels, grid_size=50, n_paths=300)
    verify_z0_consistency(results)

    df_main.to_csv("results/exp4_brownian_functional.csv", index=False)
    print("\nSaved results/exp4_brownian_functional.csv")
    print("\nResults:")
    print(df_main.to_string(index=False))

    grid = np.linspace(0, T, 51)
    plot_N_t(results, grid)
    plot_residuals_z_cases(results, grid)

    print("\nExperiment 4 complete.")
