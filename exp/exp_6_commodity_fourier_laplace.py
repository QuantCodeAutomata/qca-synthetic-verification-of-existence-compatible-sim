"""Experiment 6: Validate full Fourier-Laplace transform in the Volterra-Wishart Gibson-Schwartz model.

Validates:
    Phi_t(u1, u2, u3) = exp(C_T + int_0^t chi sqrt(V) dZ
                             - (1-rho^T rho)/2 int_0^t chi V chi^T ds
                             - Y_t^{i,H})

Scenarios:
    - rho=0, rho^T rho in (0,1), rho^T rho=1
    - Laplace reduction (u1=u2=0)
    - Markovian vs fractional kernels
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
from src.commodity_model import simulate_commodity
from src.commodity_transform import (
    build_deterministic_objects,
    solve_commodity_riccati,
    compute_stochastic_integral_chi,
    compute_quadratic_integral_chi,
    compute_Y_t_commodity,
    compute_full_transform,
)
from src.diagnostics import summarize_psd_diagnostics

os.makedirs("results", exist_ok=True)


def setup_parameters():
    T = 1.0
    d = 2
    Q = np.array([[0.3, 0.05], [0.05, 0.25]])
    alpha = Q.T @ Q
    M = np.array([[-0.6, 0.05], [0.05, -0.5]])
    V0 = np.array([[1.0, 0.15], [0.15, 0.7]])
    kappa = 2.0
    mu = 0.03
    theta = 0.02
    S0 = 100.0
    logS0 = np.log(S0)
    delta0 = 0.02
    return T, d, Q, alpha, M, V0, kappa, mu, theta, logS0, delta0


def run_transform_scenario(
    T, d, Q, alpha, M, V0, kappa, mu, theta, logS0, delta0,
    rho, kfn, kname, rho_label, u1, u2, u3, grid, n_paths=300,
):
    """Run one full transform scenario."""
    N = len(grid)
    k0 = kfn(0.0)
    b = 3.0 * k0 * alpha

    # Simulate commodity model
    sim = simulate_commodity(
        V0=V0, delta0=delta0, logS0=logS0, grid=grid,
        Q=Q, b=b, M=M, kernel_fn=kfn,
        kappa=kappa, mu=mu, theta=theta, rho=rho,
        n_paths=n_paths, seed=42, project_V=True,
    )

    # Build deterministic objects
    det_objs = build_deterministic_objects(
        grid=grid, u1=u1, u2=u2, u3=u3,
        kappa=kappa, mu=mu, theta=theta,
        delta0=delta0, logS0=logS0, rho=rho,
        Q=Q, M=M,
    )

    # Solve Riccati-Volterra
    rv = solve_commodity_riccati(grid, u3, det_objs, alpha, kfn)
    psi = rv["psi"]
    residuals = rv["residual_norms"]

    # Full transform
    f_running = det_objs["f_running"]
    M_chi_arr = det_objs["M_chi"]
    transform_result = compute_full_transform(
        commodity_sim=sim,
        det_objs=det_objs,
        psi=psi,
        grid=grid,
        u1=u1, u2=u2, u3=u3,
        rho=rho, Q=Q, M=M, alpha=alpha, b=b,
        f_running=f_running,
        M_chi_arr=M_chi_arr,
        kappa=kappa, theta=theta,
    )

    Phi_t = transform_result["Phi_t"]
    payoffs = transform_result["payoffs"]
    mc_mean = complex(np.mean(payoffs))
    mc_std_real = float(np.std(payoffs.real) / np.sqrt(n_paths))

    # Theoretical value at t=0: Phi_0
    Phi_0 = complex(np.mean(Phi_t[:, 0]))
    rel_err_real = abs(mc_mean.real - Phi_0.real) / (abs(Phi_0.real) + 1e-10)

    # PSD diagnostics for V
    diag = summarize_psd_diagnostics(sim["V_min_eigs"], grid)

    # Min eigenvalue of Re psi(t)
    min_eig_psi = float(min(np.linalg.eigvalsh(np.real(psi[j])).min() for j in range(len(grid))))

    return {
        "psi": psi, "residuals": residuals, "sim": sim,
        "transform_result": transform_result,
        "mc_mean": mc_mean, "mc_std_real": mc_std_real,
        "Phi_0": Phi_0, "rel_err_real": rel_err_real,
        "diag": diag, "min_eig_psi": min_eig_psi,
    }


def run_all_scenarios(T, d, Q, alpha, M, V0, kappa, mu, theta, logS0, delta0,
                      kernels, grid_sizes, n_paths=200):
    """Run all full transform scenarios."""
    rows = []
    results = {}

    # Fourier-Laplace arguments
    u3_psd = np.array([[0.15, 0.02], [0.02, 0.12]])
    u_cases = {
        "Laplace (u1=u2=0)": (0.0 + 0j, 0.0 + 0j, u3_psd),
        "Fourier (u1=i, u2=0.5i)": (0.0 + 1j, 0.0 + 0.5j, u3_psd),
    }
    rho_cases = {
        "rho=0": np.array([0.0, 0.0]),
        "rho=(0.3,0.2)": np.array([0.3, 0.2]),
    }
    # Add rho=1 case (edge case)
    rho_cases["rho^T rho=1"] = np.array([1.0, 0.0])

    grid_size = max(grid_sizes)
    grid = np.linspace(0, T, grid_size + 1)

    for kname, kfn in kernels.items():
        for rho_label, rho in rho_cases.items():
            for u_label, (u1, u2, u3) in u_cases.items():
                print(f"  {kname:8s} | {rho_label:15s} | {u_label:30s}", end=" ")
                try:
                    res = run_transform_scenario(
                        T, d, Q, alpha, M, V0, kappa, mu, theta, logS0, delta0,
                        rho=rho, kfn=kfn, kname=kname, rho_label=rho_label,
                        u1=u1, u2=u2, u3=u3, grid=grid, n_paths=n_paths,
                    )
                    key = (kname, rho_label, u_label)
                    results[key] = res

                    mc = res["mc_mean"]
                    phi = res["Phi_0"]
                    rows.append({
                        "Kernel": kname, "Rho": rho_label, "Transform": u_label,
                        "MC_real": f"{mc.real:.5f}",
                        "MC_imag": f"{mc.imag:.5f}",
                        "MC_2std": f"{2*res['mc_std_real']:.5f}",
                        "Phi0_real": f"{phi.real:.5f}",
                        "Phi0_imag": f"{phi.imag:.5f}",
                        "RelErr_real": f"{res['rel_err_real']:.4f}",
                        "MaxResidual": f"{np.max(res['residuals']):.2e}",
                        "MinEig_psi": f"{res['min_eig_psi']:.4f}",
                        "V_ViolRate": f"{res['diag']['violation_rate']:.4f}",
                    })
                    print(f"| MC.re={mc.real:.4f} | Phi0.re={phi.real:.4f} | "
                          f"RelErr={res['rel_err_real']:.3f} | MaxRes={np.max(res['residuals']):.2e}")
                except Exception as e:
                    print(f"| ERROR: {e}")
                    rows.append({
                        "Kernel": kname, "Rho": rho_label, "Transform": u_label,
                        "MC_real": "ERROR", "MC_imag": "ERROR",
                        "MC_2std": "ERROR", "Phi0_real": "ERROR",
                        "Phi0_imag": "ERROR", "RelErr_real": "ERROR",
                        "MaxResidual": "ERROR", "MinEig_psi": "ERROR",
                        "V_ViolRate": "ERROR",
                    })

    return results, pd.DataFrame(rows)


def plot_M_t_martingale(results: dict, grid: np.ndarray) -> None:
    """Plot |M_t| - local martingale magnitude over time."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, ((kname, rho_label, u_label), res) in zip(axes, list(results.items())[:2]):
        M_t = np.abs(res["transform_result"]["M_t"])
        n_show = min(10, M_t.shape[0])
        for p in range(n_show):
            ax.plot(grid, M_t[p], alpha=0.4)
        ax.axhline(1.0, color="red", linestyle="--", label="|M|=1 (martingale)")
        ax.set_title(f"|M_t| ({kname}, {rho_label[:8]})")
        ax.set_xlabel("$t$")
        ax.set_ylabel("|M_t|")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle("Experiment 6: Local Martingale |M_t| over Time")
    plt.tight_layout()
    plt.savefig("results/exp6_M_t_martingale.png", dpi=100)
    plt.close()
    print("Saved results/exp6_M_t_martingale.png")


def plot_commodity_paths(results: dict, grid: np.ndarray) -> None:
    """Plot logS_t, delta_t, and V_11 paths."""
    key = list(results.keys())[0]
    res = results[key]
    sim = res["sim"]
    n_show = min(10, sim["logS_paths"].shape[0])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for p in range(n_show):
        axes[0].plot(grid, np.exp(sim["logS_paths"][p]), alpha=0.4)
        axes[1].plot(grid, sim["delta_paths"][p], alpha=0.4)
        axes[2].plot(grid, sim["V_paths"][p, :, 0, 0], alpha=0.4)
    for ax, label in zip(axes, ["Spot price S_t", "Convenience yield delta_t", "V_11"]):
        ax.set_title(label)
        ax.set_xlabel("$t$")
        ax.grid(True, alpha=0.3)
    plt.suptitle("Experiment 6: Commodity Model Sample Paths")
    plt.tight_layout()
    plt.savefig("results/exp6_commodity_paths.png", dpi=100)
    plt.close()
    print("Saved results/exp6_commodity_paths.png")


def plot_Phi_t_over_time(results: dict, grid: np.ndarray) -> None:
    """Plot Re(Phi_t) over time for selected scenarios."""
    keys_to_plot = list(results.keys())[:2]
    fig, axes = plt.subplots(1, len(keys_to_plot), figsize=(14, 5))
    if len(keys_to_plot) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys_to_plot):
        res = results[key]
        Phi_t = res["transform_result"]["Phi_t"]
        n_show = min(5, Phi_t.shape[0])
        for p in range(n_show):
            ax.plot(grid, Phi_t[p].real, alpha=0.5)
        kname, rho_label, u_label = key
        ax.set_title(f"Re(Phi_t) ({kname}, {rho_label[:8]})")
        ax.set_xlabel("$t$")
        ax.grid(True, alpha=0.3)
    plt.suptitle("Experiment 6: Transform Phi_t over Time")
    plt.tight_layout()
    plt.savefig("results/exp6_Phi_t.png", dpi=100)
    plt.close()
    print("Saved results/exp6_Phi_t.png")


if __name__ == "__main__":
    print("Running Experiment 6: Full Commodity Fourier-Laplace Transform")

    T, d, Q, alpha, M, V0, kappa, mu, theta, logS0, delta0 = setup_parameters()
    kernels = {
        "markov": k_markov,
        "frac": partial(k_shifted_fractional, a=0.7, eps=0.1),
    }
    grid_sizes = [50]

    results, df_main = run_all_scenarios(
        T, d, Q, alpha, M, V0, kappa, mu, theta, logS0, delta0,
        kernels=kernels, grid_sizes=grid_sizes, n_paths=200,
    )

    df_main.to_csv("results/exp6_commodity_transform.csv", index=False)
    print("\nSaved results/exp6_commodity_transform.csv")
    print("\nResults:")
    print(df_main.to_string(index=False))

    grid = np.linspace(0, T, max(grid_sizes) + 1)
    if results:
        plot_M_t_martingale(results, grid)
        plot_commodity_paths(results, grid)
        plot_Phi_t_over_time(results, grid)

    print("\nExperiment 6 complete.")
