import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append("../src")

from delay_check import check_delay_functions
from phi_inv import compute_phi_inv_direct_grid, make_phi_inv_from_grid_pchip


# ============================================================
# User settings
# ============================================================

N_SAMPLES = 2000
T = 10.0
DT_GRID = 1e-3

OUTFILE = "../dataset/psi_dataset.pt"
SEED = 0


# ============================================================
# Delay family
# ============================================================
def sample_delay_params(rng):
    """
    Sample parameters for the delay family
        phi(t) = t - (a + b/(1+t) + alpha*sin(omega t + varphi))
    with conservative ranges for numerical stability.
    """
    a = rng.uniform(0.1, 0.5)
    b = rng.uniform(0.1, 0.5)
    alpha = rng.uniform(-0.3, 0.3)
    omega = rng.uniform(0, 2*np.pi)
    varphi = rng.uniform(0.0, np.pi)

    return a, b, alpha, omega, varphi


def make_phi(params):
    a, b, alpha, omega, varphi = params

    def phi(t):
        return t - (
            a
            + b / (1.0 + t)
            + alpha * np.sin(omega * t + varphi)
        )

    return phi


def make_dphi(params):
    a, b, alpha, omega, varphi = params

    def dphi(t):
        return (
            1.0
            + b / (1.0 + t) ** 2
            - alpha * omega * np.cos(omega * t + varphi)
        )

    return dphi


def make_D(phi):
    def D(t):
        return t - phi(t)
    return D


# ============================================================
# Helpers
# ============================================================

def make_uniform_grid(t_min, t_max, dt):
    n = int(np.round((t_max - t_min) / dt)) + 1
    grid = t_min + dt * np.arange(n, dtype=float)
    grid[-1] = t_max
    return grid


def sample_valid_delay(rng, T, dt, max_tries=200):
    """
    Sample one delay law, build phi^{-1} on [0, T] using the direct method,
    and check validity.
    """
    for _ in range(max_tries):
        params = sample_delay_params(rng)

        phi = make_phi(params)
        dphi = make_dphi(params)
        D = make_D(phi)

        # quick screening on [0, T]
        probe = np.linspace(0.0, T, 4000)
        D_vals = np.array([D(t) for t in probe], dtype=float)
        dphi_vals = np.array([dphi(t) for t in probe], dtype=float)

        if np.min(D_vals) <= 1e-8:
            continue
        if np.min(dphi_vals) <= 1e-8:
            continue

        # Need phi^{-1}(t) for t in [0, T], so 0 must lie in the range of phi.
        # Since phi is increasing and phi(0) is the minimum on t >= 0,
        # we need phi(0) <= 0.
        if phi(0.0) > 1e-10:
            continue

        try:
            t_grid_inv, rho_grid = compute_phi_inv_direct_grid(
                phi=phi,
                phi_prime=dphi,
                t_min=0.0,
                t_max=T,
                dt=dt,
                residual_print=False,
            )
            phi_inv = make_phi_inv_from_grid_pchip(t_grid_inv, rho_grid)
        except Exception:
            continue

        try:
            results = check_delay_functions(
                T=T,
                D1=D,
                D2=D,
                phi1=phi,
                phi2=phi,
                dphi1=dphi,
                dphi2=dphi,
                phi1_inv=phi_inv,
                n_grid=4000,
                tol=1e-8,
                verbose=False,
            )
        except Exception:
            continue

        if results["all_ok"]:
            return {
                "params": np.array(params, dtype=float),
                "phi": phi,
                "dphi": dphi,
                "D": D,
                "phi_inv": phi_inv,
                "t_grid_inv": t_grid_inv,
                "rho_grid": rho_grid,
                "delay_check": results,
            }

    raise RuntimeError(f"Could not find valid delay in {max_tries} tries.")


def build_single_sample(valid_delay, t_grid):
    """
    Build one dataset example on [0, T]:
        input  = D(t)
        output = Psi(t) = phi^{-1}(t) - t
    """
    phi = valid_delay["phi"]
    dphi = valid_delay["dphi"]
    D = valid_delay["D"]
    params = valid_delay["params"]
    t_grid_inv = valid_delay["t_grid_inv"]
    rho_grid = valid_delay["rho_grid"]
    delay_check = valid_delay["delay_check"]

    rho_on_grid = np.interp(t_grid, t_grid_inv, rho_grid)
    psi_grid = rho_on_grid - t_grid

    D_grid = np.array([D(t) for t in t_grid], dtype=float)
    phi_grid = np.array([phi(t) for t in t_grid], dtype=float)
    dphi_grid = np.array([dphi(t) for t in t_grid], dtype=float)

    consistency_err = np.max(np.abs(phi(t_grid + psi_grid) - t_grid))

    return {
        "D": D_grid,
        "phi": phi_grid,
        "dphi": dphi_grid,
        "psi": psi_grid,
        "params": np.array(params, dtype=float),
        "consistency_err": float(consistency_err),
        "delay_check": delay_check,
    }


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)

    rng = np.random.default_rng(SEED)
    t_grid = make_uniform_grid(0.0, T, DT_GRID)
    N_GRID = len(t_grid)

    X_list = []
    Y_list = []
    params_list = []
    err_list = []

    viz_samples = []

    for k in range(N_SAMPLES):
        valid_delay = sample_valid_delay(rng, T=T, dt=DT_GRID)
        sample = build_single_sample(valid_delay, t_grid)

        # input: D(t), output: Psi(t)
        x = sample["D"][None, :]     # shape (1, N)
        y = sample["psi"][None, :]   # shape (1, N)

        X_list.append(x.astype(np.float32))
        Y_list.append(y.astype(np.float32))
        params_list.append(sample["params"].astype(np.float32))
        err_list.append(sample["consistency_err"])

        if len(viz_samples) < 5:
            viz_samples.append((sample["D"].copy(), sample["psi"].copy()))

        if (k + 1) % max(1, N_SAMPLES // 10) == 0:
            print(f"[{k+1:5d}/{N_SAMPLES}] built samples")

    X = np.stack(X_list, axis=0)   # (N_samp, 1, N_grid)
    Y = np.stack(Y_list, axis=0)   # (N_samp, 1, N_grid)
    params = np.stack(params_list, axis=0)
    errs = np.array(err_list, dtype=np.float32)

    x_mean = X.mean(axis=(0, 2), keepdims=True)
    x_std = X.std(axis=(0, 2), keepdims=True) + 1e-8
    y_mean = Y.mean(axis=(0, 2), keepdims=True)
    y_std = Y.std(axis=(0, 2), keepdims=True) + 1e-8

    payload = {
        "X": torch.tensor(X),                 # (N_samp, 1, N_grid)
        "Y": torch.tensor(Y),                 # (N_samp, 1, N_grid)
        "t_grid": torch.tensor(t_grid, dtype=torch.float32),
        "params": torch.tensor(params),
        "consistency_err": torch.tensor(errs),
        "x_mean": torch.tensor(x_mean, dtype=torch.float32),
        "x_std": torch.tensor(x_std, dtype=torch.float32),
        "y_mean": torch.tensor(y_mean, dtype=torch.float32),
        "y_std": torch.tensor(y_std, dtype=torch.float32),
        "meta": {
            "T": T,
            "N_grid": N_GRID,
            "N_samples": N_SAMPLES,
            "DT_grid": DT_GRID,
            "input_channels": ["D"],
            "output_channels": ["psi"],
            "domain": [0.0, T],
            "phi_inv_method": "direct+pchip",
        },
    }

    torch.save(payload, OUTFILE)

    print()
    print("Saved dataset to:", OUTFILE)
    print("X shape:", tuple(payload["X"].shape))
    print("Y shape:", tuple(payload["Y"].shape))
    print("Max inverse consistency error:", float(payload["consistency_err"].max()))

    # --------------------------------------------------------
    # Small visualization
    # --------------------------------------------------------
    print("\nVisualizing a few dataset samples...")

    n_plot = len(viz_samples)
    plt.figure(figsize=(12, 4 * n_plot))

    for i, (D_plot, psi_plot) in enumerate(viz_samples):
        ax1 = plt.subplot(n_plot, 2, 2 * i + 1)
        ax1.plot(t_grid, D_plot)
        ax1.set_title(f"Sample {i}: D(t)")
        ax1.set_xlabel("t")
        ax1.set_ylabel("D(t)")
        ax1.grid(True)

        ax2 = plt.subplot(n_plot, 2, 2 * i + 2)
        ax2.plot(t_grid, psi_plot)
        ax2.set_title(f"Sample {i}: Psi(t)")
        ax2.set_xlabel("t")
        ax2.set_ylabel("Psi(t)")
        ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()