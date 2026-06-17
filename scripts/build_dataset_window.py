"""
Build a SHORT-WINDOW dataset for the inverse-delay operator.
============================================================

The main dataset (build_dataset.py) maps the delay D on the full horizon
[0, T] to the prediction horizon psi on [0, T]. That trains an operator whose
single evaluation is non-causal.

Here we instead train an operator on short windows of length W. Each sample is

    input  : D(t0 + s),                 s in [0, W]
    output : psi(t0 + s) = phi^{-1}(t0 + s) - (t0 + s)

for a random anchor t0 in [0, T0_MAX]. By translation equivariance of the
inverse-delay operator, Psi(D(t0 + .))(s) = psi(t0 + s), so an operator trained
on these local windows can be applied at any re-initialization point t_k at
runtime. Randomizing t0 makes the training distribution match the shifted
windows seen during recursive deployment (the b/(1+t) term flattens with t0).

The targets are computed by exact pointwise inversion of phi on the window,
which legitimately uses D beyond the window -- this is what lets the operator
learn the (small) right-edge behavior as well as it can.
"""

import os
import sys

import numpy as np
import torch

sys.path.append("../src")

from phi_inv import _invert_phi_pointwise


# ============================================================
# User settings
# ============================================================

N_SAMPLES = 2000
W = 2.0            # window length (operator input support)
T0_MAX = 12.0      # maximum anchor offset (covers deployment t_k in [0, T])
DT_GRID = 1e-3
PSI_GUARD = 2.0    # extra margin (>= psi_max) used only for validity screening

OUTFILE = "../dataset/psi_dataset_window.pt"
SEED = 0


# ============================================================
# Delay family (identical to build_dataset.py)
# ============================================================

def sample_delay_params(rng):
    a = rng.uniform(0.1, 0.5)
    b = rng.uniform(0.1, 0.5)
    alpha = rng.uniform(-0.3, 0.3)
    omega = rng.uniform(0, 2 * np.pi)
    varphi = rng.uniform(0.0, np.pi)
    return a, b, alpha, omega, varphi


def make_phi(params):
    a, b, alpha, omega, varphi = params

    def phi(t):
        return t - (a + b / (1.0 + t) + alpha * np.sin(omega * t + varphi))

    return phi


def make_dphi(params):
    a, b, alpha, omega, varphi = params

    def dphi(t):
        return 1.0 + b / (1.0 + t) ** 2 - alpha * omega * np.cos(omega * t + varphi)

    return dphi


def make_D(phi):
    def D(t):
        return t - phi(t)

    return D


# ============================================================
# Sampling one valid window
# ============================================================

def sample_valid_window(rng, W, dt, max_tries=500):
    """
    Sample one delay law and a random anchor t0, then build the windowed
    (D, psi) pair on [t0, t0 + W]. Screen the assumptions on a slightly larger
    interval so the exact inversion (which reaches a bit past t0 + W) is valid.
    """
    for _ in range(max_tries):
        params = sample_delay_params(rng)
        t0 = float(rng.uniform(0.0, T0_MAX))

        phi = make_phi(params)
        dphi = make_dphi(params)
        D = make_D(phi)

        # screen D > 0 and phi' > 0 on [0, t0 + W + guard]
        probe = np.linspace(0.0, t0 + W + PSI_GUARD, 4000)
        D_probe = np.array([D(t) for t in probe], dtype=float)
        dphi_probe = np.array([dphi(t) for t in probe], dtype=float)

        if np.min(D_probe) <= 1e-8:
            continue
        if np.min(dphi_probe) <= 1e-8:
            continue
        # need 0 reachable by phi so phi^{-1}(t0) exists for t0 >= 0
        if phi(0.0) > 1e-10:
            continue

        # fixed-length local grid so every sample has identical shape
        n_grid = int(round(W / dt)) + 1
        s_grid = np.linspace(0.0, W, n_grid)
        t_grid_abs = t0 + s_grid

        try:
            rho_grid = _invert_phi_pointwise(phi, t_grid_abs)
        except Exception:
            continue

        psi_grid = rho_grid - t_grid_abs
        consistency = np.max(np.abs(np.array([phi(r) for r in rho_grid]) - t_grid_abs))
        if consistency > 1e-6:
            continue

        D_window = np.array([D(t) for t in t_grid_abs], dtype=float)

        return {
            "params": np.array(params, dtype=float),
            "t0": t0,
            "s_grid": s_grid,
            "D": D_window,
            "psi": psi_grid,
            "consistency_err": float(consistency),
        }

    raise RuntimeError(f"Could not find valid window in {max_tries} tries.")


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)

    rng = np.random.default_rng(SEED)

    X_list = []
    Y_list = []
    params_list = []
    t0_list = []
    err_list = []
    s_grid = None

    for k in range(N_SAMPLES):
        sample = sample_valid_window(rng, W=W, dt=DT_GRID)

        if s_grid is None:
            s_grid = sample["s_grid"]

        x = sample["D"][None, :].astype(np.float32)     # (1, N)
        y = sample["psi"][None, :].astype(np.float32)   # (1, N)

        X_list.append(x)
        Y_list.append(y)
        params_list.append(sample["params"].astype(np.float32))
        t0_list.append(np.float32(sample["t0"]))
        err_list.append(sample["consistency_err"])

        if (k + 1) % max(1, N_SAMPLES // 10) == 0:
            print(f"[{k+1:5d}/{N_SAMPLES}] built windows")

    X = np.stack(X_list, axis=0)   # (N_samp, 1, N_grid)
    Y = np.stack(Y_list, axis=0)
    params = np.stack(params_list, axis=0)
    t0s = np.array(t0_list, dtype=np.float32)
    errs = np.array(err_list, dtype=np.float32)

    x_mean = X.mean(axis=(0, 2), keepdims=True)
    x_std = X.std(axis=(0, 2), keepdims=True) + 1e-8
    y_mean = Y.mean(axis=(0, 2), keepdims=True)
    y_std = Y.std(axis=(0, 2), keepdims=True) + 1e-8

    payload = {
        "X": torch.tensor(X),
        "Y": torch.tensor(Y),
        "t_grid": torch.tensor(s_grid, dtype=torch.float32),   # local window coord [0, W]
        "params": torch.tensor(params),
        "t0": torch.tensor(t0s),
        "consistency_err": torch.tensor(errs),
        "x_mean": torch.tensor(x_mean, dtype=torch.float32),
        "x_std": torch.tensor(x_std, dtype=torch.float32),
        "y_mean": torch.tensor(y_mean, dtype=torch.float32),
        "y_std": torch.tensor(y_std, dtype=torch.float32),
        "meta": {
            "W": W,
            "T0_MAX": T0_MAX,
            "N_grid": int(X.shape[2]),
            "N_samples": N_SAMPLES,
            "DT_grid": DT_GRID,
            "input_channels": ["D"],
            "output_channels": ["psi"],
            "domain": [0.0, W],
            "anchored": "random t0 in [0, T0_MAX]",
            "phi_inv_method": "direct",
        },
    }

    torch.save(payload, OUTFILE)

    print()
    print("Saved windowed dataset to:", OUTFILE)
    print("X shape:", tuple(payload["X"].shape))
    print("Y shape:", tuple(payload["Y"].shape))
    print("Window length W:", W, " N_grid:", int(X.shape[2]))
    print("Max inverse consistency error:", float(payload["consistency_err"].max()))


if __name__ == "__main__":
    main()
