import time
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../src")

from phi_inv import (
    compute_phi_inv_euler_grid,
    compute_phi_inv_rk4_grid,
    compute_phi_inv_direct_grid,
)

# ============================================================
# FNO settings
# ============================================================

USE_FNO = True
FNO_MODEL_PATH = "../models/fno_psi/fno_model.pt"

if USE_FNO:
    import torch
    from neuralop.models import FNO

# ============================================================
# Delay family
# ============================================================

def sample_delay_params(rng):
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


# ============================================================
# Settings
# ============================================================

SEED = 0
NUM_SAMPLES = 1000
T = 10.0
DT = 1e-3

rng = np.random.default_rng(SEED)

if USE_FNO:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(FNO_MODEL_PATH, map_location=device, weights_only=False)

    fno = FNO(
        n_modes=(32,),
        hidden_channels=64,
        in_channels=1,
        out_channels=1,
    ).to(device)

    fno.load_state_dict(ckpt["model_state_dict"])
    fno.eval()

    t_grid_fno = ckpt["t_grid"].cpu().numpy()

    x_mean = ckpt["x_mean"].to(device)
    x_std = ckpt["x_std"].to(device)
    y_mean = ckpt["y_mean"].to(device)
    y_std = ckpt["y_std"].to(device)

# ============================================================
# Aggregate statistics
# ============================================================

max_err_euler = []
max_err_rk4 = []
max_err_direct = []

mean_err_euler = []
mean_err_rk4 = []
mean_err_direct = []

max_diff_euler_vs_direct = []
max_diff_rk4_vs_direct = []

time_euler = []
time_rk4 = []
time_direct = []

if USE_FNO:
    max_err_fno = []
    mean_err_fno = []
    max_diff_fno_vs_direct = []
    time_fno = []

example_data = None

successful_samples = 0
attempts = 0
skipped_samples = 0

while successful_samples < NUM_SAMPLES:
    attempts += 1

    params = sample_delay_params(rng)
    phi = make_phi(params)
    dphi = make_dphi(params)

    t_min = float(phi(0.0))
    t_max = T

    try:
        start = time.perf_counter()
        t_e, rho_e = compute_phi_inv_euler_grid(
            phi=phi,
            phi_prime=dphi,
            t_min=t_min,
            t_max=t_max,
            dt=DT,
            residual_print=False,
        )
        runtime_e = time.perf_counter() - start

        start = time.perf_counter()
        t_rk, rho_rk = compute_phi_inv_rk4_grid(
            phi=phi,
            phi_prime=dphi,
            t_min=t_min,
            t_max=t_max,
            dt=DT,
            residual_print=False,
        )
        runtime_rk = time.perf_counter() - start

        start = time.perf_counter()
        t_d, rho_d = compute_phi_inv_direct_grid(
            phi=phi,
            phi_prime=dphi,
            t_min=t_min,
            t_max=t_max,
            dt=DT,
            residual_print=False,
        )
        runtime_d = time.perf_counter() - start

    except Exception as e:
        skipped_samples += 1
        print(f"[attempt {attempts}] skipped due to inverse failure: {e}")
        continue

    # If using FNO, evaluate it only after the reference/direct solve succeeded.
    if USE_FNO:
        try:
            D_vals = np.array([r - phi(r) for r in rho_d])
            D_interp = np.interp(t_grid_fno, t_d, D_vals)

            if device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            with torch.no_grad():
                x = torch.tensor(D_interp, dtype=torch.float32).view(1, 1, -1).to(device)
                x = (x - x_mean) / x_std

                psi_pred = fno(x)
                psi_pred = psi_pred * y_std + y_mean
                psi_pred = psi_pred.cpu().numpy()[0, 0]

            if device == "cuda":
                torch.cuda.synchronize()

            runtime_fno = time.perf_counter() - start
            rho_fno = t_grid_fno + psi_pred

        except Exception as e:
            skipped_samples += 1
            print(f"[attempt {attempts}] skipped due to FNO failure: {e}")
            continue

    # Only now is this sample accepted.
    successful_samples += 1

    err_e = np.abs(np.array([phi(r) for r in rho_e]) - t_e)
    err_rk = np.abs(np.array([phi(r) for r in rho_rk]) - t_rk)
    err_d = np.abs(np.array([phi(r) for r in rho_d]) - t_d)

    max_err_euler.append(np.max(err_e))
    max_err_rk4.append(np.max(err_rk))
    max_err_direct.append(np.max(err_d))

    mean_err_euler.append(np.mean(err_e))
    mean_err_rk4.append(np.mean(err_rk))
    mean_err_direct.append(np.mean(err_d))

    max_diff_euler_vs_direct.append(np.max(np.abs(rho_e - rho_d)))
    max_diff_rk4_vs_direct.append(np.max(np.abs(rho_rk - rho_d)))

    time_euler.append(runtime_e)
    time_rk4.append(runtime_rk)
    time_direct.append(runtime_d)

    if USE_FNO:
        err_fno = np.abs(np.array([phi(r) for r in rho_fno]) - t_grid_fno)
        rho_direct_interp = np.interp(t_grid_fno, t_d, rho_d)

        max_err_fno.append(np.max(err_fno))
        mean_err_fno.append(np.mean(err_fno))
        max_diff_fno_vs_direct.append(np.max(np.abs(rho_fno - rho_direct_interp)))
        time_fno.append(runtime_fno)

    if example_data is None:
        example_data = {
            "params": params,
            "phi": phi,
            "t_e": t_e,
            "rho_e": rho_e,
            "t_rk": t_rk,
            "rho_rk": rho_rk,
            "t_d": t_d,
            "rho_d": rho_d,
            "err_e": err_e,
            "err_rk": err_rk,
            "err_d": err_d,
        }

        if USE_FNO:
            example_data["t_fno"] = t_grid_fno
            example_data["rho_fno"] = rho_fno
            example_data["err_fno"] = err_fno

    if successful_samples % max(1, NUM_SAMPLES // 10) == 0:
        print(f"[accepted {successful_samples:4d}/{NUM_SAMPLES}] after {attempts} attempts")
# ============================================================
# Summary
# ============================================================

print()
print(f"Accepted samples: {successful_samples}/{NUM_SAMPLES}")
print(f"Total attempts:   {attempts}")
print(f"Rejected samples: {skipped_samples}")
print(f"Acceptance rate:  {successful_samples / attempts:.2%}")

n_ok = len(max_err_euler)
print()
print(f"Successful samples: {n_ok}/{NUM_SAMPLES}")

if n_ok == 0:
    raise RuntimeError("No successful samples were processed.")


def summarize(name, arr):
    arr = np.asarray(arr, dtype=float)
    print(
        f"{name:<28s} mean={np.mean(arr):.6e}   "
        f"median={np.median(arr):.6e}   max={np.max(arr):.6e}"
    )


print()
print("Consistency error summary across samples")
summarize("Euler max residual", max_err_euler)
summarize("RK4 max residual", max_err_rk4)
summarize("Direct max residual", max_err_direct)
if USE_FNO:
    summarize("FNO max residual", max_err_fno)

summarize("Euler mean residual", mean_err_euler)
summarize("RK4 mean residual", mean_err_rk4)
summarize("Direct mean residual", mean_err_direct)
if USE_FNO:
    summarize("FNO mean residual", mean_err_fno)

print()
print("Difference from direct method across samples")
summarize("Euler max |rho-rho_d|", max_diff_euler_vs_direct)
summarize("RK4 max |rho-rho_d|", max_diff_rk4_vs_direct)
if USE_FNO:
    summarize("FNO max |rho-rho_d|", max_diff_fno_vs_direct)

print()
print("Runtime summary across samples [seconds]")
summarize("Euler runtime", time_euler)
summarize("RK4 runtime", time_rk4)
summarize("Direct runtime", time_direct)
if USE_FNO:
    summarize("FNO runtime", time_fno)

# ============================================================
# Aggregate bar plots
# ============================================================

avg_max_residuals = [
    np.mean(max_err_euler),
    np.mean(max_err_rk4),
    np.mean(max_err_direct),
]

avg_mean_residuals = [
    np.mean(mean_err_euler),
    np.mean(mean_err_rk4),
    np.mean(mean_err_direct),
]

avg_runtimes = [
    np.mean(time_euler),
    np.mean(time_rk4),
    np.mean(time_direct),
]

method_names = ["Euler", "RK4", "Direct"]

if USE_FNO:
    avg_max_residuals.append(np.mean(max_err_fno))
    avg_mean_residuals.append(np.mean(mean_err_fno))
    avg_runtimes.append(np.mean(time_fno))
    method_names.append("FNO")

plt.figure(figsize=(8, 4))
plt.bar(method_names, avg_max_residuals)
plt.yscale("log")
plt.title("Average max consistency error across samples")
plt.ylabel(r"average of $\max_t |\phi(\rho(t)) - t|$")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.bar(method_names, avg_mean_residuals)
plt.yscale("log")
plt.title("Average mean consistency error across samples")
plt.ylabel(r"average of mean $|\phi(\rho(t)) - t|$")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.bar(method_names, avg_runtimes)
plt.yscale("log")
plt.title("Average runtime across samples")
plt.ylabel("seconds")
plt.tight_layout()
plt.show()

# ============================================================
# Accuracy-vs-runtime scatter
# ============================================================

plt.figure(figsize=(7, 5))
plt.scatter(np.mean(time_euler), np.mean(max_err_euler), label="Euler", s=80)
plt.scatter(np.mean(time_rk4), np.mean(max_err_rk4), label="RK4", s=80)
plt.scatter(np.mean(time_direct), np.mean(max_err_direct), label="Direct", s=80)
if USE_FNO:
    plt.scatter(np.mean(time_fno), np.mean(max_err_fno), label="FNO", s=80)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("average runtime [s]")
plt.ylabel(r"average max residual")
plt.title("Accuracy-runtime tradeoff")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# One-example plots
# ============================================================

phi = example_data["phi"]
t_e = example_data["t_e"]
rho_e = example_data["rho_e"]
t_rk = example_data["t_rk"]
rho_rk = example_data["rho_rk"]
t_d = example_data["t_d"]
rho_d = example_data["rho_d"]
err_e = example_data["err_e"]
err_rk = example_data["err_rk"]
err_d = example_data["err_d"]
params = example_data["params"]

if USE_FNO:
    t_fno = example_data["t_fno"]
    rho_fno = example_data["rho_fno"]
    err_fno = example_data["err_fno"]

print()
print("Plotted example params =", params)

plt.figure(figsize=(10, 5))
plt.plot(t_e, rho_e - t_e, label="Euler")
plt.plot(t_rk, rho_rk - t_rk, label="RK4")
plt.plot(t_d, rho_d - t_d, "--", linewidth=2, label="Direct")
if USE_FNO:
    plt.plot(t_fno, rho_fno - t_fno, linewidth=2, label="FNO")
plt.title(r"Example inverse curves $\psi(t)$")
plt.xlabel("t")
plt.ylabel(r"$\psi(t)$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(t_e, err_e, label="Euler")
plt.plot(t_rk, err_rk, label="RK4")
plt.plot(t_d, err_d, "--", linewidth=2, label="Direct")
if USE_FNO:
    plt.plot(t_fno, err_fno, linewidth=2, label="FNO")
plt.yscale("log")
plt.title(r"Example consistency error $|\phi(\rho(t)) - t|$")
plt.xlabel("t")
plt.ylabel("absolute error")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(t_e, np.abs(rho_e - rho_d), label=r"$|\rho_{\mathrm{Euler}}-\rho_{\mathrm{Direct}}|$")
plt.plot(t_rk, np.abs(rho_rk - rho_d), label=r"$|\rho_{\mathrm{RK4}}-\rho_{\mathrm{Direct}}|$")
if USE_FNO:
    rho_d_on_fno = np.interp(t_fno, t_d, rho_d)
    plt.plot(t_fno, np.abs(rho_fno - rho_d_on_fno), label=r"$|\rho_{\mathrm{FNO}}-\rho_{\mathrm{Direct}}|$")
plt.yscale("log")
plt.title(r"Example difference from direct inverse")
plt.xlabel("t")
plt.ylabel("absolute difference")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()