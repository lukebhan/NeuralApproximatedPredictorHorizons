"""
Recursive short-window operator vs. batch FNO predictor feedback.
=================================================================

The main output-feedback example evaluates a long-horizon FNO once on the whole
interval [0, T] to produce the prediction horizon psi-hat(t). Because the FNO
uses global (Fourier) layers, that single-shot evaluation makes psi-hat(t)
depend on the delay D at *all* times, including the future -- i.e. it is
non-causal.

Here we use a SHORT-window operator (trained on windows of length W) that is
applied recursively: at re-initialization points t_k = k*Delta we evaluate it on
[t_k, t_k + W] and consume only the reliable interior [t_k, t_k + Delta]. Using
translation equivariance of the inverse-delay operator,
Psi(D(t_k + .))(s) = psi(t_k + s), each window recovers psi locally. This needs
the delay only W seconds ahead (a finite, T-independent lookahead) and the
operator is evaluated in-distribution (same window length it was trained on).
This is exactly the Corollary 1 mechanism: re-approximate psi on successive
compact intervals.

We run the *same* closed-loop system with three inverses and overlay them:
  (i)   batch: one long-horizon FNO call over [0, T]            (non-causal)
  (ii)  recursive: short operator, W=2, consume interior Delta=1 (causal, clean)
  (iii) strict sliding ("Option A"): the same short operator slid forward in
        non-overlapping blocks -- feed D on [t_k, t_k+H], use psi on the whole
        window, slide by H (W=2, Delta=2). Maximally strict on causality, but
        consumes the window's underdetermined right edge, so its residual is
        expected to bump up near each block boundary.
Matching trajectories convert the causality argument into a demonstration.

Workflow:
    1. python build_dataset_window.py
    2. python train_fno.py --data ../dataset/psi_dataset_window.pt \
           --save_dir ../models/fno_psi_window --no_plots
    3. python causal_sliding_window_example.py
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator
from matplotlib.lines import Line2D

sys.path.append("../src")

from simulate import simulate
from utils import set_size
from controller import PredictorFeedbackController
from delay_check import check_delay_functions
from phi_inv import (
    load_fno_inverse_model,
    compute_phi_inv_fno_grid,
    compute_phi_inv_fno_sliding_grid,
    make_phi_inv_from_grid_linear,
)

# ============================================================
# Operator settings
# ============================================================

BATCH_MODEL_PATH = "../models/fno_psi/fno_model.pt"          # long-horizon, batch
SHORT_MODEL_PATH = "../models/fno_psi_window/fno_model.pt"   # short-window operator

W_WINDOW = 2.0        # operator input support (= training window length)
REINIT_STRIDE = 1.0   # Delta: re-initialization stride (consumed interior)

# "Option A" strict sliding: same operator, fixed window H, slid forward in
# non-overlapping blocks (feed D on [t_k, t_k+H], use psi on the whole window,
# slide by H). Set stride = window so each block is consumed in full.
SLIDE_WINDOW = 2.0
SLIDE_STRIDE = 2.0

# ============================================================
# System data (identical to output_feedback_example.py)
# ============================================================

A = np.array([[0.0, 1.0],
              [2.0, 1.0]])

B = np.array([[0.0],
              [1.0]])

C = np.array([[1.0, -1.0]])

K = np.array([[-4.0, -4.0]])

L = np.array([[-4],
              [-8]])

print("eig(A):", np.linalg.eigvals(A))
print("eig(A+BK):", np.linalg.eigvals(A + B @ K))
print("eig(A-LC):", np.linalg.eigvals(A - L @ C))

# ============================================================
# Delay family (identical sampling to output_feedback_example.py)
# ============================================================

rng = np.random.default_rng(1)


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


def sample_valid_delay_pair(
    rng, T, dt, max_tries=100,
    inverse_grid_builder=compute_phi_inv_fno_grid,
    inverse_callable_builder=make_phi_inv_from_grid_linear,
    model=None,
):
    """Same procedure (and rng draw order) as output_feedback_example.py, so the
    sampled delay pair is identical for a given seed."""
    for _ in range(max_tries):
        params1 = sample_delay_params(rng)
        params2 = sample_delay_params(rng)

        phi1 = make_phi(params1)
        phi2 = make_phi(params2)
        dphi1 = make_dphi(params1)
        dphi2 = make_dphi(params2)
        D1 = make_D(phi1)
        D2 = make_D(phi2)

        try:
            t_d, rho_d = inverse_grid_builder(
                phi=phi1, phi_prime=dphi1, t_min=0.0, t_max=T, dt=1e-3, model=model,
            )
            phi1_inv = inverse_callable_builder(t_d, rho_d)
        except (ValueError, FloatingPointError):
            print("VALUE ERROR in inverse computation, retrying...")
            continue

        results = check_delay_functions(
            T=T, D1=D1, D2=D2, phi1=phi1, phi2=phi2,
            dphi1=dphi1, dphi2=dphi2, phi1_inv=phi1_inv,
            n_grid=4000, tol=1e-2, verbose=False,
        )

        if results["all_ok"]:
            return (params1, params2, phi1, phi2, dphi1, dphi2,
                    D1, D2, phi1_inv, t_d, rho_d, results)
        else:
            print("\nDelay check failed, resampling...")

    raise RuntimeError(f"Could not sample a valid delay pair in {max_tries} tries.")


# ============================================================
# Histories (identical to output_feedback_example.py)
# ============================================================

def z_history(t):
    return np.array([-1, 1])


def u_history(t):
    return np.array([0.0])


xi0 = np.array([5, -5])

# ============================================================
# Simulation setup
# ============================================================

T = 12.0
dt = 1e-3

if not os.path.exists(SHORT_MODEL_PATH):
    raise FileNotFoundError(
        f"Short-window model not found at {SHORT_MODEL_PATH}.\n"
        "Build and train it first:\n"
        "  python build_dataset_window.py\n"
        "  python train_fno.py --data ../dataset/psi_dataset_window.pt "
        "--save_dir ../models/fno_psi_window --no_plots"
    )

batch_model = load_fno_inverse_model(BATCH_MODEL_PATH)
short_model = load_fno_inverse_model(SHORT_MODEL_PATH)

# Sample the delay pair with the batch model so it matches output_feedback_example.py
(params1, params2, phi1, phi2, dphi1, dphi2,
 D1, D2, _phi1_inv, _t_d, _rho_d, results) = sample_valid_delay_pair(
    rng=rng, T=T, dt=dt, max_tries=200,
    inverse_grid_builder=compute_phi_inv_fno_grid,
    inverse_callable_builder=make_phi_inv_from_grid_linear,
    model=batch_model,
)

print("phi1 parameters:", params1)
print("phi2 parameters:", params2)

# ============================================================
# Build the two inverse mappings
#   batch    : single long-horizon FNO evaluation on [0, T]  (non-causal)
#   recursive: short-window FNO re-evaluated on [t_k, t_k+W]  (causal)
# ============================================================

t_batch, rho_batch = compute_phi_inv_fno_grid(
    phi=phi1, phi_prime=dphi1, t_min=0.0, t_max=T, dt=dt, model=batch_model,
)
phi1_inv_batch = make_phi_inv_from_grid_linear(t_batch, rho_batch)

t_rec, rho_rec = compute_phi_inv_fno_sliding_grid(
    phi=phi1, phi_prime=dphi1, t_min=0.0, t_max=T, dt=dt, model=short_model,
    window=W_WINDOW, reinit_stride=REINIT_STRIDE,
)
phi1_inv_rec = make_phi_inv_from_grid_linear(t_rec, rho_rec)

# strict sliding (Option A): non-overlapping H-blocks of the same operator
t_sld, rho_sld = compute_phi_inv_fno_sliding_grid(
    phi=phi1, phi_prime=dphi1, t_min=0.0, t_max=T, dt=dt, model=short_model,
    window=SLIDE_WINDOW, reinit_stride=SLIDE_STRIDE,
)
phi1_inv_sld = make_phi_inv_from_grid_linear(t_sld, rho_sld)

# ============================================================
# Run the same closed loop twice
# ============================================================

system = {"A": A, "B": B, "C": C, "K": K, "L": L}
init = {"u_history": u_history, "z_history": z_history, "xi0": xi0}


def run_closed_loop(phi1_inv):
    delays = {
        "phi1": phi1, "phi2": phi2,
        "dphi1": dphi1, "dphi2": dphi2,
        "phi1_inv": phi1_inv,
    }
    controller = PredictorFeedbackController(
        system=system, delays=delays, init=init, n_quad=40,
    )
    t, Z, Y, U = simulate(
        T=T, dt=dt, system=system, delays=delays,
        history=init, controller=controller, verbose=False,
    )
    return t, Z, Y, U, controller


print("\n=== Running BATCH (single-shot long-horizon FNO) closed loop ===")
t, Z_b, Y_b, U_b, ctrl_b = run_closed_loop(phi1_inv_batch)

print(f"=== Running RECURSIVE short-window FNO (W={W_WINDOW}, "
      f"Delta={REINIT_STRIDE}) closed loop ===")
t, Z_s, Y_s, U_s, ctrl_s = run_closed_loop(phi1_inv_rec)

print(f"=== Running STRICT SLIDING (Option A: W={SLIDE_WINDOW}, "
      f"Delta={SLIDE_STRIDE}) closed loop ===")
t, Z_a, Y_a, U_a, ctrl_a = run_closed_loop(phi1_inv_sld)

# ============================================================
# Curves and differences
# ============================================================

D1_vals = np.array([D1(tt) for tt in t])
D2_vals = np.array([D2(tt) for tt in t])

rho_b_on_t = np.interp(t, t_batch, rho_batch)
rho_r_on_t = np.interp(t, t_rec, rho_rec)
rho_a_on_t = np.interp(t, t_sld, rho_sld)

psi_b = rho_b_on_t - t   # batch (long-horizon) prediction horizon
psi_s = rho_r_on_t - t   # recursive short-window prediction horizon
psi_a = rho_a_on_t - t   # strict sliding (Option A) prediction horizon

# consistency residual of each APPLIED horizon: phi(psi_hat(t) + t) - t
resid_b = np.array([phi1(r) - tt for tt, r in zip(t, rho_b_on_t)])
resid_s = np.array([phi1(r) - tt for tt, r in zip(t, rho_r_on_t)])
resid_a = np.array([phi1(r) - tt for tt, r in zip(t, rho_a_on_t)])

reinit_pts = np.arange(0.0, T + 1e-9, SLIDE_STRIDE)

print("\n=== Batch vs. recursive vs. strict sliding (Option A) ===")
print(f"max |Z_batch - Z_rec|         : {np.max(np.abs(Z_b - Z_s)):.3e}")
print(f"max |Z_batch - Z_slide|       : {np.max(np.abs(Z_b - Z_a)):.3e}")
print(f"max residual (batch)          : {np.max(np.abs(resid_b)):.3e}")
print(f"max residual (recursive)      : {np.max(np.abs(resid_s)):.3e}")
print(f"max residual (strict sliding) : {np.max(np.abs(resid_a)):.3e}")
print(f"final |Z_batch|               : {np.linalg.norm(Z_b[-1]):.3e}")
print(f"final |Z_rec|                 : {np.linalg.norm(Z_s[-1]):.3e}")
print(f"final |Z_slide|               : {np.linalg.norm(Z_a[-1]):.3e}")

# ============================================================
# Styling
# ============================================================

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.titlesize": 10,
    "lines.linewidth": 1.6,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.fancybox": False,
    "legend.edgecolor": "0.8",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

TEXTWIDTH = 522


def format_ax(ax, xlabel=None, ylabel=None, title=None, yscale=None):
    if title is not None:
        ax.set_title(title, pad=6)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.margins(x=0.01)


state_colors = ["#1b9e77", "#7570b3"]   # teal, purple (state components Z1, Z2)

# method encoding (used wherever curves are colored by method)
method_color = {"batch": "#8c2d04", "recursive": "#e6550d", "sliding": "#3182bd"}
method_ls = {"batch": "-", "recursive": "--", "sliding": ":"}
method_label = {
    "batch": "batch",
    "recursive": r"recursive ($\Delta$=1)",
    "sliding": r"sliding ($\Delta$=2)",
}

# ============================================================
# Figure: state | control  (top),  delays+horizon | residual (bottom)
# overlaying batch, recursive (Delta=1), and strict sliding (Delta=2)
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=set_size(TEXTWIDTH, fraction=1.0, subplots=(2, 2), height_add=0.2),
    sharex=True,
    constrained_layout=True,
)

for ax in axes.ravel():
    ax.tick_params(axis="x", labelbottom=True)


def mark_reinit(ax):
    for tk in reinit_pts:
        ax.axvline(tk, color="0.8", linestyle=":", linewidth=0.6, zorder=0)


# ---- (0,0) State overlay (color = component, linestyle = method) --------
ax = axes[0, 0]
for i in range(Z_b.shape[1]):
    ax.plot(t, Z_b[:, i], color=state_colors[i], ls=method_ls["batch"],
            linewidth=2.0, zorder=3)
    ax.plot(t, Z_s[:, i], color=state_colors[i], ls=method_ls["recursive"],
            linewidth=1.4, zorder=4)
    ax.plot(t, Z_a[:, i], color=state_colors[i], ls=method_ls["sliding"],
            linewidth=1.4, zorder=5)
comp_handles = [
    Line2D([0], [0], color=state_colors[0], lw=2, label=r"$Z_1$"),
    Line2D([0], [0], color=state_colors[1], lw=2, label=r"$Z_2$"),
]
meth_handles = [
    Line2D([0], [0], color="0.3", ls=method_ls[m], lw=1.6, label=method_label[m])
    for m in ("batch", "recursive", "sliding")
]
format_ax(ax, title=r"State: batch vs.\ recursive vs.\ sliding")
ax.legend(handles=comp_handles + meth_handles, ncol=2, loc="best")
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

# ---- (0,1) Control input overlay ----------------------------------------
ax = axes[0, 1]
ax.plot(t, U_b[:, 0], color=method_color["batch"], ls=method_ls["batch"],
        linewidth=2.0, label=method_label["batch"], zorder=3)
ax.plot(t, U_s[:, 0], color=method_color["recursive"], ls=method_ls["recursive"],
        linewidth=1.4, label=method_label["recursive"], zorder=4)
ax.plot(t, U_a[:, 0], color=method_color["sliding"], ls=method_ls["sliding"],
        linewidth=1.4, label=method_label["sliding"], zorder=5)
format_ax(ax, title=r"Control input $U(t)$")
ax.legend(loc="best")
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

# ---- (1,0) Delays and applied prediction horizons -----------------------
ax = axes[1, 0]
ax.plot(t, D1_vals, color="#1b9e77", linewidth=1.8, label=r"$D_1(t)$")
ax.plot(t, D2_vals, color="#7570b3", linewidth=1.8, label=r"$D_2(t)$")
ax.plot(t, psi_b, color=method_color["batch"], ls=method_ls["batch"],
        linewidth=1.6, label=r"$\hat{\psi}$ batch")
ax.plot(t, psi_s, color=method_color["recursive"], ls=method_ls["recursive"],
        linewidth=1.6, label=r"$\hat{\psi}$ recursive")
ax.plot(t, psi_a, color=method_color["sliding"], ls=method_ls["sliding"],
        linewidth=1.6, label=r"$\hat{\psi}$ sliding")
format_ax(ax, xlabel=r"$t$", title="Delays and applied prediction horizon")
ax.legend(loc="best", ncol=2)
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

# ---- (1,1) Applied-horizon consistency residual -------------------------
ax = axes[1, 1]
mark_reinit(ax)
for resid, m in ((resid_b, "batch"), (resid_s, "recursive"), (resid_a, "sliding")):
    ax.plot(t, np.maximum(np.abs(resid), 1e-16), color=method_color[m],
            ls=method_ls[m], linewidth=1.4, label=method_label[m])
format_ax(ax, xlabel=r"$t$", title=r"Residual $\phi(\hat{\psi}(t)+t)-t$",
          yscale="log")
ax.legend(loc="best")
ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))

fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.95,
                    wspace=0.28, hspace=0.35)
plt.savefig("../figures/recursive_short_operator_fig.pdf", dpi=300, bbox_inches="tight")
print("\nSaved figure to ../figures/recursive_short_operator_fig.pdf")
