import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LogLocator, MaxNLocator
import matplotlib.colors as mcolors
import sys
import torch

sys.path.append("../src")

from simulate import simulate
from utils import set_size
from controller import StatePredictorFeedbackController
from delay_check import check_delay_functions
from phi_inv import (
    make_phi_inv_from_grid_pchip,
    load_fno_inverse_model,
    compute_phi_inv_fno_grid,
    compute_phi_inv_direct_grid,
    make_phi_inv_from_grid_linear,
    compute_phi_inv_euler_grid,
    compute_phi_inv_rk4_grid,
)

# ============================================================
# System data
# ============================================================

A = np.array([
    [0.0, 1.0],
    [1.0, 1.0],
])

B = np.array([
    [0.0],
    [1.0],
])

K = np.array([[-5.0, -5.0]])

print("eig(A):", np.linalg.eigvals(A))
print("eig(A+BK):", np.linalg.eigvals(A + B @ K))

# ============================================================
# Delay family
# phi(t) = t - (a + b/(1+t) + alpha sin(omega t + varphi))
# ============================================================

rng = np.random.default_rng(1)


def sample_delay_params(rng):
    """
    Sample parameters for the delay family
        phi(t) = t - (a + b/(1+t) + alpha*sin(omega t + varphi))
    with conservative ranges for numerical stability.
    """
    a = rng.uniform(0.1, 0.5)
    b = rng.uniform(0.1, 0.5)
    alpha = rng.uniform(-0.3, 0.3)
    omega = rng.uniform(0, 2 * np.pi)
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


def sample_valid_delay(
    rng,
    T,
    dt,
    max_tries=100,
    inverse_grid_builder=compute_phi_inv_direct_grid,
    inverse_callable_builder=make_phi_inv_from_grid_pchiBp,
    model=None,
):
    for _ in range(max_tries):
        params1 = sample_delay_params(rng)

        phi1 = make_phi(params1)
        dphi1 = make_dphi(params1)
        D1 = make_D(phi1)

        # For the state-feedback case, we do not need phi2 for simulation,
        # but check_delay_functions still expects two delays, so use phi2 = phi1.
        phi2 = phi1
        dphi2 = dphi1
        D2 = D1

        try:
            t_d, rho_d = inverse_grid_builder(
                phi=phi1,
                phi_prime=dphi1,
                t_min=0.0,
                t_max=T,
                dt=1e-3,
                model=model,
            )
            phi1_inv = inverse_callable_builder(t_d, rho_d)
        except (ValueError, FloatingPointError):
            print("VALUE ERROR in inverse computation, retrying...")
            continue

        results = check_delay_functions(
            T=T,
            D1=D1,
            D2=D2,
            phi1=phi1,
            phi2=phi2,
            dphi1=dphi1,
            dphi2=dphi2,
            phi1_inv=phi1_inv,
            n_grid=4000,
            tol=1e-2,
            verbose=False,
        )

        if results["all_ok"]:
            return (
                params1,
                phi1,
                dphi1,
                D1,
                phi1_inv,
                t_d,
                rho_d,
                results,
            )
        else:
            print("\nDelay check failed:")
            for key, val in results.items():
                if key != "all_ok" and val is not True:
                    print(f"  {key}: {val}")
            print("params1:", params1)
            print("-" * 60)

    raise RuntimeError(f"Could not sample a valid delay in {max_tries} tries.")


# ============================================================
# Histories
# ============================================================

def z_history(t):
    return np.array([-5.0, 5.0])


def u_history(t):
    return np.array([0.0])


# ============================================================
# Simulation setup
# ============================================================

T = 12.0
dt = 1e-3

fno_model = load_fno_inverse_model("../models/fno_psi/fno_model.pt")

(
    params1,
    phi1,
    dphi1,
    D1,
    phi1_inv,
    t_inv_grid,
    phi1_inv_grid,
    results,
) = sample_valid_delay(
    rng=rng,
    T=T,
    dt=dt,
    max_tries=200,
    inverse_grid_builder=compute_phi_inv_fno_grid,
    inverse_callable_builder=make_phi_inv_from_grid_linear,
    model=fno_model,
)

print("phi1 parameters:", params1)

# for checking only
check_delay_functions(
    T=T,
    D1=D1,
    D2=D1,
    phi1=phi1,
    phi2=phi1,
    dphi1=dphi1,
    dphi2=dphi1,
    phi1_inv=phi1_inv,
    n_grid=4000,
    tol=1e-8,
    verbose=True,
)

# ============================================================
# Run simulation
# ============================================================

system = {
    "A": A,
    "B": B,
    "K": K,
}

delays = {
    "phi1": phi1,
    "dphi1": dphi1,
    "phi1_inv": phi1_inv,
}

init = {
    "u_history": u_history,
    "z_history": z_history,
}

controller = StatePredictorFeedbackController(
    system=system,
    delays=delays,
    init=init,
    n_quad=40,
)

# run simulation
t, Z, Z_meas, U = simulate(
    T=T,
    dt=dt,
    system=system,
    delays=delays,
    history=init,
    controller=controller,
    delayed_measurement=False,   # traditional direct state feedback measurement
    verbose=True,
)

# recover controller histories
t_ctrl = np.array(controller.t_hist)
HAT_Z = np.array(controller.hat_Z_hist)
HAT_P = np.array(controller.hat_P_hist)

# delay-related curves
PSI = np.array([phi1_inv(tt) - tt for tt in t])
D1_vals = np.array([D1(tt) for tt in t])

# ============================================================
# Build error curves for consistency errors for all methods
# ============================================================

t_e, rho_e = compute_phi_inv_euler_grid(
    phi=phi1,
    phi_prime=dphi1,
    t_max=T,
    dt=dt,
    t_min=0.0,
    residual_print=False,
)

t_rk, rho_rk = compute_phi_inv_rk4_grid(
    phi=phi1,
    phi_prime=dphi1,
    t_max=T,
    dt=dt,
    t_min=0.0,
    residual_print=False,
)

t_d, rho_d = compute_phi_inv_direct_grid(
    phi=phi1,
    phi_prime=dphi1,
    t_max=T,
    dt=dt,
    t_min=0.0,
    residual_print=False,
)

t_fno, rho_fno = compute_phi_inv_fno_grid(
    phi=phi1,
    phi_prime=dphi1,
    t_max=T,
    dt=dt,
    t_min=0.0,
    model=fno_model,
)

rho_e_on_t = np.interp(t, t_e, rho_e)
rho_rk_on_t = np.interp(t, t_rk, rho_rk)
rho_d_on_t = np.interp(t, t_d, rho_d)
rho_fno_on_t = np.interp(t, t_fno, rho_fno)

psi_e = rho_e_on_t - t
psi_rk = rho_rk_on_t - t
psi_d = rho_d_on_t - t
psi_fno = rho_fno_on_t - t

consistency_err_euler = np.array([
    phi1(rho) - tt
    for tt, rho in zip(t, rho_e_on_t)
])

consistency_err_rk4 = np.array([
    phi1(rho) - tt
    for tt, rho in zip(t, rho_rk_on_t)
])

consistency_err_direct = np.array([
    phi1(rho) - tt
    for tt, rho in zip(t, rho_d_on_t)
])

consistency_err_fno = np.array([
    phi1(rho) - tt
    for tt, rho in zip(t, rho_fno_on_t)
])

abs_consistency_err_euler = np.abs(consistency_err_euler)
abs_consistency_err_rk4 = np.abs(consistency_err_rk4)
abs_consistency_err_direct = np.abs(consistency_err_direct)
abs_consistency_err_fno = np.abs(consistency_err_fno)

consistency_curves = {
    "Euler": abs_consistency_err_euler,
    "RK4": abs_consistency_err_rk4,
    # "Direct": abs_consistency_err_direct,
    "FNO": abs_consistency_err_fno,
}

# --------------------------------------------------
# Styling helpers
# --------------------------------------------------

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
    "grid.linewidth": 0.5,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
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


def lighten_color(color, amount=0.75):
    c = np.array(mcolors.to_rgb(color))
    return tuple(1 - amount * (1 - c))


# --------------------------------------------------
# Figure
# --------------------------------------------------

fig, axes = plt.subplots(
    3,
    2,
    figsize=set_size(TEXTWIDTH, fraction=1.0, subplots=(3, 2), height_add=0.5),
    sharex=True,
    constrained_layout=True,
)

for ax in axes.ravel():
    ax.tick_params(axis="x", labelbottom=True)

state_colors = ["#1b9e77", "#7570b3"]
meas_colors = ["#66c2a5", "#bcbddc"]
control_colors = ["#1b9e77"] * U.shape[1]

# ==================================================
# (1,1) State and measured state
# ==================================================

ax = axes[0, 0]

for i in range(Z.shape[1]):
    ax.plot(
        t,
        Z[:, i],
        color=state_colors[i],
        linestyle="-",
        linewidth=2.0,
        label=fr"$Z_{i+1}(t)$",
        zorder=3,
    )

# for i in range(Z_meas.shape[1]):
#     ax.plot(
#         t,
#         Z_meas[:, i],
#         color=meas_colors[i],
#         linestyle="--",
#         linewidth=1.5,
#         label=fr"$Z^m_{i+1}(t)$",
#         zorder=2,
#     )

format_ax(
    ax,
    title=r"State $Z(t)$",
)
ax.legend(ncol=2, loc="best")
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

# ==================================================
# (1,2) Control input
# ==================================================

ax = axes[0, 1]

for i in range(U.shape[1]):
    ax.plot(
        t,
        U[:, i],
        color=control_colors[i],
        linewidth=1.8,
        label=fr"$U_{i+1}(t)$",
    )

format_ax(
    ax,
    title=r"Control input $U(t)$",
)

if U.shape[1] > 1:
    ax.legend(loc="best")
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

# ==================================================
# (2,1) Predictor state
# ==================================================

ax = axes[1, 0]

for i in range(HAT_P.shape[1]):
    ax.plot(
        t_ctrl,
        HAT_P[:, i],
        linewidth=1.8,
        label=fr"$\hat{{P}}_{i+1}(t)$",
    )

format_ax(
    ax,
    title=r"Predictor state $\hat{P}(t)$",
)

if HAT_P.shape[1] > 1:
    ax.legend(loc="best")
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

# ==================================================
# (2,2) Delay function
# ==================================================

ax = axes[1, 1]

ax.plot(t, D1_vals, color="#1b9e77", linewidth=1.8, label=r"$D_1(t)$")

format_ax(
    ax,
    title="Input delay function",
)
ax.legend(loc="best")
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

# ==================================================
# (3,1) Prediction horizon
# ==================================================

ax = axes[2, 0]

ax.plot(t, PSI, color="#1b9e77", linewidth=1.8)

format_ax(
    ax,
    xlabel=r"$t$",
    title=r"Prediction horizon $\hat{\psi}_{\mathrm{FNO}}(t)$",
)
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

# ==================================================
# (3,2) Inverse consistency error
# ==================================================

ax = axes[2, 1]

line_styles = {
    "FNO": "--",
    "RK4": "-.",
    "Euler": "-",
}

line_widths = {
    "Euler": 1.4,
    "RK4": 1.4,
    "FNO": 1.4,
}

zorders = {
    "FNO": 2,
    "RK4": 4,
    "Euler": 1,
}

colors = {
    "FNO": "#8c2d04",
    "RK4": "#7b3294",
    "Euler": "#1b9e77",
}

for name, curve in consistency_curves.items():
    ax.plot(
        t,
        np.maximum(np.abs(curve), 1e-16),
        linestyle=line_styles.get(name, "-"),
        linewidth=line_widths.get(name, 1.6),
        color=colors.get(name, None),
        zorder=zorders.get(name, 2),
        label=name,
    )

format_ax(
    ax,
    xlabel=r"$t$",
    title=r"Residual $\phi(\hat{\psi}(t)+t)-t$",
    yscale="log",
)
ax.legend(loc="lower right")
ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))

# --------------------------------------------------
# Save
# --------------------------------------------------

fig.subplots_adjust(
    left=0.10,
    right=0.98,
    bottom=0.10,
    top=0.97,
    wspace=0.28,
    hspace=0.35,
)
plt.savefig("../figures/state_feedback_main_fig.pdf", dpi=300, bbox_inches="tight")

# ============================================================
# CONCISE FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=set_size(TEXTWIDTH, fraction=1.0, subplots=(2, 2), height_add=-0.5),
    sharex=True,
    constrained_layout=True,
)

for ax in axes.ravel():
    ax.tick_params(axis="x", labelbottom=True)

state_colors = ["#1b9e77", "#7570b3"]
pred_colors = ["#66c2a5", "#bcbddc"]
meas_colors = ["#66c2a5", "#bcbddc"]
control_colors = ["#1b9e77"] * U.shape[1]

# ==================================================
# (1,1) State and measured state
# ==================================================

ax = axes[0, 0]

for i in range(Z.shape[1]):
    ax.plot(
        t,
        Z[:, i],
        color=state_colors[i],
        linestyle="-",
        linewidth=1.4,
        label=fr"$Z_{i+1}(t)$",
        zorder=3,
    )

for i in range(HAT_P.shape[1]):
    ax.plot(
        t_ctrl,
        HAT_P[:, i],
        color=pred_colors[i],
        linestyle="--",
        linewidth=1.4,
        label=fr"$\hat{{P}}_{i+1}(t)$",
    )

format_ax(
    ax,
    title=r"State and predictor state",
)
ax.legend(ncol=2, loc="best")
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

# ==================================================
# (1,2) Control input
# ==================================================

ax = axes[0, 1]

for i in range(U.shape[1]):
    ax.plot(
        t,
        U[:, i],
        color=control_colors[i],
        linewidth=1.4,
        label=fr"$U_{i+1}(t)$",
    )

format_ax(
    ax,
    title=r"Control input $U(t)$",
)

if U.shape[1] > 1:
    ax.legend(loc="best")
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

# ==================================================
# (2,1) Delay and prediction horizon
# ==================================================

ax = axes[1, 0]

ax.plot(t, D1_vals, color="#1b9e77", linewidth=1.8, label=r"$D_1(t)$")
ax.plot(
    t,
    PSI,
    color="#8c2d04",
    linestyle="--",
    linewidth=2.0,
    label=r"$\hat{\psi}_{\mathrm{FNO}}(t)$",
)

format_ax(
    ax,
    xlabel=r"$t$",
    title="Input delay and learned prediction horizon",
)

ax.legend(loc="upper right", ncol=2)
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

# ==================================================
# (2,2) Inverse consistency error
# ==================================================

ax = axes[1, 1]

for name, curve in consistency_curves.items():
    ax.plot(
        t,
        np.maximum(np.abs(curve), 1e-16),
        linestyle=line_styles.get(name, "-"),
        linewidth=line_widths.get(name, 1.6),
        color=colors.get(name, None),
        zorder=zorders.get(name, 2),
        label=name,
    )

format_ax(
    ax,
    xlabel=r"$t$",
    title=r"Residual $\phi(\hat{\psi}(t)+t)-t$",
    yscale="log",
)
ax.legend(loc="lower right", ncol=3 )
ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))

# --------------------------------------------------
# Save
# --------------------------------------------------

fig.subplots_adjust(
    left=0.10,
    right=0.98,
    bottom=0.10,
    top=0.97,
    wspace=0.28,
    hspace=0.35,
)
plt.savefig("../figures/state_feedback_main_concise.pdf", dpi=300, bbox_inches="tight")