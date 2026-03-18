import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.ticker import LogLocator,  MaxNLocator
import sys
import torch


sys.path.append("../src")

from simulate import simulate
from utils import set_size
from controller import PredictorFeedbackController
from delay_check import check_delay_functions
from phi_inv import make_phi_inv_from_grid_pchip, load_fno_inverse_model, compute_phi_inv_fno_grid, compute_phi_inv_direct_grid, make_phi_inv_from_grid, make_phi_inv_from_grid_linear, compute_phi_inv_euler_grid, compute_phi_inv_rk4_grid
# ============================================================
# System data
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
# Delay family
# phi(t) = t - a + b/(1+t) + alpha sin(omega t + varphi)
# ============================================================

rng = np.random.default_rng(1)

# ============================================================
# Plant
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


# ============================================================
# Build phi(t)
# ============================================================

def make_phi(params):
    a, b, alpha, omega, varphi = params

    def phi(t):
        return t - (
            a
            + b / (1.0 + t)
            + alpha * np.sin(omega * t + varphi)
        )

    return phi


# ============================================================
# Build phi'(t)
# ============================================================

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

def make_D(phi):
    def D(t):
        return t - phi(t)

    return D

def sample_valid_delay_pair(rng, T, dt, max_tries=100, inverse_grid_builder=compute_phi_inv_direct_grid,
    inverse_callable_builder=make_phi_inv_from_grid_pchip, model=None):
    for _ in range(max_tries):
        params1 = sample_delay_params(rng)
        params2 = sample_delay_params(rng)

        phi1 = make_phi(params1)
        phi2 = make_phi(params2)

        dphi1 = make_dphi(params1)
        dphi2 = make_dphi(params2)

        D1 = make_D(phi1)
        D2 = make_D(phi2)

        # build phi1^{-1} on the full interval where it may be queried
        t_inv_min = phi1(phi2(0.0))
        t_inv_max = T

        # this does a pointwise solve for every single t - expensive but guarentees machine precision
        # good for building datasets
        try:
            t_d, rho_d = inverse_grid_builder(
                phi=phi1,
                phi_prime=dphi1,   
                t_min=0.0,
                t_max=T,
                dt=1e-3,
                model=model
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
                params2,
                phi1,
                phi2,
                dphi1,
                dphi2,
                D1,
                D2,
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
            print("params2:", params2)
            print("-" * 60)

    raise RuntimeError(f"Could not sample a valid delay pair in {max_tries} tries.")

# ============================================================
# Histories
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
dt = 1e-2

fno_model = load_fno_inverse_model("../models/fno_psi/fno_model.pt")

params1, params2, phi1, phi2, dphi1, dphi2, D1, D2, phi1_inv, t_inv_grid, phi1_inv_grid, results = sample_valid_delay_pair(
    rng=rng,
    T=T,
    dt=dt,
    max_tries=200,
    inverse_grid_builder=compute_phi_inv_fno_grid,
    inverse_callable_builder=make_phi_inv_from_grid_linear,
    model = fno_model
)

# ============================================================
# Check delay assumptions
# ============================================================

print("phi1 parameters:", params1)
print("phi2 parameters:", params2)

check_delay_functions(
    T=T,
    D1=D1,
    D2=D2,
    phi1=phi1,
    phi2=phi2,
    dphi1=dphi1,
    dphi2=dphi2,
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
    "C": C,
    "K": K,
    "L": L,
}

delays = {
    "phi1": phi1,
    "phi2": phi2,
    "dphi1": dphi1,
    "dphi2": dphi2,
    "phi1_inv": phi1_inv,
}

init = {
    "u_history": u_history,
    "z_history": z_history,
    "xi0": xi0,
}

controller = PredictorFeedbackController(
    system=system,
    delays=delays,
    init=init,
    n_quad=40,
)

# run simulation
t, Z, Y, U = simulate(
    T=T,
    dt=dt,
    system=system,
    delays=delays,
    history=init,
    controller=controller,
    verbose=True,
)

# recover controller histories
t_ctrl = np.array(controller.t_hist)

# delay-related curves
XI = np.array(controller.xi_hist)
HAT_Z = np.array(controller.hat_Z_hist)
PSI = np.array([phi1_inv(tt) - tt for tt in t])
D1_vals = np.array([D1(tt) for tt in t])
D2_vals = np.array([D2(tt) for tt in t])

# Build error curves for consistency errors for all the methods
# ============================================================
# Build inverse grids for all methods
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

# ============================================================
# Interpolate all inverse curves onto the common simulation grid t
# ============================================================

rho_e_on_t = np.interp(t, t_e, rho_e)
rho_rk_on_t = np.interp(t, t_rk, rho_rk)
rho_d_on_t = np.interp(t, t_d, rho_d)
rho_fno_on_t = np.interp(t, t_fno, rho_fno)

# ============================================================
# Convert to psi if desired
# ============================================================

psi_e = rho_e_on_t - t
psi_rk = rho_rk_on_t - t
psi_d = rho_d_on_t - t
psi_fno = rho_fno_on_t - t

# ============================================================
# Consistency errors:
#     phi1(rho(t)) - t
# equivalently:
#     phi1(t + psi(t)) - t
# ============================================================

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

# ============================================================
# Absolute versions for plotting on log-scale
# ============================================================

abs_consistency_err_euler = np.abs(consistency_err_euler)
abs_consistency_err_rk4 = np.abs(consistency_err_rk4)
abs_consistency_err_direct = np.abs(consistency_err_direct)
abs_consistency_err_fno = np.abs(consistency_err_fno)

# ============================================================
# Optional dictionary for cleaner plotting
# ============================================================

consistency_curves = {
    "Euler": abs_consistency_err_euler,
    "RK4": abs_consistency_err_rk4,
    #"Direct": abs_consistency_err_direct,
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
    """
    Lighten a matplotlib color.
    amount=0 -> original color
    amount=1 -> white
    """
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

# consistent colors for state / estimate pairing
state_colors = ["#1b9e77", "#7570b3"]        # teal, purple
observer_colors = ["#66c2a5", "#bcbddc"]     # lighter versions

output_colors = ["#1b9e77"] * Y.shape[1]
control_colors = ["#1b9e77"] * U.shape[1]
# ==================================================
# (1,1) State and estimate
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

for i in range(HAT_Z.shape[1]):
    ax.plot(
        t,
        HAT_Z[:, i],
        color=observer_colors[i],
        linestyle="--",
        linewidth=1.5,
        label=fr"$\hat{{Z}}_{i+1}(t)$",
        zorder=2,
    )

format_ax(
    ax,
    ylabel="State",
    title=r"State and estimate",
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
# (2,1) Measured output
# ==================================================

ax = axes[1, 0]

for i in range(Y.shape[1]):
    ax.plot(
        t,
        Y[:, i],
        color=output_colors[i],
        linewidth=1.8,
        label=fr"$Y_{i+1}(t)$",
    )

format_ax(
    ax,
    title=r"Measured output $Y(t)$",
)

if Y.shape[1] > 1:
    ax.legend(loc="best")
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


# ==================================================
# (2,2) Delay functions
# ==================================================
delay_colors = ["#1b9e77", "#7570b3"]  # teal, purple
yticks = np.arange(0.3, 0.71, 0.1)

ax = axes[1, 1]

ax.plot(t, D1_vals, color=delay_colors[0], linewidth=1.8, label=r"$D_1(t)$")
ax.plot(t, D2_vals, color=delay_colors[1], linewidth=1.8, label=r"$D_2(t)$")

format_ax(
    ax,
    title="Delay functions",
)
ax.legend(loc="best")
ax.set_yticks(yticks)

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
ax.set_yticks(yticks)
# ==================================================
# (3,2) Inverse consistency error
# ==================================================

ax = axes[2, 1]

line_styles = {
    "Euler": "--",
    "RK4": "-.",
    "FNO": "-",
}

line_widths = {
    "Euler": 1.4,
    "RK4": 1.4,
    "FNO": 1.4,
}

zorders = {
    "Euler": 2,
    "RK4": 4,
    "FNO": 1,   # plotted on top
}

colors = {
    "Euler": "#8c2d04",   # dark red
    "RK4": "#7b3294",     # muted purple
    "FNO": "#1b9e77",     # teal
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
    title=r"Residual $\phi(\psi(t)+t)$",
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
    bottom=0.1,
    top=0.97,
    wspace=0.28,
    hspace=0.35,
)
plt.savefig("../figures/output_feedback_main_fig.pdf", dpi=300, bbox_inches="tight")



# CONCISE FIGURE
fig, axes = plt.subplots(
    2,
    2,
    figsize=set_size(TEXTWIDTH, fraction=1.0, subplots=(2, 2), height_add=-0.5),
    sharex=True,
    constrained_layout=True,
)

for ax in axes.ravel():
    ax.tick_params(axis="x", labelbottom=True)

# consistent colors for state / estimate pairing
state_colors = ["#1b9e77", "#7570b3"]        # teal, purple
observer_colors = ["#66c2a5", "#bcbddc"]     # lighter versions

output_colors = ["#1b9e77"] * Y.shape[1]
control_colors = ["#1b9e77"] * U.shape[1]
# ==================================================
# (1,1) State and estimate
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

for i in range(HAT_Z.shape[1]):
    ax.plot(
        t,
        HAT_Z[:, i],
        color=observer_colors[i],
        linestyle="--",
        linewidth=1.5,
        label=fr"$\hat{{Z}}_{i+1}(t)$",
        zorder=2,
    )

format_ax(
    ax,
    title=r"State and estimate",
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
# (2, 1) DEelay functions and prediction horizon
# ==================================================

ax = axes[1,0]

ax.plot(t, D1_vals, color="#1b9e77", linewidth=1.8, label=r"$D_1(t)$")
ax.plot(t, D2_vals, color="#7570b3", linewidth=1.8, label=r"$D_2(t)$")

ax.plot(
    t,
    PSI,
    color="#8c2d04",
    linestyle="--",
    linewidth=2.0,
    label=r"$\hat{\psi}_{\mathrm{FNO}}(t)$",
)

ax.set_yticks(np.arange(0.1, 0.71, 0.2))

format_ax(
    ax,
    xlabel=r"$t$",
    title="Delays and learned prediction horizon",
)

ax.legend(loc="lower right", ncol=3)
# ==================================================
# (2,2) Inverse consistency error
# ==================================================

ax = axes[1, 1]

line_styles = {
    "Euler": "--",
    "RK4": "-.",
    "FNO": "-",
}

line_widths = {
    "Euler": 1.4,
    "RK4": 1.4,
    "FNO": 1.4,
}

zorders = {
    "Euler": 2,
    "RK4": 4,
    "FNO": 1,   # plotted on top
}

colors = {
    "Euler": "#8c2d04",   # dark red
    "RK4": "#7b3294",     # muted purple
    "FNO": "#1b9e77",     # teal
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
    title=r"Residual $\phi(\hat{\psi}(t)+t)$",
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
    bottom=0.1,
    top=0.97,
    wspace=0.28,
    hspace=0.35,
)
plt.savefig("../figures/output_feedback_main_concise.pdf", dpi=300, bbox_inches="tight")