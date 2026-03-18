import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator
import torch
from neuralop.models import FNO

# ============================================================
# Common helpers
# ============================================================

def secant_inverse(phi, y, x0, x1, tol=1e-12, max_iter=100):
    x0 = float(x0)
    x1 = float(x1)

    f0 = float(phi(x0) - y)
    f1 = float(phi(x1) - y)

    if abs(f0) <= tol:
        return x0
    if abs(f1) <= tol:
        return x1

    for _ in range(max_iter):
        denom = f1 - f0
        if abs(denom) < 1e-16:
            raise FloatingPointError(
                f"Secant breakdown: denominator too small with x0={x0}, x1={x1}"
            )

        x2 = x1 - f1 * (x1 - x0) / denom
        f2 = float(phi(x2) - y)

        if abs(f2) <= tol:
            return x2

        x0, f0 = x1, f1
        x1, f1 = x2, f2

    return x1


def _build_uniform_grid(t_min, t_max, dt):
    if t_max < t_min:
        raise ValueError(f"Need t_max >= t_min, got t_min={t_min}, t_max={t_max}")
    if dt <= 0.0:
        raise ValueError(f"Need dt > 0, got dt={dt}")

    n_steps = int(np.ceil((t_max - t_min) / dt))
    t_grid = np.linspace(float(t_min), float(t_max), n_steps + 1, dtype=float)
    h_grid = np.diff(t_grid)
    return t_grid, h_grid, n_steps


def _compute_rho0(phi, t_min, rho0, secant_guess, tol):
    if rho0 is not None:
        return float(rho0)

    phi0 = float(phi(0.0))
    if abs(t_min - phi0) <= tol:
        return 0.0

    return secant_inverse(
        phi=phi,
        y=t_min,
        x0=secant_guess[0],
        x1=secant_guess[1],
        tol=tol,
        max_iter=200,
    )


def _print_residual_report(method_name, t_grid, rho_grid, phi):
    residual = np.abs(np.array([phi(r) for r in rho_grid], dtype=float) - t_grid)
    max_residual = float(np.max(residual))
    mean_residual = float(np.mean(residual))

    print(f"=== phi^{{-1}} construction ({method_name}) ===")
    print(f"target interval: [{t_grid[0]:.12f}, {t_grid[-1]:.12f}]")
    print(f"number of steps: {len(t_grid) - 1}")
    print(f"initial rho(t_min): {rho_grid[0]:.12f}")
    print(f"max residual |phi(rho(t)) - t|:  {max_residual:.6e}")
    print(f"mean residual |phi(rho(t)) - t|: {mean_residual:.6e}")


# ============================================================
# Euler inverse propagation
# ============================================================

def compute_phi_inv_euler_grid(
    phi,
    phi_prime,
    t_max,
    dt,
    t_min=None,
    rho0=None,
    secant_guess=(0.0, 5.0),
    clamp_phip_min=1e-12,
    residual_print=True,
    model=None
):
    """
    Construct phi^{-1} on a grid using forward Euler applied to
        rho'(t) = 1 / phi'(rho(t)).
    """
    if t_min is None:
        t_min = float(phi(0.0))

    t_min = float(t_min)
    t_max = float(t_max)
    dt = float(dt)

    t_grid, h_grid, _ = _build_uniform_grid(t_min, t_max, dt)
    rho_grid = np.zeros_like(t_grid)

    rho_grid[0] = _compute_rho0(
        phi=phi,
        t_min=t_min,
        rho0=rho0,
        secant_guess=secant_guess,
        tol=1e-12,
    )

    def f(r):
        phip = float(phi_prime(r))
        if (not np.isfinite(phip)) or (phip <= clamp_phip_min):
            raise FloatingPointError(
                f"Bad phi'(rho)={phip} at rho={r}. "
                "Cannot propagate inverse ODE safely."
            )
        return 1.0 / phip

    for k in range(len(t_grid) - 1):
        h = float(h_grid[k])
        r = float(rho_grid[k])
        rho_grid[k + 1] = r + h * f(r)

    if residual_print:
        _print_residual_report("Euler", t_grid, rho_grid, phi)

    return t_grid, rho_grid


# ============================================================
# RK4 inverse propagation
# ============================================================

def compute_phi_inv_rk4_grid(
    phi,
    phi_prime,
    t_max,
    dt,
    t_min=None,
    rho0=None,
    secant_guess=(0.0, 5.0),
    clamp_phip_min=1e-12,
    residual_print=True,
    model=None,
):
    """
    Construct phi^{-1} on a grid using classical RK4 applied to
        rho'(t) = 1 / phi'(rho(t)).
    """
    if t_min is None:
        t_min = float(phi(0.0))

    t_min = float(t_min)
    t_max = float(t_max)
    dt = float(dt)

    t_grid, h_grid, _ = _build_uniform_grid(t_min, t_max, dt)
    rho_grid = np.zeros_like(t_grid)

    rho_grid[0] = _compute_rho0(
        phi=phi,
        t_min=t_min,
        rho0=rho0,
        secant_guess=secant_guess,
        tol=1e-12,
    )

    def f(r):
        phip = float(phi_prime(r))
        if (not np.isfinite(phip)) or (phip <= clamp_phip_min):
            raise FloatingPointError(
                f"Bad phi'(rho)={phip} at rho={r}. "
                "Cannot propagate inverse ODE safely."
            )
        return 1.0 / phip

    for k in range(len(t_grid) - 1):
        h = float(h_grid[k])
        r = float(rho_grid[k])

        k1 = f(r)
        k2 = f(r + 0.5 * h * k1)
        k3 = f(r + 0.5 * h * k2)
        k4 = f(r + h * k3)

        rho_grid[k + 1] = r + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    if residual_print:
        _print_residual_report("RK4", t_grid, rho_grid, phi)

    return t_grid, rho_grid


# ============================================================
# High-accuracy direct inversion
# ============================================================

def _bracket_inverse(phi, y, x_lo=0.0, x_hi=1.0, growth=2.0, max_expand=100):
    x_lo = float(x_lo)
    x_hi = float(x_hi)
    y = float(y)

    f_lo = float(phi(x_lo) - y)
    f_hi = float(phi(x_hi) - y)

    if f_lo > 0.0:
        raise ValueError(
            f"Cannot bracket inverse at y={y}: phi({x_lo})={phi(x_lo)} > y."
        )

    k = 0
    while f_hi < 0.0:
        if k >= max_expand:
            raise RuntimeError(
                f"Failed to bracket inverse for y={y} after {max_expand} expansions."
            )
        x_hi = growth * x_hi + 1.0
        f_hi = float(phi(x_hi) - y)
        k += 1

    return x_lo, x_hi


def _invert_phi_pointwise(
    phi,
    t_grid,
    x_lo=0.0,
    x_hi_init=1.0,
    xtol=1e-14,
    rtol=1e-14,
    maxiter=200,
):
    t_grid = np.asarray(t_grid, dtype=float)
    rho_grid = np.empty_like(t_grid)

    rho_prev = 0.0

    for k, t in enumerate(t_grid):
        lo = rho_prev
        if float(phi(lo) - t) > 0.0:
            lo = x_lo

        hi = max(x_hi_init, lo + 1.0)
        lo, hi = _bracket_inverse(phi, t, x_lo=lo, x_hi=hi)

        rho = brentq(
            lambda x: float(phi(x) - t),
            lo,
            hi,
            xtol=xtol,
            rtol=rtol,
            maxiter=maxiter,
        )

        rho_grid[k] = rho
        rho_prev = rho

    return rho_grid


def compute_phi_inv_direct_grid(
    phi,
    phi_prime,  # kept only for API compatibility
    t_max,
    dt,
    t_min=None,
    rho0=None,  # kept only for API compatibility
    secant_guess=(0.0, 5.0),  # kept only for API compatibility
    clamp_phip_min=1e-12,     # kept only for API compatibility
    residual_print=True,
    xtol=1e-14,
    rtol=1e-14,
    model=None,  # kept only for API compatibility
):
    """
    Construct phi^{-1} on a grid by solving phi(rho)=t independently at
    every grid point using a bracketed root finder.
    """
    del phi_prime, rho0, secant_guess, clamp_phip_min

    if t_min is None:
        t_min = float(phi(0.0))

    t_min = float(t_min)
    t_max = float(t_max)
    dt = float(dt)

    t_grid, _, _ = _build_uniform_grid(t_min, t_max, dt)

    rho_grid = _invert_phi_pointwise(
        phi=phi,
        t_grid=t_grid,
        x_lo=0.0,
        x_hi_init=1.0,
        xtol=xtol,
        rtol=rtol,
        maxiter=200,
    )

    if residual_print:
        _print_residual_report("Direct", t_grid, rho_grid, phi)

    return t_grid, rho_grid


# ============================================================
# Interpolant builders
# ============================================================

def make_phi_inv_from_grid_linear(t_grid, rho_grid, tol=1e-10):
    """
    Linear interpolant for phi^{-1}; same behavior as np.interp.
    """
    t_grid = np.asarray(t_grid, dtype=float)
    rho_grid = np.asarray(rho_grid, dtype=float)

    t_min = float(t_grid[0])
    t_max = float(t_grid[-1])

    def phi_inv(t):
        t = float(t)

        if t < t_min - tol or t > t_max + tol:
            raise ValueError(
                f"phi_inv queried at t={t}, outside tabulated range "
                f"[{t_min}, {t_max}]."
            )

        t_clamped = min(max(t, t_min), t_max)
        return float(np.interp(t_clamped, t_grid, rho_grid))

    phi_inv.t_min = t_min
    phi_inv.t_max = t_max

    return phi_inv


def make_phi_inv_from_grid_pchip(t_grid, rho_grid, tol=1e-10):
    """
    Monotone cubic interpolant for phi^{-1}; usually more accurate than linear.
    """
    t_grid = np.asarray(t_grid, dtype=float)
    rho_grid = np.asarray(rho_grid, dtype=float)

    t_min = float(t_grid[0])
    t_max = float(t_grid[-1])

    interpolant = PchipInterpolator(t_grid, rho_grid, extrapolate=False)

    def phi_inv(t):
        t = float(t)

        if t < t_min - tol or t > t_max + tol:
            raise ValueError(
                f"phi_inv queried at t={t}, outside tabulated range "
                f"[{t_min}, {t_max}]."
            )

        t_clamped = min(max(t, t_min), t_max)
        return float(interpolant(t_clamped))

    phi_inv.t_min = t_min
    phi_inv.t_max = t_max

    return phi_inv


# Backward-compatible default
def make_phi_inv_from_grid(t_grid, rho_grid, tol=1e-10):
    return make_phi_inv_from_grid_linear(t_grid, rho_grid, tol=tol)


# ============================================================
# Optional comparison utility
# ============================================================

def compare_phi_inv_methods(
    phi,
    phi_prime,
    t_max,
    dt,
    t_min=None,
    rho0=None,
    secant_guess=(0.0, 5.0),
    clamp_phip_min=1e-12,
):
    methods = {
        "euler": compute_phi_inv_euler_grid,
        "rk4": compute_phi_inv_rk4_grid,
        "direct": compute_phi_inv_direct_grid,
    }

    out = {}

    for name, fn in methods.items():
        t_grid, rho_grid = fn(
            phi=phi,
            phi_prime=phi_prime,
            t_min=t_min,
            t_max=t_max,
            dt=dt,
            rho0=rho0,
            secant_guess=secant_guess,
            clamp_phip_min=clamp_phip_min,
            residual_print=False,
        )

        residual = np.abs(np.array([phi(r) for r in rho_grid], dtype=float) - t_grid)

        out[name] = {
            "t_grid": t_grid,
            "rho_grid": rho_grid,
            "max_residual": float(np.max(residual)),
            "mean_residual": float(np.mean(residual)),
        }

    return out

def compute_phi_inv_fno_grid(
    phi,
    phi_prime,  # API compatibility only
    t_max,
    dt,
    t_min=None,
    rho0=None,
    model=None,
):
    """
    FNO approximation of rho(t) = phi^{-1}(t).

    Expects `model` to be a dictionary containing:
        net
        x_mean
        x_std
        y_mean
        y_std
        device
    """

    if model is None:
        raise ValueError("model must be provided for FNO inverse")

    if t_min is None:
        t_min = 0.0

    net = model["net"]
    x_mean = model["x_mean"]
    x_std = model["x_std"]
    y_mean = model["y_mean"]
    y_std = model["y_std"]
    device = model["device"]

    # ------------------------------------------------
    # build uniform t grid
    # ------------------------------------------------

    n = int(np.round((t_max - t_min) / dt)) + 1
    t_grid = t_min + dt * np.arange(n, dtype=float)
    t_grid[-1] = t_max

    # ------------------------------------------------
    # build D(t)
    # ------------------------------------------------

    D_grid = np.array([t - phi(t) for t in t_grid], dtype=np.float32)

    x = torch.tensor(D_grid, dtype=torch.float32, device=device).view(1, 1, -1)

    x = (x - x_mean.to(device)) / x_std.to(device)

    # ------------------------------------------------
    # FNO prediction
    # ------------------------------------------------

    net.eval()

    with torch.no_grad():

        psi = net(x)

        psi = psi * y_std.to(device) + y_mean.to(device)

        psi = psi[0, 0].cpu().numpy()

    rho_grid = t_grid + psi

    return t_grid, rho_grid

def load_fno_inverse_model(path, device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(path, map_location=device, weights_only=False)

    model = FNO(
        n_modes=(32,),
        hidden_channels=64,
        in_channels=1,
        out_channels=1,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    bundle = {
        "net": model,
        "x_mean": ckpt["x_mean"],
        "x_std": ckpt["x_std"],
        "y_mean": ckpt["y_mean"],
        "y_std": ckpt["y_std"],
        "device": device,
        "t_grid_train": ckpt["t_grid"],
    }

    return bundle