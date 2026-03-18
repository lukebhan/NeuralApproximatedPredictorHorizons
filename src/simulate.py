import numpy as np
from utils import interp_history


def simulate(T, dt, system, delays, history, controller, verbose=False):
    A = system["A"]
    B = system["B"]
    C = system["C"]

    phi1 = delays["phi1"]
    phi2 = delays["phi2"]

    z_history = history["z_history"]
    u_history = history["u_history"]

    N = int(np.round(T / dt))
    t_grid = np.linspace(0.0, N * dt, N + 1)

    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]

    Z = np.zeros((N + 1, n))
    Y = np.zeros((N + 1, p))
    U = np.zeros((N + 1, m))

    Z[0] = np.asarray(z_history(0.0), dtype=float).reshape(-1)

    controller.reset()

    if verbose:
        print("=== Simulation start ===")
        print(f"T = {T}")
        print(f"dt = {dt}")
        print(f"N steps = {N}")
        print(f"state dim = {n}")
        print(f"input dim = {m}")
        print(f"output dim = {p}")
        print()

    # progress reporting interval (~10 updates)
    report_every = max(1, N // 10)

    for k in range(N + 1):
        t = t_grid[k]

        if verbose and (k % report_every == 0 or k == N):
            pct = 100.0 * k / N
            print(f"[{pct:6.2f}%] step {k:6d}/{N}, t = {t:.4f}")

        z_times = t_grid[:max(k, 1)]
        z_vals = Z[:max(k, 1)]
        u_times = t_grid[:max(k, 1)]
        u_vals = U[:max(k, 1)]

        # measurement Y(t) = C Z(phi2(t))
        z_del = interp_history(phi2(t), z_times, z_vals, z_history)
        Y[k] = C @ z_del

        # controller update
        U[k] = controller.step(
            t=t,
            Y=Y[k],
            t_U=u_times,
            U=u_vals,
            dt=dt,
        )

        if k == N:
            break

        # plant dynamics
        def fz(tau, z):
            u_del = interp_history(phi1(tau), t_grid[:k + 1], U[:k + 1], u_history)
            return A @ z + B @ u_del

        z = Z[k]

        k1 = fz(t, z)
        k2 = fz(t + 0.5 * dt, z + 0.5 * dt * k1)
        k3 = fz(t + 0.5 * dt, z + 0.5 * dt * k2)
        k4 = fz(t + dt, z + dt * k3)

        Z[k + 1] = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    if verbose:
        print()
        print("=== Simulation finished ===")
        print(f"final time = {t_grid[-1]:.6f}")
        print(f"final state norm = {np.linalg.norm(Z[-1]):.6e}")
        print()

    return t_grid, Z, Y, U