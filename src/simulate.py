import numpy as np
from utils import interp_history


def simulate(
    T,
    dt,
    system,
    delays,
    history,
    controller,
    delayed_measurement=True,
    verbose=False,
):
    A = system["A"]
    B = system["B"]
    if delayed_measurement:
        C = system["C"]

    phi1 = delays["phi1"]
    phi2 = delays.get("phi2", None)

    z_history = history["z_history"]
    u_history = history["u_history"]

    N = int(np.round(T / dt))
    t_grid = np.linspace(0.0, N * dt, N + 1)

    n = A.shape[0]
    m = B.shape[1]
    if delayed_measurement:
        p = C.shape[0]

    Z = np.zeros((N + 1, n))
    if delayed_measurement:
        Y = np.zeros((N + 1, p))
    else:
        Y = np.zeros((N + 1, n))
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
        if delayed_measurement:
            print(f"output dim = {p}")
        print(f"delayed_measurement = {delayed_measurement}")
        print()

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

        # measurement model
        if delayed_measurement:
            # output feedback with delay
            if phi2 is None:
                raise ValueError("delayed_measurement=True requires delays['phi2']")
            z_meas = interp_history(phi2(t), z_times, z_vals, z_history)
            Y[k] = C @ z_meas
        else:
            # full state feedback
            z_meas = Z[k]
            Y[k] = z_meas

        # controller update
        U[k] = controller.step(
            t,
            Y[k],
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