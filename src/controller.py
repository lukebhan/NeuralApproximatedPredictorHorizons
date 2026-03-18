import numpy as np
from scipy.linalg import expm
from utils import interp_history


class PredictorFeedbackController:
    def __init__(self, system, delays, init, n_quad=40):
        """
        Predictor-feedback output controller.

        system:
            {
                "A": A,
                "B": B,
                "C": C,
                "K": K,
                "L": L,
            }

        delays:
            {
                "phi1": phi1,
                "phi2": phi2,
                "dphi1": dphi1,
                "dphi2": dphi2,
                "phi1_inv": phi1_inv,
            }

        init:
            {
                "u_history": u_history,
                "z_history": z_history,
                "xi0": xi0,
            }
        """
        self.A = np.asarray(system["A"], dtype=float)
        self.B = np.asarray(system["B"], dtype=float)
        self.C = np.asarray(system["C"], dtype=float)
        self.K = np.asarray(system["K"], dtype=float)
        self.L = np.asarray(system["L"], dtype=float)

        self.phi1 = delays["phi1"]
        self.phi2 = delays["phi2"]
        self.dphi1 = delays["dphi1"]
        self.dphi2 = delays["dphi2"]
        self.phi1_inv = delays["phi1_inv"]

        self.u_history = init["u_history"]
        self.z_history = init["z_history"]
        self.xi0 = np.asarray(init["xi0"], dtype=float).reshape(-1)

        self.n_quad = int(n_quad)
        self.m = self.B.shape[1]

        self.reset()

    def reset(self):
        self.xi = self.xi0.copy()
        self.hat_Z = None
        self.hat_P = None
        self.last_u = np.zeros(self.m, dtype=float)

        self.t_hist = []
        self.xi_hist = []
        self.hat_Z_hist = []
        self.hat_P_hist = []
        self.u_hist = []

    def zero_control(self):
        return np.zeros(self.m, dtype=float)

    def predictor_available(self, t):
        """
        Only use the predictor-based control once phi1(t) >= 0.
        Before that, hold the control at zero.
        """
        return self.phi1(t) >= 0.0

    def xi_rhs(self, t, xi, Y, t_U, U):
        """
        xi_dot(t) = phi2'(t) [ A xi(t) + B U(phi1(phi2(t))) + L(Y(t) - C xi(t)) ].
        """
        xi = np.asarray(xi, dtype=float).reshape(-1)
        Y = np.asarray(Y, dtype=float).reshape(-1)

        u_del = np.asarray(
            interp_history(self.phi1(self.phi2(t)), t_U, U, self.u_history),
            dtype=float,
        ).reshape(-1)

        return self.dphi2(t) * (
            self.A @ xi + self.B @ u_del + self.L @ (Y - self.C @ xi)
        )

    def compute_hat_Z(self, t, xi, t_U, U):
        """
        hatZ(t) = exp(A(t-phi2(t))) xi(t)
                  + int_{phi2(t)}^t exp(A(t-tau)) B U(phi1(tau)) dtau.
        """
        xi = np.asarray(xi, dtype=float).reshape(-1)

        a = self.phi2(t)
        b = t

        out = expm(self.A * (t - a)) @ xi

        if abs(b - a) < 1e-14:
            return out

        taus = np.linspace(a, b, self.n_quad)
        vals = []

        for tau in taus:
            u_val = np.asarray(
                interp_history(self.phi1(tau), t_U, U, self.u_history),
                dtype=float,
            ).reshape(-1)
            vals.append(expm(self.A * (t - tau)) @ (self.B @ u_val))

        vals = np.asarray(vals, dtype=float)
        integ = np.trapz(vals, taus, axis=0)

        return out + integ

    def compute_hat_P(self, t, hat_Z, t_U, U):
        """
        hatP(t) = exp(A(phi1^{-1}(t)-t)) hatZ(t)
                  + int_{phi1(t)}^t exp(A(phi1^{-1}(t)-phi1^{-1}(theta))) B U(theta)
                    / phi1'(phi1^{-1}(theta)) dtheta.

        This should only be called when phi1(t) >= 0.
        """
        if not self.predictor_available(t):
            raise ValueError(
                f"compute_hat_P called at t={t}, but phi1(t)={self.phi1(t)} < 0."
            )

        hat_Z = np.asarray(hat_Z, dtype=float).reshape(-1)

        a = self.phi1(t)
        b = t

        rho_t = self.phi1_inv(t)
        out = expm(self.A * (rho_t - t)) @ hat_Z

        if abs(b - a) < 1e-14:
            return out

        thetas = np.linspace(a, b, self.n_quad)
        vals = []

        for theta in thetas:
            u_val = np.asarray(
                interp_history(theta, t_U, U, self.u_history),
                dtype=float,
            ).reshape(-1)

            rho_theta = self.phi1_inv(theta)
            denom = self.dphi1(rho_theta)

            vals.append(
                expm(self.A * (rho_t - rho_theta)) @ (self.B @ u_val) / denom
            )

        vals = np.asarray(vals, dtype=float)
        integ = np.trapz(vals, thetas, axis=0)

        return out + integ

    def compute_control(self, t, xi, t_U, U):
        """
        Hold the control at zero until phi1(t) >= 0, then switch to
        predictor feedback.
        """
        hat_Z = self.compute_hat_Z(t, xi, t_U, U)

        if not self.predictor_available(t):
            u = self.zero_control()
            hat_P = hat_Z.copy()
            return u, hat_Z, hat_P

        hat_P = self.compute_hat_P(t, hat_Z, t_U, U)
        u = self.K @ hat_P
        return np.asarray(u, dtype=float).reshape(-1), hat_Z, hat_P

    def step(self, t, Y, t_U, U, dt):
        """
        Advance the controller one time step.

        Inputs
        ------
        t   : current time
        Y   : current measurement Y(t)
        t_U : stored control times up to current time
        U   : stored control values up to current time
        dt  : step size

        Returns
        -------
        u_now : control U(t)
        """
        Y = np.asarray(Y, dtype=float).reshape(-1)
        xi_now = self.xi.copy()

        # compute control using current xi(t)
        u_now, hat_Z, hat_P = self.compute_control(t, xi_now, t_U, U)

        # RK4 update for observer state xi
        def f(tau, xi_tau):
            return self.xi_rhs(tau, xi_tau, Y, t_U, U)

        q1 = f(t, xi_now)
        q2 = f(t + 0.5 * dt, xi_now + 0.5 * dt * q1)
        q3 = f(t + 0.5 * dt, xi_now + 0.5 * dt * q2)
        q4 = f(t + dt, xi_now + dt * q3)

        self.xi = xi_now + (dt / 6.0) * (q1 + 2 * q2 + 2 * q3 + q4)

        self.hat_Z = hat_Z
        self.hat_P = hat_P
        self.last_u = np.asarray(u_now, dtype=float).reshape(-1)

        self.t_hist.append(float(t))
        self.xi_hist.append(self.xi.copy())
        self.hat_Z_hist.append(hat_Z.copy())
        self.hat_P_hist.append(hat_P.copy())
        self.u_hist.append(self.last_u.copy())

        return self.last_u