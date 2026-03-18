import numpy as np


def check_delay_functions(
    T,
    D1,
    D2,
    phi1,
    phi2,
    dphi1,
    dphi2,
    phi1_inv,
    n_grid=2000,
    tol=1e-8,
    verbose=True,
):
    """
    Numerically validate the main delay assumptions over [0, T].

    Checks:
      1. D1(t), D2(t) >= 0
      2. phi1(t) = t - D1(t), phi2(t) = t - D2(t)
      3. dphi1(t) > 0, dphi2(t) > 0
      4. phi1, phi2 are increasing on [0, T]
      5. phi1_inv(phi1(t)) ~= t on [0, T]
      6. phi2(0) <= 0 and phi1(phi2(0)) <= 0 for history initialization

    Returns
    -------
    results : dict
        Dictionary of booleans and summary statistics.
    """

    t_grid = np.linspace(0.0, T, n_grid)

    D1_vals = np.array([D1(t) for t in t_grid], dtype=float)
    D2_vals = np.array([D2(t) for t in t_grid], dtype=float)

    min_D1 = np.min(D1_vals)
    min_D2 = np.min(D2_vals)

    phi1_vals = np.array([phi1(t) for t in t_grid], dtype=float)
    phi2_vals = np.array([phi2(t) for t in t_grid], dtype=float)

    dphi1_vals = np.array([dphi1(t) for t in t_grid], dtype=float)
    dphi2_vals = np.array([dphi2(t) for t in t_grid], dtype=float)

    # 1. Nonnegative delays
    D1_nonnegative = np.all(D1_vals >= -tol)
    D2_nonnegative = np.all(D2_vals >= -tol)

    # 2. Consistency phi_i(t) = t - D_i(t)
    phi1_consistency_err = np.max(np.abs(phi1_vals - (t_grid - D1_vals)))
    phi2_consistency_err = np.max(np.abs(phi2_vals - (t_grid - D2_vals)))

    phi1_consistent = phi1_consistency_err <= 100 * tol
    phi2_consistent = phi2_consistency_err <= 100 * tol

    # 3. Positive derivatives
    dphi1_positive = np.all(dphi1_vals > tol)
    dphi2_positive = np.all(dphi2_vals > tol)

    min_dphi1 = np.min(dphi1_vals)
    min_dphi2 = np.min(dphi2_vals)

    # 4. Increasing phi_i
    phi1_diffs = np.diff(phi1_vals)
    phi2_diffs = np.diff(phi2_vals)

    phi1_increasing = np.all(phi1_diffs > -tol)
    phi2_increasing = np.all(phi2_diffs > -tol)

    min_phi1_step = np.min(phi1_diffs) if len(phi1_diffs) > 0 else np.nan
    min_phi2_step = np.min(phi2_diffs) if len(phi2_diffs) > 0 else np.nan

    # 5. Inverse consistency for phi1
    phi1_of_t = np.array([phi1(t) for t in t_grid], dtype=float)

    valid_mask = (
        (phi1_of_t >= phi1_inv.t_min) &
        (phi1_of_t <= phi1_inv.t_max)
    )

    if np.any(valid_mask):
        phi1_inv_err = np.max(
            np.abs([
                phi1_inv(phi1(t)) - t
                for t in t_grid[valid_mask]
            ])
        )
        phi1_inv_consistent = phi1_inv_err <= 1
    else:
        phi1_inv_err = np.nan
        phi1_inv_consistent = False

    # 6. Initialization interval checks
    phi2_0 = phi2(0.0)
    phi1_phi2_0 = phi1(phi2_0)

    phi2_0_valid = phi2_0 <= tol
    phi1_phi2_0_valid = phi1_phi2_0 <= tol

    all_ok = all([
        D1_nonnegative,
        D2_nonnegative,
        phi1_consistent,
        phi2_consistent,
        dphi1_positive,
        dphi2_positive,
        phi1_increasing,
        phi2_increasing,
        phi1_inv_consistent,
        phi2_0_valid,
        phi1_phi2_0_valid,
    ])

    results = {
        "all_ok": all_ok,
        "D1_nonnegative": D1_nonnegative,
        "D2_nonnegative": D2_nonnegative,
        "min_D1": min_D1,
        "min_D2": min_D2,
        "phi1_consistent_with_D1": phi1_consistent,
        "phi2_consistent_with_D2": phi2_consistent,
        "phi1_consistency_error": phi1_consistency_err,
        "phi2_consistency_error": phi2_consistency_err,
        "dphi1_positive": dphi1_positive,
        "dphi2_positive": dphi2_positive,
        "min_dphi1": min_dphi1,
        "min_dphi2": min_dphi2,
        "phi1_increasing": phi1_increasing,
        "phi2_increasing": phi2_increasing,
        "min_phi1_step": min_phi1_step,
        "min_phi2_step": min_phi2_step,
        "phi1_inv_consistent": phi1_inv_consistent,
        "phi1_inv_error": phi1_inv_err,
        "phi2_0": phi2_0,
        "phi1_phi2_0": phi1_phi2_0,
        "phi2_0_valid_for_history": phi2_0_valid,
        "phi1_phi2_0_valid_for_history": phi1_phi2_0_valid,
    }

    if verbose:
        print("=== Delay Assumption Check ===")
        print(f"all_ok: {results['all_ok']}")
        print(f"D1_nonnegative: {D1_nonnegative}")
        print(f"D2_nonnegative: {D2_nonnegative}")
        print(f"min D1(t): {min_D1:.6e}")
        print(f"min D2(t): {min_D2:.6e}")
        print(f"phi1_consistent_with_D1: {phi1_consistent} (max err = {phi1_consistency_err:.3e})")
        print(f"phi2_consistent_with_D2: {phi2_consistent} (max err = {phi2_consistency_err:.3e})")
        print(f"dphi1_positive: {dphi1_positive} (min = {min_dphi1:.6e})")
        print(f"dphi2_positive: {dphi2_positive} (min = {min_dphi2:.6e})")
        print(f"phi1_increasing: {phi1_increasing} (min step = {min_phi1_step:.6e})")
        print(f"phi2_increasing: {phi2_increasing} (min step = {min_phi2_step:.6e})")
        print(f"phi1_inv_consistent: {phi1_inv_consistent} (max err = {phi1_inv_err:.3e})")
        print(f"phi2(0): {phi2_0:.6e} -> valid history interval: {phi2_0_valid}")
        print(f"phi1(phi2(0)): {phi1_phi2_0:.6e} -> valid input history interval: {phi1_phi2_0_valid}")

    return results