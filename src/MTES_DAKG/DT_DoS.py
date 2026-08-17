"""
Dynamic Distance Threshold Domain KGxS with Gradient Correction (DT_DoS)

Algorithm 4 in CCMTO:
Transfers domain knowledge from a source task distribution to a target task,
using region-based dynamic distance thresholds and numerical gradient correction.
"""

from typing import Callable, Optional
import numpy as np


def dt_dos(
    X_t: np.ndarray,
    F_t: np.ndarray,
    m_t: np.ndarray,
    C_t: np.ndarray,
    m_s: np.ndarray,
    C_s: np.ndarray,
    eval_target: Callable[[np.ndarray], float],
    k_knn: int = 5,
    beta: float = 1e-5,
    phi: float = 1.0,
    sigma_s: float = 1.0,
    C_s_sqrt: Optional[np.ndarray] = None,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate external sample using Dynamic Distance Threshold Domain KGxS with Gradient Correction.

    Args:
        X_t: Current generation samples of target task (shape: [lambda, dim])
        F_t: Objective values of target task samples (shape: [lambda])
        m_t: Target distribution mean vector
        C_t: Target covariance matrix
        m_s: Source distribution mean vector
        C_s: Source covariance matrix
        eval_target: Target objective function evaluator f(x)
        k_knn: Number of nearest neighbors for region affiliation
        beta: Finite difference step for gradient estimation
        phi: Gradient correction coefficient
        sigma_s: Step size of source distribution
        C_s_sqrt: Precomputed C_s^(1/2) matrix from source optimizer
        lower: Lower bound array (optional)
        upper: Upper bound array (optional)

    Returns:
        hat_x: Knowledge-guided external sample vector (shape: [dim])
    """
    dim = len(m_t)
    popsize = len(F_t)

    # Proportion coefficients
    if popsize <= 6:
        mu1 = 0.4
        mu2 = 0.6
    else:
        mu1 = 0.3
        mu2 = 0.4

    # 1. Sort X_t by objective value ascending (high fitness = low objective)
    order = np.argsort(F_t)
    sorted_X = X_t[order]

    idx1 = max(1, int(np.floor(mu1 * popsize)))
    idx2 = max(idx1 + 1, int(np.floor((mu1 + mu2) * popsize)))
    idx2 = min(idx2, popsize - 1) if popsize >= 3 else popsize

    S_high = sorted_X[:idx1]
    S_mid = sorted_X[idx1:idx2] if idx2 > idx1 else sorted_X[idx1:]
    S_low = sorted_X[idx2:] if idx2 < popsize else sorted_X[-1:]

    regions = [S_high, S_mid, S_low]

    # 2. Calculate dynamic distance thresholds
    thresholds = []
    for reg in regions:
        if len(reg) == 0:
            thresholds.append(1e-3)
        else:
            dists = np.linalg.norm(reg - m_t, axis=1)
            thresholds.append(max(float(np.mean(dists)), 1e-6))

    d_high, d_mid, d_low = thresholds

    # 3. Sample z ~ N(m_s, sigma_s^2 * C_s) without SVD non-convergence risk
    dim_s = len(m_s)
    if C_s_sqrt is not None:
        z_raw = m_s + sigma_s * (C_s_sqrt @ np.random.randn(dim_s))
    else:
        eigenvals, eigenvecs = np.linalg.eigh(C_s)
        eigenvals = np.clip(np.nan_to_num(eigenvals, nan=1.0), 1e-14, 1e14)
        c_sqrt = eigenvecs @ np.diag(np.sqrt(eigenvals)) @ eigenvecs.T
        z_raw = m_s + sigma_s * (c_sqrt @ np.random.randn(dim_s))

    z_raw = np.nan_to_num(z_raw, nan=0.0)

    # Ensure dimension matches target
    if len(z_raw) != dim:
        if len(z_raw) > dim:
            z = z_raw[:dim]
        else:
            z = np.zeros(dim)
            z[: len(z_raw)] = z_raw
    else:
        z = z_raw

    # 4. Region affiliation using KNN
    avg_dists = []
    for reg in regions:
        if len(reg) == 0:
            avg_dists.append(float("inf"))
        else:
            dists = np.linalg.norm(reg - z, axis=1)
            sorted_d = np.sort(dists)
            k_val = min(k_knn, len(sorted_d))
            avg_dists.append(float(np.mean(sorted_d[:k_val])))

    best_region_idx = int(np.argmin(avg_dists))
    d_g_star = thresholds[best_region_idx]

    # 5. Estimate gradient at m_t
    g = np.zeros(dim)
    for i in range(dim):
        ei = np.zeros(dim)
        ei[i] = beta
        f_plus = eval_target(m_t + ei)
        f_minus = eval_target(m_t - ei)
        g[i] = (f_plus - f_minus) / (2.0 * beta)

    g = np.nan_to_num(g, nan=0.0)

    # Optimal direction: gradient descent direction
    g_opt = -g
    norm_g = np.linalg.norm(g_opt)
    if norm_g > 1e-12:
        g_grad_opt = g_opt / norm_g
    else:
        g_grad_opt = np.ones(dim) / np.sqrt(dim)

    # 6. Direction vector from m_t to z
    diff_z = z - m_t
    dist_z = np.linalg.norm(diff_z)
    if dist_z > 1e-12:
        v = diff_z / dist_z
    else:
        v = g_grad_opt.copy()

    # 7. Angle theta between v and g_grad_opt
    dot_val = np.clip(np.dot(v, g_grad_opt), -1.0, 1.0)
    theta = np.arccos(dot_val)

    # 8. External sample calculation
    if theta < (np.pi / 2.0):  # Acute angle: points in improvement direction
        if dist_z < d_g_star:
            hat_x = z.copy()
        else:
            hat_x = m_t + d_g_star * v
    else:  # Obtuse angle: requires gradient correction
        v_prime_raw = v + phi * g_grad_opt
        norm_v_prime = np.linalg.norm(v_prime_raw)
        v_prime = v_prime_raw / norm_v_prime if norm_v_prime > 1e-12 else g_grad_opt
        hat_x = m_t + d_g_star * v_prime

    hat_x = np.nan_to_num(hat_x, nan=0.0)

    if lower is not None and upper is not None:
        hat_x = np.clip(hat_x, lower, upper)

    return hat_x
