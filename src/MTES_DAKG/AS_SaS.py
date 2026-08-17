"""
Adaptive Elite Sampling Shape KGxS (AS_SaS)

Algorithm 5 in CCMTO:
Transfers function shape knowledge from source task elite samples to target task,
using rank-based exponential weighting and covariance alignment.
"""

from typing import Optional
import numpy as np


def as_sas(
    X_s: np.ndarray,
    F_s: np.ndarray,
    m_t: np.ndarray,
    C_t_sqrt: np.ndarray,
    m_s: np.ndarray,
    C_s_inv_sqrt: np.ndarray,
    gen: int,
    max_gen: int,
    popsize: int,
    a: float = 2.0,
    gamma: float = 2.0,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate external sample using Adaptive Elite Sampling Shape KGxS.

    Args:
        X_s: Source task population samples (shape: [lambda, dim_s])
        F_s: Objective values of source task samples (shape: [lambda])
        m_t: Target distribution mean vector (dim_t)
        C_t_sqrt: Matrix C_t^(1/2) of target task (dim_t x dim_t)
        m_s: Source distribution mean vector (dim_s)
        C_s_inv_sqrt: Matrix C_s^(-1/2) of source task (dim_s x dim_s)
        gen: Current generation index
        max_gen: Maximum generations in current cycle
        popsize: Population size (lambda)
        a: Rate parameter for dynamic elite sample growth (default: 2.0)
        gamma: Weighting coefficient (default: 2.0)
        lower: Target lower bounds
        upper: Target upper bounds

    Returns:
        hat_x: Knowledge-guided external sample vector (shape: [dim_t])
    """
    dim_t = len(m_t)
    dim_s = len(m_s)
    actual_popsize = len(F_s)

    n_min = max(2, int(np.floor(0.3 * popsize)))
    n_max = max(n_min, int(np.floor(0.8 * popsize)))
    M = max(1, int(np.floor(0.6 * max_gen)))

    # 1. Dynamic elite sample count
    if gen <= M:
        n = int(np.round(((gen / M) ** a) * (n_max - n_min) + n_min))
    else:
        n = n_max

    n = max(2, min(n, actual_popsize))

    # 2. Sort source samples by fitness ascending (objective values ascending)
    order = np.argsort(F_s)
    elite_idx = order[:n]
    Z_s = X_s[elite_idx]

    # 3. Random exclusion index j
    j_ex = np.random.randint(0, n)
    indices = [i for i in range(n) if i != j_ex]

    # Weights calculation
    raw_w = np.zeros(len(indices))
    for idx_pos, i in enumerate(indices):
        raw_w[idx_pos] = np.exp(gamma * (n - (i + 1)) / n)

    w_sum = np.sum(raw_w)
    weights = raw_w / (w_sum if w_sum > 1e-14 else 1.0)

    # 4. Weighted center position relative to source mean m_s
    diff_s = Z_s[indices] - m_s  # shape: [len(indices), dim_s]
    y_S = np.sum(weights[:, np.newaxis] * diff_s, axis=0)  # shape: [dim_s]

    # 5. Transform center position to target space: C_t^(1/2) * C_s^(-1/2) * y_S
    # Handle dimension matching if source and target dimensions differ
    if dim_s == dim_t:
        inv_y = C_s_inv_sqrt @ y_S
        trans_y = C_t_sqrt @ inv_y
    else:
        # Align via projection/padding
        inv_y_s = C_s_inv_sqrt @ y_S
        if dim_s > dim_t:
            aligned_y = inv_y_s[:dim_t]
        else:
            aligned_y = np.zeros(dim_t)
            aligned_y[:dim_s] = inv_y_s
        trans_y = C_t_sqrt @ aligned_y

    hat_x = m_t + trans_y

    if lower is not None and upper is not None:
        hat_x = np.clip(hat_x, lower, upper)

    return hat_x
