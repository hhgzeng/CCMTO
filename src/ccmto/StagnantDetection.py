"""
Stagnant Subtask Detection

Implements Equations (10)-(18) from the CCMTO paper to identify subtasks whose
fitness improvement and population diversity have stagnated.
"""

from typing import Optional
import numpy as np


class StagnantDetection:
    def __init__(self, dim: int, epsilon: float = 1e-6):
        """
        Initialize stagnant detector for a single subtask.

        Args:
            dim: Dimension of subtask variables
            epsilon: Threshold for relative change in fitness and gene diversity
        """
        self.dim = dim
        self.epsilon = epsilon

        self.last_best_f: Optional[float] = None
        self.last_mean: Optional[np.ndarray] = None
        self.last_std: Optional[np.ndarray] = None

        self.v_G: int = 0      # Successive generations with no fitness change
        self.eta_G: int = 0    # Successive generations with no diversity change across all dimensions
        self.is_stagnant: bool = False

    def reset(self):
        """Reset internal counters for a new co-evolution cycle."""
        self.last_best_f = None
        self.last_mean = None
        self.last_std = None
        self.v_G = 0
        self.eta_G = 0
        self.is_stagnant = False

    def check(self, current_best_f: float, current_pop: np.ndarray, max_gen: int = 50) -> bool:
        """
        Update statistics and determine if subtask is stagnant.

        Args:
            current_best_f: Best objective value of subtask at current generation
            current_pop: Population array of shape (N, dim)
            max_gen: Max generations per cycle for calculating threshold U

        Returns:
            is_stagnant (rho_G): True if subtask is stagnant, else False
        """
        if self.is_stagnant:
            return True

        U = max(1, min(self.dim, max_gen))

        # 1. Fitness stagnation check (Eq. 10 & 13)
        if self.last_best_f is not None:
            denom_f = abs(self.last_best_f) if abs(self.last_best_f) > 1e-12 else 1.0
            delta_f = abs(self.last_best_f - current_best_f) / denom_f
            if delta_f < self.epsilon:
                self.v_G += 1
            else:
                self.v_G = 0
        self.last_best_f = current_best_f

        # 2. Population diversity stagnation check across dimensions (Eq. 11, 12, 14, 15, 16)
        curr_mean = np.mean(current_pop, axis=0)
        curr_std = np.std(current_pop, axis=0)

        if self.last_mean is not None and self.last_std is not None:
            sigma_count = 0
            for d in range(self.dim):
                denom_m = abs(self.last_mean[d]) if abs(self.last_mean[d]) > 1e-12 else 1.0
                denom_std = abs(self.last_std[d]) if abs(self.last_std[d]) > 1e-12 else 1.0

                delta_m = abs(self.last_mean[d] - curr_mean[d]) / denom_m
                delta_std = abs(self.last_std[d] - curr_std[d]) / denom_std

                if delta_m < self.epsilon and delta_std < self.epsilon:
                    sigma_count += 1

            if sigma_count == self.dim:
                self.eta_G += 1
            else:
                self.eta_G = 0

        self.last_mean = curr_mean.copy()
        self.last_std = curr_std.copy()

        # 3. Detection flag rho_G (Eq. 17)
        if self.v_G >= U and self.eta_G >= U:
            self.is_stagnant = True

        return self.is_stagnant
