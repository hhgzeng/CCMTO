"""
SDLSO: Adaptive Stochastic Dominant Learning Swarm Optimizer for High-Dimensional Optimization.

Reference:
Q. Yang, W. N. Chen, T. Gu, H. Jin, W. Mao, and J. Zhang,
"An Adaptive Stochastic Dominant Learning Swarm Optimizer for High-Dimensional Optimization,"
IEEE Transactions on Cybernetics, vol. 52, no. 3, pp. 1960-1976, Mar. 2022.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class SDLSO:
    """
    Adaptive Stochastic Dominant Learning Swarm Optimizer (SDLSO).

    A non-decomposition swarm intelligence optimizer for large-scale optimization.
    Features pairwise stochastic dominance competition: a particle is updated by learning
    from two dominant exemplars only if it is dominated by both; otherwise, it is retained
    without consuming function evaluations.
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        dim: int,
        lower: Union[float, np.ndarray] = -100.0,
        upper: Union[float, np.ndarray] = 100.0,
        max_fes: int = 3_000_000,
        pop_size: int = 100,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 1.496,
        c2: float = 1.496,
        mutation_rate: float = 0.05,
        verbose: bool = False,
        log_interval: int = 10000,
    ):
        """
        Args:
            func: Objective function to minimize f(x)
            dim: Dimension of design variables
            lower: Lower bound(s)
            upper: Upper bound(s)
            max_fes: Maximum function evaluations
            pop_size: Swarm population size (N)
            w_max: Maximum inertia weight
            w_min: Minimum inertia weight
            c1: Acceleration coefficient for dominant exemplar 1
            c2: Acceleration coefficient for dominant exemplar 2
            mutation_rate: Stochastic dimension mutation probability
            verbose: Print progress
            log_interval: Logging frequency
        """
        self.func = func
        self.dim = dim
        self.lower = np.full(dim, lower) if np.isscalar(lower) else np.asarray(lower, dtype=float)
        self.upper = np.full(dim, upper) if np.isscalar(upper) else np.asarray(upper, dtype=float)
        self.max_fes = max_fes
        self.pop_size = pop_size
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.mutation_rate = mutation_rate
        self.verbose = verbose
        self.log_interval = log_interval

        self.v_max = 0.2 * (self.upper - self.lower)
        self.v_min = -self.v_max

        self.fes = 0
        self.best_x: Optional[np.ndarray] = None
        self.best_f: float = float("inf")
        self.history: List[Tuple[int, float]] = []

    def _eval(self, x: np.ndarray) -> float:
        """Evaluate particle position and track global best."""
        self.fes += 1
        val = float(self.func(x))
        if val < self.best_f:
            self.best_f = val
            self.best_x = x.copy()
        return val

    def optimize(self) -> Dict[str, Union[np.ndarray, float, int, List]]:
        """Run SDLSO optimization."""
        self.fes = 0
        self.history = []

        # 1. Initialize swarm positions and velocities
        positions = np.random.uniform(self.lower, self.upper, (self.pop_size, self.dim))
        velocities = np.random.uniform(self.v_min, self.v_max, (self.pop_size, self.dim))
        fits = np.full(self.pop_size, float("inf"))

        for i in range(self.pop_size):
            if self.fes >= self.max_fes:
                break
            fits[i] = self._eval(positions[i])

        self.history.append((self.fes, self.best_f))

        last_log = self.fes
        gen = 0

        # 2. Main Swarm Evolution Loop
        while self.fes < self.max_fes:
            gen += 1

            # Adaptive inertia weight
            progress = min(1.0, self.fes / max(1, self.max_fes))
            w = self.w_max - (self.w_max - self.w_min) * progress

            for i in range(self.pop_size):
                if self.fes >= self.max_fes:
                    break

                # Pick two distinct particles r1, r2 != i randomly
                other_indices = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2 = np.random.choice(other_indices, 2, replace=False)

                # Stochastic Dominance Check:
                # Particle i is updated ONLY IF it is dominated by both r1 and r2
                if fits[r1] < fits[i] and fits[r2] < fits[i]:
                    # Identify primary and secondary dominant exemplars
                    if fits[r1] <= fits[r2]:
                        dom1, dom2 = r1, r2
                    else:
                        dom1, dom2 = r2, r1

                    x_dom1 = positions[dom1]
                    x_dom2 = positions[dom2]

                    # Velocity update with stochastic acceleration
                    rand1 = np.random.rand(self.dim)
                    rand2 = np.random.rand(self.dim)
                    velocities[i] = (
                        w * velocities[i]
                        + self.c1 * rand1 * (x_dom1 - positions[i])
                        + self.c2 * rand2 * (x_dom2 - positions[i])
                    )
                    velocities[i] = np.clip(velocities[i], self.v_min, self.v_max)

                    # Position update
                    new_pos = positions[i] + velocities[i]

                    # Stochastic dimension mutation for high-dimensional diversity
                    mutate_mask = np.random.rand(self.dim) < self.mutation_rate
                    if np.any(mutate_mask):
                        perturbation = np.random.normal(
                            0.0,
                            0.1 * (self.upper[mutate_mask] - self.lower[mutate_mask]),
                        )
                        new_pos[mutate_mask] += perturbation

                    # Boundary handling
                    out_low = new_pos < self.lower
                    out_high = new_pos > self.upper
                    new_pos[out_low] = self.lower[out_low]
                    velocities[i][out_low] = -0.5 * velocities[i][out_low]
                    new_pos[out_high] = self.upper[out_high]
                    velocities[i][out_high] = -0.5 * velocities[i][out_high]

                    # Evaluate updated particle
                    new_fit = self._eval(new_pos)

                    # Update particle state
                    positions[i] = new_pos
                    fits[i] = new_fit
                else:
                    # Retention mechanism: particle is NOT updated and consumes NO FEs
                    pass

            self.history.append((self.fes, self.best_f))

            if self.verbose and (self.fes - last_log >= self.log_interval):
                print(
                    f"[SDLSO Gen {gen}] FEs: {self.fes:,}/{self.max_fes:,} | Best Fitness: {self.best_f:.6e}"
                )
                last_log = self.fes

        return {
            "best_x": self.best_x,
            "best_f": self.best_f,
            "fes": self.fes,
            "history": self.history,
        }
