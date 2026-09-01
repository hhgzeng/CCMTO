"""
Efficient Differential Grouping (EDG / RDG)

Decomposes a large-scale optimization problem into separable and nonseparable subproblems
using recursive differential grouping to minimize function evaluations.
"""

from typing import Callable, List, Optional, Tuple, Union, cast
import numpy as np


class EDG:
    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        dim: int,
        lower: Union[float, np.ndarray] = -100.0,
        upper: Union[float, np.ndarray] = 100.0,
        delta: Optional[float] = None,
        epsilon: float = 1e-2,
    ):
        """
        Initialize EDG decomposition solver.

        Args:
            func: Objective function to evaluate f(x)
            dim: Total number of design variables
            lower: Lower bound(s) of variables
            upper: Upper bound(s) of variables
            delta: Perturbation size (default: 0.1 * (upper - lower))
            epsilon: Relative threshold for detecting variable interaction
        """
        self.func = func
        self.dim = dim

        if np.isscalar(lower):
            self.lower = np.full(dim, float(cast(float, lower)))
        else:
            self.lower = np.asarray(lower, dtype=float)

        if np.isscalar(upper):
            self.upper = np.full(dim, float(cast(float, upper)))
        else:
            self.upper = np.asarray(upper, dtype=float)

        if delta is None:
            self.delta = (self.upper - self.lower) * 0.1
        elif np.isscalar(delta):
            self.delta = np.full(dim, float(cast(float, delta)))
        else:
            self.delta = np.asarray(delta, dtype=float)

        self.epsilon = epsilon
        self.fe_count = 0

    def _eval(self, x: np.ndarray) -> float:
        """Evaluate function and track FE count."""
        self.fe_count += 1
        return float(self.func(x))

    def _interact(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        i: int,
        var_subset: List[int],
        delta_1: float,
    ) -> bool:
        """
        Test whether variable i interacts with any variable in var_subset.
        delta_1 = f(p1) - f(p1 + delta[i]*e_i)
        We evaluate:
            y3 = p1 with var_subset perturbed by delta
            y4 = y3 with variable i also perturbed by delta[i]
            delta_2 = f(y3) - f(y4)
        Uses relative interaction threshold to prevent floating-point scale noise.
        """
        if not var_subset:
            return False

        # y3: perturb var_subset
        y3 = p1.copy()
        y3[var_subset] = p2[var_subset]
        f3 = self._eval(y3)

        # y4: perturb var_subset and variable i
        y4 = y3.copy()
        y4[i] = p2[i]
        f4 = self._eval(y4)

        delta_2 = f3 - f4
        diff = abs(delta_1 - delta_2)
        scale = max(abs(delta_1), abs(delta_2), 1e-12)
        return (diff / scale) > self.epsilon

    def _find_interacting_variables(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        i: int,
        candidates: List[int],
        delta_1: float,
    ) -> List[int]:
        """
        Recursively find all variables in candidates that interact with variable i.
        """
        if not candidates:
            return []

        # Check if i interacts with the whole candidate set
        if not self._interact(p1, p2, i, candidates, delta_1):
            return []

        if len(candidates) == 1:
            return list(candidates)

        mid = len(candidates) // 2
        left = candidates[:mid]
        right = candidates[mid:]

        inter_left = self._find_interacting_variables(p1, p2, i, left, delta_1)
        inter_right = self._find_interacting_variables(p1, p2, i, right, delta_1)

        return inter_left + inter_right

    def run(self) -> Tuple[List[List[int]], int]:
        """
        Execute EDG variable grouping.

        Returns:
            groups: List of subproblems, where each subproblem is a list of variable indices.
            fe_count: Number of function evaluations consumed during grouping.
        """
        self.fe_count = 0

        # Base vectors selected in search space interior
        p1 = self.lower + 0.2 * (self.upper - self.lower)
        p2 = p1 + self.delta

        # Base evaluation at p1
        f_p1 = self._eval(p1)

        # Precompute f(p1 with x[i] perturbed) for all i
        delta_1_cache = {}
        for i in range(self.dim):
            y2 = p1.copy()
            y2[i] = p2[i]
            f_p2_i = self._eval(y2)
            delta_1_cache[i] = f_p1 - f_p2_i

        remaining = set(range(self.dim))
        nonseparable_groups = []
        separable_vars = []

        while remaining:
            seed = min(remaining)
            remaining.remove(seed)

            # Check if seed interacts with any remaining variables
            candidates = sorted(list(remaining))
            interacting = self._find_interacting_variables(
                p1, p2, seed, candidates, delta_1_cache[seed]
            )

            if not interacting:
                # seed is separable
                separable_vars.append(seed)
            else:
                # seed belongs to a non-separable component
                component = [seed] + interacting
                for var in interacting:
                    remaining.remove(var)

                # Check if other members in the component interact with remaining variables
                queue = list(interacting)
                while queue and remaining:
                    curr_var = queue.pop(0)
                    rem_candidates = sorted(list(remaining))
                    new_interacting = self._find_interacting_variables(
                        p1, p2, curr_var, rem_candidates, delta_1_cache[curr_var]
                    )
                    for nvar in new_interacting:
                        component.append(nvar)
                        queue.append(nvar)
                        remaining.remove(nvar)

                nonseparable_groups.append(sorted(component))

        # Build final subproblems list: nonseparable groups + 1-D separable groups
        all_groups = nonseparable_groups + [[v] for v in sorted(separable_vars)]

        return all_groups, self.fe_count


def edg(
    func: Callable[[np.ndarray], float],
    dim: int,
    lower: Union[float, np.ndarray] = -100.0,
    upper: Union[float, np.ndarray] = 100.0,
    delta: Optional[float] = None,
    epsilon: float = 1e-2,
) -> Tuple[List[List[int]], int]:
    """Convenience functional wrapper for EDG."""
    solver = EDG(func, dim, lower=lower, upper=upper, delta=delta, epsilon=epsilon)
    return solver.run()
