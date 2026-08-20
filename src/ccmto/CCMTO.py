"""
CCMTO (Cooperative Co-Evolutionary Multitask Optimization) Framework

Algorithm 2 in CCMTO:
Integrates Efficient Differential Grouping (EDG), Multitask Optimization Problem (MTOP) Construction,
Contribution-Based Resource Allocation, and MTES-DAKG to solve Large-Scale Optimization Problems.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from decomposition.edg import EDG
from .MTOPConstruction import MTOPConstruction
from .ResourceAllocation import ResourceAllocation


class CCMTO:
    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        dim: int,
        lower: Union[float, np.ndarray] = -100.0,
        upper: Union[float, np.ndarray] = 100.0,
        max_fes: int = 3_000_000,
        n_sub: int = 5,
        d_max: float = 2.0,
        epsilon: float = 1e-6,
        max_gen_per_cycle: int = 50,
        alpha: float = 0.5,
        tau: int = 1,
        fre_ratio: float = 0.1,
        edg_epsilon: float = 1e-2,
        custom_subproblems: Optional[List[List[int]]] = None,
        verbose: bool = True,
        log_interval: int = 5000,
    ):
        """
        Initialize the CCMTO optimization framework.

        Args:
            func: Objective function to minimize f(x)
            dim: Dimension of design variables
            lower: Lower bound(s)
            upper: Upper bound(s)
            max_fes: Maximum allowed function evaluations (MaxFEs)
            n_sub: Maximum subtasks per MTOP
            d_max: Maximum dimension ratio for grouping
            epsilon: Stagnation detection threshold
            max_gen_per_cycle: Number of generations per subtask optimization cycle
            alpha: Probability of Domain vs Shape KGxS in MTES-DAKG
            tau: Number of external transferred samples
            edg_epsilon: EDG interaction detection threshold
            custom_subproblems: Pre-computed subproblems (skip EDG if provided)
            verbose: Whether to print optimization progress
            log_interval: Interval of FEs for logging progress
        """
        self.func = func
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.max_fes = max_fes
        self.n_sub = n_sub
        self.d_max = d_max
        self.epsilon = epsilon
        self.max_gen_per_cycle = max_gen_per_cycle
        self.alpha = alpha
        self.tau = tau
        self.fre_ratio = fre_ratio
        self.edg_epsilon = edg_epsilon
        self.custom_subproblems = custom_subproblems
        self.verbose = verbose
        self.log_interval = log_interval

        # Tracking state
        self.fe_counter = [0]
        self.best_x: Optional[np.ndarray] = None
        self.best_f: float = float("inf")
        self.history: List[Tuple[int, float]] = []

    def _eval(self, x: np.ndarray) -> float:
        """Wrapper to evaluate objective function and track FEs."""
        self.fe_counter[0] += 1
        val = float(self.func(x))
        if val < self.best_f:
            self.best_f = val
            self.best_x = x.copy()
        return val

    def optimize(self) -> Dict[str, Union[np.ndarray, float, int, List]]:
        """
        Run the complete CCMTO optimization workflow.

        Returns:
            Dictionary containing:
                - 'best_x': Optimal decision variable vector
                - 'best_f': Optimal objective value
                - 'fes': Total function evaluations used
                - 'history': List of (fe_count, best_fitness) records
                - 'subproblems': List of decomposed subproblems
                - 'mtops': List of constructed MTOPs
        """
        self.fe_counter[0] = 0
        self.history = []

        # 1. Variable Grouping (EDG)
        if self.custom_subproblems is not None:
            subproblems = self.custom_subproblems
            if self.verbose:
                print(f"[CCMTO] Using {len(subproblems)} provided subproblems.")
        else:
            if self.verbose:
                print(f"[CCMTO] Starting EDG decomposition for {self.dim}-dimensional problem...")
            edg_solver = EDG(
                func=self._eval,
                dim=self.dim,
                lower=self.lower,
                upper=self.upper,
                epsilon=self.edg_epsilon,
            )
            subproblems, edg_fes = edg_solver.run()
            if self.verbose:
                print(
                    f"[CCMTO] EDG completed: {len(subproblems)} subproblems identified using {edg_fes} FEs."
                )

        # 2. MTOP Construction (Algorithm 3)
        constructor = MTOPConstruction(n_sub=self.n_sub, d_max=self.d_max)
        mtops = constructor.construct(subproblems)
        if self.verbose:
            print(f"[CCMTO] MTOP construction produced {len(mtops)} MTOPs.")

        # 3. Initialize global best solution
        low = np.full(self.dim, self.lower) if np.isscalar(self.lower) else np.asarray(self.lower)
        high = np.full(self.dim, self.upper) if np.isscalar(self.upper) else np.asarray(self.upper)
        init_x = np.random.uniform(low, high, self.dim)
        init_f = self._eval(init_x)
        self.best_x = init_x.copy()
        self.best_f = init_f
        self.history.append((self.fe_counter[0], self.best_f))

        # 4. Resource Allocation Manager (Algorithm 6)
        allocator = ResourceAllocation(
            mtops=mtops,
            eval_func=self._eval,
            dim=self.dim,
            lower=self.lower,
            upper=self.upper,
            max_gen=self.max_gen_per_cycle,
            epsilon=self.epsilon,
            alpha=self.alpha,
            tau=self.tau,
            fre_ratio=self.fre_ratio,
        )

        last_logged_fe = self.fe_counter[0]
        coevo_cycle = 0

        # 5. Main Co-Evolutionary Loop
        while self.fe_counter[0] < self.max_fes:
            coevo_cycle += 1
            allocator.reset_cycle()

            # Phase 1: Sequential sweep over all MTOPs
            for i in range(len(mtops)):
                if self.fe_counter[0] >= self.max_fes:
                    break
                self.best_x, self.best_f = allocator.optimize_mtop(
                    mtop_idx=i,
                    best_x=self.best_x,
                    best_f=self.best_f,
                    max_fes=self.max_fes,
                    fe_counter=self.fe_counter,
                )
                self.history.append((self.fe_counter[0], self.best_f))

                if self.verbose and (self.fe_counter[0] - last_logged_fe >= self.log_interval):
                    print(
                        f"[Cycle {coevo_cycle} - Phase 1] FEs: {self.fe_counter[0]:,}/{self.max_fes:,} | Best Fitness: {self.best_f:.6e}"
                    )
                    last_logged_fe = self.fe_counter[0]

            # Phase 2: Greedy selection based on contributions
            while (
                (np.max(allocator.contributions) - np.min(allocator.contributions) > self.epsilon)
                and self.fe_counter[0] < self.max_fes
            ):
                best_mtop_idx = int(np.argmax(allocator.contributions))
                self.best_x, self.best_f = allocator.optimize_mtop(
                    mtop_idx=best_mtop_idx,
                    best_x=self.best_x,
                    best_f=self.best_f,
                    max_fes=self.max_fes,
                    fe_counter=self.fe_counter,
                )
                self.history.append((self.fe_counter[0], self.best_f))

                if self.verbose and (self.fe_counter[0] - last_logged_fe >= self.log_interval):
                    print(
                        f"[Cycle {coevo_cycle} - Phase 2] FEs: {self.fe_counter[0]:,}/{self.max_fes:,} | Best Fitness: {self.best_f:.6e}"
                    )
                    last_logged_fe = self.fe_counter[0]

        if self.verbose:
            print(
                f"[CCMTO] Finished! Final Best Fitness: {self.best_f:.6e} after {self.fe_counter[0]} FEs."
            )

        return {
            "best_x": self.best_x,
            "best_f": self.best_f,
            "fes": self.fe_counter[0],
            "history": self.history,
            "subproblems": subproblems,
            "mtops": mtops,
        }


def ccmto(
    func: Callable[[np.ndarray], float],
    dim: int,
    lower: Union[float, np.ndarray] = -100.0,
    upper: Union[float, np.ndarray] = 100.0,
    max_fes: int = 3_000_000,
    **kwargs,
) -> Dict[str, Union[np.ndarray, float, int, List]]:
    """Convenience functional wrapper for CCMTO."""
    solver = CCMTO(func, dim, lower=lower, upper=upper, max_fes=max_fes, **kwargs)
    return solver.optimize()
