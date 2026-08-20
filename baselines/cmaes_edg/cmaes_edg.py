"""
CMAES-EDG: Cooperative Co-evolution with Efficient Differential Grouping and CMA-ES optimizer.

Decomposes large-scale optimization problems into subcomponents using EDG,
and sequentially optimizes each subproblem using single-task CMA-ES with
round-robin cooperative co-evolution.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from decomposition.edg import EDG
from src.MTES_DAKG.CMAES import CMAES


class CMAES_EDG:
    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        dim: int,
        lower: Union[float, np.ndarray] = -100.0,
        upper: Union[float, np.ndarray] = 100.0,
        max_fes: int = 3_000_000,
        max_gen_per_cycle: int = 50,
        edg_epsilon: float = 1e-3,
        custom_subproblems: Optional[List[List[int]]] = None,
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
            max_gen_per_cycle: Generations per subproblem in each co-evolutionary cycle
            edg_epsilon: EDG interaction detection threshold
            custom_subproblems: Pre-computed subproblems (skip EDG if provided)
            verbose: Print progress
            log_interval: Logging frequency
        """
        self.func = func
        self.dim = dim
        self.lower = np.full(dim, lower) if np.isscalar(lower) else np.asarray(lower, dtype=float)
        self.upper = np.full(dim, upper) if np.isscalar(upper) else np.asarray(upper, dtype=float)
        self.max_fes = max_fes
        self.max_gen_per_cycle = max_gen_per_cycle
        self.edg_epsilon = edg_epsilon
        self.custom_subproblems = custom_subproblems
        self.verbose = verbose
        self.log_interval = log_interval

        self.fes = 0
        self.best_x: Optional[np.ndarray] = None
        self.best_f: float = float("inf")
        self.history: List[Tuple[int, float]] = []

    def _eval(self, x: np.ndarray) -> float:
        """Evaluate full candidate solution and track best."""
        self.fes += 1
        val = float(self.func(x))
        if val < self.best_f:
            self.best_f = val
            self.best_x = x.copy()
        return val

    def optimize(self) -> Dict[str, Union[np.ndarray, float, int, List]]:
        """Run CMAES-EDG optimization."""
        self.fes = 0
        self.history = []

        # 1. Variable Grouping via EDG
        if self.custom_subproblems is not None:
            subproblems = self.custom_subproblems
        else:
            edg_solver = EDG(
                func=self._eval,
                dim=self.dim,
                lower=self.lower,
                upper=self.upper,
                epsilon=self.edg_epsilon,
            )
            subproblems, _ = edg_solver.run()

        # 2. Initialize global collaborator
        init_x = np.random.uniform(self.lower, self.upper, self.dim)
        init_f = self._eval(init_x)
        self.best_x = init_x.copy()
        self.best_f = init_f
        self.history.append((self.fes, self.best_f))

        # 3. Create CMA-ES solvers for each subproblem
        cmaes_pool: List[CMAES] = []
        for sp in subproblems:
            sp_dim = len(sp)
            sp_low = self.lower[sp]
            sp_high = self.upper[sp]
            sp_mean = self.best_x[sp].copy()
            cma = CMAES(
                dim=sp_dim,
                lower=sp_low,
                upper=sp_high,
                mean=sp_mean,
            )
            cmaes_pool.append(cma)

        last_log = self.fes
        cycle = 0

        # 4. Cooperative Co-evolution Loop
        while self.fes < self.max_fes:
            cycle += 1

            for k, (sp, cma) in enumerate(zip(subproblems, cmaes_pool)):
                if self.fes >= self.max_fes:
                    break

                for g in range(self.max_gen_per_cycle):
                    if self.fes >= self.max_fes:
                        break

                    # Sample offspring from CMA-ES
                    samples, _ = cma.sample()
                    fitnesses = np.zeros(len(samples))

                    for i in range(len(samples)):
                        if self.fes >= self.max_fes:
                            fitnesses[i] = self.best_f
                            continue

                        cand_x = self.best_x.copy()
                        cand_x[sp] = samples[i]
                        cand_x = np.clip(cand_x, self.lower, self.upper)
                        f_val = self._eval(cand_x)
                        fitnesses[i] = f_val

                    # Update CMA-ES distribution
                    cma.update(samples, fitnesses)
                    self.history.append((self.fes, self.best_f))

                    if self.verbose and (self.fes - last_log >= self.log_interval):
                        print(
                            f"[CMAES-EDG Cycle {cycle}] FEs: {self.fes:,}/{self.max_fes:,} | Best Fitness: {self.best_f:.6e}"
                        )
                        last_log = self.fes

        return {
            "best_x": self.best_x,
            "best_f": self.best_f,
            "fes": self.fes,
            "history": self.history,
            "subproblems": subproblems,
        }
