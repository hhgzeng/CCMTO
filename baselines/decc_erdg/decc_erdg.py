"""
DECC-ERDG: Cooperative Co-evolution with Efficient Recursive Differential Grouping and Differential Evolution (DE).

Reference:
M. Yang, A. Zhou, C. Li, and X. Yao,
"An Efficient Recursive Differential Grouping for Large-Scale Continuous Problems,"
IEEE Transactions on Evolutionary Computation, vol. 25, no. 1, pp. 159-171, Feb. 2021.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from decomposition.erdg import ERDG


class DECC_ERDG:
    """
    Cooperative Co-evolution algorithm with Efficient Recursive Differential Grouping (DECC-ERDG).

    Decomposes large-scale optimization problems into subcomponents using ERDG,
    and sequentially optimizes each subproblem using Differential Evolution (DE/rand/1/bin).
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        dim: int,
        lower: Union[float, np.ndarray] = -100.0,
        upper: Union[float, np.ndarray] = 100.0,
        max_fes: int = 3_000_000,
        pop_size: int = 50,
        f_weight: float = 0.5,
        cr_prob: float = 0.9,
        gen_per_cycle: int = 10,
        erdg_epsilon: float = 1e-3,
        custom_subproblems: Optional[List[List[int]]] = None,
        verbose: bool = False,
        log_interval: int = 10000,
    ):
        """
        Args:
            func: Objective function f(x)
            dim: Total dimension of design variables
            lower: Lower bound(s)
            upper: Upper bound(s)
            max_fes: Maximum function evaluations
            pop_size: Subpopulation size for each subproblem
            f_weight: DE mutation scaling factor F
            cr_prob: DE crossover probability CR
            gen_per_cycle: Number of DE generations per subproblem in each CC cycle
            erdg_epsilon: ERDG interaction detection threshold
            custom_subproblems: Pre-computed subproblems (skip ERDG if provided)
            verbose: Print progress
            log_interval: Logging frequency
        """
        self.func = func
        self.dim = dim
        self.lower = np.full(dim, lower) if np.isscalar(lower) else np.asarray(lower, dtype=float)
        self.upper = np.full(dim, upper) if np.isscalar(upper) else np.asarray(upper, dtype=float)
        self.max_fes = max_fes
        self.pop_size = pop_size
        self.f_weight = f_weight
        self.cr_prob = cr_prob
        self.gen_per_cycle = gen_per_cycle
        self.erdg_epsilon = erdg_epsilon
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
        """Run DECC-ERDG optimization."""
        self.fes = 0
        self.history = []

        # 1. Variable Grouping via ERDG
        if self.custom_subproblems is not None:
            subproblems = self.custom_subproblems
        else:
            erdg_solver = ERDG(
                func=self._eval,
                dim=self.dim,
                lower=self.lower,
                upper=self.upper,
                epsilon=self.erdg_epsilon,
            )
            subproblems, _ = erdg_solver.run()

        # 2. Initialize global collaborator
        init_x = np.random.uniform(self.lower, self.upper, self.dim)
        init_f = self._eval(init_x)
        self.best_x = init_x.copy()
        self.best_f = init_f
        self.history.append((self.fes, self.best_f))

        # 3. Initialize subpopulation for each subproblem
        subpops: List[np.ndarray] = []
        subfits: List[np.ndarray] = []

        for sp in subproblems:
            sp_dim = len(sp)
            sp_low = self.lower[sp]
            sp_high = self.upper[sp]

            # Generate population (NP x sp_dim)
            pop = np.random.uniform(sp_low, sp_high, (self.pop_size, sp_dim))
            # Include current best collaborator in initial population
            pop[0] = self.best_x[sp].copy()

            fits = np.full(self.pop_size, float("inf"))
            for i in range(self.pop_size):
                if self.fes >= self.max_fes:
                    break
                cand_x = self.best_x.copy()
                cand_x[sp] = pop[i]
                cand_x = np.clip(cand_x, self.lower, self.upper)
                fits[i] = self._eval(cand_x)

            subpops.append(pop)
            subfits.append(fits)

        last_log = self.fes
        cycle = 0

        # 4. Cooperative Co-evolution Loop
        while self.fes < self.max_fes:
            cycle += 1

            for k, sp in enumerate(subproblems):
                if self.fes >= self.max_fes:
                    break

                pop = subpops[k]
                fits = subfits[k]
                sp_dim = len(sp)
                sp_low = self.lower[sp]
                sp_high = self.upper[sp]

                for g in range(self.gen_per_cycle):
                    if self.fes >= self.max_fes:
                        break

                    for i in range(self.pop_size):
                        if self.fes >= self.max_fes:
                            break

                        # Select 3 distinct random candidates != i
                        idxs = [idx for idx in range(self.pop_size) if idx != i]
                        if len(idxs) < 3:
                            continue
                        r1, r2, r3 = np.random.choice(idxs, 3, replace=False)

                        # DE Mutation: DE/rand/1
                        mutant = pop[r1] + self.f_weight * (pop[r2] - pop[r3])
                        mutant = np.clip(mutant, sp_low, sp_high)

                        # Binomial Crossover
                        trial = pop[i].copy()
                        cross_points = np.random.rand(sp_dim) < self.cr_prob
                        if not np.any(cross_points):
                            cross_points[np.random.randint(0, sp_dim)] = True
                        trial[cross_points] = mutant[cross_points]

                        # Evaluate trial vector embedded in collaborator
                        cand_x = self.best_x.copy()
                        cand_x[sp] = trial
                        cand_x = np.clip(cand_x, self.lower, self.upper)
                        trial_f = self._eval(cand_x)

                        # Selection
                        if trial_f <= fits[i]:
                            pop[i] = trial
                            fits[i] = trial_f

                    self.history.append((self.fes, self.best_f))

                    if self.verbose and (self.fes - last_log >= self.log_interval):
                        print(
                            f"[DECC-ERDG Cycle {cycle}] FEs: {self.fes:,}/{self.max_fes:,} | Best Fitness: {self.best_f:.6e}"
                        )
                        last_log = self.fes

        return {
            "best_x": self.best_x,
            "best_f": self.best_f,
            "fes": self.fes,
            "history": self.history,
            "subproblems": subproblems,
        }
