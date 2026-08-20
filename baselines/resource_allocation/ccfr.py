"""
CCFR: Cooperative Co-evolution with Fast Resource Allocation (Yang et al. 2017).

Identifies stagnant subcomponents and freezes them for several cycles
to concentrate computational budget on non-stagnant ones.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from src.CCMTO.CCMTO import CCMTO
from src.CCMTO.ResourceAllocation import ResourceAllocation
from decomposition.edg import EDG
from src.CCMTO.MTOPConstruction import MTOPConstruction


class CCFR(CCMTO):
    """
    CCMTO variant with CCFR resource allocation strategy.
    Freezes stagnant MTOPs for a number of freeze cycles (t_freeze=3).
    """

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
        edg_epsilon: float = 1e-3,
        freeze_cycles: int = 3,
        custom_subproblems: Optional[List[List[int]]] = None,
        verbose: bool = False,
        log_interval: int = 5000,
    ):
        super().__init__(
            func=func,
            dim=dim,
            lower=lower,
            upper=upper,
            max_fes=max_fes,
            n_sub=n_sub,
            d_max=d_max,
            epsilon=epsilon,
            max_gen_per_cycle=max_gen_per_cycle,
            alpha=alpha,
            tau=tau,
            fre_ratio=fre_ratio,
            edg_epsilon=edg_epsilon,
            custom_subproblems=custom_subproblems,
            verbose=verbose,
            log_interval=log_interval,
        )
        self.freeze_cycles = freeze_cycles

    def optimize(self) -> Dict[str, Union[np.ndarray, float, int, List]]:
        self.fe_counter[0] = 0
        self.history = []

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

        constructor = MTOPConstruction(n_sub=self.n_sub, d_max=self.d_max)
        mtops = constructor.construct(subproblems)
        k_mtops = len(mtops)

        low = np.full(self.dim, self.lower) if np.isscalar(self.lower) else np.asarray(self.lower)
        high = np.full(self.dim, self.upper) if np.isscalar(self.upper) else np.asarray(self.upper)
        init_x = np.random.uniform(low, high, self.dim)
        init_f = self._eval(init_x)
        self.best_x = init_x.copy()
        self.best_f = init_f
        self.history.append((self.fe_counter[0], self.best_f))

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

        frozen_counters = np.zeros(k_mtops, dtype=int)

        while self.fe_counter[0] < self.max_fes:
            allocator.reset_cycle()

            for i in range(k_mtops):
                if self.fe_counter[0] >= self.max_fes:
                    break

                if frozen_counters[i] > 0:
                    frozen_counters[i] -= 1
                    continue

                f_before = self.best_f
                self.best_x, self.best_f = allocator.optimize_mtop(
                    mtop_idx=i,
                    best_x=self.best_x,
                    best_f=self.best_f,
                    max_fes=self.max_fes,
                    fe_counter=self.fe_counter,
                )
                self.history.append((self.fe_counter[0], self.best_f))

                delta = abs(f_before - self.best_f)
                if delta < self.epsilon:
                    frozen_counters[i] = self.freeze_cycles
                else:
                    frozen_counters[i] = 0

            # If all are frozen, reset frozen counters
            if np.all(frozen_counters > 0):
                frozen_counters.fill(0)

        return {
            "best_x": self.best_x,
            "best_f": self.best_f,
            "fes": self.fe_counter[0],
            "history": self.history,
            "subproblems": subproblems,
            "mtops": mtops,
        }
