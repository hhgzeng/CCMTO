"""
CCFR3: Cooperative Co-evolution with Efficient Resource Allocation (Yang et al. 2022).

Uses an exponential moving average (EMA) of historical contributions to adaptively
schedule MTOPs across multiple co-evolutionary stages.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from src.CCMTO.CCMTO import CCMTO
from src.CCMTO.ResourceAllocation import ResourceAllocation
from decomposition.edg import EDG
from src.CCMTO.MTOPConstruction import MTOPConstruction


class CCFR3(CCMTO):
    """
    CCMTO variant with CCFR3 resource allocation strategy.
    Multi-stage EMA contribution tracking and dynamic adaptive budget allocation.
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
        ema_weight: float = 0.7,
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
        self.ema_weight = ema_weight

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

        ema_contribs = np.zeros(k_mtops)

        while self.fe_counter[0] < self.max_fes:
            allocator.reset_cycle()

            # Sweep all MTOPs
            for i in range(k_mtops):
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

            # Update EMA contributions
            ema_contribs = self.ema_weight * ema_contribs + (1.0 - self.ema_weight) * allocator.contributions

            # Adaptive Stage: optimize top contributing MTOPs
            sorted_indices = np.argsort(ema_contribs)[::-1]
            top_count = max(1, k_mtops // 2)
            for idx in sorted_indices[:top_count]:
                if self.fe_counter[0] >= self.max_fes:
                    break
                self.best_x, self.best_f = allocator.optimize_mtop(
                    mtop_idx=int(idx),
                    best_x=self.best_x,
                    best_f=self.best_f,
                    max_fes=self.max_fes,
                    fe_counter=self.fe_counter,
                )
                self.history.append((self.fe_counter[0], self.best_f))

        return {
            "best_x": self.best_x,
            "best_f": self.best_f,
            "fes": self.fe_counter[0],
            "history": self.history,
            "subproblems": subproblems,
            "mtops": mtops,
        }
