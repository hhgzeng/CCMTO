"""
wo-SD: CCMTO-MTES-DAKG without Stagnant Subtask Detection.

Uses full MTES-DAKG (with DT-DoS and AS-SaS), but Stagnant Subtask Detection is DISABLED.
All subtasks continue to receive computational resource allocations across all co-evolutionary cycles.
"""

from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from src.CCMTO.CCMTO import CCMTO
from src.CCMTO.MTOPConstruction import MTOPConstruction
from src.MTES_DAKG.MTES_DAKG import MTES_DAKG
from decomposition.edg import EDG


class WO_SD(CCMTO):
    """
    CCMTO variant with Stagnant Detection disabled (wo-SD).
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
            custom_subproblems=custom_subproblems,
            verbose=verbose,
            log_interval=log_interval,
        )

    def optimize(self) -> Dict[str, Union[np.ndarray, float, int, List]]:
        self.fe_counter[0] = 0
        self.history = []

        if self.custom_subproblems is not None:
            subproblems = self.custom_subproblems
        else:
            edg_solver = EDG(func=self._eval, dim=self.dim, lower=self.lower, upper=self.upper, epsilon=self.edg_epsilon)
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

        # MTES-DAKG optimizers without stagnant detectors
        optimizers = [
            MTES_DAKG(
                subtasks_vars=mtop_subs,
                eval_func=self._eval,
                lower=self.lower,
                upper=self.upper,
                alpha=self.alpha,
                tau=self.tau,
                max_gen=self.max_gen_per_cycle,
                fre_ratio=self.fre_ratio,
            )
            for mtop_subs in mtops
        ]
        contributions = np.zeros(k_mtops)
        empty_stagnant_set: Set[int] = set()

        while self.fe_counter[0] < self.max_fes:
            contributions.fill(0.0)

            # Phase 1: Sequential sweep
            for i in range(k_mtops):
                if self.fe_counter[0] >= self.max_fes:
                    break
                f_last = self.best_f
                best_sols, _, _ = optimizers[i].optimize(
                    collaborator=self.best_x,
                    stagnant_set=empty_stagnant_set,  # Never prune subtasks
                    stagnant_detectors=None,          # No stagnant detection
                    max_evals=self.max_fes,
                    eval_counter=self.fe_counter,
                )
                cand_x = self.best_x.copy()
                for j, sub_vars in enumerate(mtops[i]):
                    cand_x[sub_vars] = best_sols[j]
                cand_f = self._eval(cand_x)
                if cand_f < self.best_f:
                    self.best_f = cand_f
                    self.best_x = cand_x
                contributions[i] = abs(f_last - self.best_f)
                self.history.append((self.fe_counter[0], self.best_f))

            # Phase 2: Contribution-based greedy selection
            while (np.max(contributions) - np.min(contributions) > self.epsilon) and self.fe_counter[0] < self.max_fes:
                best_idx = int(np.argmax(contributions))
                f_last = self.best_f
                best_sols, _, _ = optimizers[best_idx].optimize(
                    collaborator=self.best_x,
                    stagnant_set=empty_stagnant_set,
                    stagnant_detectors=None,
                    max_evals=self.max_fes,
                    eval_counter=self.fe_counter,
                )
                cand_x = self.best_x.copy()
                for j, sub_vars in enumerate(mtops[best_idx]):
                    cand_x[sub_vars] = best_sols[j]
                cand_f = self._eval(cand_x)
                if cand_f < self.best_f:
                    self.best_f = cand_f
                    self.best_x = cand_x
                contributions[best_idx] = abs(f_last - self.best_f)
                self.history.append((self.fe_counter[0], self.best_f))

        return {
            "best_x": self.best_x,
            "best_f": self.best_f,
            "fes": self.fe_counter[0],
            "history": self.history,
            "subproblems": subproblems,
            "mtops": mtops,
        }
