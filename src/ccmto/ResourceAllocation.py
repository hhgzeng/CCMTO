"""
Contribution-Based Resource Allocation Strategy of MTOPs and Subtasks

Algorithm 6 in CCMTO:
Dynamically allocates computational resources to MTOPs with the highest fitness contributions,
while pruning stagnant subtasks to conserve function evaluations.
"""

from typing import Callable, Dict, List, Optional, Set, Tuple
import numpy as np

from .StagnantDetection import StagnantDetection
from ..MTES_DAKG.MTES_DAKG import MTES_DAKG


class ResourceAllocation:
    def __init__(
        self,
        mtops: List[List[List[int]]],
        eval_func: Callable[[np.ndarray], float],
        dim: int,
        lower: float = -100.0,
        upper: float = 100.0,
        max_gen: int = 50,
        epsilon: float = 1e-6,
        alpha: float = 0.5,
        tau: int = 1,
        fre_ratio: float = 0.1,
    ):
        """
        Initialize Resource Allocation manager.

        Args:
            mtops: List of constructed MTOPs [T_1, ..., T_k]
            eval_func: Global objective function f(x)
            dim: Total dimension of global solution
            lower: Lower bound
            upper: Upper bound
            max_gen: Generations per MTES-DAKG optimization call
            epsilon: Stagnation threshold
            alpha: Knowledge type probability in MTES-DAKG
            tau: Number of external samples in MTES-DAKG
            fre_ratio: Frequency ratio of external sampling (fre = fre_ratio * max_gen)
        """
        self.mtops = mtops
        self.k_mtops = len(mtops)
        self.eval_func = eval_func
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.max_gen = max_gen
        self.epsilon = epsilon
        self.alpha = alpha
        self.tau = tau
        self.fre_ratio = fre_ratio

        # Instantiate MTES-DAKG optimizers and Stagnant Detectors for each MTOP
        self.optimizers: List[MTES_DAKG] = []
        self.stagnant_detectors: List[List[StagnantDetection]] = []
        self.stagnant_sets: List[Set[int]] = []
        self.contributions: np.ndarray = np.zeros(self.k_mtops)

        for mtop_subs in self.mtops:
            opt = MTES_DAKG(
                subtasks_vars=mtop_subs,
                eval_func=self.eval_func,
                lower=self.lower,
                upper=self.upper,
                alpha=self.alpha,
                tau=self.tau,
                max_gen=self.max_gen,
                fre_ratio=self.fre_ratio,
            )
            self.optimizers.append(opt)

            detectors = [
                StagnantDetection(dim=len(sub), epsilon=self.epsilon) for sub in mtop_subs
            ]
            self.stagnant_detectors.append(detectors)
            self.stagnant_sets.append(set())

    def reset_cycle(self):
        """Reset stagnant sets and detectors for a new co-evolutionary cycle (Step 3)."""
        self.contributions.fill(0.0)
        for i in range(self.k_mtops):
            self.stagnant_sets[i].clear()
            for det in self.stagnant_detectors[i]:
                det.reset()
            # Synchronize optimizer mean with best_x without destroying adapted covariance and sigma
            for opt in self.optimizers[i].optimizers:
                opt.m = opt.best_x.copy()

    def optimize_mtop(
        self,
        mtop_idx: int,
        best_x: np.ndarray,
        best_f: float,
        max_fes: Optional[int] = None,
        fe_counter: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize a single MTOP and update the global best solution and collaborator.

        Returns:
            updated_best_x: Global solution vector with MTOP components updated
            updated_best_f: Global best fitness
        """
        f_last = best_f
        opt = self.optimizers[mtop_idx]
        detectors = self.stagnant_detectors[mtop_idx]
        stag_set = self.stagnant_sets[mtop_idx]
        subtasks = self.mtops[mtop_idx]

        # Optimize active subtasks
        best_sols, best_fits, stag_set = opt.optimize(
            collaborator=best_x,
            stagnant_set=stag_set,
            stagnant_detectors=detectors,
            max_evals=max_fes,
            eval_counter=fe_counter,
        )
        self.stagnant_sets[mtop_idx] = stag_set

        # Update candidate global solution
        cand_x = best_x.copy()
        for j, sub_vars in enumerate(subtasks):
            cand_x[sub_vars] = best_sols[j]

        cand_f = float(self.eval_func(cand_x))

        if cand_f < best_f:
            best_x = cand_x
            best_f = cand_f

        # Update contribution (Eq. 19)
        delta_F = abs(f_last - best_f)
        if len(stag_set) == len(subtasks):
            # All subtasks stagnant
            delta_F = 0.0

        self.contributions[mtop_idx] = delta_F

        return best_x, best_f
