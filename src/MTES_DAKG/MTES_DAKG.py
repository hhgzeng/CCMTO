"""
MTES-DAKG (Multitask Evolution Strategy with Dynamic Distance Threshold and Adaptive Elite Sampling KGxS)

Optimizer for Multitask Optimization Problems (MTOPs) within the CCMTO framework.
Integrates CMA-ES, DT_DoS domain knowledge transfer, and AS_SaS shape knowledge transfer.
"""

from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from .CMAES import CMAES
from .DT_DoS import dt_dos
from .AS_SaS import as_sas


class MTES_DAKG:
    def __init__(
        self,
        subtasks_vars: List[List[int]],
        eval_func: Callable[[np.ndarray], float],
        lower: Union[float, np.ndarray] = -100.0,
        upper: Union[float, np.ndarray] = 100.0,
        alpha: float = 0.5,
        tau: int = 1,
        max_gen: int = 50,
        fre_ratio: float = 0.1,
        k_knn: int = 5,
        beta: float = 1e-5,
        phi: float = 1.0,
        a: float = 2.0,
        gamma: float = 2.0,
    ):
        """
        Initialize MTES-DAKG solver for a MTOP.

        Args:
            subtasks_vars: List of variable index lists for each subtask in the MTOP
            eval_func: Global evaluation function f(x)
            lower: Lower bound(s)
            upper: Upper bound(s)
            alpha: Probability of domain KGxS (DT_DoS) vs shape KGxS (AS_SaS)
            tau: Number of external samples per transfer
            max_gen: Maximum generations per optimization call
            fre_ratio: Frequency ratio for external sampling (fre = fre_ratio * max_gen)
            k_knn: KNN count for DT_DoS
            beta: Perturbation step for DT_DoS
            phi: Gradient correction coefficient for DT_DoS
            a: Rate parameter for AS_SaS
            gamma: Weighting parameter for AS_SaS
        """
        self.subtasks_vars = subtasks_vars
        self.num_tasks = len(subtasks_vars)
        self.eval_func = eval_func

        self.lower = lower
        self.upper = upper
        self.alpha = alpha
        self.tau = tau
        self.max_gen = max_gen
        self.fre = max(1, int(np.round(fre_ratio * max_gen)))

        self.k_knn = k_knn
        self.beta = beta
        self.phi = phi
        self.a = a
        self.gamma = gamma

        # Initialize CMAES instances for each subtask
        self.optimizers: List[CMAES] = []
        for var_indices in self.subtasks_vars:
            d = len(var_indices)
            sub_lower = lower if np.isscalar(lower) else np.asarray(lower)[var_indices]
            sub_upper = upper if np.isscalar(upper) else np.asarray(upper)[var_indices]
            self.optimizers.append(CMAES(dim=d, lower=sub_lower, upper=sub_upper))

    def optimize(
        self,
        collaborator: np.ndarray,
        stagnant_set: Set[int],
        stagnant_detectors: Optional[List[object]] = None,
        max_evals: Optional[int] = None,
        eval_counter: Optional[List[int]] = None,
    ) -> Tuple[List[np.ndarray], List[float], Set[int]]:
        """
        Execute MTES-DAKG optimization on active subtasks.

        Args:
            collaborator: Full solution vector used as collaborator context
            stagnant_set: Set of indices of subtasks that are currently stagnant
            stagnant_detectors: List of StagnantDetection instances for each subtask
            max_evals: Optional budget limit for FEs
            eval_counter: Single-element list [current_fes] for global tracking

        Returns:
            best_solutions: List of best local solution vectors for each subtask
            best_fitnesses: List of best objective values for each subtask
            updated_stagnant_set: Set of stagnant subtask indices
        """
        active_tasks = [k for k in range(self.num_tasks) if k not in stagnant_set]
        if not active_tasks:
            best_sols = [opt.best_x for opt in self.optimizers]
            best_fits = [opt.best_f for opt in self.optimizers]
            return best_sols, best_fits, stagnant_set

        def evaluate_subtask(k: int, sub_x: np.ndarray) -> float:
            full_x = collaborator.copy()
            full_x[self.subtasks_vars[k]] = sub_x
            if eval_counter is not None:
                eval_counter[0] += 1
            return float(self.eval_func(full_x))

        # Synchronize optimizer mean with current global collaborator and establish baseline
        for k in range(self.num_tasks):
            opt = self.optimizers[k]
            opt.m = np.asarray(collaborator[self.subtasks_vars[k]], dtype=float).copy()
            opt.best_x = opt.m.copy()
            opt.best_f = evaluate_subtask(k, opt.m)

        for gen in range(1, self.max_gen + 1):
            if max_evals is not None and eval_counter is not None and eval_counter[0] >= max_evals:
                break

            pop_X: Dict[int, np.ndarray] = {}
            pop_F: Dict[int, np.ndarray] = {}

            # 1. Internal sampling and evaluation
            for k in active_tasks:
                if max_evals is not None and eval_counter is not None and eval_counter[0] >= max_evals:
                    break
                opt = self.optimizers[k]
                X_k, _ = opt.sample()
                F_k = np.zeros(len(X_k))
                for i in range(len(X_k)):
                    if max_evals is not None and eval_counter is not None and eval_counter[0] >= max_evals:
                        F_k[i:] = float("inf")
                        break
                    F_k[i] = evaluate_subtask(k, X_k[i])

                pop_X[k] = X_k
                pop_F[k] = F_k

            # 2. Knowledge transfer (External sampling)
            if (gen % self.fre == 0) and (len(active_tasks) > 1):
                for k in active_tasks:
                    if k not in pop_X:
                        continue
                    if max_evals is not None and eval_counter is not None and eval_counter[0] >= max_evals:
                        break

                    other_tasks = [s for s in active_tasks if s != k and s in pop_X]
                    if not other_tasks:
                        continue
                    s = int(np.random.choice(other_tasks))

                    opt_t = self.optimizers[k]
                    opt_s = self.optimizers[s]

                    for _ in range(self.tau):
                        if max_evals is not None and eval_counter is not None and eval_counter[0] >= max_evals:
                            break

                        if np.random.rand() < self.alpha:
                            # Domain KGxS (DT_DoS)
                            hat_x = dt_dos(
                                X_t=pop_X[k],
                                F_t=pop_F[k],
                                m_t=opt_t.m,
                                C_t=opt_t.C,
                                m_s=opt_s.m,
                                C_s=opt_s.C,
                                eval_target=lambda x, task_idx=k: evaluate_subtask(task_idx, x),
                                k_knn=self.k_knn,
                                beta=self.beta,
                                phi=self.phi,
                                sigma_s=opt_s.sigma,
                                C_s_sqrt=opt_s.C_sqrt,
                                lower=opt_t.lower,
                                upper=opt_t.upper,
                            )
                        else:
                            # Shape KGxS (AS_SaS)
                            hat_x = as_sas(
                                X_s=pop_X[s],
                                F_s=pop_F[s],
                                m_t=opt_t.m,
                                C_t_sqrt=opt_t.C_sqrt,
                                m_s=opt_s.m,
                                C_s_inv_sqrt=opt_s.C_inv_sqrt,
                                gen=gen,
                                max_gen=self.max_gen,
                                popsize=opt_t.lambda_,
                                a=self.a,
                                gamma=self.gamma,
                                lower=opt_t.lower,
                                upper=opt_t.upper,
                            )

                        f_hat = evaluate_subtask(k, hat_x)
                        pop_X[k] = np.vstack([pop_X[k], hat_x])
                        pop_F[k] = np.append(pop_F[k], f_hat)

            # 3. Parameter update & Stagnant detection
            for k in list(active_tasks):
                if k not in pop_X or len(pop_F[k]) == 0:
                    continue
                opt = self.optimizers[k]
                opt.update(pop_X[k], pop_F[k])

                # Check stagnant status
                if stagnant_detectors is not None and k < len(stagnant_detectors):
                    detector = stagnant_detectors[k]
                    is_stag = detector.check(
                        current_best_f=opt.best_f,
                        current_pop=pop_X[k],
                        max_gen=self.max_gen,
                    )
                    if is_stag:
                        stagnant_set.add(k)
                        active_tasks.remove(k)

        best_sols = [opt.best_x for opt in self.optimizers]
        best_fits = [opt.best_f for opt in self.optimizers]
        return best_sols, best_fits, stagnant_set
