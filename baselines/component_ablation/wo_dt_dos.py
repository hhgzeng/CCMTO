"""
wo-DT-DoS: CCMTO-MTES-DAKG without DT-DoS (uses AS-SaS only).

Uses standard Domain KGxS (fixed mean distance, no gradient correction)
along with proposed Adaptive Elite Sampling Shape KGxS (AS-SaS).
Stagnant Subtask Detection is ENABLED.
"""

from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from src.CCMTO.CCMTO import CCMTO
from src.CCMTO.StagnantDetection import StagnantDetection
from src.CCMTO.MTOPConstruction import MTOPConstruction
from src.MTES_DAKG.CMAES import CMAES
from src.MTES_DAKG.AS_SaS import as_sas
from decomposition.edg import EDG
from .wo_da import standard_domain_kgxs


class MTES_wo_DT_DoS_Optimizer:
    """MTES optimizer without DT-DoS (standard domain KGxS + AS-SaS)."""

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
        a: float = 2.0,
        gamma: float = 2.0,
    ):
        self.subtasks_vars = subtasks_vars
        self.num_tasks = len(subtasks_vars)
        self.eval_func = eval_func
        self.lower = lower
        self.upper = upper
        self.alpha = alpha
        self.tau = tau
        self.max_gen = max_gen
        self.fre = max(1, int(np.round(fre_ratio * max_gen)))
        self.a = a
        self.gamma = gamma

        self.optimizers: List[CMAES] = []
        for var_indices in self.subtasks_vars:
            d = len(var_indices)
            sub_low = lower if np.isscalar(lower) else np.asarray(lower)[var_indices]
            sub_high = upper if np.isscalar(upper) else np.asarray(upper)[var_indices]
            self.optimizers.append(CMAES(dim=d, lower=sub_low, upper=sub_high))

    def optimize(
        self,
        collaborator: np.ndarray,
        stagnant_set: Set[int],
        stagnant_detectors: Optional[List[StagnantDetection]] = None,
        max_evals: Optional[int] = None,
        eval_counter: Optional[List[int]] = None,
    ) -> Tuple[List[np.ndarray], List[float], Set[int]]:
        active_tasks = [k for k in range(self.num_tasks) if k not in stagnant_set]
        if not active_tasks:
            best_sols = [opt.best_x for opt in self.optimizers]
            best_fits = [opt.best_f for opt in self.optimizers]
            return best_sols, best_fits, stagnant_set

        def evaluate_subtask(k: int, sub_x: np.ndarray) -> float:
            full_x = collaborator.copy()
            full_x[self.subtasks_vars[k]] = sub_x
            return float(self.eval_func(full_x))

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
                            # Standard Domain KGxS
                            hat_x = standard_domain_kgxs(
                                X_t=pop_X[k],
                                m_t=opt_t.m,
                                m_s=opt_s.m,
                                sigma_s=opt_s.sigma,
                                C_s_sqrt=opt_s.C_sqrt,
                                lower=opt_t.lower,
                                upper=opt_t.upper,
                            )
                        else:
                            # Proposed AS-SaS
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

            for k in list(active_tasks):
                if k not in pop_X or len(pop_F[k]) == 0:
                    continue
                opt = self.optimizers[k]
                opt.update(pop_X[k], pop_F[k])

                if stagnant_detectors is not None and k < len(stagnant_detectors):
                    det = stagnant_detectors[k]
                    if det.check(current_best_f=opt.best_f, current_pop=pop_X[k], max_gen=self.max_gen):
                        stagnant_set.add(k)
                        active_tasks.remove(k)

        best_sols = [opt.best_x for opt in self.optimizers]
        best_fits = [opt.best_f for opt in self.optimizers]
        return best_sols, best_fits, stagnant_set


class WO_DT_DoS(CCMTO):
    """CCMTO with wo-DT-DoS (without DT-DoS, with AS-SaS only)."""

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

        opts = [
            MTES_wo_DT_DoS_Optimizer(
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
        stagnant_detectors = [
            [StagnantDetection(dim=len(sub), epsilon=self.epsilon) for sub in mtop_subs]
            for mtop_subs in mtops
        ]
        stagnant_sets = [set() for _ in range(k_mtops)]
        contributions = np.zeros(k_mtops)

        while self.fe_counter[0] < self.max_fes:
            contributions.fill(0.0)
            for s_set in stagnant_sets:
                s_set.clear()
            for dets in stagnant_detectors:
                for d in dets:
                    d.reset()

            for i in range(k_mtops):
                if self.fe_counter[0] >= self.max_fes:
                    break
                f_last = self.best_f
                best_sols, _, stagnant_sets[i] = opts[i].optimize(
                    collaborator=self.best_x,
                    stagnant_set=stagnant_sets[i],
                    stagnant_detectors=stagnant_detectors[i],
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

            while (np.max(contributions) - np.min(contributions) > self.epsilon) and self.fe_counter[0] < self.max_fes:
                best_idx = int(np.argmax(contributions))
                f_last = self.best_f
                best_sols, _, stagnant_sets[best_idx] = opts[best_idx].optimize(
                    collaborator=self.best_x,
                    stagnant_set=stagnant_sets[best_idx],
                    stagnant_detectors=stagnant_detectors[best_idx],
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
