"""
wo-DA: CCMTO-MTES-DAKG without DT-DoS and without AS-SaS (Equivalent to MTES-KG, Li et al. 2024).

Uses standard Domain KGxS (fixed mean distance threshold without gradient correction)
and standard Shape KGxS (unweighted top mu elite center without adaptive sample count).
Stagnant Subtask Detection is ENABLED.
"""

from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from src.CCMTO.CCMTO import CCMTO
from src.CCMTO.StagnantDetection import StagnantDetection
from src.CCMTO.MTOPConstruction import MTOPConstruction
from src.MTES_DAKG.CMAES import CMAES
from decomposition.edg import EDG


def standard_domain_kgxs(
    X_t: np.ndarray,
    m_t: np.ndarray,
    m_s: np.ndarray,
    sigma_s: float,
    C_s_sqrt: np.ndarray,
    lower: Union[float, np.ndarray],
    upper: Union[float, np.ndarray],
) -> np.ndarray:
    """Standard Domain KGxS (Algorithm 1, Steps 16-22) with dimension alignment."""
    dim_t = len(m_t)
    dim_s = len(m_s)

    # Mean distance of target population
    dists = np.linalg.norm(X_t - m_t, axis=1)
    d_mean = float(np.mean(dists)) if len(dists) > 0 else 1.0

    # Sample z ~ N(m_s, sigma_s^2 * C_s) in source space
    z_raw = m_s + sigma_s * (C_s_sqrt @ np.random.randn(dim_s))
    z_raw = np.nan_to_num(z_raw, nan=0.0)

    # Align dimension to target task
    if dim_s == dim_t:
        z = z_raw
    elif dim_s > dim_t:
        z = z_raw[:dim_t]
    else:
        z = m_t.copy()
        z[:dim_s] = z_raw

    dist_z = float(np.linalg.norm(z - m_t))
    if dist_z < d_mean:
        hat_x = z
    else:
        direction = (z - m_t) / (dist_z + 1e-12)
        hat_x = m_t + d_mean * direction

    return np.clip(hat_x, lower, upper)


def standard_shape_kgxs(
    X_s: np.ndarray,
    F_s: np.ndarray,
    m_t: np.ndarray,
    C_t_sqrt: np.ndarray,
    m_s: np.ndarray,
    C_s_inv_sqrt: np.ndarray,
    lower: Union[float, np.ndarray],
    upper: Union[float, np.ndarray],
) -> np.ndarray:
    """Standard Shape KGxS (Algorithm 1, Steps 23-27) with dimension alignment."""
    dim_t = len(m_t)
    dim_s = len(m_s)
    popsize = len(X_s)
    mu = max(2, popsize // 2)

    # Sort source samples by fitness
    order = np.argsort(F_s)
    elite_X = X_s[order[:mu]]

    # Randomly exclude one elite sample j
    j = np.random.randint(mu)
    remain_indices = [idx for idx in range(mu) if idx != j]
    remain_elites = elite_X[remain_indices]

    # Unweighted average displacement
    y_S = np.mean(remain_elites - m_s, axis=0)  # shape (dim_s,)

    # Transform to target space with cross-dimension handling
    inv_y_s = C_s_inv_sqrt @ y_S
    if dim_s == dim_t:
        trans_y = C_t_sqrt @ inv_y_s
    else:
        if dim_s > dim_t:
            aligned_y = inv_y_s[:dim_t]
        else:
            aligned_y = np.zeros(dim_t)
            aligned_y[:dim_s] = inv_y_s
        trans_y = C_t_sqrt @ aligned_y

    hat_x = m_t + trans_y
    return np.clip(hat_x, lower, upper)


class MTES_KG_Optimizer:
    """MTES-KG solver (without DT-DoS and without AS-SaS)."""

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
            if eval_counter is not None:
                eval_counter[0] += 1
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
                            hat_x = standard_shape_kgxs(
                                X_s=pop_X[s],
                                F_s=pop_F[s],
                                m_t=opt_t.m,
                                C_t_sqrt=opt_t.C_sqrt,
                                m_s=opt_s.m,
                                C_s_inv_sqrt=opt_s.C_inv_sqrt,
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


class WO_DA(CCMTO):
    """CCMTO with wo-DA (original MTES-KG without DT-DoS and without AS-SaS)."""

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
            MTES_KG_Optimizer(
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
