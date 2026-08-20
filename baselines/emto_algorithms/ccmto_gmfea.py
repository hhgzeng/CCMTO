"""
CCMTO-G-MFEA: Generalized Multifactorial Evolutionary Algorithm (Ding et al. 2019) inside CCMTO.

Replaces MTES-DAKG with G-MFEA as the EMTO optimizer for solving constructed MTOPs.
Uses unified continuous search space, assortative mating with random mating probability (rmp),
and vertical cultural transmission.
"""

from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from src.CCMTO.CCMTO import CCMTO
from src.CCMTO.StagnantDetection import StagnantDetection
from src.CCMTO.MTOPConstruction import MTOPConstruction
from decomposition.edg import EDG


class GMFEA_Optimizer:
    """Generalized Multifactorial Evolutionary Algorithm for a single MTOP."""

    def __init__(
        self,
        subtasks_vars: List[List[int]],
        eval_func: Callable[[np.ndarray], float],
        lower: Union[float, np.ndarray] = -100.0,
        upper: Union[float, np.ndarray] = 100.0,
        pop_per_task: int = 25,
        max_gen: int = 50,
        rmp: float = 0.3,
        mu_c: float = 2.0,
        mu_m: float = 5.0,
    ):
        self.subtasks_vars = subtasks_vars
        self.num_tasks = len(subtasks_vars)
        self.eval_func = eval_func
        self.lower = lower
        self.upper = upper
        self.pop_per_task = pop_per_task
        self.total_pop_size = pop_per_task * self.num_tasks
        self.max_gen = max_gen
        self.rmp = rmp
        self.mu_c = mu_c
        self.mu_m = mu_m

        self.dims = [len(v) for v in subtasks_vars]
        self.max_dim = max(self.dims)

        # Unified search space [0, 1]^max_dim
        self.pop = np.random.rand(self.total_pop_size, self.max_dim)
        self.skill_factors = np.repeat(np.arange(self.num_tasks), self.pop_per_task)
        self.factorial_costs = np.full(self.total_pop_size, float("inf"))

        self.best_x: List[np.ndarray] = [np.zeros(d) for d in self.dims]
        self.best_f: List[float] = [float("inf") for _ in range(self.num_tasks)]

    def _decode(self, u_vec: np.ndarray, task_idx: int) -> np.ndarray:
        """Decode unified [0, 1] representation to real domain."""
        sub_vars = self.subtasks_vars[task_idx]
        sub_low = self.lower if np.isscalar(self.lower) else np.asarray(self.lower)[sub_vars]
        sub_high = self.upper if np.isscalar(self.upper) else np.asarray(self.upper)[sub_vars]
        d = self.dims[task_idx]
        return sub_low + u_vec[:d] * (sub_high - sub_low)

    def _encode(self, real_vec: np.ndarray, task_idx: int) -> np.ndarray:
        """Encode real vector into unified [0, 1] representation."""
        sub_vars = self.subtasks_vars[task_idx]
        sub_low = self.lower if np.isscalar(self.lower) else np.asarray(self.lower)[sub_vars]
        sub_high = self.upper if np.isscalar(self.upper) else np.asarray(self.upper)[sub_vars]
        d = self.dims[task_idx]
        u = np.random.rand(self.max_dim)
        u[:d] = np.clip((real_vec - sub_low) / (sub_high - sub_low + 1e-12), 0.0, 1.0)
        return u

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
            return self.best_x, self.best_f, stagnant_set

        def eval_candidate(u_vec: np.ndarray, task_idx: int) -> float:
            real_sub = self._decode(u_vec, task_idx)
            full = collaborator.copy()
            full[self.subtasks_vars[task_idx]] = real_sub
            return float(self.eval_func(full))

        # Re-center populations around collaborator
        for idx in range(self.total_pop_size):
            k = self.skill_factors[idx]
            if k in active_tasks:
                sub_vars = self.subtasks_vars[k]
                base_u = self._encode(collaborator[sub_vars], k)
                if idx % self.pop_per_task == 0:
                    self.pop[idx] = base_u
                else:
                    self.pop[idx] = np.clip(base_u + np.random.normal(0, 0.1, self.max_dim), 0.0, 1.0)

                f = eval_candidate(self.pop[idx], k)
                self.factorial_costs[idx] = f
                if f < self.best_f[k]:
                    self.best_f[k] = f
                    self.best_x[k] = self._decode(self.pop[idx], k)

        for gen in range(1, self.max_gen + 1):
            if max_evals is not None and eval_counter is not None and eval_counter[0] >= max_evals:
                break

            offspring_list = []
            offspring_skills = []

            # Assortative mating and offspring generation
            for _ in range(self.total_pop_size // 2):
                if max_evals is not None and eval_counter is not None and eval_counter[0] >= max_evals:
                    break

                p1_idx = np.random.randint(self.total_pop_size)
                p2_idx = np.random.randint(self.total_pop_size)
                p1 = self.pop[p1_idx]
                p2 = self.pop[p2_idx]
                sf1 = self.skill_factors[p1_idx]
                sf2 = self.skill_factors[p2_idx]

                if sf1 == sf2 or np.random.rand() < self.rmp:
                    # SBX Crossover
                    u = np.random.rand(self.max_dim)
                    beta = np.where(
                        u <= 0.5,
                        (2 * u) ** (1.0 / (self.mu_c + 1)),
                        (1.0 / (2 * (1 - u))) ** (1.0 / (self.mu_c + 1)),
                    )
                    c1 = np.clip(0.5 * ((1 + beta) * p1 + (1 - beta) * p2), 0.0, 1.0)
                    c2 = np.clip(0.5 * ((1 - beta) * p1 + (1 + beta) * p2), 0.0, 1.0)

                    # Vertical cultural transmission
                    skill1 = sf1 if np.random.rand() < 0.5 else sf2
                    skill2 = sf1 if np.random.rand() < 0.5 else sf2
                else:
                    # Polynomial Mutation
                    delta = np.random.normal(0, 0.1, (2, self.max_dim))
                    c1 = np.clip(p1 + delta[0], 0.0, 1.0)
                    c2 = np.clip(p2 + delta[1], 0.0, 1.0)
                    skill1 = sf1
                    skill2 = sf2

                if skill1 in active_tasks:
                    offspring_list.append(c1)
                    offspring_skills.append(skill1)
                if skill2 in active_tasks:
                    offspring_list.append(c2)
                    offspring_skills.append(skill2)

            if not offspring_list:
                break

            offspring_pop = np.array(offspring_list)
            offspring_skills = np.array(offspring_skills)
            offspring_costs = np.zeros(len(offspring_pop))

            for o_idx in range(len(offspring_pop)):
                if max_evals is not None and eval_counter is not None and eval_counter[0] >= max_evals:
                    offspring_costs[o_idx:] = float("inf")
                    break
                k = offspring_skills[o_idx]
                f = eval_candidate(offspring_pop[o_idx], k)
                offspring_costs[o_idx] = f
                if f < self.best_f[k]:
                    self.best_f[k] = f
                    self.best_x[k] = self._decode(offspring_pop[o_idx], k)

            # Task-specific selection to maintain balanced population
            for k in list(active_tasks):
                # Gather parent and offspring candidates for task k
                p_mask = self.skill_factors == k
                o_mask = offspring_skills == k

                k_pop = np.vstack([self.pop[p_mask], offspring_pop[o_mask]])
                k_costs = np.concatenate([self.factorial_costs[p_mask], offspring_costs[o_mask]])

                survivor_order = np.argsort(k_costs)[: self.pop_per_task]
                selected_pop = k_pop[survivor_order]
                selected_costs = k_costs[survivor_order]

                # Put back into main population
                p_indices = np.where(p_mask)[0]
                n_put = min(len(p_indices), len(selected_pop))
                self.pop[p_indices[:n_put]] = selected_pop[:n_put]
                self.factorial_costs[p_indices[:n_put]] = selected_costs[:n_put]

                # Stagnant check
                if stagnant_detectors is not None and k < len(stagnant_detectors):
                    det = stagnant_detectors[k]
                    real_pop_k = np.array([self._decode(u, k) for u in selected_pop])
                    if det.check(current_best_f=self.best_f[k], current_pop=real_pop_k, max_gen=self.max_gen):
                        stagnant_set.add(k)
                        active_tasks.remove(k)

        return self.best_x, self.best_f, stagnant_set


class CCMTO_GMFEA(CCMTO):
    """CCMTO with G-MFEA (Generalized MFEA) as EMTO solver."""

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

        gmfea_opts = [
            GMFEA_Optimizer(
                subtasks_vars=mtop_subs,
                eval_func=self._eval,
                lower=self.lower,
                upper=self.upper,
                max_gen=self.max_gen_per_cycle,
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
                best_sols, _, stagnant_sets[i] = gmfea_opts[i].optimize(
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
                best_sols, _, stagnant_sets[best_idx] = gmfea_opts[best_idx].optimize(
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
