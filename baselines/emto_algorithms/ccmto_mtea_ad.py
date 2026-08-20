"""
CCMTO-MTEA-AD: Multitask Evolutionary Algorithm with Anomaly Detection (Wang et al. 2022) inside CCMTO.

Replaces MTES-DAKG with MTEA-AD as the EMTO optimizer.
Filters negative knowledge transfer by evaluating an anomaly detection metric (e.g. Mahalanobis distance)
before admitting transferred solutions into the target subtask population.
"""

from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from src.CCMTO.CCMTO import CCMTO
from src.CCMTO.StagnantDetection import StagnantDetection
from src.CCMTO.MTOPConstruction import MTOPConstruction
from decomposition.edg import EDG


class MTEA_AD_Optimizer:
    """Multitask Evolutionary Algorithm with Anomaly Detection for an MTOP."""

    def __init__(
        self,
        subtasks_vars: List[List[int]],
        eval_func: Callable[[np.ndarray], float],
        lower: Union[float, np.ndarray] = -100.0,
        upper: Union[float, np.ndarray] = 100.0,
        pop_size: int = 30,
        max_gen: int = 50,
        cr_prob: float = 0.9,
        f_weight: float = 0.5,
        anomaly_threshold: float = 2.5,
    ):
        self.subtasks_vars = subtasks_vars
        self.num_tasks = len(subtasks_vars)
        self.eval_func = eval_func
        self.lower = lower
        self.upper = upper
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.CR = cr_prob
        self.F = f_weight
        self.anomaly_threshold = anomaly_threshold

        self.dims = [len(v) for v in subtasks_vars]
        self.populations: List[np.ndarray] = []
        self.fitnesses: List[np.ndarray] = []
        self.best_x: List[np.ndarray] = []
        self.best_f: List[float] = []

        for k, d in enumerate(self.dims):
            sub_low = lower if np.isscalar(lower) else np.asarray(lower)[self.subtasks_vars[k]]
            sub_high = upper if np.isscalar(upper) else np.asarray(upper)[self.subtasks_vars[k]]
            pop = np.random.uniform(sub_low, sub_high, (self.pop_size, d))
            self.populations.append(pop)
            self.fitnesses.append(np.full(self.pop_size, float("inf")))
            self.best_x.append(pop[0].copy())
            self.best_f.append(float("inf"))

    def _is_normal(self, sample: np.ndarray, target_pop: np.ndarray) -> bool:
        """Check if candidate transfer sample is non-anomalous relative to target task."""
        mean = np.mean(target_pop, axis=0)
        std = np.std(target_pop, axis=0) + 1e-8
        z_scores = np.abs((sample - mean) / std)
        max_z = np.max(z_scores)
        return bool(max_z <= self.anomaly_threshold)

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

        def eval_sub(k: int, x_sub: np.ndarray) -> float:
            full = collaborator.copy()
            full[self.subtasks_vars[k]] = x_sub
            return float(self.eval_func(full))

        for k in active_tasks:
            sub_vars = self.subtasks_vars[k]
            sub_low = self.lower if np.isscalar(self.lower) else np.asarray(self.lower)[sub_vars]
            sub_high = self.upper if np.isscalar(self.upper) else np.asarray(self.upper)[sub_vars]

            self.populations[k][0] = collaborator[sub_vars].copy()
            f0 = eval_sub(k, self.populations[k][0])
            self.fitnesses[k][0] = f0
            self.best_x[k] = self.populations[k][0].copy()
            self.best_f[k] = f0

            for i in range(1, self.pop_size):
                if max_evals is not None and eval_counter is not None and eval_counter[0] >= max_evals:
                    break
                self.populations[k][i] = np.clip(
                    collaborator[sub_vars] + np.random.normal(0, 0.1 * (sub_high - sub_low), len(sub_vars)),
                    sub_low,
                    sub_high,
                )
                fi = eval_sub(k, self.populations[k][i])
                self.fitnesses[k][i] = fi
                if fi < self.best_f[k]:
                    self.best_f[k] = fi
                    self.best_x[k] = self.populations[k][i].copy()

        for gen in range(1, self.max_gen + 1):
            if max_evals is not None and eval_counter is not None and eval_counter[0] >= max_evals:
                break

            for k in list(active_tasks):
                if max_evals is not None and eval_counter is not None and eval_counter[0] >= max_evals:
                    break

                pop = self.populations[k]
                fits = self.fitnesses[k]
                d = self.dims[k]
                sub_low = self.lower if np.isscalar(self.lower) else np.asarray(self.lower)[self.subtasks_vars[k]]
                sub_high = self.upper if np.isscalar(self.upper) else np.asarray(self.upper)[self.subtasks_vars[k]]

                for i in range(self.pop_size):
                    if max_evals is not None and eval_counter is not None and eval_counter[0] >= max_evals:
                        break

                    # Check for transfer candidate
                    transfer_accepted = False
                    if len(active_tasks) > 1 and np.random.rand() < 0.4:
                        other_k = int(np.random.choice([s for s in active_tasks if s != k]))
                        donor_best = self.best_x[other_k]
                        donor_proj = np.zeros(d)
                        copy_len = min(d, len(donor_best))
                        donor_proj[:copy_len] = donor_best[:copy_len]
                        if d > copy_len:
                            donor_proj[copy_len:] = pop[i, copy_len:]

                        # Anomaly Detection filter
                        if self._is_normal(donor_proj, pop):
                            # Transfer accepted
                            cross_points = np.random.rand(d) < self.CR
                            cross_points[np.random.randint(d)] = True
                            trial = np.where(cross_points, donor_proj, pop[i])
                            transfer_accepted = True

                    if not transfer_accepted:
                        # DE/rand/1
                        idxs = [idx for idx in range(self.pop_size) if idx != i]
                        r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                        mutant = pop[r1] + self.F * (pop[r2] - pop[r3])
                        cross_points = np.random.rand(d) < self.CR
                        cross_points[np.random.randint(d)] = True
                        trial = np.where(cross_points, mutant, pop[i])

                    trial = np.clip(trial, sub_low, sub_high)
                    f_trial = eval_sub(k, trial)

                    if f_trial <= fits[i]:
                        pop[i] = trial
                        fits[i] = f_trial
                        if f_trial < self.best_f[k]:
                            self.best_f[k] = f_trial
                            self.best_x[k] = trial.copy()

                # Stagnant detection
                if stagnant_detectors is not None and k < len(stagnant_detectors):
                    det = stagnant_detectors[k]
                    if det.check(current_best_f=self.best_f[k], current_pop=pop, max_gen=self.max_gen):
                        stagnant_set.add(k)
                        active_tasks.remove(k)

        return self.best_x, self.best_f, stagnant_set


class CCMTO_MTEA_AD(CCMTO):
    """CCMTO with MTEA-AD (Anomaly Detection MTEA) as EMTO solver."""

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

        ad_opts = [
            MTEA_AD_Optimizer(
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
                best_sols, _, stagnant_sets[i] = ad_opts[i].optimize(
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
                best_sols, _, stagnant_sets[best_idx] = ad_opts[best_idx].optimize(
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
