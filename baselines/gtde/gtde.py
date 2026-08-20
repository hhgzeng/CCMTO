"""
GTDE: Gene Targeting Differential Evolution for Large-Scale Global Optimization.

Reference:
Z. J. Wang, J. R. Jian, Z. H. Zhan, Y. Li, S. Kwong, and J. Zhang,
"Gene Targeting Differential Evolution: A Simple and Efficient Method for Large-Scale Optimization,"
IEEE Transactions on Evolutionary Computation, vol. 27, no. 4, pp. 964-979, Aug. 2023.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class GTDE:
    """
    Gene Targeting Differential Evolution (GTDE).

    A non-decomposition large-scale optimization algorithm that couples standard
    population differential evolution with a specialized Gene Targeting (GT) operator
    applied to the global best individual (gbest) to break bottleneck dimensions.
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
        pbest_rate: float = 0.1,
        target_group_size: Optional[int] = None,
        learning_rate: float = 0.1,
        verbose: bool = False,
        log_interval: int = 10000,
    ):
        """
        Args:
            func: Objective function to minimize f(x)
            dim: Dimension of design variables
            lower: Lower bound(s)
            upper: Upper bound(s)
            max_fes: Maximum function evaluations
            pop_size: Population size (NP)
            f_weight: Mutation scaling factor F
            cr_prob: Crossover rate CR
            pbest_rate: Top p-best ratio for current-to-pbest mutation
            target_group_size: Number of bottleneck dimensions targeted in each GT step
            learning_rate: Adaptation rate for gene targeting probabilities
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
        self.pbest_rate = pbest_rate

        # If not specified, default to min(50, 0.1 * dim)
        if target_group_size is None:
            self.target_group_size = max(1, min(50, int(0.1 * dim)))
        else:
            self.target_group_size = max(1, min(target_group_size, dim))

        self.learning_rate = learning_rate
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
        """Run GTDE optimization."""
        self.fes = 0
        self.history = []

        # 1. Initialize population
        pop = np.random.uniform(self.lower, self.upper, (self.pop_size, self.dim))
        fits = np.full(self.pop_size, float("inf"))

        for i in range(self.pop_size):
            if self.fes >= self.max_fes:
                break
            fits[i] = self._eval(pop[i])

        self.history.append((self.fes, self.best_f))

        # 2. Gene targeting statistics and probability distribution
        # Initialize uniform targeting probability across all dimensions
        gene_probs = np.ones(self.dim) / self.dim
        dim_stagnation = np.zeros(self.dim)
        gene_success = np.zeros(self.dim)
        gene_trials = np.zeros(self.dim)

        pbest_num = max(2, int(np.ceil(self.pop_size * self.pbest_rate)))
        last_log = self.fes
        gen = 0

        # 3. Main Evolution Loop
        while self.fes < self.max_fes:
            gen += 1

            # Sort population by fitness
            sorted_indices = np.argsort(fits)
            pbest_indices = sorted_indices[:pbest_num]
            gbest_idx = sorted_indices[0]

            # Track variable movement for stagnation detection
            dim_delta = np.zeros(self.dim)

            # Phase A: Standard Population Differential Evolution (Full Dimensional Search)
            for i in range(self.pop_size):
                if self.fes >= self.max_fes:
                    break

                # Pick pbest and distinct random r1, r2 != i
                pbest_idx = np.random.choice(pbest_indices)
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2 = np.random.choice(idxs, 2, replace=False)

                x_i = pop[i]
                x_pbest = pop[pbest_idx]
                x_r1 = pop[r1]
                x_r2 = pop[r2]

                # DE/current-to-pbest/1 mutation across all dimensions
                mutant = (
                    x_i
                    + self.f_weight * (x_pbest - x_i)
                    + self.f_weight * (x_r1 - x_r2)
                )
                mutant = np.clip(mutant, self.lower, self.upper)

                # Binomial Crossover across all dimensions
                trial = x_i.copy()
                cross_mask = np.random.rand(self.dim) < self.cr_prob
                if not np.any(cross_mask):
                    cross_mask[np.random.randint(0, self.dim)] = True
                trial[cross_mask] = mutant[cross_mask]

                # Evaluate trial vector
                trial_f = self._eval(trial)

                # Selection
                if trial_f <= fits[i]:
                    dim_delta += np.abs(trial - pop[i])
                    pop[i] = trial
                    fits[i] = trial_f

            # Update dimension stagnation count based on population movement
            low_movement_mask = dim_delta < 1e-6 * (self.upper - self.lower)
            dim_stagnation[low_movement_mask] += 1.0
            dim_stagnation[~low_movement_mask] = np.maximum(0.0, dim_stagnation[~low_movement_mask] - 0.5)

            # Phase B: Gene Targeting (GT) Operator on Global Best Individual (gbest)
            if self.fes < self.max_fes and self.best_x is not None:
                # 1. Target bottleneck dimensions probabilistically
                # Blend stagnation score and learned gene probabilities
                targeting_score = gene_probs * (1.0 + dim_stagnation / (gen + 1.0))
                norm_score = targeting_score / np.sum(targeting_score)

                if np.random.rand() < 0.8:
                    targeted_dims = np.random.choice(
                        self.dim,
                        size=self.target_group_size,
                        replace=False,
                        p=norm_score,
                    )
                else:
                    targeted_dims = np.random.choice(
                        self.dim,
                        size=self.target_group_size,
                        replace=False,
                    )

                gene_trials[targeted_dims] += 1

                # 2. Construct homologous targeting vector
                # Pick two distinct individuals from population
                r_cand = np.random.choice(self.pop_size, 2, replace=False)
                xr1, xr2 = pop[r_cand[0]], pop[r_cand[1]]

                # 3. Gene Targeting Insertion into gbest
                gt_trial = self.best_x.copy()
                gt_mutant = (
                    self.best_x[targeted_dims]
                    + self.f_weight * (xr1[targeted_dims] - xr2[targeted_dims])
                )
                gt_mutant = np.clip(
                    gt_mutant,
                    self.lower[targeted_dims],
                    self.upper[targeted_dims],
                )

                gt_trial[targeted_dims] = gt_mutant

                # 4. Evaluate GT candidate solution
                gt_f = self._eval(gt_trial)

                if gt_f < self.best_f:
                    # Success: breakthrough in bottleneck dimensions
                    gene_success[targeted_dims] += (self.best_f - gt_f)
                    self.best_f = gt_f
                    self.best_x = gt_trial.copy()
                    # Also replace gbest in population
                    pop[gbest_idx] = gt_trial.copy()
                    fits[gbest_idx] = gt_f

            # Phase C: Periodically adapt gene targeting probabilities
            if gen % 10 == 0:
                valid_mask = gene_trials > 0
                success_rates = np.zeros(self.dim)
                if np.any(valid_mask):
                    success_rates[valid_mask] = gene_success[valid_mask] / (gene_trials[valid_mask] + 1e-12)

                if np.sum(success_rates) > 0:
                    normalized_sr = success_rates / np.sum(success_rates)
                    gene_probs = (1.0 - self.learning_rate) * gene_probs + self.learning_rate * normalized_sr
                else:
                    # Stagnation-based fallback
                    if np.sum(dim_stagnation) > 0:
                        norm_stag = dim_stagnation / np.sum(dim_stagnation)
                        gene_probs = (1.0 - self.learning_rate) * gene_probs + self.learning_rate * norm_stag

                gene_probs = np.maximum(gene_probs, 1e-4 / self.dim)
                gene_probs = gene_probs / np.sum(gene_probs)

                # Decay historical success and trial buffers
                gene_success *= 0.5
                gene_trials *= 0.5

            self.history.append((self.fes, self.best_f))

            if self.verbose and (self.fes - last_log >= self.log_interval):
                print(
                    f"[GTDE Gen {gen}] FEs: {self.fes:,}/{self.max_fes:,} | Best Fitness: {self.best_f:.6e}"
                )
                last_log = self.fes

        return {
            "best_x": self.best_x,
            "best_f": self.best_f,
            "fes": self.fes,
            "history": self.history,
        }
