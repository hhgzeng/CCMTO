"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

Single-task optimizer core for MTES-DAKG.
Supports sampling, external solution injection, and distribution parameter updates.
Includes numerical safeguards (exponent clipping, covariance matrix conditioning, sigma clamping).
"""

from typing import Optional, Tuple
import numpy as np


class CMAES:
    def __init__(
        self,
        dim: int,
        lower: float = 0.0,
        upper: float = 1.0,
        mean: Optional[np.ndarray] = None,
        sigma: Optional[float] = None,
        popsize: Optional[int] = None,
    ):
        """
        Initialize CMA-ES instance.

        Args:
            dim: Dimension of decision variables
            lower: Lower bound in search space
            upper: Upper bound in search space
            mean: Initial mean vector (default: middle of bounds or random)
            sigma: Initial step size (default: 0.3 * (upper - lower))
            popsize: Number of samples per generation (lambda)
        """
        self.dim = dim
        self.lower = np.full(dim, lower) if np.isscalar(lower) else np.asarray(lower, dtype=float)
        self.upper = np.full(dim, upper) if np.isscalar(upper) else np.asarray(upper, dtype=float)

        # Initial mean
        if mean is not None:
            self.m = np.asarray(mean, dtype=float).copy()
        else:
            self.m = self.lower + 0.5 * (self.upper - self.lower)

        # Initial sigma
        if sigma is not None:
            self.sigma = float(sigma)
        else:
            self.sigma = float(np.mean(0.3 * (self.upper - self.lower)))

        # Population size lambda
        if popsize is not None:
            self.lambda_ = int(popsize)
        else:
            self.lambda_ = int(4 + np.floor(3 * np.log(max(dim, 1))))

        # Number of parents mu
        self.mu = max(1, self.lambda_ // 2)

        # Recombination weights
        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_weights / np.sum(raw_weights)
        self.mueff = 1.0 / np.sum(self.weights ** 2)

        # Strategy parameter adaptation
        self.cc = (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
        self.cs = (self.mueff + 2.0) / (self.dim + self.mueff + 5.0)
        self.c1 = 2.0 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1.0 - self.c1,
            2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((self.dim + 2.0) ** 2 + self.mueff),
        )
        self.damps = 1.0 + 2.0 * max(0.0, np.sqrt((self.mueff - 1.0) / (self.dim + 1.0)) - 1.0) + self.cs
        self.chiN = np.sqrt(self.dim) * (1.0 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * (self.dim ** 2)))

        # Dynamic strategy parameters
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.C = np.eye(self.dim)
        self.invsqrtC = np.eye(self.dim)

        self.gen = 0
        self.best_x = self.m.copy()
        self.best_f = float("inf")

    def _decompose_C(self):
        """Update eigendecomposition of covariance matrix C with strict numerical conditioning."""
        self.C = np.triu(self.C) + np.triu(self.C, 1).T  # Enforce symmetry
        if np.any(np.isnan(self.C)) or np.any(np.isinf(self.C)):
            self.C = np.eye(self.dim)

        eigenvalues, eigenvectors = np.linalg.eigh(self.C)
        eigenvalues = np.nan_to_num(eigenvalues, nan=1.0)
        eigenvalues = np.clip(eigenvalues, 1e-14, 1e14)
        self.D = np.sqrt(eigenvalues)
        self.B = eigenvectors
        self.invsqrtC = self.B @ np.diag(1.0 / self.D) @ self.B.T
        self.C = self.B @ np.diag(self.D ** 2) @ self.B.T

    @property
    def C_sqrt(self) -> np.ndarray:
        """Return C^(1/2)."""
        return self.B @ np.diag(self.D) @ self.B.T

    @property
    def C_inv_sqrt(self) -> np.ndarray:
        """Return C^(-1/2)."""
        return self.invsqrtC

    def sample(self, count: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample candidate solutions from N(m, sigma^2 * C).

        Returns:
            X: Array of shape (count, dim), clamped to bounds
            Z: Normalized standard normal samples z ~ N(0, I)
        """
        k = count if count is not None else self.lambda_
        z = np.random.randn(k, self.dim)
        # y = B * D * z
        y = np.dot(z, np.diag(self.D)) @ self.B.T
        x = self.m + self.sigma * y

        # Clamping
        x_clamped = np.clip(x, self.lower, self.upper)
        return x_clamped, y

    def update(self, pop: np.ndarray, fitness: np.ndarray):
        """
        Update distribution parameters based on evaluated population.

        Args:
            pop: Evaluated candidate population (shape: [K, dim])
            fitness: Array of fitness/objective values (shape: [K]), lower is better
        """
        self.gen += 1
        pop_size = len(fitness)

        # Sort solutions by fitness ascending (minimization)
        order = np.argsort(fitness)
        if fitness[order[0]] < self.best_f:
            self.best_f = fitness[order[0]]
            self.best_x = pop[order[0]].copy()

        # Selection: top mu
        top_indices = order[: self.mu]
        selected_pop = pop[top_indices]

        # Compute displacements y_i = (x_i - m) / sigma
        y = (selected_pop - self.m) / max(self.sigma, 1e-14)
        y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        y_w = np.sum(self.weights[:, np.newaxis] * y, axis=0)

        # Update mean
        old_m = self.m.copy()
        self.m = old_m + self.sigma * y_w
        if np.any(np.isnan(self.m)) or np.any(np.isinf(self.m)):
            self.m = self.best_x.copy()
        self.m = np.clip(self.m, self.lower, self.upper)

        # Update evolution paths
        # C^(-1/2) * y_w = invsqrtC * y_w
        inv_y_w = self.invsqrtC @ y_w
        inv_y_w = np.nan_to_num(inv_y_w, nan=0.0)
        self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * inv_y_w

        # Heuristic for h_sig
        norm_ps = np.linalg.norm(self.ps)
        denominator = np.sqrt(1.0 - (1.0 - self.cs) ** (2 * self.gen))
        hsig_thresh = (1.4 + 2.0 / (self.dim + 1.0)) * self.chiN
        hsig = 1.0 if (norm_ps / max(denominator, 1e-14)) < hsig_thresh else 0.0

        self.pc = (1.0 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2.0 - self.cc) * self.mueff) * y_w

        # Update covariance matrix C
        # Rank-1 update
        rank1 = self.c1 * (
            np.outer(self.pc, self.pc) + (1.0 - hsig) * self.cc * (2.0 - self.cc) * self.C
        )
        # Rank-mu update (vectorized)
        weighted_y = y[:self.mu] * self.weights[:, np.newaxis]
        rank_mu = self.cmu * (weighted_y.T @ y[:self.mu])

        self.C = (1.0 - self.c1 - self.cmu) * self.C + rank1 + rank_mu

        # Update step size sigma with exponent bounding
        exponent = (self.cs / self.damps) * (norm_ps / self.chiN - 1.0)
        exponent = float(np.clip(exponent, -20.0, 2.0))
        self.sigma = self.sigma * np.exp(exponent)

        span = float(np.max(self.upper - self.lower)) if np.all(np.isfinite(self.upper - self.lower)) else 1000.0
        self.sigma = float(np.clip(self.sigma, 1e-20, span))

        # Lazy eigendecomposition for high-dimensional efficiency (Hansen standard)
        if self.dim <= 20:
            lazy_gap = 1
        else:
            lazy_gap = max(1, min(20, int(self.dim / 20)))

        if self.gen % lazy_gap == 0 or self.gen == 1:
            self._decompose_C()
