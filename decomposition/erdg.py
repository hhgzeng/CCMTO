"""
Efficient Recursive Differential Grouping (ERDG)

Reference:
M. Yang, A. Zhou, C. Li, and X. Yao,
"An Efficient Recursive Differential Grouping for Large-Scale Continuous Problems,"
IEEE Transactions on Evolutionary Computation, vol. 25, no. 1, pp. 159-171, Feb. 2021.
"""

from typing import Callable, List, Optional, Tuple, Union
import numpy as np

from .edg import EDG


class ERDG(EDG):
    """
    Efficient Recursive Differential Grouping (ERDG).

    Inherits from EDG and provides explicit ERDG nomenclature according to Yang et al. (2021).
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        dim: int,
        lower: Union[float, np.ndarray] = -100.0,
        upper: Union[float, np.ndarray] = 100.0,
        delta: Optional[float] = None,
        epsilon: float = 1e-2,
    ):
        super().__init__(
            func=func,
            dim=dim,
            lower=lower,
            upper=upper,
            delta=delta,
            epsilon=epsilon,
        )


def erdg(
    func: Callable[[np.ndarray], float],
    dim: int,
    lower: Union[float, np.ndarray] = -100.0,
    upper: Union[float, np.ndarray] = 100.0,
    delta: Optional[float] = None,
    epsilon: float = 1e-2,
) -> Tuple[List[List[int]], int]:
    """Convenience functional wrapper for ERDG."""
    solver = ERDG(func, dim, lower=lower, upper=upper, delta=delta, epsilon=epsilon)
    return solver.run()
