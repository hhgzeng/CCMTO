"""
Construction Strategy of Multitask Optimization Problems (MTOPs Construction)

Algorithm 3 in CCMTO:
Groups decomposed subproblems by dimension and partitions them into MTOPs
with suitable task counts and bounded dimension disparities.
"""

from typing import List
import numpy as np


class MTOPConstruction:
    def __init__(self, n_sub: int = 5, d_max: float = 2.0):
        """
        Initialize MTOP constructor.

        Args:
            n_sub: Maximum number of subtasks per MTOP (default: 5)
            d_max: Maximum allowed dimension ratio between tasks in a MTOP (default: 2.0)
        """
        self.n_sub = n_sub
        self.d_max = d_max

    def construct(self, subproblems: List[List[int]]) -> List[List[List[int]]]:
        """
        Construct MTOPs from decomposed subproblems.

        Args:
            subproblems: List of subproblems, where each subproblem is a list of variable indices

        Returns:
            mtops: List of MTOPs, where each MTOP is a list of subproblems [T_1, ..., T_k]
        """
        if not subproblems:
            return []

        if len(subproblems) == 1:
            return [[subproblems[0]]]

        # 1. Sort subproblems by dimension ascending
        sorted_subs = sorted(subproblems, key=lambda s: len(s))

        # Group by identical dimensions
        dim_to_subs = {}
        for s in sorted_subs:
            d = len(s)
            if d not in dim_to_subs:
                dim_to_subs[d] = []
            dim_to_subs[d].append(s)

        unique_dims = sorted(list(dim_to_subs.keys()))
        groups = [dim_to_subs[d] for d in unique_dims]
        group_dims = [float(d) for d in unique_dims]

        # 2. Adjust singleton groups
        j = len(groups)
        for i in range(j):
            if len(groups[i]) == 1:
                sub = groups[i][0]
                d_curr = group_dims[i]

                # Check ratios with neighbors
                ratio_left = (d_curr / group_dims[i - 1]) if i > 0 else float("inf")
                ratio_right = (group_dims[i + 1] / d_curr) if (i < j - 1) else float("inf")

                if ratio_left > self.d_max and ratio_right > self.d_max:
                    # Treat as single task, keep as is
                    pass
                elif ratio_right <= self.d_max and i < j - 1:
                    # Add into right neighbor
                    groups[i + 1].append(sub)
                    groups[i] = []
                elif ratio_left <= self.d_max and i > 0:
                    # Add into left neighbor
                    groups[i - 1].append(sub)
                    groups[i] = []

        # Remove empty groups
        valid_groups = [g for g in groups if len(g) > 0]

        # 3. Partition into MTOPs
        mtops = []
        for g in valid_groups:
            card = len(g)
            if card == 1:
                mtops.append([g[0]])
            elif 1 < card <= self.n_sub:
                mtops.append(list(g))
            else:
                # Partition randomly into chunks of size n_sub
                shuffled = list(g)
                np.random.shuffle(shuffled)
                for start_idx in range(0, card, self.n_sub):
                    chunk = shuffled[start_idx : start_idx + self.n_sub]
                    if chunk:
                        mtops.append(chunk)

        return mtops


def construct_mtops(
    subproblems: List[List[int]], n_sub: int = 5, d_max: float = 2.0
) -> List[List[List[int]]]:
    """Convenience functional wrapper for MTOP construction."""
    constructor = MTOPConstruction(n_sub=n_sub, d_max=d_max)
    return constructor.construct(subproblems)
