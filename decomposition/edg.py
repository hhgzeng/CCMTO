"""
Efficient Differential Grouping (EDG)

For large-scale optimization variable decomposition.

Input:
    func:
        objective function

    dim:
        number of variables

    lower:
        lower bound

    upper:
        upper bound

Output:
    groups:
        variable interaction groups

"""

import numpy as np


class EDG:
    def __init__(self, func, dim, lower=-100, upper=100, delta=1e-4, epsilon=1e-3):

        self.func = func
        self.dim = dim

        self.lower = lower
        self.upper = upper

        self.delta = delta
        self.epsilon = epsilon

    def _interaction_test(self, i, j):
        """
        Differential grouping test
        Determine whether variable i and j interact
        """

        x = np.random.uniform(self.lower, self.upper, self.dim)
        f0 = self.func(x)

        # xi + delta
        x1 = x.copy()
        x1[i] += self.delta
        f1 = self.func(x1)

        # xj + delta
        x2 = x.copy()
        x2[j] += self.delta
        f2 = self.func(x2)

        # xi,xj + delta
        x3 = x.copy()
        x3[i] += self.delta
        x3[j] += self.delta
        f3 = self.func(x3)

        diff = f3 - f1 - f2 + f0

        return abs(diff) > self.epsilon

    def run(self):
        """
        EDG decomposition
        """
        groups = []
        ungrouped = set(range(self.dim))

        while ungrouped:
            seed = min(ungrouped)
            current = [seed]
            ungrouped.remove(seed)
            changed = True

            while changed:
                changed = False

                for j in list(ungrouped):
                    flag = False

                    for i in current:
                        if self._interaction_test(i, j):
                            flag = True
                            break

                    if flag:
                        current.append(j)
                        ungrouped.remove(j)
                        changed = True

            groups.append(current)

        return groups


def edg(func, dim, lower=-100, upper=100):
    solver = EDG(func, dim, lower, upper)
    return solver.run()


if __name__ == "__main__":
    # test function
    def sphere(x):
        return np.sum(x * x)

    groups = edg(sphere, dim=10)
    print(groups)
