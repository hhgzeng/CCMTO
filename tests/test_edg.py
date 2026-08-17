import unittest
import numpy as np
from decomposition.edg import EDG, edg


class TestEDG(unittest.TestCase):
    def test_fully_separable(self):
        """Test EDG on fully separable Sphere function."""
        def sphere(x):
            return float(np.sum(x ** 2))

        dim = 10
        groups, fes = edg(sphere, dim=dim, lower=-10, upper=10)

        # In a fully separable problem, all subproblems should have dimension 1
        self.assertEqual(len(groups), dim)
        for g in groups:
            self.assertEqual(len(g), 1)
        self.assertGreater(fes, 0)

    def test_partially_separable(self):
        """Test EDG on partially separable function with known non-separable blocks."""
        def part_sep(x):
            # [0, 1] interact, [2, 3, 4] interact, [5], [6] separable
            term1 = (x[0] + x[1]) ** 2
            term2 = (x[2] * x[3] + x[4]) ** 2
            term3 = x[5] ** 2 + x[6] ** 2
            return float(term1 + term2 + term3)

        dim = 7
        groups, fes = edg(part_sep, dim=dim, lower=-10, upper=10)

        # Normalize group representation for checking
        sorted_groups = sorted([sorted(g) for g in groups], key=lambda g: (len(g), g[0]))
        expected_groups = sorted([[0, 1], [2, 3, 4], [5], [6]], key=lambda g: (len(g), g[0]))

        self.assertEqual(sorted_groups, expected_groups)
        self.assertGreater(fes, 0)


if __name__ == "__main__":
    unittest.main()
