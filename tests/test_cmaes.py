import unittest
import numpy as np
from src.MTES_DAKG.CMAES import CMAES


class TestCMAES(unittest.TestCase):
    def test_cmaes_optimization(self):
        """Test that CMA-ES optimizes a simple Sphere function."""
        np.random.seed(42)
        dim = 5
        init_m = np.ones(dim) * 5.0
        cma = CMAES(dim=dim, lower=-10.0, upper=10.0, mean=init_m)

        def sphere(x):
            return float(np.sum(x ** 2))

        init_val = sphere(init_m)

        for _ in range(80):
            X, _ = cma.sample()
            F = np.array([sphere(x) for x in X])
            cma.update(X, F)

        final_val = sphere(cma.best_x)
        self.assertLess(final_val, init_val)
        self.assertLess(final_val, 1e-3)


if __name__ == "__main__":
    unittest.main()
