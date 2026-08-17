import unittest
import numpy as np
from src.CCMTO.CCMTO import CCMTO, ccmto


class TestCCMTO(unittest.TestCase):
    def test_ccmto_end_to_end_synthetic(self):
        """Test complete CCMTO workflow on a 20-D partially separable synthetic problem."""
        dim = 20

        def synthetic_func(x):
            # 2 non-separable blocks of size 5 + 10 separable variables
            block1 = np.sum((x[0:5] - 1.0) ** 2) + (x[0] * x[1])
            block2 = np.sum((x[5:10] + 2.0) ** 2) + (x[5] * x[6])
            sep = np.sum(x[10:20] ** 2)
            return float(block1 + block2 + sep)

        result = ccmto(
            func=synthetic_func,
            dim=dim,
            lower=-10.0,
            upper=10.0,
            max_fes=20_000,
            n_sub=5,
            d_max=2.0,
            verbose=False,
        )

        self.assertIn("best_x", result)
        self.assertIn("best_f", result)
        self.assertIn("fes", result)
        self.assertIn("history", result)
        self.assertLessEqual(result["fes"], 25_000)
        self.assertLess(result["best_f"], 100.0)


if __name__ == "__main__":
    unittest.main()
