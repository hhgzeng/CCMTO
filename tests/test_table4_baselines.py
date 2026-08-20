"""
Unit tests for Table IV baseline algorithms across all 3 modules.
"""

import unittest
import numpy as np

from baselines.resource_allocation import CBCC1, CBCC2, CBCC3, CCFR, CCFR2, CCFR3
from baselines.emto_algorithms import CCMTO_MaTDE, CCMTO_GMFEA, CCMTO_MTEA_AD
from baselines.component_ablation import WO_DA, WO_DT_DoS, WO_AS_SaS, WO_SD


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


class TestTable4Baselines(unittest.TestCase):
    def setUp(self):
        self.dim = 20
        self.max_fes = 1000
        self.subproblems = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]

    def test_module1_resource_allocation(self):
        algorithms = [CBCC1, CBCC2, CBCC3, CCFR, CCFR2, CCFR3]
        for algo_cls in algorithms:
            solver = algo_cls(
                func=sphere,
                dim=self.dim,
                lower=-100.0,
                upper=100.0,
                max_fes=self.max_fes,
                max_gen_per_cycle=5,
                custom_subproblems=self.subproblems,
                verbose=False,
            )
            res = solver.optimize()
            self.assertIn("best_f", res)
            self.assertIn("best_x", res)
            self.assertLessEqual(res["fes"], self.max_fes + 200)
            self.assertIsInstance(res["best_f"], float)

    def test_module2_emto_algorithms(self):
        algorithms = [CCMTO_MaTDE, CCMTO_GMFEA, CCMTO_MTEA_AD]
        for algo_cls in algorithms:
            solver = algo_cls(
                func=sphere,
                dim=self.dim,
                lower=-100.0,
                upper=100.0,
                max_fes=self.max_fes,
                max_gen_per_cycle=5,
                custom_subproblems=self.subproblems,
                verbose=False,
            )
            res = solver.optimize()
            self.assertIn("best_f", res)
            self.assertIn("best_x", res)
            self.assertLessEqual(res["fes"], self.max_fes + 200)
            self.assertIsInstance(res["best_f"], float)

    def test_module3_component_ablation(self):
        algorithms = [WO_DA, WO_DT_DoS, WO_AS_SaS, WO_SD]
        for algo_cls in algorithms:
            solver = algo_cls(
                func=sphere,
                dim=self.dim,
                lower=-100.0,
                upper=100.0,
                max_fes=self.max_fes,
                max_gen_per_cycle=5,
                custom_subproblems=self.subproblems,
                verbose=False,
            )
            res = solver.optimize()
            self.assertIn("best_f", res)
            self.assertIn("best_x", res)
            self.assertLessEqual(res["fes"], self.max_fes + 200)
            self.assertIsInstance(res["best_f"], float)


if __name__ == "__main__":
    unittest.main()
