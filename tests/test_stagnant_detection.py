import unittest
import numpy as np
from src.CCMTO.StagnantDetection import StagnantDetection


class TestStagnantDetection(unittest.TestCase):
    def test_stagnation_detection(self):
        """Test that stagnant detector flags stagnation when population and fitness do not change."""
        dim = 3
        detector = StagnantDetection(dim=dim, epsilon=1e-5)

        # Constant population and fitness for several generations
        pop = np.ones((10, dim)) * 2.0
        fitness = 5.0

        is_stag = False
        for gen in range(10):
            is_stag = detector.check(current_best_f=fitness, current_pop=pop, max_gen=5)
            if is_stag:
                break

        self.assertTrue(is_stag)

    def test_active_evolution_no_stagnation(self):
        """Test that stagnant detector stays False while fitness is constantly improving."""
        dim = 3
        detector = StagnantDetection(dim=dim, epsilon=1e-5)

        for gen in range(20):
            pop = np.random.randn(10, dim) * (1.0 / (gen + 1))
            fitness = 100.0 / (gen + 1)
            is_stag = detector.check(current_best_f=fitness, current_pop=pop, max_gen=50)
            self.assertFalse(is_stag)


if __name__ == "__main__":
    unittest.main()
