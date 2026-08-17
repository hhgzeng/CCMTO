import unittest
from src.CCMTO.MTOPConstruction import MTOPConstruction, construct_mtops


class TestMTOPConstruction(unittest.TestCase):
    def test_construction_uniform_dims(self):
        """Test constructing MTOPs from 10 50-D subproblems with n_sub=5."""
        subproblems = [list(range(i * 50, (i + 1) * 50)) for i in range(10)]
        mtops = construct_mtops(subproblems, n_sub=5, d_max=2.0)

        # 10 subproblems of equal dimension partitioned into chunks of 5 -> 2 MTOPs
        self.assertEqual(len(mtops), 2)
        self.assertEqual(len(mtops[0]), 5)
        self.assertEqual(len(mtops[1]), 5)

    def test_construction_variable_coverage(self):
        """Test that all original variables and subproblems are preserved."""
        subproblems = [
            [0, 1],
            [2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10],
            [11],
            [12],
        ]
        mtops = construct_mtops(subproblems, n_sub=5, d_max=2.0)

        all_constructed_subs = []
        for mtop in mtops:
            for sub in mtop:
                all_constructed_subs.append(sub)

        self.assertEqual(len(all_constructed_subs), len(subproblems))
        flattened_orig = sorted([v for sub in subproblems for v in sub])
        flattened_constructed = sorted([v for sub in all_constructed_subs for v in sub])
        self.assertEqual(flattened_orig, flattened_constructed)


if __name__ == "__main__":
    unittest.main()
