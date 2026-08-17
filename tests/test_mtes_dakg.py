import unittest
import numpy as np
from src.MTES_DAKG.MTES_DAKG import MTES_DAKG
from src.MTES_DAKG.DT_DoS import dt_dos
from src.MTES_DAKG.AS_SaS import as_sas


class TestMTESDAKG(unittest.TestCase):
    def test_dt_dos_generation(self):
        """Test DT_DoS external sample generation."""
        np.random.seed(42)
        dim = 4
        X_t = np.random.randn(8, dim)
        F_t = np.sum(X_t ** 2, axis=1)
        m_t = np.mean(X_t, axis=0)
        C_t = np.eye(dim)

        m_s = np.ones(dim) * 0.5
        C_s = np.eye(dim)

        hat_x = dt_dos(
            X_t=X_t,
            F_t=F_t,
            m_t=m_t,
            C_t=C_t,
            m_s=m_s,
            C_s=C_s,
            eval_target=lambda x: float(np.sum(x ** 2)),
            k_knn=3,
        )
        self.assertEqual(len(hat_x), dim)
        self.assertFalse(np.any(np.isnan(hat_x)))

    def test_as_sas_generation(self):
        """Test AS_SaS external sample generation."""
        np.random.seed(42)
        dim = 4
        X_s = np.random.randn(8, dim)
        F_s = np.sum(X_s ** 2, axis=1)
        m_t = np.zeros(dim)
        C_t_sqrt = np.eye(dim)
        m_s = np.mean(X_s, axis=0)
        C_s_inv_sqrt = np.eye(dim)

        hat_x = as_sas(
            X_s=X_s,
            F_s=F_s,
            m_t=m_t,
            C_t_sqrt=C_t_sqrt,
            m_s=m_s,
            C_s_inv_sqrt=C_s_inv_sqrt,
            gen=10,
            max_gen=50,
            popsize=8,
        )
        self.assertEqual(len(hat_x), dim)
        self.assertFalse(np.any(np.isnan(hat_x)))

    def test_mtes_dakg_optimization(self):
        """Test MTES_DAKG optimization on a 2-task synthetic subproblem."""
        def eval_global(x):
            return float(np.sum(x ** 2))

        # MTOP with 2 tasks of 3 dimensions each
        subtasks = [[0, 1, 2], [3, 4, 5]]
        mtes = MTES_DAKG(
            subtasks_vars=subtasks,
            eval_func=eval_global,
            lower=-10.0,
            upper=10.0,
            max_gen=30,
        )

        collaborator = np.ones(6) * 5.0
        init_fit = eval_global(collaborator)

        best_sols, best_fits, stag_set = mtes.optimize(
            collaborator=collaborator,
            stagnant_set=set(),
        )

        for sub_vars, sub_sol in zip(subtasks, best_sols):
            collaborator[sub_vars] = sub_sol

        final_fit = eval_global(collaborator)
        self.assertLess(final_fit, init_fit)


if __name__ == "__main__":
    unittest.main()
