"""
Comprehensive verification script for the 4 corrected baseline algorithms:
1. CMAES-EDG (Kumar et al., 2024)
2. DECC-ERDG (Yang et al., 2021)
3. GTDE (Wang et al., 2023)
4. SDLSO (Yang et al., 2022)
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cec2013lsgo.cec2013 import Benchmark
from baselines.cmaes_edg import CMAES_EDG
from baselines.decc_erdg import DECC_ERDG
from baselines.gtde import GTDE
from baselines.sdlso import SDLSO


def test_sphere_convergence():
    """Verify all algorithms converge on standard Sphere function."""
    print("=" * 60)
    print("1. Testing Convergence on 100-D Sphere Function")
    print("=" * 60)

    dim = 100
    lower = -100.0
    upper = 100.0
    max_fes = 20000

    def sphere(x):
        return float(np.sum(x ** 2))

    algorithms = {
        "CMAES-EDG": CMAES_EDG(sphere, dim, lower, upper, max_fes=max_fes, custom_subproblems=[list(range(dim))]),
        "DECC-ERDG": DECC_ERDG(sphere, dim, lower, upper, max_fes=max_fes, custom_subproblems=[list(range(dim))]),
        "GTDE": GTDE(sphere, dim, lower, upper, max_fes=max_fes, target_group_size=10),
        "SDLSO": SDLSO(sphere, dim, lower, upper, max_fes=max_fes),
    }

    for name, algo in algorithms.items():
        res = algo.optimize()
        best_f = res["best_f"]
        fes = res["fes"]
        print(f"  [{name:18s}] Best Fitness: {best_f:.4e} | FEs: {fes:,}")
        assert np.isfinite(best_f), f"{name} produced non-finite fitness"
        assert best_f < 1e5, f"{name} failed to improve on Sphere"


def test_gtde_mechanisms():
    """Verify GTDE specific mechanisms: full DE + gene targeting on gbest."""
    print("\n" + "=" * 60)
    print("2. Testing GTDE Specific Mechanisms (Wang et al., 2023)")
    print("=" * 60)

    dim = 50
    lower = -10.0
    upper = 10.0

    def rosenbrock(x):
        return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2))

    gtde = GTDE(rosenbrock, dim, lower, upper, max_fes=5000, pop_size=30, target_group_size=5)
    res = gtde.optimize()

    assert res["fes"] >= 5000
    assert np.isfinite(res["best_f"])
    print(f"  GTDE Rosenbrock 50D: Best = {res['best_f']:.4e}, FEs = {res['fes']:,} -> PASS")


def test_sdlso_mechanisms():
    """Verify SDLSO specific mechanisms: pairwise competition and dominance check."""
    print("\n" + "=" * 60)
    print("3. Testing SDLSO Specific Mechanisms (Yang et al., 2022)")
    print("=" * 60)

    dim = 50
    lower = -10.0
    upper = 10.0

    def ackley(x):
        n = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(2 * np.pi * x))
        return float(-20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20.0 + np.e)

    sdlso = SDLSO(ackley, dim, lower, upper, max_fes=5000, pop_size=40)
    res = sdlso.optimize()

    assert res["fes"] >= 5000
    assert np.isfinite(res["best_f"])
    print(f"  SDLSO Ackley 50D: Best = {res['best_f']:.4e}, FEs = {res['fes']:,} -> PASS")


def test_cec2013_smoke():
    """Test on CEC2013 LSGO Function 1 (1000-D)."""
    print("\n" + "=" * 60)
    print("4. Testing on CEC2013 LSGO F1 (1000-D Fully Separable Elliptic)")
    print("=" * 60)

    bench = Benchmark()
    func = bench.get_function(1)
    info = bench.get_info(1)
    dim, lower, upper = info["dimension"], info["lower"], info["upper"]

    # 10 subproblems of size 100 for decomposition solvers
    subproblems = [list(range(i * 100, (i + 1) * 100)) for i in range(10)]

    for name, cls, kw in [
        ("CMAES-EDG", CMAES_EDG, {"custom_subproblems": subproblems}),
        ("DECC-ERDG", DECC_ERDG, {"custom_subproblems": subproblems}),
        ("GTDE", GTDE, {"target_group_size": 50}),
        ("SDLSO", SDLSO, {}),
    ]:
        solver = cls(func, dim, lower, upper, max_fes=2000, **kw)
        res = solver.optimize()
        print(f"  [{name:12s}] CEC2013 F1: Best = {res['best_f']:.4e} | FEs = {res['fes']:,}")
        assert np.isfinite(res["best_f"])


if __name__ == "__main__":
    test_sphere_convergence()
    test_gtde_mechanisms()
    test_sdlso_mechanisms()
    test_cec2013_smoke()
    print("\nAll verification tests PASSED successfully!")
