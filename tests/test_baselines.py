"""
Smoke tests for all baseline algorithms to verify interface compatibility,
numerical correctness, and evaluation limits.
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
from src.CCMTO.CCMTO import CCMTO


def run_tests():
    bench = Benchmark()
    func = bench.get_function(1)
    info = bench.get_info(1)
    dim, lower, upper = info["dimension"], info["lower"], info["upper"]

    custom_sub = [[i] for i in range(10)] + [list(range(10, dim))]

    print("Testing CMAES_EDG...")
    solver_cmaes = CMAES_EDG(
        func=func,
        dim=dim,
        lower=lower,
        upper=upper,
        max_fes=1000,
        custom_subproblems=custom_sub,
    )
    res_cmaes = solver_cmaes.optimize()
    assert res_cmaes["fes"] >= 1000
    assert np.isfinite(res_cmaes["best_f"])
    print(f"  CMAES_EDG ok, best_f = {res_cmaes['best_f']:.4e}, fes = {res_cmaes['fes']}")

    print("Testing DECC_ERDG...")
    solver_decc_erdg = DECC_ERDG(
        func=func,
        dim=dim,
        lower=lower,
        upper=upper,
        max_fes=1000,
        pop_size=20,
        custom_subproblems=custom_sub,
    )
    res_decc_erdg = solver_decc_erdg.optimize()
    assert res_decc_erdg["fes"] >= 1000
    assert np.isfinite(res_decc_erdg["best_f"])
    print(f"  DECC_ERDG ok, best_f = {res_decc_erdg['best_f']:.4e}, fes = {res_decc_erdg['fes']}")

    print("Testing GTDE...")
    solver_gtde = GTDE(
        func=func,
        dim=dim,
        lower=lower,
        upper=upper,
        max_fes=1000,
        pop_size=20,
        target_group_size=20,
    )
    res_gtde = solver_gtde.optimize()
    assert res_gtde["fes"] >= 1000
    assert np.isfinite(res_gtde["best_f"])
    print(f"  GTDE ok, best_f = {res_gtde['best_f']:.4e}, fes = {res_gtde['fes']}")

    print("Testing SDLSO...")
    solver_sdlso = SDLSO(
        func=func,
        dim=dim,
        lower=lower,
        upper=upper,
        max_fes=1000,
        pop_size=20,
    )
    res_sdlso = solver_sdlso.optimize()
    assert res_sdlso["fes"] >= 1000
    assert np.isfinite(res_sdlso["best_f"])
    print(f"  SDLSO ok, best_f = {res_sdlso['best_f']:.4e}, fes = {res_sdlso['fes']}")

    print("Testing CCMTO-MTES-DAKG...")
    solver_ccmto = CCMTO(
        func=func,
        dim=dim,
        lower=lower,
        upper=upper,
        max_fes=1000,
        custom_subproblems=custom_sub,
        verbose=False,
    )
    res_ccmto = solver_ccmto.optimize()
    assert res_ccmto["fes"] >= 1000
    assert np.isfinite(res_ccmto["best_f"])
    print(f"  CCMTO ok, best_f = {res_ccmto['best_f']:.4e}, fes = {res_ccmto['fes']}")

    print("\nAll baseline algorithms smoke tested successfully!")


if __name__ == "__main__":
    run_tests()
