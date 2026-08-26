"""
Precompute and cache EDG subproblem decomposition for CEC2013 LSGO functions F1 to F11.
Since benchmark objective landscape interactions are deterministic, running EDG once per
benchmark function saves massive computational overhead across repeated Monte Carlo runs.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json

from cec2013lsgo.cec2013 import Benchmark

from decomposition.edg import EDG
from src.utils import cleanup_benchmark_csv, register_csv_cleanup, save_json

register_csv_cleanup()

CACHE_FILE = os.path.join(os.path.dirname(__file__), "edg_subproblems_cec2013.json")


def get_or_compute_edg_subproblems(func_id: int, epsilon: float = 1e-2):
    """Load cached EDG subproblems if available, otherwise compute and cache."""
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    fid_str = str(func_id)
    if fid_str in cache:
        return cache[fid_str]["subproblems"], cache[fid_str]["fes"]

    print(f"[EDG Cache] Computing EDG decomposition for CEC2013 F{func_id}...")
    bench = Benchmark()
    info = bench.get_info(func_id)
    func = bench.get_function(func_id)
    dim = info["dimension"]
    lower = info["lower"]
    upper = info["upper"]

    try:
        start_t = time.time()
        edg_solver = EDG(func=func, dim=dim, lower=lower, upper=upper, epsilon=epsilon)
        subproblems, edg_fes = edg_solver.run()
        elapsed = time.time() - start_t

        print(
            f"[EDG Cache] F{func_id} decomposition done in {elapsed:.2f}s: {len(subproblems)} subproblems found using {edg_fes} FEs."
        )

        cache[fid_str] = {
            "subproblems": subproblems,
            "fes": edg_fes,
            "num_subproblems": len(subproblems),
            "sizes": [len(s) for s in subproblems],
        }

        save_json(cache, CACHE_FILE, format_prettier=True)

        return subproblems, edg_fes
    finally:
        cleanup_benchmark_csv()


def precompute_all(functions=range(1, 12)):
    print("Precomputing EDG decomposition for CEC2013 functions...")
    try:
        for fid in functions:
            get_or_compute_edg_subproblems(fid)
        print(f"All EDG decompositions cached in: {CACHE_FILE}")
    finally:
        cleanup_benchmark_csv()


if __name__ == "__main__":
    precompute_all()
