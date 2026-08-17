"""
Experiment runner for CCMTO on CEC2013 LSGO benchmarks.
Supports all parameter sensitivity settings from TABLE III (n_sub, d_max, tau, fre)
with default values set to the best settings identified in the paper.
Saves optimization progress and final results into the results/ folder.
"""

import argparse
import json
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from cec2013lsgo.cec2013 import Benchmark

from src.CCMTO.CCMTO import CCMTO


def parse_d_max(d_max_val: str) -> float:
    """Parse d_max value, supporting 'limitless' / 'inf' as float('inf')."""
    if str(d_max_val).lower() in ["limitless", "inf", "none"]:
        return float("inf")
    return float(d_max_val)


def parse_fre(fre_val: str) -> float:
    """Parse frequency parameter (e.g. '0.1Maxgen', 'everygen', '0.05Maxgen', 0.1)."""
    val_str = str(fre_val).lower().replace("maxgen", "").strip()
    if val_str in ["everygen", "every", "0", "0.0"]:
        return 0.0  # fre = 1 every generation
    return float(val_str)


def run_cec2013_experiment(
    func_id: int = 1,
    max_fes: int = 3_000_000,
    n_sub: int = 5,
    d_max: float = 2.0,
    tau: int = 1,
    fre: str = "0.1Maxgen",
    seed: int = 42,
    output_dir: str = "results",
    log_interval: int = 10_000,
):
    """
    Execute CCMTO on a specified CEC2013 benchmark problem.
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    d_max_parsed = parse_d_max(d_max)
    fre_ratio = parse_fre(fre)

    bench = Benchmark()
    info = bench.get_info(func_id)
    func = bench.get_function(func_id)

    dim = info["dimension"]
    lower = info["lower"]
    upper = info["upper"]
    best_known = info["best"]

    print("=" * 70)
    print(f"Running CCMTO-MTES-DAKG on CEC2013 Function F{func_id}")
    print(f"Dimension: {dim} | Bounds: [{lower}, {upper}] | Best Known: {best_known}")
    print(f"MaxFEs: {max_fes:,} | Seed: {seed}")
    print(
        f"Sensitivity Parameters (Table III): n_sub={n_sub}, d_max={d_max}, tau={tau}, fre={fre}"
    )
    print("=" * 70)

    start_time = time.time()

    solver = CCMTO(
        func=func,
        dim=dim,
        lower=lower,
        upper=upper,
        max_fes=max_fes,
        n_sub=n_sub,
        d_max=d_max_parsed,
        tau=tau,
        fre_ratio=fre_ratio,
        verbose=True,
        log_interval=log_interval,
    )

    result = solver.optimize()
    elapsed_time = time.time() - start_time

    best_f = result["best_f"]
    error = abs(best_f - best_known)
    total_fes = result["fes"]
    history = result["history"]
    subproblems = result["subproblems"]
    mtops = result["mtops"]

    print("\n" + "=" * 70)
    print(f"Optimization Finished for CEC2013 F{func_id}!")
    print(f"Elapsed Time: {elapsed_time:.2f} s")
    print(f"Total FEs: {total_fes:,}")
    print(f"Best Fitness: {best_f:.6e}")
    print(f"Error to Best: {error:.6e}")
    print(f"Decomposed Subproblems: {len(subproblems)}")
    print(f"Constructed MTOPs: {len(mtops)}")
    print("=" * 70)

    # Convert results to serializable format
    res_data = {
        "benchmark": "CEC2013 LSGO",
        "func_id": func_id,
        "dimension": dim,
        "max_fes": max_fes,
        "parameters": {
            "n_sub": n_sub,
            "d_max": str(d_max),
            "tau": tau,
            "fre": str(fre),
        },
        "seed": seed,
        "elapsed_seconds": elapsed_time,
        "total_fes": total_fes,
        "best_fitness": best_f,
        "best_known": best_known,
        "error": error,
        "num_subproblems": len(subproblems),
        "subproblem_sizes": [len(s) for s in subproblems],
        "num_mtops": len(mtops),
        "mtop_task_counts": [len(m) for m in mtops],
        "history": history,
    }

    out_path = os.path.join(output_dir, f"cec2013_f{func_id}_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res_data, f, indent=2)

    print(f"Results successfully saved to: {out_path}")
    return res_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CCMTO on CEC2013 LSGO benchmark with Table III parameter support"
    )
    parser.add_argument("--func", type=int, default=1, help="CEC2013 Function ID (1-15)")
    parser.add_argument("--max_fes", type=int, default=100_000, help="Max function evaluations")
    parser.add_argument(
        "--n_sub",
        type=int,
        default=5,
        choices=[2, 3, 5, 7, 10, 20],
        help="Subtasks in a MTOP (default: 5 - Best in Table III)",
    )
    parser.add_argument(
        "--d_max",
        type=str,
        default="2",
        choices=["1", "2", "4", "limitless"],
        help="Max dimension ratio (default: 2 - Best in Table III)",
    )
    parser.add_argument(
        "--tau",
        type=int,
        default=1,
        choices=[0, 1, 2, 3, 4, 5],
        help="External sample number (default: 1 - Best in Table III)",
    )
    parser.add_argument(
        "--fre",
        type=str,
        default="0.1Maxgen",
        choices=["everygen", "0.05Maxgen", "0.1Maxgen", "0.2Maxgen", "0.3Maxgen", "0.5Maxgen"],
        help="Frequency of external sampling (default: 0.1Maxgen - Best in Table III)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_interval", type=int, default=5_000, help="Logging interval")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")

    args = parser.parse_args()
    run_cec2013_experiment(
        func_id=args.func,
        max_fes=args.max_fes,
        n_sub=args.n_sub,
        d_max=args.d_max,
        tau=args.tau,
        fre=args.fre,
        seed=args.seed,
        output_dir=args.output_dir,
        log_interval=args.log_interval,
    )
