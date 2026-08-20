"""
Experiment runner for Table II reproduction in CCMTO paper.

Runs 5 algorithms on CEC2013 LSGO benchmark suite (F1 to F11):
1. CCMTO-MTES-DAKG (Proposed)
2. CMAES-EDG
3. DECC-ERDG
4. GTDE
5. SDLSO

Each algorithm is evaluated for 10 independent runs per benchmark function.
Results are saved under results/TABLE II/<algorithm_name>/cec2013_f<func_id>.json.
Supports multi-process parallelism for fast execution.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cec2013lsgo.cec2013 import Benchmark
from baselines.cmaes_edg import CMAES_EDG
from baselines.decc_erdg import DECC_ERDG
from baselines.gtde import GTDE
from baselines.sdlso import SDLSO
from src.CCMTO.CCMTO import CCMTO
from decomposition.precompute_edg import get_or_compute_edg_subproblems


ALGORITHMS = {
    "CCMTO-MTES-DAKG": CCMTO,
    "CMAES-EDG": CMAES_EDG,
    "DECC-ERDG": DECC_ERDG,
    "GTDE": GTDE,
    "SDLSO": SDLSO,
}


def create_solver(algo_name: str, func, dim: int, lower: float, upper: float, max_fes: int, subproblems=None):
    """Instantiate the optimizer with appropriate parameters."""
    if algo_name == "CCMTO-MTES-DAKG":
        return CCMTO(
            func=func,
            dim=dim,
            lower=lower,
            upper=upper,
            max_fes=max_fes,
            n_sub=5,
            d_max=2.0,
            tau=1,
            fre_ratio=0.1,
            custom_subproblems=subproblems,
            verbose=False,
        )
    elif algo_name == "CMAES-EDG":
        return CMAES_EDG(
            func=func,
            dim=dim,
            lower=lower,
            upper=upper,
            max_fes=max_fes,
            max_gen_per_cycle=50,
            custom_subproblems=subproblems,
            verbose=False,
        )
    elif algo_name == "DECC-ERDG":
        return DECC_ERDG(
            func=func,
            dim=dim,
            lower=lower,
            upper=upper,
            max_fes=max_fes,
            pop_size=50,
            f_weight=0.5,
            cr_prob=0.9,
            gen_per_cycle=10,
            custom_subproblems=subproblems,
            verbose=False,
        )
    elif algo_name == "GTDE":
        return GTDE(
            func=func,
            dim=dim,
            lower=lower,
            upper=upper,
            max_fes=max_fes,
            pop_size=50,
            f_weight=0.5,
            cr_prob=0.9,
            target_group_size=50,
            verbose=False,
        )
    elif algo_name == "SDLSO":
        return SDLSO(
            func=func,
            dim=dim,
            lower=lower,
            upper=upper,
            max_fes=max_fes,
            pop_size=100,
            w_max=0.9,
            w_min=0.4,
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def run_single_task(
    algo_name: str,
    func_id: int,
    run_idx: int,
    seed: int,
    max_fes: int,
    subproblems: Optional[List[List[int]]] = None,
) -> Dict[str, Any]:
    """Execute a single run of an algorithm on a benchmark function."""
    np.random.seed(seed)
    bench = Benchmark()
    info = bench.get_info(func_id)
    func = bench.get_function(func_id)

    dim = info["dimension"]
    lower = info["lower"]
    upper = info["upper"]
    best_known = info["best"]

    start_time = time.time()
    solver = create_solver(algo_name, func, dim, lower, upper, max_fes, subproblems=subproblems)
    result = solver.optimize()
    elapsed_time = time.time() - start_time

    best_f = float(result["best_f"])
    error = float(abs(best_f - best_known))
    total_fes = int(result["fes"])

    # Sample history to keep file size reasonable (max 100 points)
    full_history = result.get("history", [])
    if len(full_history) > 100:
        step = max(1, len(full_history) // 100)
        sampled_history = [full_history[i] for i in range(0, len(full_history), step)]
        if sampled_history[-1] != full_history[-1]:
            sampled_history.append(full_history[-1])
    else:
        sampled_history = full_history

    return {
        "run_idx": run_idx,
        "seed": seed,
        "best_fitness": best_f,
        "best_known": best_known,
        "error": error,
        "total_fes": total_fes,
        "elapsed_seconds": elapsed_time,
        "history": sampled_history,
    }


def run_experiments(
    algorithms: Optional[List[str]] = None,
    functions: Optional[List[int]] = None,
    num_runs: int = 10,
    max_fes: int = 3_000_000,
    base_seed: int = 42,
    num_workers: int = 4,
    output_base_dir: str = "results/TABLE II",
):
    """Run all benchmark comparison experiments and save structured results."""
    if algorithms is None:
        algorithms = list(ALGORITHMS.keys())
    if functions is None:
        functions = list(range(1, 12))  # F1 to F11

    os.makedirs(output_base_dir, exist_ok=True)

    print("=" * 80)
    print("STARTING TABLE II BENCHMARK REPRODUCTION EXPERIMENTS")
    print(f"Algorithms ({len(algorithms)}): {algorithms}")
    print(f"Functions ({len(functions)}): {functions}")
    print(f"Runs per function: {num_runs} | MaxFEs: {max_fes:,} | Workers: {num_workers}")
    print(f"Output Directory: {output_base_dir}")
    print("=" * 80)

    for algo in algorithms:
        algo_dir = os.path.join(output_base_dir, algo)
        os.makedirs(algo_dir, exist_ok=True)

        for fid in functions:
            json_path = os.path.join(algo_dir, f"cec2013_f{fid}.json")

            # Check if already completed
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    if len(existing.get("runs", [])) >= num_runs and existing.get("max_fes") == max_fes:
                        print(f"[{algo}] F{fid} already completed ({num_runs} runs). Skipping.")
                        continue
                except Exception:
                    pass

            print(f"\n>>> Running [{algo}] on CEC2013 F{fid} ({num_runs} runs, MaxFEs={max_fes:,})...")
            runs_data = []

            # Preload EDG subproblems for decomposition-based algorithms
            subproblems = None
            if algo in ["CCMTO-MTES-DAKG", "CMAES-EDG", "DECC-ERDG"]:
                subproblems, _ = get_or_compute_edg_subproblems(fid)

            # Multi-process execution for runs of this function
            tasks = [
                (algo, fid, r + 1, base_seed + r * 100 + fid, max_fes, subproblems)
                for r in range(num_runs)
            ]

            if num_workers > 1:
                with ProcessPoolExecutor(max_workers=min(num_workers, num_runs)) as executor:
                    futures = {
                        executor.submit(run_single_task, *task): task[2]
                        for task in tasks
                    }
                    for future in as_completed(futures):
                        run_id = futures[future]
                        try:
                            res = future.result()
                            runs_data.append(res)
                            print(
                                f"  [{algo} | F{fid} | Run {res['run_idx']}/{num_runs}] "
                                f"Error: {res['error']:.6e} | Time: {res['elapsed_seconds']:.2f}s | FEs: {res['total_fes']:,}"
                            )
                        except Exception as e:
                            print(f"  [ERROR in {algo} | F{fid} | Run {run_id}]: {e}")
            else:
                for task in tasks:
                    res = run_single_task(*task)
                    runs_data.append(res)
                    print(
                        f"  [{algo} | F{fid} | Run {res['run_idx']}/{num_runs}] "
                        f"Error: {res['error']:.6e} | Time: {res['elapsed_seconds']:.2f}s | FEs: {res['total_fes']:,}"
                    )

            # Sort runs by run_idx
            runs_data.sort(key=lambda x: x["run_idx"])

            errors = [r["error"] for r in runs_data]
            fitnesses = [r["best_fitness"] for r in runs_data]
            times = [r["elapsed_seconds"] for r in runs_data]

            bench = Benchmark()
            info = bench.get_info(fid)

            summary_data = {
                "algorithm": algo,
                "benchmark": "CEC2013 LSGO",
                "func_id": fid,
                "func_name": info.get("name", f"F{fid}"),
                "dimension": info["dimension"],
                "max_fes": max_fes,
                "num_runs": len(runs_data),
                "mean_error": float(np.mean(errors)) if errors else float("inf"),
                "std_error": float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0,
                "best_error": float(np.min(errors)) if errors else float("inf"),
                "worst_error": float(np.max(errors)) if errors else float("inf"),
                "median_error": float(np.median(errors)) if errors else float("inf"),
                "mean_fitness": float(np.mean(fitnesses)) if fitnesses else float("inf"),
                "std_fitness": float(np.std(fitnesses, ddof=1)) if len(fitnesses) > 1 else 0.0,
                "mean_time_seconds": float(np.mean(times)) if times else 0.0,
                "runs": runs_data,
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2)

            print(
                f"[{algo} | F{fid} COMPLETED] Mean Error: {summary_data['mean_error']:.6e} ± {summary_data['std_error']:.6e} -> Saved: {json_path}"
            )

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Table II Reproduction Experiments")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(ALGORITHMS.keys()),
        choices=list(ALGORITHMS.keys()),
        help="List of algorithms to run",
    )
    # User specified benchmark functions: F1, F2, F4, F5, F9
    DEFAULT_FUNCTIONS = [1, 2, 4, 5, 9]

    parser.add_argument(
        "--functions",
        nargs="+",
        type=int,
        default=DEFAULT_FUNCTIONS,
        help="List of CEC2013 function IDs to run (default: 1, 2, 4, 5, 9)",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="Number of runs per function (default: 10)",
    )
    parser.add_argument(
        "--max_fes",
        type=int,
        default=1_000_000,
        help="Maximum fitness evaluations (default: 1,000,000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/TABLE II",
        help="Output directory for results",
    )

    args = parser.parse_args()
    run_experiments(
        algorithms=args.algorithms,
        functions=args.functions,
        num_runs=args.num_runs,
        max_fes=args.max_fes,
        num_workers=args.workers,
        output_base_dir=args.output_dir,
    )
