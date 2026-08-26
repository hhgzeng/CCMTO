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
from src.utils import cleanup_benchmark_csv, register_csv_cleanup, save_json

register_csv_cleanup()


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
            verbose=False,
        )
    elif algo_name == "SDLSO":
        return SDLSO(
            func=func,
            dim=dim,
            lower=lower,
            upper=upper,
            max_fes=max_fes,
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def run_single_experiment(
    algo_name: str,
    func_id: int,
    run_idx: int,
    seed: int,
    max_fes: int,
    subproblems: Optional[List[List[int]]] = None,
) -> Dict[str, Any]:
    """Execute a single Monte Carlo run of a specified algorithm on a CEC2013 function."""
    np.random.seed(seed)

    bench = Benchmark()
    info = bench.get_info(func_id)
    func = bench.get_function(func_id)
    dim = info["dimension"]
    lower = info["lower"]
    upper = info["upper"]
    best_known = info["best"]

    t0 = time.time()
    try:
        solver = create_solver(
            algo_name=algo_name,
            func=func,
            dim=dim,
            lower=lower,
            upper=upper,
            max_fes=max_fes,
            subproblems=subproblems,
        )

        res = solver.optimize()
        elapsed = time.time() - t0

        best_f = res["best_f"]
        error = abs(best_f - best_known)
        fes_used = res["fes"]

        return {
            "algo": algo_name,
            "func_id": func_id,
            "run_idx": run_idx,
            "seed": seed,
            "best_fitness": float(best_f),
            "best_known": float(best_known),
            "error": float(error),
            "total_fes": int(fes_used),
            "elapsed_seconds": float(elapsed),
            "history": res.get("history", []),
        }
    finally:
        cleanup_benchmark_csv()


def run_experiments(
    algorithms: List[str],
    functions: List[int],
    num_runs: int = 10,
    max_fes: int = 1_000_000,
    num_workers: int = 4,
    output_base_dir: str = "results/TABLE II",
):
    """Orchestrate all Table II experiments with multi-process parallel worker pool."""
    print("=" * 80)
    print("TABLE II REPRODUCTION EXPERIMENTS")
    print(f"Algorithms: {algorithms}")
    print(f"Functions: F{[f for f in functions]}")
    print(f"Independent Runs per setting: {num_runs}")
    print(f"Max FEs: {max_fes:,}")
    print(f"Parallel Workers: {num_workers}")
    print(f"Output Directory: {output_base_dir}")
    print("=" * 80)

    # 1. Precompute/Load EDG decompositions for required functions
    print("\n[Step 1/3] Loading EDG subproblem decompositions...")
    subproblems_map = {}
    for fid in functions:
        subproblems_map[fid], _ = get_or_compute_edg_subproblems(fid)
    print("All EDG subproblems ready.\n")

    # 2. Prepare task list (skipping existing runs if already done)
    tasks = []
    base_seed = 42

    for algo in algorithms:
        algo_dir = os.path.join(output_base_dir, algo)
        os.makedirs(algo_dir, exist_ok=True)

        for fid in functions:
            json_path = os.path.join(algo_dir, f"cec2013_f{fid}.json")

            existing_runs = {}
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for r in data.get("runs", []):
                            existing_runs[r["run_idx"]] = r
                except Exception:
                    existing_runs = {}

            for run_idx in range(1, num_runs + 1):
                if run_idx not in existing_runs:
                    seed = base_seed + run_idx * 1000 + fid * 10
                    tasks.append((algo, fid, run_idx, seed, max_fes, subproblems_map[fid]))

    print(f"[Step 2/3] Total tasks to run: {len(tasks)} (already completed tasks skipped)")

    # 3. Execute tasks in parallel or sequentially
    results_collected: Dict[str, Dict[int, List[Dict[str, Any]]]] = {
        algo: {fid: [] for fid in functions} for algo in algorithms
    }

    # Load existing runs into results_collected
    for algo in algorithms:
        algo_dir = os.path.join(output_base_dir, algo)
        for fid in functions:
            json_path = os.path.join(algo_dir, f"cec2013_f{fid}.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        results_collected[algo][fid] = data.get("runs", [])
                except Exception:
                    pass

    if tasks:
        print(f"[Step 3/3] Executing with {num_workers} processes...")
        completed_count = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_task = {
                executor.submit(
                    run_single_experiment,
                    algo_name=t[0],
                    func_id=t[1],
                    run_idx=t[2],
                    seed=t[3],
                    max_fes=t[4],
                    subproblems=t[5],
                ): t
                for t in tasks
            }

            for future in as_completed(future_to_task):
                task_info = future_to_task[future]
                algo_name, fid, run_idx = task_info[0], task_info[1], task_info[2]
                try:
                    res = future.result()
                    results_collected[algo_name][fid].append(res)
                    completed_count += 1
                    print(
                        f"[{completed_count}/{len(tasks)}] DONE: {algo_name} | F{fid} | Run {run_idx}/{num_runs} | "
                        f"Best Fitness: {res['best_fitness']:.6e} | Error: {res['error']:.6e} | Time: {res['elapsed_seconds']:.2f}s"
                    )
                except Exception as exc:
                    print(f"[ERROR] {algo_name} | F{fid} | Run {run_idx} generated an exception: {exc}")
                finally:
                    cleanup_benchmark_csv()

    # 4. Save aggregated summary files for each algorithm and function
    print("\nSaving summary results...")
    for algo in algorithms:
        algo_dir = os.path.join(output_base_dir, algo)
        for fid in functions:
            json_path = os.path.join(algo_dir, f"cec2013_f{fid}.json")
            runs_data = results_collected[algo][fid]

            if not runs_data:
                continue

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

            save_json(summary_data, json_path, format_prettier=True)

            print(
                f"[{algo} | F{fid} COMPLETED] Mean Error: {summary_data['mean_error']:.6e} ± {summary_data['std_error']:.6e} -> Saved: {json_path}"
            )

    cleanup_benchmark_csv()
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
    try:
        run_experiments(
            algorithms=args.algorithms,
            functions=args.functions,
            num_runs=args.num_runs,
            max_fes=args.max_fes,
            num_workers=args.workers,
            output_base_dir=args.output_dir,
        )
    finally:
        cleanup_benchmark_csv()
