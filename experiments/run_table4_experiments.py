"""
Experiment runner for Table IV (Representative Component Ablation Analysis) in CCMTO paper.

Evaluates on selected CEC2013 LSGO benchmarks:
Functions: [1, 2, 4, 5, 9]

Modules & Selected Representative Algorithms:
1. Module 1 (Resource Allocation): CBCC1, CCFR3, CCMTO-MTES-DAKG
2. Module 2 (EMTO Algorithms): CCMTO-MaTDE, CCMTO-MTES-DAKG
3. Module 3 (Component Ablation): wo-DA, wo-DT-DoS, wo-AS-SaS, wo-SD, CCMTO-MTES-DAKG

Parameters:
- Runs per benchmark: 5
- MaxFEs: 1,000,000
- Parallel workers: 5

Results are saved under:
results/TABLE IV/<module_name>/<algorithm_name>/cec2013_f<func_id>.json

Supports multi-process parallelism for fast execution.
"""

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cec2013lsgo.cec2013 import Benchmark
from decomposition.precompute_edg import get_or_compute_edg_subproblems

from src.CCMTO.CCMTO import CCMTO
from baselines.resource_allocation import CBCC1, CCFR3
from baselines.emto_algorithms import CCMTO_MaTDE
from baselines.component_ablation import WO_DA, WO_DT_DoS, WO_AS_SaS, WO_SD


# Selected benchmark functions (F1, F2, F4, F5, F9)
DEFAULT_FUNCTIONS = [1, 2, 4, 5, 9]

# Module algorithms mapping based on user selection
MODULE_ALGORITHMS = {
    "resource_allocation": {
        "CBCC1": CBCC1,
        "CCFR3": CCFR3,
        "CCMTO-MTES-DAKG": CCMTO,
    },
    "emto_algorithms": {
        "CCMTO-MaTDE": CCMTO_MaTDE,
        "CCMTO-MTES-DAKG": CCMTO,
    },
    "component_ablation": {
        "wo-DA": WO_DA,
        "wo-DT-DoS": WO_DT_DoS,
        "wo-AS-SaS": WO_AS_SaS,
        "wo-SD": WO_SD,
        "CCMTO-MTES-DAKG": CCMTO,
    },
}


def create_solver(
    algo_name: str,
    algo_cls: Any,
    func,
    dim: int,
    lower: float,
    upper: float,
    max_fes: int,
    subproblems: Optional[List[List[int]]] = None,
):
    """Instantiate optimizer solver instance with standardized parameters."""
    return algo_cls(
        func=func,
        dim=dim,
        lower=lower,
        upper=upper,
        max_fes=max_fes,
        n_sub=5,
        d_max=2.0,
        custom_subproblems=subproblems,
        verbose=False,
    )


def run_single_run(
    algo_name: str,
    algo_cls: Any,
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
    solver = create_solver(algo_name, algo_cls, func, dim, lower, upper, max_fes, subproblems)
    result = solver.optimize()
    elapsed_time = time.time() - start_time

    best_f = float(result["best_f"])
    error = float(abs(best_f - best_known))
    total_fes = int(result["fes"])

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


def run_module_algorithm(
    module_name: str,
    algo_name: str,
    algo_cls: Any,
    functions: List[int],
    num_runs: int = 5,
    max_fes: int = 1_000_000,
    base_seed: int = 42,
    num_workers: int = 5,
    output_base_dir: str = "results/TABLE IV",
):
    """Run all benchmark functions for a specific algorithm within a module."""
    algo_dir = os.path.join(output_base_dir, module_name, algo_name)
    os.makedirs(algo_dir, exist_ok=True)

    print(f"\n======================================================================")
    print(f"Running [{module_name.upper()}] Algorithm: [{algo_name}]")
    print(f"Functions ({len(functions)}): {functions} | Runs: {num_runs} | MaxFEs: {max_fes:,}")
    print(f"Target Directory: {algo_dir}")
    print(f"======================================================================")

    for fid in functions:
        json_path = os.path.join(algo_dir, f"cec2013_f{fid}.json")

        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if (
                    len(existing.get("runs", [])) >= num_runs
                    and existing.get("max_fes") == max_fes
                    and not np.isinf(existing.get("mean_error", float("inf")))
                ):
                    print(f"  [Skipping] [{algo_name}] F{fid} already completed ({num_runs} runs).")
                    continue
            except Exception:
                pass

        print(f"\n>>> [{algo_name}] Executing CEC2013 F{fid} ({num_runs} runs, MaxFEs={max_fes:,})...")
        subproblems, _ = get_or_compute_edg_subproblems(fid)

        tasks = [
            (algo_name, algo_cls, fid, r + 1, base_seed + r * 100 + fid, max_fes, subproblems)
            for r in range(num_runs)
        ]

        runs_data = []
        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=min(num_workers, num_runs)) as executor:
                futures = {
                    executor.submit(run_single_run, *task): task[3]
                    for task in tasks
                }
                for future in as_completed(futures):
                    run_idx = futures[future]
                    try:
                        res = future.result()
                        runs_data.append(res)
                        print(
                            f"    [{algo_name} | F{fid} | Run {res['run_idx']:2d}/{num_runs}] "
                            f"Error: {res['error']:.6e} | Time: {res['elapsed_seconds']:.2f}s | FEs: {res['total_fes']:,}"
                        )
                    except Exception as e:
                        print(f"    [ERROR in Run {run_idx}]: {e}")
        else:
            for task in tasks:
                res = run_single_run(*task)
                runs_data.append(res)
                print(
                    f"    [{algo_name} | F{fid} | Run {res['run_idx']:2d}/{num_runs}] "
                    f"Error: {res['error']:.6e} | Time: {res['elapsed_seconds']:.2f}s | FEs: {res['total_fes']:,}"
                )

        runs_data.sort(key=lambda x: x["run_idx"])

        errors = [r["error"] for r in runs_data]
        fitnesses = [r["best_fitness"] for r in runs_data]
        times = [r["elapsed_seconds"] for r in runs_data]

        bench = Benchmark()
        info = bench.get_info(fid)

        summary_data = {
            "module": module_name,
            "algorithm": algo_name,
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
            f"  [{algo_name} | F{fid} COMPLETED] Mean Error: {summary_data['mean_error']:.6e} ± {summary_data['std_error']:.6e} -> {json_path}"
        )


def sync_proposed_baseline(output_base_dir: str, num_runs: int, max_fes: int, functions: List[int]):
    """
    Syncs CCMTO-MTES-DAKG results across all three module directories to avoid duplicate runs.
    """
    source_mod = "resource_allocation"
    src_dir = os.path.join(output_base_dir, source_mod, "CCMTO-MTES-DAKG")
    if not os.path.exists(src_dir):
        return

    target_mods = ["emto_algorithms", "component_ablation"]
    for tgt_mod in target_mods:
        dst_dir = os.path.join(output_base_dir, tgt_mod, "CCMTO-MTES-DAKG")
        os.makedirs(dst_dir, exist_ok=True)
        for fid in functions:
            src_file = os.path.join(src_dir, f"cec2013_f{fid}.json")
            dst_file = os.path.join(dst_dir, f"cec2013_f{fid}.json")
            if os.path.exists(src_file) and not os.path.exists(dst_file):
                try:
                    with open(src_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    data["module"] = tgt_mod
                    with open(dst_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                except Exception as e:
                    print(f"Warning syncing baseline to {dst_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run Table IV Component Ablation Reproduction Experiments")
    parser.add_argument(
        "--modules",
        nargs="+",
        default=["resource_allocation", "emto_algorithms", "component_ablation"],
        choices=["resource_allocation", "emto_algorithms", "component_ablation"],
        help="Modules to evaluate",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help="Specific algorithms to run (e.g. --algorithms wo-DA wo-DT-DoS wo-AS-SaS)",
    )
    parser.add_argument(
        "--functions",
        nargs="+",
        type=int,
        default=DEFAULT_FUNCTIONS,
        help="CEC2013 function IDs (default: [1, 2, 4, 5, 9])",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of independent runs per benchmark function (default: 5)",
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
        default=min(5, os.cpu_count() or 4),
        help="Number of parallel worker processes (default: 5)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/TABLE IV",
        help="Root output directory for results",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("STARTING TABLE IV REPRESENTATIVE ABLATION EXPERIMENTS")
    print(f"Modules: {args.modules}")
    if args.algorithms:
        print(f"Selected Algorithms: {args.algorithms}")
    print(f"Functions ({len(args.functions)}): {args.functions}")
    print(f"Runs per benchmark: {args.num_runs} | MaxFEs: {args.max_fes:,} | Workers: {args.workers}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 80)

    # First, run baseline CCMTO-MTES-DAKG in resource_allocation if requested
    if "resource_allocation" in args.modules and (args.algorithms is None or "CCMTO-MTES-DAKG" in args.algorithms):
        run_module_algorithm(
            module_name="resource_allocation",
            algo_name="CCMTO-MTES-DAKG",
            algo_cls=CCMTO,
            functions=args.functions,
            num_runs=args.num_runs,
            max_fes=args.max_fes,
            num_workers=args.workers,
            output_base_dir=args.output_dir,
        )
        sync_proposed_baseline(args.output_dir, args.num_runs, args.max_fes, args.functions)

    for mod_name in args.modules:
        algos_dict = MODULE_ALGORITHMS[mod_name]
        for algo_name, algo_cls in algos_dict.items():
            if args.algorithms is not None and algo_name not in args.algorithms:
                continue

            if algo_name == "CCMTO-MTES-DAKG" and mod_name == "resource_allocation":
                continue  # already ran above

            run_module_algorithm(
                module_name=mod_name,
                algo_name=algo_name,
                algo_cls=algo_cls,
                functions=args.functions,
                num_runs=args.num_runs,
                max_fes=args.max_fes,
                num_workers=args.workers,
                output_base_dir=args.output_dir,
            )
            if algo_name == "CCMTO-MTES-DAKG":
                sync_proposed_baseline(args.output_dir, args.num_runs, args.max_fes, args.functions)

    sync_proposed_baseline(args.output_dir, args.num_runs, args.max_fes, args.functions)

    print("\n" + "=" * 80)
    print("ALL TABLE IV EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
