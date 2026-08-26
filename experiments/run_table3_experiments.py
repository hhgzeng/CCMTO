"""
Experiment runner for Table III reproduction in CCMTO paper.

Conducts parameter sensitivity analysis across parameters:
1. n_sub: [2, 3, 5, 7] (Tested on CEC2013 F4-F7)
2. d_max: [1, 2, 4, limitless] (Tested on CEC2013 F4-F7)

Each setting is evaluated for 10 independent runs per benchmark function.
Results are saved under:
results/TABLE III/<parameter_name>/<setting_value>/cec2013_f<func_id>.json

Supports multi-process parallelism for fast execution.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List

import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cec2013lsgo.cec2013 import Benchmark
from src.CCMTO.CCMTO import CCMTO
from src.utils import cleanup_benchmark_csv, register_csv_cleanup, save_json

register_csv_cleanup()


# Default baseline parameters (Setting with best performance in Table I & Table III)
DEFAULT_PARAMS = {
    "n_sub": 5,
    "d_max": 2.0,
    "tau": 1,
    "fre_ratio": 0.1,  # 0.1Maxgen
}

# Parameter definitions and tested settings for Table III (F4-F7)
PARAMETER_CONFIGS = {
    "n_sub": {
        "settings": [2, 3, 5, 7],  # Settings 2, 3, 5, 7 as requested
        "functions": [4, 5, 6, 7],  # CEC2013 F4 to F7
        "default_setting": 5,
    },
    "d_max": {
        "settings": ["1", "2", "4", "limitless"],
        "functions": [4, 5, 6, 7],  # CEC2013 F4 to F7
        "default_setting": "2",
    },
}


def parse_d_max(d_max_val: Any) -> float:
    """Parse d_max value, supporting 'limitless' / 'inf' as float('inf')."""
    if str(d_max_val).lower() in ["limitless", "inf", "none"]:
        return float("inf")
    return float(d_max_val)


def parse_fre(fre_val: Any) -> float:
    """Parse frequency parameter (e.g. '0.1Maxgen', 'everygen', '0.05Maxgen', 0.1)."""
    val_str = str(fre_val).lower().replace("maxgen", "").strip()
    if val_str in ["everygen", "every", "0", "0.0"]:
        return 0.0  # fre = 1 every generation
    return float(val_str)


def run_single_run(
    func_id: int,
    run_idx: int,
    seed: int,
    max_fes: int,
    n_sub: int,
    d_max: float,
    tau: int,
    fre_ratio: float,
) -> Dict[str, Any]:
    """Execute a single run of CCMTO with specified parameter settings."""
    np.random.seed(seed)
    bench = Benchmark()
    info = bench.get_info(func_id)
    func = bench.get_function(func_id)

    dim = info["dimension"]
    lower = info["lower"]
    upper = info["upper"]
    best_known = info["best"]

    try:
        start_time = time.time()
        solver = CCMTO(
            func=func,
            dim=dim,
            lower=lower,
            upper=upper,
            max_fes=max_fes,
            n_sub=n_sub,
            d_max=d_max,
            tau=tau,
            fre_ratio=fre_ratio,
            verbose=False,
        )
        result = solver.optimize()
        elapsed_time = time.time() - start_time

        best_f = float(result["best_f"])
        error = float(abs(best_f - best_known))
        total_fes = int(result["fes"])

        # Sample history to keep json lightweight (max 100 points)
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
    finally:
        cleanup_benchmark_csv()


def run_parameter_experiments(
    param_name: str,
    setting_val: Any,
    functions: List[int],
    num_runs: int = 10,
    max_fes: int = 1_000_000,
    base_seed: int = 42,
    num_workers: int = 8,
    output_dir: str = "results/TABLE III",
):
    """Run all benchmarks for a specific parameter setting."""
    setting_str = str(setting_val)
    target_dir = os.path.join(output_dir, param_name, setting_str)
    os.makedirs(target_dir, exist_ok=True)

    # Determine parameter values for solver
    n_sub = DEFAULT_PARAMS["n_sub"]
    d_max = DEFAULT_PARAMS["d_max"]
    tau = DEFAULT_PARAMS["tau"]
    fre_ratio = DEFAULT_PARAMS["fre_ratio"]

    if param_name == "n_sub":
        n_sub = int(setting_val)
    elif param_name == "d_max":
        d_max = parse_d_max(setting_val)
    elif param_name == "tau":
        tau = int(setting_val)
    elif param_name == "fre":
        fre_ratio = parse_fre(setting_val)

    print(f"\n======================================================================")
    print(f"Running Table III: Parameter [{param_name}] = {setting_str}")
    print(f"Configuration: n_sub={n_sub}, d_max={d_max}, tau={tau}, fre_ratio={fre_ratio}")
    print(f"Functions ({len(functions)}): {functions} | Runs: {num_runs} | MaxFEs: {max_fes:,}")
    print(f"Target Directory: {target_dir}")
    print(f"======================================================================")

    for fid in functions:
        json_path = os.path.join(target_dir, f"cec2013_f{fid}.json")

        # Checkpoint: skip if already completed
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if len(existing.get("runs", [])) >= num_runs and existing.get("max_fes") == max_fes:
                    print(f"  [Skipping] F{fid} already completed ({num_runs} runs with MaxFEs={max_fes:,}).")
                    continue
            except Exception:
                pass

        print(f"\n>>> [{param_name}={setting_str}] Executing CEC2013 F{fid} ({num_runs} runs)...")
        tasks = [
            (fid, r + 1, base_seed + r * 100 + fid, max_fes, n_sub, d_max, tau, fre_ratio)
            for r in range(num_runs)
        ]

        runs_data = []
        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=min(num_workers, num_runs)) as executor:
                futures = {
                    executor.submit(run_single_run, *task): task[1]
                    for task in tasks
                }
                for future in as_completed(futures):
                    run_idx = futures[future]
                    try:
                        res = future.result()
                        runs_data.append(res)
                        print(
                            f"    [Run {res['run_idx']:2d}/{num_runs}] "
                            f"Error: {res['error']:.6e} | Time: {res['elapsed_seconds']:.2f}s | FEs: {res['total_fes']:,}"
                        )
                    except Exception as e:
                        print(f"    [ERROR in Run {run_idx}]: {e}")
                    finally:
                        cleanup_benchmark_csv()
        else:
            for task in tasks:
                res = run_single_run(*task)
                runs_data.append(res)
                print(
                    f"    [Run {res['run_idx']:2d}/{num_runs}] "
                    f"Error: {res['error']:.6e} | Time: {res['elapsed_seconds']:.2f}s | FEs: {res['total_fes']:,}"
                )
                cleanup_benchmark_csv()

        runs_data.sort(key=lambda x: x["run_idx"])

        errors = [r["error"] for r in runs_data]
        fitnesses = [r["best_fitness"] for r in runs_data]
        times = [r["elapsed_seconds"] for r in runs_data]

        bench = Benchmark()
        info = bench.get_info(fid)

        summary_data = {
            "parameter": param_name,
            "setting": setting_str,
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
            f"  [Completed F{fid}] Mean Error: {summary_data['mean_error']:.6e} ± {summary_data['std_error']:.6e} -> {json_path}"
        )

    cleanup_benchmark_csv()


def sync_baseline_results(output_dir: str, num_runs: int, max_fes: int):
    """
    Ensure the baseline configuration (n_sub=5, d_max=2)
    is synchronized between n_sub/5 and d_max/2 directories to prevent redundant computation.
    """
    nsub5_dir = os.path.join(output_dir, "n_sub", "5")
    dmax2_dir = os.path.join(output_dir, "d_max", "2")

    # If n_sub/5 exists and d_max/2 doesn't, copy to d_max/2
    if os.path.exists(nsub5_dir):
        os.makedirs(dmax2_dir, exist_ok=True)
        for fid in [4, 5, 6, 7]:
            src_file = os.path.join(nsub5_dir, f"cec2013_f{fid}.json")
            dst_file = os.path.join(dmax2_dir, f"cec2013_f{fid}.json")
            if os.path.exists(src_file) and not os.path.exists(dst_file):
                try:
                    with open(src_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    data["parameter"] = "d_max"
                    data["setting"] = "2"
                    save_json(data, dst_file, format_prettier=True)
                except Exception as e:
                    print(f"Warning syncing baseline to {dst_file}: {e}")

    # If d_max/2 exists and n_sub/5 doesn't, copy to n_sub/5
    elif os.path.exists(dmax2_dir):
        os.makedirs(nsub5_dir, exist_ok=True)
        for fid in [4, 5, 6, 7]:
            src_file = os.path.join(dmax2_dir, f"cec2013_f{fid}.json")
            dst_file = os.path.join(nsub5_dir, f"cec2013_f{fid}.json")
            if os.path.exists(src_file) and not os.path.exists(dst_file):
                try:
                    with open(src_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    data["parameter"] = "n_sub"
                    data["setting"] = "5"
                    save_json(data, dst_file, format_prettier=True)
                except Exception as e:
                    print(f"Warning syncing baseline to {dst_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run Table III Parameter Sensitivity Reproduction (n_sub [2,3,5,7] & d_max on CEC2013 F4-F7)")
    parser.add_argument(
        "--parameters",
        nargs="+",
        default=["n_sub", "d_max"],
        choices=["n_sub", "d_max"],
        help="Parameters to evaluate",
    )
    parser.add_argument(
        "--functions",
        nargs="+",
        type=int,
        default=[4, 5, 6, 7],
        help="CEC2013 Function IDs (default: 4 5 6 7)",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="Number of runs per function (default: 10 as requested)",
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
        default=min(10, os.cpu_count() or 4),
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/TABLE III",
        help="Root output directory for results",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("STARTING TABLE III PARAMETER SENSITIVITY EXPERIMENTS (n_sub [2,3,5,7] & d_max on F4-F7)")
    print(f"Selected Parameters: {args.parameters}")
    print(f"Functions: {args.functions}")
    print(f"Runs per Benchmark: {args.num_runs} | MaxFEs: {args.max_fes:,} | Workers: {args.workers}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 80)

    try:
        # First, run baseline (n_sub=5)
        if "n_sub" in args.parameters:
            run_parameter_experiments(
                param_name="n_sub",
                setting_val=5,
                functions=args.functions,
                num_runs=args.num_runs,
                max_fes=args.max_fes,
                num_workers=args.workers,
                output_dir=args.output_dir,
            )
            sync_baseline_results(args.output_dir, args.num_runs, args.max_fes)

        # Run remaining settings for all selected parameters
        for param in args.parameters:
            cfg = PARAMETER_CONFIGS[param]
            for setting in cfg["settings"]:
                if param == "n_sub" and setting == 5:
                    continue  # already ran above

                run_parameter_experiments(
                    param_name=param,
                    setting_val=setting,
                    functions=args.functions,
                    num_runs=args.num_runs,
                    max_fes=args.max_fes,
                    num_workers=args.workers,
                    output_dir=args.output_dir,
                )
                # Sync baseline if setting is the default
                sync_baseline_results(args.output_dir, args.num_runs, args.max_fes)

        print("\n" + "=" * 80)
        print("ALL TABLE III EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
    finally:
        cleanup_benchmark_csv()


if __name__ == "__main__":
    main()
