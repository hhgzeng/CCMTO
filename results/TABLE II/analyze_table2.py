"""
Statistical Analysis and Visualization Script for Table II Reproduction.

Generates:
1. table2_summary.md: Summary table matching Table II layout + detailed benchmark results
2. table2_results.csv: Complete CSV dataset of experimental results
3. algorithm_rankings.png: Visual bar chart of algorithm average rankings
4. benchmark_performance.png: Visual performance comparison across benchmarks F1-F11
5. convergence_curves.png: Convergence curves for representative functions
"""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, ranksums

# Set matplotlib style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 1.0


def load_all_results(results_dir: str) -> Dict[str, Dict[int, Dict]]:
    """Load JSON result files for all algorithms and functions."""
    data = {}
    if not os.path.exists(results_dir):
        return data

    for item in os.listdir(results_dir):
        algo_dir = os.path.join(results_dir, item)
        if not os.path.isdir(algo_dir):
            continue

        algo_name = item
        data[algo_name] = {}

        for f_file in os.listdir(algo_dir):
            if f_file.startswith("cec2013_f") and f_file.endswith(".json"):
                f_path = os.path.join(algo_dir, f_file)
                try:
                    with open(f_path, "r", encoding="utf-8") as f:
                        f_data = json.load(f)
                    fid = f_data.get("func_id")
                    if fid is not None:
                        data[algo_name][fid] = f_data
                except Exception as e:
                    print(f"Warning: Could not load {f_path}: {e}")

    return data


def perform_statistical_analysis(
    data: Dict[str, Dict[int, Dict]],
    target_algo: str = "CCMTO-MTES-DAKG",
    functions: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Perform Wilcoxon rank-sum test and Friedman ranking.
    """
    all_algos = sorted(list(data.keys()))
    if target_algo in all_algos:
        # Move target_algo to front
        all_algos.remove(target_algo)
        all_algos = [target_algo] + all_algos

    if functions is None:
        all_fids = set()
        for algo in all_algos:
            all_fids.update(data[algo].keys())
        functions = sorted(list(all_fids))

    # Detailed per-function records
    detailed_rows = []
    # Matrix for Friedman ranking: shape (num_functions, num_algos)
    mean_errors_matrix = np.zeros((len(functions), len(all_algos)))

    for f_idx, fid in enumerate(functions):
        target_runs = data.get(target_algo, {}).get(fid, {}).get("runs", [])
        target_errors = [r["error"] for r in target_runs] if target_runs else []
        target_mean = np.mean(target_errors) if target_errors else float("inf")

        for a_idx, algo in enumerate(all_algos):
            algo_res = data.get(algo, {}).get(fid, {})
            runs = algo_res.get("runs", [])
            errors = [r["error"] for r in runs] if runs else []

            mean_err = np.mean(errors) if errors else float("inf")
            std_err = np.std(errors, ddof=1) if len(errors) > 1 else 0.0
            best_err = np.min(errors) if errors else float("inf")
            worst_err = np.max(errors) if errors else float("inf")
            median_err = np.median(errors) if errors else float("inf")

            mean_errors_matrix[f_idx, a_idx] = mean_err

            # Statistical comparison vs target_algo
            stat_outcome = "\\ "
            p_val = 1.0

            if algo == target_algo:
                stat_outcome = "\\ "
            else:
                if len(target_errors) >= 3 and len(errors) >= 3:
                    try:
                        # Wilcoxon rank-sum / Mann-Whitney U test
                        stat, p_val = mannwhitneyu(target_errors, errors, alternative="two-sided")
                        if p_val < 0.05:
                            if target_mean < mean_err:
                                stat_outcome = "+"  # CCMTO significantly better
                            else:
                                stat_outcome = "-"  # CCMTO significantly worse
                        else:
                            stat_outcome = "≈"  # No significant difference
                    except Exception:
                        stat_outcome = "≈"
                else:
                    stat_outcome = "≈"

            detailed_rows.append({
                "Function": f"F{fid}",
                "Algorithm": algo,
                "Mean Error": mean_err,
                "Std Error": std_err,
                "Best Error": best_err,
                "Median Error": median_err,
                "Worst Error": worst_err,
                "Wilcoxon Outcome": stat_outcome,
                "p-value": p_val,
            })

    df_detailed = pd.DataFrame(detailed_rows)

    # Compute Friedman rankings per function
    # rankdata: 1 for lowest error (best), ties get average rank
    ranks_matrix = np.zeros_like(mean_errors_matrix)
    for f_idx in range(len(functions)):
        ranks_matrix[f_idx, :] = rankdata(mean_errors_matrix[f_idx, :])

    avg_ranks = np.mean(ranks_matrix, axis=0)

    # Compile Table II summary
    summary_rows = []
    for a_idx, algo in enumerate(all_algos):
        if algo == target_algo:
            plus_cnt, approx_cnt, minus_cnt = "\\", "\\", "\\"
        else:
            algo_details = df_detailed[df_detailed["Algorithm"] == algo]
            plus_cnt = int(np.sum(algo_details["Wilcoxon Outcome"] == "+"))
            approx_cnt = int(np.sum(algo_details["Wilcoxon Outcome"] == "≈"))
            minus_cnt = int(np.sum(algo_details["Wilcoxon Outcome"] == "-"))

        summary_rows.append({
            "Algorithm": algo,
            "CEC2013 (+)": plus_cnt,
            "CEC2013 (≈)": approx_cnt,
            "CEC2013 (-)": minus_cnt,
            "Average Ranking": round(float(avg_ranks[a_idx]), 2),
        })

    df_summary = pd.DataFrame(summary_rows)

    # Rank details dictionary
    ranking_dict = {
        algo: {
            "avg_rank": float(avg_ranks[i]),
            "ranks_per_func": {f"F{functions[f_idx]}": float(ranks_matrix[f_idx, i]) for f_idx in range(len(functions))},
        }
        for i, algo in enumerate(all_algos)
    }

    return df_summary, df_detailed, ranking_dict


def generate_table2_markdown(
    df_summary: pd.DataFrame,
    df_detailed: pd.DataFrame,
    output_path: str,
):
    """Generate professional Markdown report matching Table II format."""
    lines = []
    lines.append("# TABLE II REPRODUCTION RESULTS")
    lines.append("")
    lines.append("## 1. Average Rankings and Statistical Significance Comparison")
    lines.append("")
    func_names_str = ", ".join([f"F{fid}" for fid in sorted(list(set(df_detailed['Function'])), key=lambda x: int(x[1:]))])
    lines.append(f"This table presents the average rankings across the tested CEC2013 LSGO benchmarks ({func_names_str}) and Wilcoxon rank-sum test outcomes (`+/≈/-`) comparing `CCMTO-MTES-DAKG` against each baseline algorithm at significance level $\\alpha = 0.05$.")
    lines.append("")
    lines.append("| Algorithm | CEC2013 (+) | CEC2013 (≈) | CEC2013 (-) | Average Ranking |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")

    for _, row in df_summary.iterrows():
        algo = row["Algorithm"]
        plus = row["CEC2013 (+)"]
        approx = row["CEC2013 (≈)"]
        minus = row["CEC2013 (-)"]
        rank = f"{row['Average Ranking']:.2f}"
        if algo == "CCMTO-MTES-DAKG":
            lines.append(f"| **{algo}** | **{plus}** | **{approx}** | **{minus}** | **{rank}** |")
        else:
            lines.append(f"| {algo} | {plus} | {approx} | {minus} | {rank} |")

    lines.append("")
    lines.append("`+`: Proposed CCMTO-MTES-DAKG is significantly better ($p < 0.05$).")
    lines.append("`≈`: No significant difference ($p \\ge 0.05$).")
    lines.append("`-`: Proposed CCMTO-MTES-DAKG is significantly worse ($p < 0.05$).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Detailed Performance on CEC2013 LSGO Benchmarks (F1 - F11)")
    lines.append("")
    lines.append("Statistical metrics (Mean Error ± Std Error) across 10 independent runs:")
    lines.append("")

    # Pivot table for clean display
    fids = sorted(list(set(df_detailed["Function"])), key=lambda x: int(x[1:]))
    algos = list(df_summary["Algorithm"])

    header = "| Function | " + " | ".join(algos) + " |"
    sep = "| :--- | " + " | ".join([":---:"] * len(algos)) + " |"
    lines.append(header)
    lines.append(sep)

    for fid in fids:
        row_str = f"| **{fid}** |"
        for algo in algos:
            subset = df_detailed[(df_detailed["Function"] == fid) & (df_detailed["Algorithm"] == algo)]
            if not subset.empty:
                m = subset.iloc[0]["Mean Error"]
                s = subset.iloc[0]["Std Error"]
                outcome = subset.iloc[0]["Wilcoxon Outcome"]
                val_str = f"{m:.2e}±{s:.2e}"
            else:
                val_str = "N/A"
            row_str += f" {val_str} |"
        lines.append(row_str)

    lines.append("")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Generated summary report at: {output_path}")


def generate_visualizations(
    df_summary: pd.DataFrame,
    df_detailed: pd.DataFrame,
    data: Dict[str, Dict[int, Dict]],
    output_dir: str,
):
    """Create publication-quality charts for ranking, performance, and convergence."""
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Algorithm Average Rankings Bar Chart
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    df_sorted = df_summary.sort_values(by="Average Ranking", ascending=True)

    algos = df_sorted["Algorithm"].tolist()
    ranks = df_sorted["Average Ranking"].tolist()

    # Color palette: Highlight CCMTO in primary vibrant color
    colors = ["#2b5c8f" if "CCMTO" in a else "#7fa8d1" if "EDG" in a else "#d9822b" if "GTDE" in a else "#48a999" for a in algos]

    bars = ax.barh(algos, ranks, color=colors, height=0.55, edgecolor="none")
    ax.invert_yaxis()  # Best on top

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.08,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold" if "CCMTO" in algos[int(bar.get_y())] else "normal",
            color="#222222",
        )

    ax.set_xlabel("Average Friedman Rank (Lower is Better)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("Algorithm Average Rankings on CEC2013 LSGO (Table II Reproduction)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, max(ranks) + 1.2)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()

    rank_fig_path = os.path.join(output_dir, "algorithm_rankings.png")
    fig.savefig(rank_fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved ranking chart to: {rank_fig_path}")

    # -------------------------------------------------------------
    # 2. Benchmark Performance Comparison (Log10 Mean Error)
    # -------------------------------------------------------------
    fids = sorted(list(set(df_detailed["Function"])), key=lambda x: int(x[1:]))
    algos_list = list(df_summary["Algorithm"])

    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    x = np.arange(len(fids))
    width = 0.8 / len(algos_list)

    palette = {
        "CCMTO-MTES-DAKG": "#1f77b4",
        "CMAES-EDG": "#aec7e8",
        "DECC-ERDG": "#ff7f0e",
        "GTDE": "#2ca02c",
        "SDLSO": "#9467bd",
    }

    for idx, algo in enumerate(algos_list):
        algo_data = df_detailed[df_detailed["Algorithm"] == algo]
        log_errors = []
        for fid in fids:
            row = algo_data[algo_data["Function"] == fid]
            if not row.empty:
                val = row.iloc[0]["Mean Error"]
                # Log10 scale with safeguard
                log_val = np.log10(max(val, 1e-16))
            else:
                log_val = 0
            log_errors.append(log_val)

        offset = (idx - len(algos_list) / 2 + 0.5) * width
        color = palette.get(algo, "#888888")
        ax.bar(x + offset, log_errors, width, label=algo, color=color, alpha=0.9, edgecolor="white", linewidth=0.5)

    ax.set_ylabel(r"$\log_{10}(\mathrm{Mean\ Error})$", fontsize=12, fontweight="bold")
    ax.set_title("Performance Comparison across CEC2013 LSGO Benchmarks (F1 - F11)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(fids, fontsize=11, fontweight="bold")
    ax.legend(frameon=True, fontsize=10, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    perf_fig_path = os.path.join(output_dir, "benchmark_performance.png")
    fig.savefig(perf_fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved benchmark performance chart to: {perf_fig_path}")

    # -------------------------------------------------------------
    # 3. Convergence Curves for Tested Functions (e.g. F1, F2, F4, F5, F9)
    # -------------------------------------------------------------
    available_fids = sorted(list(set(int(f[1:]) for f in df_detailed["Function"])))
    rep_functions = available_fids[:6] if len(available_fids) <= 6 else [1, 2, 4, 5, 9]

    if rep_functions:
        num_cols = min(2, len(rep_functions))
        num_rows = int(np.ceil(len(rep_functions) / num_cols))
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(7 * num_cols, 5 * num_rows), dpi=300)
        axes = np.atleast_1d(axes).flatten()

        for ax_idx, fid in enumerate(rep_functions):
            ax = axes[ax_idx]
            for algo in algos_list:
                f_data = data.get(algo, {}).get(fid, {})
                runs = f_data.get("runs", [])
                if not runs:
                    continue

                # Find run with median performance for representative convergence curve
                sorted_runs = sorted(runs, key=lambda r: r["error"])
                median_run = sorted_runs[len(sorted_runs) // 2]
                history = median_run.get("history", [])

                if history:
                    fes = [h[0] for h in history]
                    # Compute error to best known
                    best_known = median_run.get("best_known", 0.0)
                    errs = [max(1e-16, abs(h[1] - best_known)) for h in history]

                    color = palette.get(algo, "#888888")
                    lw = 2.5 if "CCMTO" in algo else 1.6
                    ls = "-" if "CCMTO" in algo else "--"
                    ax.plot(fes, errs, label=algo, color=color, linewidth=lw, linestyle=ls)

            ax.set_yscale("log")
            ax.set_xlabel("Fitness Evaluations (FEs)", fontsize=10, fontweight="bold")
            ax.set_ylabel("Error to Optimum", fontsize=10, fontweight="bold")
            ax.set_title(f"CEC2013 Function F{fid} Convergence", fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.5)
            if ax_idx == 0:
                ax.legend(frameon=True, fontsize=9, loc="upper right")

        # Hide any unused subplots
        for ax_idx in range(len(rep_functions), len(axes)):
            axes[ax_idx].set_visible(False)

        plt.tight_layout()
        conv_fig_path = os.path.join(output_dir, "convergence_curves.png")
        fig.savefig(conv_fig_path, dpi=300)
        plt.close(fig)
        print(f"Saved convergence curves to: {conv_fig_path}")


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "TABLE II")
    results_dir = os.path.abspath(results_dir)

    print(f"Loading results from: {results_dir}")
    data = load_all_results(results_dir)
    if not data:
        print("No result data found in results/TABLE II! Please run experiments first.")
        return

    print(f"Found results for algorithms: {list(data.keys())}")
    df_summary, df_detailed, ranking_dict = perform_statistical_analysis(data)

    print("\n" + "=" * 60)
    print("SUMMARY OF AVERAGE RANKINGS (TABLE II)")
    print("=" * 60)
    print(df_summary.to_string(index=False))
    print("=" * 60 + "\n")

    # Output paths
    summary_md_path = os.path.join(results_dir, "table2_summary.md")
    results_csv_path = os.path.join(results_dir, "table2_results.csv")

    # Generate Markdown & CSV
    generate_table2_markdown(df_summary, df_detailed, summary_md_path)
    df_detailed.to_csv(results_csv_path, index=False, encoding="utf-8")
    print(f"Saved detailed CSV results to: {results_csv_path}")

    # Generate Visualizations
    generate_visualizations(df_summary, df_detailed, data, results_dir)
    print("All Table II reproduction analysis and figures successfully created!")


if __name__ == "__main__":
    main()
