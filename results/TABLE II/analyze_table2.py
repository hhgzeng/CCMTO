"""
Statistical Analysis and Visualization Script for Table II Reproduction in CCMTO paper.

Generates:
1. table2_summary.md: Summary table matching Table II layout + detailed benchmark results
2. algorithm_rankings.csv: Algorithm ranking & Wilcoxon significance test summary table
3. algorithm_scores.csv: Algorithm benchmark scores (Mean Error ± Std Error)
4. table2_results.csv: Complete raw dataset of experimental results across all runs
5. algorithm_rankings.png (and algorithm_rankings_table.png): Academic 3-line table for rankings
6. algorithm_scores.png (and algorithm_scores_table.png): Academic 3-line table for benchmark scores
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata

# Publication-grade typography styling matching academic standards
plt.rcParams["font.sans-serif"] = [
    "DejaVu Serif",
    "STIXGeneral",
    "DejaVu Sans",
    "Arial",
    "SimHei",
    "STHeiti",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "cm"

FONT_SERIF = "DejaVu Serif"


def format_sci(mean_val: float, std_val: float) -> str:
    """Format scientific notation (Mean ± Std) matching academic standards."""
    if not np.isfinite(mean_val):
        return "N/A"
    if not np.isfinite(std_val):
        std_val = 0.0
    return f"{mean_val:.2e}±{std_val:.2e}"


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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Perform Wilcoxon rank-sum test and Friedman ranking.
    Returns:
    - df_summary: Algorithm ranking summary table
    - df_detailed: Detailed per-function records
    - df_rankings: Formatted rankings DataFrame matching Table 1 layout
    - df_scores: Formatted scores DataFrame matching Table 2 layout
    - plot_data: Dictionary containing precomputed stats for figure generation
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

    algo_labels = {algo: f"{algo} (Ours)" if algo == target_algo else algo for algo in all_algos}
    func_names = [f"F{fid}" for fid in functions]

    # Detailed per-function records
    detailed_rows = []
    # Matrix for Friedman ranking: shape (num_functions, num_algos)
    mean_errors_matrix = np.zeros((len(functions), len(all_algos)))

    means_dict = {f"F{fid}": {} for fid in functions}
    stds_dict = {f"F{fid}": {} for fid in functions}

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
            means_dict[f"F{fid}"][algo] = mean_err
            stds_dict[f"F{fid}"][algo] = std_err

            # Statistical comparison vs target_algo
            stat_outcome = "\\ "
            p_val = 1.0

            if algo == target_algo:
                stat_outcome = "\\ "
            else:
                if len(target_errors) >= 3 and len(errors) >= 3:
                    try:
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

    # Compute Friedman rankings per function (1 for lowest error)
    ranks_matrix = np.zeros_like(mean_errors_matrix)
    for f_idx in range(len(functions)):
        ranks_matrix[f_idx, :] = rankdata(mean_errors_matrix[f_idx, :])

    avg_ranks = np.mean(ranks_matrix, axis=0)

    # Compile Table II summary & rankings table
    summary_rows = []
    ranking_csv_rows = []
    wilcoxon_stats = {}

    for a_idx, algo in enumerate(all_algos):
        label = algo_labels[algo]
        if algo == target_algo:
            plus_cnt, approx_cnt, minus_cnt = "\\", "\\", "\\"
        else:
            algo_details = df_detailed[df_detailed["Algorithm"] == algo]
            plus_cnt = int(np.sum(algo_details["Wilcoxon Outcome"] == "+"))
            approx_cnt = int(np.sum(algo_details["Wilcoxon Outcome"] == "≈"))
            minus_cnt = int(np.sum(algo_details["Wilcoxon Outcome"] == "-"))
            wilcoxon_stats[algo] = {"+": plus_cnt, "≈": approx_cnt, "-": minus_cnt}

        r_val = round(float(avg_ranks[a_idx]), 2)
        summary_rows.append({
            "Algorithm": algo,
            "CEC2013 (+)": plus_cnt,
            "CEC2013 (≈)": approx_cnt,
            "CEC2013 (-)": minus_cnt,
            "Average Ranking": r_val,
        })

        ranking_csv_rows.append({
            "Algorithm": label,
            "CEC2013 (+)": plus_cnt,
            "CEC2013 (≈)": approx_cnt,
            "CEC2013 (-)": minus_cnt,
            "Average Ranking": f"{r_val:.2f}",
        })

    df_summary = pd.DataFrame(summary_rows)
    df_rankings = pd.DataFrame(ranking_csv_rows)

    # Compile Scores table (Function/Benchmark as rows, Algorithms as columns)
    score_csv_rows = []
    for fid in functions:
        fn = f"F{fid}"
        row_dict = {"Function": fn}
        for algo in all_algos:
            label = algo_labels[algo]
            m = means_dict[fn][algo]
            s = stds_dict[fn][algo]
            row_dict[label] = format_sci(m, s)
        score_csv_rows.append(row_dict)

    df_scores = pd.DataFrame(score_csv_rows)

    # Calculate best value per function
    best_means = {
        f"F{fid}": min(means_dict[f"F{fid}"][a] for a in all_algos)
        for fid in functions
    }

    plot_data = {
        "algorithms": all_algos,
        "algo_labels": algo_labels,
        "target_algo": target_algo,
        "functions": func_names,
        "means": means_dict,
        "stds": stds_dict,
        "best_means": best_means,
        "avg_ranks": {algo: float(avg_ranks[i]) for i, algo in enumerate(all_algos)},
        "best_rank": float(np.min(avg_ranks)),
        "wilcoxon": wilcoxon_stats,
    }

    return df_summary, df_detailed, df_rankings, df_scores, plot_data


def generate_table2_markdown(
    df_rankings: pd.DataFrame,
    df_scores: pd.DataFrame,
    plot_data: Dict[str, Any],
    output_path: str,
):
    """Generate professional Markdown report matching Table II format."""
    lines = []
    lines.append("# TABLE II REPRODUCTION RESULTS")
    lines.append("")
    lines.append("## 1. Average Rankings and Statistical Significance Comparison")
    lines.append("")
    func_names_str = ", ".join(plot_data["functions"])
    target_algo = plot_data["target_algo"]
    lines.append(
        f"This table presents the average rankings across the tested CEC2013 LSGO benchmarks ({func_names_str}) "
        f"and Wilcoxon rank-sum test outcomes (`+/≈/-`) comparing `{target_algo}` against each baseline algorithm "
        f"at significance level $\\alpha = 0.05$."
    )
    lines.append("")
    lines.append("| Algorithm | CEC2013 (+) | CEC2013 (≈) | CEC2013 (-) | Average Ranking |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")

    best_rank = plot_data["best_rank"]
    for _, row in df_rankings.iterrows():
        algo = row["Algorithm"]
        plus = row["CEC2013 (+)"]
        approx = row["CEC2013 (≈)"]
        minus = row["CEC2013 (-)"]
        rank_val = float(row["Average Ranking"])
        rank_str = f"{rank_val:.2f}"
        is_best = abs(rank_val - best_rank) < 1e-4

        rank_display = f"**{rank_str}**" if is_best else rank_str
        if target_algo in algo:
            lines.append(f"| **{algo}** | {plus} | {approx} | {minus} | {rank_display} |")
        else:
            lines.append(f"| {algo} | {plus} | {approx} | {minus} | {rank_display} |")

    lines.append("")
    lines.append(f"`+`: Proposed {target_algo} is significantly better ($p < 0.05$).")
    lines.append("`≈`: No significant difference ($p \\ge 0.05$).")
    lines.append(f"`-`: Proposed {target_algo} is significantly worse ($p < 0.05$).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Detailed Performance on CEC2013 LSGO Benchmarks (F1 - F11)")
    lines.append("")
    lines.append("Statistical metrics (Mean Error ± Std Error) across 10 independent runs (best values in **bold**):")
    lines.append("")

    # Scores table header (Function as rows, Algorithms as columns)
    algos = plot_data["algorithms"]
    algo_labels = plot_data["algo_labels"]
    func_cols = plot_data["functions"]
    header = "| Function | " + " | ".join([f"**{algo_labels[a]}**" if a == target_algo else algo_labels[a] for a in algos]) + " |"
    sep = "| :--- | " + " | ".join([":---:"] * len(algos)) + " |"
    lines.append(header)
    lines.append(sep)

    for fn in func_cols:
        row_str = f"| **{fn}** |"
        best_m = plot_data["best_means"][fn]
        for algo in algos:
            m = plot_data["means"][fn][algo]
            s = plot_data["stds"][fn][algo]
            val_txt = format_sci(m, s)
            if abs(m - best_m) < 1e-12 or (best_m != 0 and abs((m - best_m) / best_m) < 1e-6):
                val_txt = f"**{val_txt}**"
            row_str += f" {val_txt} |"
        lines.append(row_str)

    lines.append("")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Generated summary report at: {output_path}")


def render_rankings_table(
    plot_data: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Render academic 3-line table for algorithm rankings & Wilcoxon test.
    Single-tier header without benchmark suite super-header.
    """
    algorithms = plot_data["algorithms"]
    algo_labels = plot_data["algo_labels"]
    target_algo = plot_data["target_algo"]
    avg_ranks = plot_data["avg_ranks"]
    best_rank = plot_data["best_rank"]
    wilcoxon = plot_data["wilcoxon"]

    n_rows = len(algorithms)
    fig, ax = plt.subplots(figsize=(9.0, 1.4 + 0.42 * n_rows), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.axis("off")

    # 5 columns: Algorithm | + | ≈ | - | Ranking
    col_x = [0.18, 0.45, 0.58, 0.71, 0.86]

    top_y = 0.86
    header_y = 0.70
    midrule_y = 0.58
    row_height = 0.48 / max(n_rows, 1)

    # Top line (Thick Top Rule)
    ax.plot([0.04, 0.96], [top_y, top_y], color="black", linewidth=2.2, clip_on=False)

    # Single Header row (bold font)
    font_header = {"fontsize": 11.0, "fontfamily": FONT_SERIF, "fontweight": "bold", "color": "#000000"}
    ax.text(col_x[0], header_y, "Algorithm", ha="center", va="center", **font_header)
    sub_headers = ["+", r"$\approx$", "-", "Ranking"]
    for idx, text in enumerate(sub_headers):
        ax.text(col_x[idx + 1], header_y, text, ha="center", va="center", **font_header)

    # Midline
    ax.plot([0.04, 0.96], [midrule_y, midrule_y], color="black", linewidth=1.0, clip_on=False)

    # Data rows
    current_y = midrule_y - 0.08
    for algo in algorithms:
        label = algo_labels[algo]
        is_target = algo == target_algo

        ax.text(
            col_x[0],
            current_y,
            label,
            ha="center",
            va="center",
            fontsize=10.5,
            fontfamily=FONT_SERIF,
            color="#000000",
        )

        if is_target:
            for c_idx in (1, 2, 3):
                ax.text(
                    col_x[c_idx],
                    current_y,
                    r"$\backslash$",
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    fontfamily=FONT_SERIF,
                    color="#000000",
                )
        else:
            ax.text(
                col_x[1],
                current_y,
                str(wilcoxon[algo]["+"]),
                ha="center",
                va="center",
                fontsize=10.5,
                fontfamily=FONT_SERIF,
                color="#000000",
            )
            ax.text(
                col_x[2],
                current_y,
                str(wilcoxon[algo]["≈"]),
                ha="center",
                va="center",
                fontsize=10.5,
                fontfamily=FONT_SERIF,
                color="#000000",
            )
            ax.text(
                col_x[3],
                current_y,
                str(wilcoxon[algo]["-"]),
                ha="center",
                va="center",
                fontsize=10.5,
                fontfamily=FONT_SERIF,
                color="#000000",
            )

        rank_val = avg_ranks[algo]
        is_best = abs(rank_val - best_rank) < 1e-4
        ax.text(
            col_x[4],
            current_y,
            f"{rank_val:.2f}",
            ha="center",
            va="center",
            fontsize=10.5,
            fontfamily=FONT_SERIF,
            fontweight="bold" if is_best else "normal",
            color="#000000",
        )

        current_y -= row_height

    bottom_y = current_y + row_height - 0.06
    # Bottom line (Thick Bottom Rule)
    ax.plot([0.04, 0.96], [bottom_y, bottom_y], color="black", linewidth=2.2, clip_on=False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)
    print(f"Saved ranking table image to: {output_path}")


def render_scores_table(
    plot_data: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Render academic 3-line table for algorithm benchmark scores.
    Algorithms are columns (horizontal axis), benchmarks/functions are rows (vertical axis).
    """
    algorithms = plot_data["algorithms"]
    algo_labels = plot_data["algo_labels"]
    func_cols = plot_data["functions"]
    means = plot_data["means"]
    stds = plot_data["stds"]
    best_means = plot_data["best_means"]

    n_algos = len(algorithms)
    n_funcs = len(func_cols)

    fig_width = max(14.0, 2.3 * n_algos + 3.0)
    fig_height = 1.4 + 0.45 * n_funcs
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.axis("off")

    func_x = 0.08
    algo_xs = np.linspace(0.24, 0.92, n_algos)

    top_y = 0.86
    header_y = 0.70
    midrule_y = 0.58
    row_height = 0.48 / max(n_funcs, 1)

    # Top line
    ax.plot([0.03, 0.97], [top_y, top_y], color="black", linewidth=2.2, clip_on=False)

    # Column headers
    font_header = {"fontsize": 10.5, "fontfamily": FONT_SERIF, "fontweight": "bold", "color": "#000000"}
    ax.text(func_x, header_y, "Function", ha="center", va="center", **font_header)

    for idx, algo in enumerate(algorithms):
        label = algo_labels[algo]
        ax.text(
            algo_xs[idx],
            header_y,
            label,
            ha="center",
            va="center",
            **font_header,
        )

    # Midline
    ax.plot([0.03, 0.97], [midrule_y, midrule_y], color="black", linewidth=1.0, clip_on=False)

    # Data rows (Benchmarks as rows)
    current_y = midrule_y - 0.08
    for fn in func_cols:
        best_m = best_means[fn]
        ax.text(
            func_x,
            current_y,
            fn,
            ha="center",
            va="center",
            fontsize=10.5,
            fontfamily=FONT_SERIF,
            fontweight="bold",
            color="#000000",
        )

        for a_idx, algo in enumerate(algorithms):
            m = means[fn][algo]
            s = stds[fn][algo]
            is_best = abs(m - best_m) < 1e-12 or (best_m != 0 and abs((m - best_m) / best_m) < 1e-6)
            txt = format_sci(m, s)

            ax.text(
                algo_xs[a_idx],
                current_y,
                txt,
                ha="center",
                va="center",
                fontsize=10.0,
                fontfamily=FONT_SERIF,
                fontweight="bold" if is_best else "normal",
                color="#000000",
            )

        current_y -= row_height

    bottom_y = current_y + row_height - 0.06
    # Bottom line
    ax.plot([0.03, 0.97], [bottom_y, bottom_y], color="black", linewidth=2.2, clip_on=False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)
    print(f"Saved scores table image to: {output_path}")


def generate_visualizations(
    plot_data: Dict[str, Any],
    output_dir: str,
):
    """Create publication-quality three-line table images for ranking and scores."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Algorithm Rankings Table Image
    rank_fig_path = Path(output_dir) / "algorithm_rankings.png"
    render_rankings_table(plot_data, rank_fig_path)

    # 2. Algorithm Scores Table Image
    scores_fig_path = Path(output_dir) / "algorithm_scores.png"
    render_scores_table(plot_data, scores_fig_path)


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "TABLE II")
    results_dir = os.path.abspath(results_dir)

    print(f"Loading results from: {results_dir}")
    data = load_all_results(results_dir)
    if not data:
        print("No result data found in results/TABLE II! Please run experiments first.")
        return

    print(f"Found results for algorithms: {list(data.keys())}")
    df_summary, df_detailed, df_rankings, df_scores, plot_data = perform_statistical_analysis(data)

    print("\n" + "=" * 60)
    print("SUMMARY OF AVERAGE RANKINGS (TABLE II)")
    print("=" * 60)
    print(df_rankings.to_string(index=False))
    print("=" * 60 + "\n")

    # Output paths
    summary_md_path = os.path.join(results_dir, "table2_summary.md")
    rankings_csv_path = os.path.join(results_dir, "algorithm_rankings.csv")
    scores_csv_path = os.path.join(results_dir, "algorithm_scores.csv")
    results_csv_path = os.path.join(results_dir, "table2_results.csv")

    # 1. Generate Markdown report
    generate_table2_markdown(df_rankings, df_scores, plot_data, summary_md_path)

    # 2. Save CSV files
    df_rankings.to_csv(rankings_csv_path, index=False, encoding="utf-8")
    print(f"Saved algorithm rankings CSV to: {rankings_csv_path}")

    df_scores.to_csv(scores_csv_path, index=False, encoding="utf-8")
    print(f"Saved algorithm scores CSV to: {scores_csv_path}")

    df_detailed.to_csv(results_csv_path, index=False, encoding="utf-8")
    print(f"Saved detailed raw CSV results to: {results_csv_path}")

    # 3. Clean up obsolete convergence curves if present
    conv_fig_path = os.path.join(results_dir, "convergence_curves.png")
    if os.path.exists(conv_fig_path):
        try:
            os.remove(conv_fig_path)
            print(f"Removed obsolete convergence curve file: {conv_fig_path}")
        except Exception as e:
            print(f"Notice: Could not remove {conv_fig_path}: {e}")

    # 4. Generate Visualizations (Rankings Table Image & Scores Table Image)
    generate_visualizations(plot_data, results_dir)
    print("All Table II reproduction analysis, CSVs, and figures successfully created!")


if __name__ == "__main__":
    main()
