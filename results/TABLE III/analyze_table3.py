"""
Statistical Analysis and Visualization Script for Table III Reproduction in CCMTO paper.
(Evaluates n_sub [2, 3, 5, 7] and d_max [1, 2, 4, limitless] on CEC2013 LSGO F4-F7)

Generates:
1. table3_summary.md: Summary table matching Table III layout + detailed benchmark results
2. table3_results.csv: Complete CSV dataset of experimental results
3. chart1_n_sub_ranking.png: Ranking bar chart for n_sub sensitivity
4. chart2_n_sub_benchmarks.png: Benchmark performance comparison chart for n_sub
5. chart3_d_max_ranking.png: Ranking bar chart for d_max sensitivity
6. chart4_d_max_benchmarks.png: Benchmark performance comparison chart for d_max
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata

# Professional plotting styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 1.0


PARAMS_INFO = {
    "n_sub": {
        "latex_name": "$n_{\\mathrm{sub}}$",
        "display_name": "Number of Subtasks in a MTOP (n_sub)",
        "settings": ["2", "3", "5", "7"],
        "baseline_setting": "5",
        "functions": [4, 5, 6, 7],
        "ranking_functions": [4, 5, 6, 7],
        "rank_chart_filename": "chart1_n_sub_ranking.png",
        "perf_chart_filename": "chart2_n_sub_benchmarks.png",
    },
    "d_max": {
        "latex_name": "$d_{\\max}$",
        "display_name": "Maximum Dimension Ratio (d_max)",
        "settings": ["1", "2", "4", "limitless"],
        "baseline_setting": "2",
        "functions": [4, 5, 6, 7],
        "ranking_functions": [4, 5, 6, 7],
        "rank_chart_filename": "chart3_d_max_ranking.png",
        "perf_chart_filename": "chart4_d_max_benchmarks.png",
    },
}


def load_table3_results(results_root: str) -> Dict[str, Dict[str, Dict[int, Dict]]]:
    """
    Load JSON result files into nested dict structure:
    data[param][setting][func_id] -> dict
    """
    data = {}
    for param in PARAMS_INFO:
        data[param] = {}
        param_dir = os.path.join(results_root, param)
        if not os.path.exists(param_dir):
            continue

        for setting_dir_name in os.listdir(param_dir):
            setting_path = os.path.join(param_dir, setting_dir_name)
            if not os.path.isdir(setting_path):
                continue

            setting_key = str(setting_dir_name)
            data[param][setting_key] = {}

            for fname in os.listdir(setting_path):
                if fname.startswith("cec2013_f") and fname.endswith(".json"):
                    fpath = os.path.join(setting_path, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            res = json.load(f)
                        fid = res.get("func_id")
                        if fid is not None:
                            data[param][setting_key][fid] = res
                    except Exception as e:
                        print(f"Warning loading {fpath}: {e}")

    return data


def analyze_parameter(
    param: str,
    param_data: Dict[str, Dict[int, Dict]],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Compute statistical test counts (+/≈/-) and average rankings for one parameter.
    """
    info = PARAMS_INFO[param]
    settings = info["settings"]
    baseline_setting = info["baseline_setting"]
    ranking_fids = info["ranking_functions"]
    all_fids = info["functions"]

    # Filter settings that exist in data
    available_settings = [s for s in settings if s in param_data and any(fid in param_data[s] for fid in all_fids)]
    if not available_settings:
        return {}, pd.DataFrame()

    detailed_rows = []
    mean_errors_matrix = np.zeros((len(ranking_fids), len(available_settings)))
    wilcoxon_outcomes = {s: {"+": 0, "≈": 0, "-": 0} for s in available_settings}

    for f_idx, fid in enumerate(ranking_fids):
        base_runs = param_data.get(baseline_setting, {}).get(fid, {}).get("runs", [])
        base_errors = [r["error"] for r in base_runs] if base_runs else []
        base_mean = np.mean(base_errors) if base_errors else float("inf")

        for s_idx, s in enumerate(available_settings):
            s_runs = param_data.get(s, {}).get(fid, {}).get("runs", [])
            s_errors = [r["error"] for r in s_runs] if s_runs else []

            mean_err = np.mean(s_errors) if s_errors else float("inf")
            std_err = np.std(s_errors, ddof=1) if len(s_errors) > 1 else 0.0
            best_err = np.min(s_errors) if s_errors else float("inf")
            worst_err = np.max(s_errors) if s_errors else float("inf")
            median_err = np.median(s_errors) if s_errors else float("inf")

            mean_errors_matrix[f_idx, s_idx] = mean_err

            if s == baseline_setting:
                outcome = "\\"
                p_val = 1.0
            else:
                if len(base_errors) >= 3 and len(s_errors) >= 3:
                    try:
                        stat, p_val = mannwhitneyu(base_errors, s_errors, alternative="two-sided")
                        if p_val < 0.05:
                            if base_mean < mean_err:
                                outcome = "+"  # Baseline significantly better
                            else:
                                outcome = "-"  # Setting significantly better
                        else:
                            outcome = "≈"
                    except Exception:
                        outcome = "≈"
                        p_val = 1.0
                else:
                    outcome = "≈"
                    p_val = 1.0

                wilcoxon_outcomes[s][outcome] += 1

            detailed_rows.append({
                "Parameter": param,
                "Setting": s,
                "Function": f"F{fid}",
                "Mean Error": mean_err,
                "Std Error": std_err,
                "Best Error": best_err,
                "Median Error": median_err,
                "Worst Error": worst_err,
                "p-value vs Baseline": p_val,
                "Outcome vs Baseline": outcome,
            })

    # Compute Friedman rankings per function
    ranks_matrix = np.zeros_like(mean_errors_matrix)
    for f_idx in range(len(ranking_fids)):
        ranks_matrix[f_idx, :] = rankdata(mean_errors_matrix[f_idx, :])

    avg_ranks = np.mean(ranks_matrix, axis=0)

    summary_result = {
        "parameter": param,
        "settings": settings,
        "available_settings": available_settings,
        "baseline_setting": baseline_setting,
        "wilcoxon_outcomes": wilcoxon_outcomes,
        "avg_ranks": {s: float(avg_ranks[i]) for i, s in enumerate(available_settings)},
        "ranks_matrix": ranks_matrix,
        "ranking_fids": ranking_fids,
        "mean_errors_matrix": mean_errors_matrix,
    }

    df_detailed = pd.DataFrame(detailed_rows)
    return summary_result, df_detailed


def generate_table3_markdown(
    summaries: Dict[str, Dict[str, Any]],
    df_all_detailed: pd.DataFrame,
    output_path: str,
):
    """
    Generate professional Markdown report matching Table III in the paper.
    """
    lines = []
    lines.append("# TABLE III REPRODUCTION RESULTS")
    lines.append("")
    lines.append("## 1. Results of Parameter Sensitivity Analysis")
    lines.append("")
    lines.append("Comparison results of parameter sensitivity analysis across CEC2013 LSGO benchmarks F4-F7 (10 independent runs per benchmark, MaxFEs = 1,000,000).")
    lines.append("Significance test: Wilcoxon rank-sum test at $\\alpha = 0.05$ with `+/≈/-` indicating that the recommended baseline setting is significantly better / equal / worse than the corresponding parameter setting.")
    lines.append("")

    for param in ["n_sub", "d_max"]:
        info = PARAMS_INFO[param]
        summary = summaries.get(param, {})
        settings = info["settings"]
        baseline_setting = info["baseline_setting"]

        lines.append(f"### Parameter: {info['display_name']}")
        lines.append("")

        setting_headers = [f"Setting {i+1}" for i in range(len(settings))]
        lines.append("| Metric | " + " | ".join(setting_headers) + " |")
        lines.append("| :--- | " + " | ".join([":---:"] * len(settings)) + " |")

        # Values
        val_cells = [f"**{s}**" + (" (Base)" if s == baseline_setting else "") for s in settings]
        lines.append(f"| **{info['latex_name']} Setting** | " + " | ".join(val_cells) + " |")

        # + / ≈ / -
        outcome_cells = []
        for s in settings:
            if s == baseline_setting:
                outcome_cells.append("\\")
            elif s in summary.get("wilcoxon_outcomes", {}):
                out = summary["wilcoxon_outcomes"][s]
                outcome_cells.append(f"{out['+']}/{out['≈']}/{out['-']}")
            else:
                outcome_cells.append("N/A")
        lines.append("| + / $\\approx$ / - | " + " | ".join(outcome_cells) + " |")

        # Ranking
        rank_cells = []
        for s in settings:
            if s in summary.get("avg_ranks", {}):
                r_val = summary["avg_ranks"][s]
                rank_cells.append(f"**{r_val:.2f}**" if s == baseline_setting else f"{r_val:.2f}")
            else:
                rank_cells.append("N/A")
        lines.append("| Average Ranking | " + " | ".join(rank_cells) + " |")
        lines.append("")

    lines.append("`+ / ≈ / -`: Indicates that the recommended baseline setting (marked with `\\`) is significantly better / statistically equivalent / significantly worse than the comparison setting.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Detailed Per-Benchmark Performance Breakdown (F4 - F7)")
    lines.append("")

    for param in ["n_sub", "d_max"]:
        info = PARAMS_INFO[param]
        lines.append(f"### Parameter: {info['display_name']}")
        lines.append("")

        sub_df = df_all_detailed[df_all_detailed["Parameter"] == param]
        if sub_df.empty:
            lines.append("No data available.")
            lines.append("")
            continue

        fids = sorted(list(set(sub_df["Function"])), key=lambda x: int(x[1:]))
        settings = info["settings"]
        baseline = info["baseline_setting"]

        header = "| Function | " + " | ".join([f"**{s}**" + (" (Base)" if s == baseline else "") for s in settings]) + " |"
        sep = "| :--- | " + " | ".join([":---:"] * len(settings)) + " |"
        lines.append(header)
        lines.append(sep)

        for fid in fids:
            row_str = f"| **{fid}** |"
            for s in settings:
                subset = sub_df[(sub_df["Function"] == fid) & (sub_df["Setting"] == s)]
                if not subset.empty:
                    m = subset.iloc[0]["Mean Error"]
                    sd = subset.iloc[0]["Std Error"]
                    out = subset.iloc[0]["Outcome vs Baseline"]
                    if s == baseline:
                        val_str = f"{m:.2e} ± {sd:.2e}"
                    else:
                        val_str = f"{m:.2e} ± {sd:.2e} ({out})"
                else:
                    val_str = "N/A"
                row_str += f" {val_str} |"
            lines.append(row_str)
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Generated Table III markdown summary: {output_path}")


def generate_charts(
    summaries: Dict[str, Dict[str, Any]],
    df_detailed: pd.DataFrame,
    output_dir: str,
):
    """
    Generate 4 distinct publication-quality charts:
    Chart 1: n_sub Average Ranking Bar Chart
    Chart 2: n_sub Performance across Benchmarks F4-F7
    Chart 3: d_max Average Ranking Bar Chart
    Chart 4: d_max Performance across Benchmarks F4-F7
    """
    # -------------------------------------------------------------
    # Chart 1: n_sub Average Friedman Ranking
    # -------------------------------------------------------------
    if "n_sub" in summaries:
        info = PARAMS_INFO["n_sub"]
        summary = summaries["n_sub"]
        settings = info["settings"]
        available_settings = summary.get("available_settings", settings)
        baseline = info["baseline_setting"]
        avg_ranks = summary.get("avg_ranks", {})

        fig, ax = plt.subplots(figsize=(7, 4.8), dpi=300)
        ranks_list = [avg_ranks.get(s, 0.0) for s in available_settings]
        colors = [
            "#1f77b4" if s == baseline else "#6baed6" if avg_ranks.get(s, 99) < 2.5 else "#9ecae1"
            for s in available_settings
        ]

        bars = ax.bar(
            [f"n_sub = {s}" for s in available_settings],
            ranks_list,
            color=colors,
            edgecolor="#333333",
            linewidth=1.0,
            width=0.5,
        )

        for bar, s in zip(bars, available_settings):
            h = bar.get_height()
            label_text = f"{h:.2f}"
            if s == baseline:
                label_text += " (Best)"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.06,
                label_text,
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold" if s == baseline else "normal",
                color="#0b3c5d" if s == baseline else "#333333",
            )

        ax.set_title("Average Friedman Ranking for $n_{\\mathrm{sub}}$ Sensitivity (CEC2013 F4-F7)\n(Lower is Better)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Parameter Setting ($n_{\\mathrm{sub}}$)", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_ylabel("Average Ranking", fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(ranks_list) + 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()

        c1_path = os.path.join(output_dir, info["rank_chart_filename"])
        fig.savefig(c1_path, dpi=300)
        plt.close(fig)
        print(f"Saved Chart 1: {c1_path}")

    # -------------------------------------------------------------
    # Chart 2: n_sub Benchmark Performance (F4-F7)
    # -------------------------------------------------------------
    if "n_sub" in summaries:
        info = PARAMS_INFO["n_sub"]
        summary = summaries["n_sub"]
        available_settings = summary.get("available_settings", info["settings"])
        baseline = info["baseline_setting"]
        ranking_fids = summary.get("ranking_fids", [4, 5, 6, 7])

        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        fid_labels = [f"F{fid}" for fid in ranking_fids]
        x = np.arange(len(fid_labels))
        n_settings = len(available_settings)
        width = 0.8 / n_settings
        cmap = plt.get_cmap("tab10")

        for s_idx, s in enumerate(available_settings):
            log_errors = []
            for fid in ranking_fids:
                row = df_detailed[(df_detailed["Parameter"] == "n_sub") & (df_detailed["Setting"] == s) & (df_detailed["Function"] == f"F{fid}")]
                if not row.empty:
                    val = row.iloc[0]["Mean Error"]
                    log_val = np.log10(max(val, 1e-16))
                else:
                    log_val = 0
                log_errors.append(log_val)

            offset = (s_idx - n_settings / 2 + 0.5) * width
            label_name = f"$n_{{\\mathrm{{sub}}}} = {s}$" + (" (Baseline)" if s == baseline else "")
            ax.bar(
                x + offset,
                log_errors,
                width,
                label=label_name,
                color=cmap(s_idx % 10),
                alpha=0.9,
                edgecolor="white",
                linewidth=0.5,
            )

        ax.set_title("Performance Comparison across Benchmarks for $n_{\\mathrm{sub}}$ (CEC2013 F4-F7)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("CEC2013 Benchmark Function", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_ylabel(r"$\log_{10}(\mathrm{Mean\ Error})$", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(fid_labels, fontsize=11, fontweight="bold")
        ax.legend(frameon=True, fontsize=9.5, loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()

        c2_path = os.path.join(output_dir, info["perf_chart_filename"])
        fig.savefig(c2_path, dpi=300)
        plt.close(fig)
        print(f"Saved Chart 2: {c2_path}")

    # -------------------------------------------------------------
    # Chart 3: d_max Average Friedman Ranking
    # -------------------------------------------------------------
    if "d_max" in summaries:
        info = PARAMS_INFO["d_max"]
        summary = summaries["d_max"]
        settings = info["settings"]
        available_settings = summary.get("available_settings", settings)
        baseline = info["baseline_setting"]
        avg_ranks = summary.get("avg_ranks", {})

        fig, ax = plt.subplots(figsize=(7, 4.8), dpi=300)
        ranks_list = [avg_ranks.get(s, 0.0) for s in available_settings]
        colors = [
            "#2ca02c" if s == baseline else "#74c476" if avg_ranks.get(s, 99) < 2.5 else "#a1d99b"
            for s in available_settings
        ]

        bars = ax.bar(
            [f"d_max = {s}" for s in available_settings],
            ranks_list,
            color=colors,
            edgecolor="#333333",
            linewidth=1.0,
            width=0.5,
        )

        for bar, s in zip(bars, available_settings):
            h = bar.get_height()
            label_text = f"{h:.2f}"
            if s == baseline:
                label_text += " (Best)"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.06,
                label_text,
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold" if s == baseline else "normal",
                color="#0f5c1e" if s == baseline else "#333333",
            )

        ax.set_title("Average Friedman Ranking for $d_{\\max}$ Sensitivity (CEC2013 F4-F7)\n(Lower is Better)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Parameter Setting ($d_{\\max}$)", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_ylabel("Average Ranking", fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(ranks_list) + 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()

        c3_path = os.path.join(output_dir, info["rank_chart_filename"])
        fig.savefig(c3_path, dpi=300)
        plt.close(fig)
        print(f"Saved Chart 3: {c3_path}")

    # -------------------------------------------------------------
    # Chart 4: d_max Benchmark Performance (F4-F7)
    # -------------------------------------------------------------
    if "d_max" in summaries:
        info = PARAMS_INFO["d_max"]
        summary = summaries["d_max"]
        available_settings = summary.get("available_settings", info["settings"])
        baseline = info["baseline_setting"]
        ranking_fids = summary.get("ranking_fids", [4, 5, 6, 7])

        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        fid_labels = [f"F{fid}" for fid in ranking_fids]
        x = np.arange(len(fid_labels))
        n_settings = len(available_settings)
        width = 0.8 / n_settings
        cmap = plt.get_cmap("tab10")

        for s_idx, s in enumerate(available_settings):
            log_errors = []
            for fid in ranking_fids:
                row = df_detailed[(df_detailed["Parameter"] == "d_max") & (df_detailed["Setting"] == s) & (df_detailed["Function"] == f"F{fid}")]
                if not row.empty:
                    val = row.iloc[0]["Mean Error"]
                    log_val = np.log10(max(val, 1e-16))
                else:
                    log_val = 0
                log_errors.append(log_val)

            offset = (s_idx - n_settings / 2 + 0.5) * width
            label_name = f"$d_{{\\max}} = {s}$" + (" (Baseline)" if s == baseline else "")
            ax.bar(
                x + offset,
                log_errors,
                width,
                label=label_name,
                color=cmap(s_idx % 10),
                alpha=0.9,
                edgecolor="white",
                linewidth=0.5,
            )

        ax.set_title("Performance Comparison across Benchmarks for $d_{\\max}$ (CEC2013 F4-F7)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("CEC2013 Benchmark Function", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_ylabel(r"$\log_{10}(\mathrm{Mean\ Error})$", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(fid_labels, fontsize=11, fontweight="bold")
        ax.legend(frameon=True, fontsize=9.5, loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()

        c4_path = os.path.join(output_dir, info["perf_chart_filename"])
        fig.savefig(c4_path, dpi=300)
        plt.close(fig)
        print(f"Saved Chart 4: {c4_path}")


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "TABLE III")
    results_dir = os.path.abspath(results_dir)

    print("=" * 80)
    print(f"Loading Table III experimental results from: {results_dir}")
    print("=" * 80)

    data = load_table3_results(results_dir)
    if not data or not any(data[p] for p in data):
        print(f"No result files found in {results_dir}! Please run experiments first.")
        return

    summaries = {}
    detailed_dfs = []

    for param in ["n_sub", "d_max"]:
        param_data = data.get(param, {})
        if not param_data:
            print(f"Warning: No data for parameter {param}")
            continue

        summary, df_detailed = analyze_parameter(param, param_data)
        summaries[param] = summary
        if not df_detailed.empty:
            detailed_dfs.append(df_detailed)

    df_all_detailed = pd.concat(detailed_dfs, ignore_index=True) if detailed_dfs else pd.DataFrame()

    # Generate Markdown Summary
    summary_md_path = os.path.join(results_dir, "table3_summary.md")
    generate_table3_markdown(summaries, df_all_detailed, summary_md_path)

    # Generate CSV dataset
    results_csv_path = os.path.join(results_dir, "table3_results.csv")
    df_all_detailed.to_csv(results_csv_path, index=False, encoding="utf-8")
    print(f"Saved complete CSV dataset: {results_csv_path}")

    # Generate 4 Visualizations
    generate_charts(summaries, df_all_detailed, results_dir)

    print("\n" + "=" * 80)
    print("ALL TABLE III STATISTICAL ANALYSIS AND 4 CHARTS GENERATED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
