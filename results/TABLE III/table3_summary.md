# TABLE III REPRODUCTION RESULTS

## 1. Results of Parameter Sensitivity Analysis

Comparison results of parameter sensitivity analysis across CEC2013 LSGO benchmarks F4-F7 (10 independent runs per benchmark, MaxFEs = 1,000,000).
Significance test: Wilcoxon rank-sum test at $\alpha = 0.05$ with `+/≈/-` indicating that the recommended baseline setting is significantly better / equal / worse than the corresponding parameter setting.

### Parameter: Number of Subtasks in a MTOP (n_sub)

| Metric | Setting 1 | Setting 2 | Setting 3 | Setting 4 |
| :--- | :---: | :---: | :---: | :---: |
| **$n_{\mathrm{sub}}$ Setting** | **2** | **3** | **5** (Base) | **7** |
| + / $\approx$ / - | 0/4/0 | 0/4/0 | \ | 0/4/0 |
| Average Ranking | 3.00 | 3.25 | **1.75** | 2.00 |

### Parameter: Maximum Dimension Ratio (d_max)

| Metric | Setting 1 | Setting 2 | Setting 3 | Setting 4 |
| :--- | :---: | :---: | :---: | :---: |
| **$d_{\max}$ Setting** | **1** | **2** (Base) | **4** | **limitless** |
| + / $\approx$ / - | 0/4/0 | \ | 0/4/0 | 0/4/0 |
| Average Ranking | 2.25 | **3.00** | 2.50 | 2.25 |

`+ / ≈ / -`: Indicates that the recommended baseline setting (marked with `\`) is significantly better / statistically equivalent / significantly worse than the comparison setting.

---

## 2. Detailed Per-Benchmark Performance Breakdown (F4 - F7)

### Parameter: Number of Subtasks in a MTOP (n_sub)

| Function | **2** | **3** | **5** (Base) | **7** |
| :--- | :---: | :---: | :---: | :---: |
| **F4** | 5.05e+06 ± 7.87e+06 (≈) | 5.85e+09 ± 8.53e+09 (≈) | 1.85e+06 ± 3.84e+06 | 1.04e+07 ± 1.53e+07 (≈) |
| **F5** | 2.69e+06 ± 1.88e+05 (≈) | 2.11e+06 ± 4.82e+05 (≈) | 2.39e+06 ± 3.57e+05 | 2.32e+06 ± 4.31e+05 (≈) |
| **F6** | 1.07e+06 ± 1.51e+03 (≈) | 1.07e+06 ± 7.48e+02 (≈) | 1.06e+06 ± 1.27e+03 | 1.07e+06 ± 1.76e+03 (≈) |
| **F7** | 1.80e+08 ± 1.11e+08 (≈) | 2.55e+08 ± 2.36e+08 (≈) | 1.58e+08 ± 1.12e+08 | 1.39e+08 ± 5.64e+07 (≈) |

### Parameter: Maximum Dimension Ratio (d_max)

| Function | **1** | **2** (Base) | **4** | **limitless** |
| :--- | :---: | :---: | :---: | :---: |
| **F4** | 1.84e+06 ± 3.84e+06 (≈) | 1.85e+06 ± 3.84e+06 | 1.85e+06 ± 3.84e+06 (≈) | 1.85e+06 ± 3.84e+06 (≈) |
| **F5** | 2.35e+06 ± 3.00e+05 (≈) | 2.39e+06 ± 3.57e+05 | 2.39e+06 ± 3.57e+05 (≈) | 2.39e+06 ± 3.57e+05 (≈) |
| **F6** | 1.07e+06 ± 1.58e+03 (≈) | 1.06e+06 ± 1.27e+03 | 1.06e+06 ± 1.27e+03 (≈) | 1.06e+06 ± 1.27e+03 (≈) |
| **F7** | 1.39e+08 ± 1.13e+08 (≈) | 1.58e+08 ± 1.12e+08 | 1.39e+08 ± 5.26e+07 (≈) | 6.95e+07 ± 9.55e+07 (≈) |
