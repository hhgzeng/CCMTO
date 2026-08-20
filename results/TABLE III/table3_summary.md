# TABLE III REPRODUCTION RESULTS

## 1. Results of Parameter Sensitivity Analysis

Comparison results of parameter sensitivity analysis across CEC2013 LSGO benchmarks F4-F7 (10 independent runs per benchmark, MaxFEs = 1,000,000).
Significance test: Wilcoxon rank-sum test at $\alpha = 0.05$ with `+/≈/-` indicating that the recommended baseline setting is significantly better / equal / worse than the corresponding parameter setting.

### Parameter: Number of Subtasks in a MTOP (n_sub)

| Metric | Setting 1 | Setting 2 | Setting 3 | Setting 4 | Setting 5 | Setting 6 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$n_{\mathrm{sub}}$ Setting** | **2** | **3** | **5** (Base) | **7** | **10** | **20** |
| + / $\approx$ / - | 0/4/0 | 0/4/0 | \ | 0/4/0 | 0/3/1 | 0/4/0 |
| Average Ranking | 3.25 | 4.50 | **3.25** | 2.75 | 3.25 | 4.00 |

### Parameter: Maximum Dimension Ratio (d_max)

| Metric | Setting 1 | Setting 2 | Setting 3 | Setting 4 |
| :--- | :---: | :---: | :---: | :---: |
| **$d_{\max}$ Setting** | **1** | **2** (Base) | **4** | **limitless** |
| + / $\approx$ / - | 0/4/0 | \ | 0/4/0 | 0/4/0 |
| Average Ranking | 1.50 | **2.75** | 3.25 | 2.50 |

`+ / ≈ / -`: Indicates that the recommended baseline setting (marked with `\`) is significantly better / statistically equivalent / significantly worse than the comparison setting.

---

## 2. Detailed Per-Benchmark Performance Breakdown (F4 - F7)

### Parameter: Number of Subtasks in a MTOP (n_sub)

| Function | **2** | **3** | **5** (Base) | **7** | **10** | **20** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **F4** | 5.94e+06 ± 7.43e+06 (≈) | 5.85e+09 ± 8.53e+09 (≈) | 1.87e+06 ± 3.87e+06 | 1.04e+07 ± 1.53e+07 (≈) | 3.04e+08 ± 6.64e+08 (≈) | 1.49e+08 ± 3.30e+08 (≈) |
| **F5** | 2.47e+06 ± 2.73e+05 (≈) | 2.72e+06 ± 4.65e+05 (≈) | 2.77e+06 ± 3.21e+05 | 2.46e+06 ± 5.76e+05 (≈) | 1.93e+06 ± 3.16e+05 (-) | 2.49e+06 ± 4.24e+05 (≈) |
| **F6** | 1.07e+06 ± 1.04e+03 (≈) | 1.06e+06 ± 1.82e+03 (≈) | 1.07e+06 ± 1.50e+03 | 1.07e+06 ± 1.13e+03 (≈) | 1.06e+06 ± 1.82e+03 (≈) | 1.07e+06 ± 8.07e+02 (≈) |
| **F7** | 1.52e+08 ± 3.71e+07 (≈) | 2.87e+08 ± 2.02e+08 (≈) | 1.26e+08 ± 1.01e+08 | 1.57e+08 ± 4.89e+07 (≈) | 2.18e+08 ± 5.21e+07 (≈) | 1.80e+08 ± 7.58e+07 (≈) |

### Parameter: Maximum Dimension Ratio (d_max)

| Function | **1** | **2** (Base) | **4** | **limitless** |
| :--- | :---: | :---: | :---: | :---: |
| **F4** | 1.84e+06 ± 3.84e+06 (≈) | 1.87e+06 ± 3.87e+06 | 1.87e+06 ± 3.87e+06 (≈) | 1.87e+06 ± 3.87e+06 (≈) |
| **F5** | 2.35e+06 ± 3.00e+05 (≈) | 2.77e+06 ± 3.21e+05 | 2.77e+06 ± 3.21e+05 (≈) | 2.77e+06 ± 3.21e+05 (≈) |
| **F6** | 1.07e+06 ± 1.58e+03 (≈) | 1.07e+06 ± 1.50e+03 | 1.07e+06 ± 1.50e+03 (≈) | 1.07e+06 ± 1.50e+03 (≈) |
| **F7** | 1.39e+08 ± 1.13e+08 (≈) | 1.26e+08 ± 1.01e+08 | 1.48e+08 ± 4.70e+07 (≈) | 2.13e+07 ± 4.04e+07 (≈) |
