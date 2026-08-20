# TABLE II REPRODUCTION RESULTS

## 1. Average Rankings and Statistical Significance Comparison

This table presents the average rankings across the tested CEC2013 LSGO benchmarks (FF1, FF2, FF4, FF5, FF9) and Wilcoxon rank-sum test outcomes (`+/≈/-`) comparing `CCMTO-MTES-DAKG` against each baseline algorithm at significance level $\alpha = 0.05$.

| Algorithm | CEC2013 (+) | CEC2013 (≈) | CEC2013 (-) | Average Ranking |
| :--- | :---: | :---: | :---: | :---: |
| **CCMTO-MTES-DAKG** | **\** | **\** | **\** | **1.40** |
| CMAES-EDG | 2 | 2 | 1 | 2.00 |
| DECC-ERDG | 4 | 0 | 1 | 3.40 |
| GTDE | 5 | 0 | 0 | 4.60 |
| SDLSO | 5 | 0 | 0 | 3.60 |

`+`: Proposed CCMTO-MTES-DAKG is significantly better ($p < 0.05$).
`≈`: No significant difference ($p \ge 0.05$).
`-`: Proposed CCMTO-MTES-DAKG is significantly worse ($p < 0.05$).

---

## 2. Detailed Performance on CEC2013 LSGO Benchmarks (F1 - F11)

Statistical metrics (Mean Error ± Std Error) across 10 independent runs:

| Function | CCMTO-MTES-DAKG | CMAES-EDG | DECC-ERDG | GTDE | SDLSO |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **F1** | 2.07e-08 ± 4.32e-08 | 3.35e-30 ± 7.49e-30 (-) | 9.24e+03 ± 5.30e+03 (+) | 1.32e+11 ± 6.92e+09 (+) | 2.36e+10 ± 1.10e+09 (+) |
| **F2** | 3.30e+02 ± 5.82e+01 | 4.42e+02 ± 3.75e+01 (+) | 1.79e+00 ± 5.33e-01 (-) | 4.33e+04 ± 1.97e+03 (+) | 1.65e+04 ± 1.36e+02 (+) |
| **F4** | 1.86e+06 ± 3.86e+06 | 4.84e+10 ± 1.69e+10 (+) | 3.92e+12 ± 1.09e+12 (+) | 1.04e+12 ± 1.45e+11 (+) | 8.35e+11 ± 9.70e+10 (+) |
| **F5** | 2.77e+06 ± 3.21e+05 | 2.81e+06 ± 9.46e+05 (≈) | 2.61e+07 ± 1.13e+06 (+) | 1.78e+07 ± 1.91e+06 (+) | 1.07e+07 ± 2.74e+05 (+) |
| **F9** | 1.57e+08 ± 3.95e+07 | 1.65e+08 ± 2.59e+07 (≈) | 6.14e+08 ± 3.53e+07 (+) | 1.25e+09 ± 2.11e+08 (+) | 7.89e+08 ± 1.60e+07 (+) |
