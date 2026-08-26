# TABLE IV REPRODUCTION RESULTS

## RESULTS OF COMPONENT ANALYSIS IN CCMTO-MTES-DAKG

Empirical results of component analysis on CEC2013 benchmarks [F1, F2, F4, F5, F9].
Significance test: Wilcoxon rank-sum test at $\alpha = 0.05$. `+/≈/-` indicates that the proposed method (`CCMTO-MTES-DAKG`) is significantly better / statistically equal / significantly worse than the comparison algorithm, respectively.

| Algorithm                                                                   |  +  | $\approx$ |  -  | Ranking  |
| :-------------------------------------------------------------------------- | :-: | :-------: | :-: | :------: |
| **Results on different resource allocation strategies for CCMTO-MTES-DAKG** |     |           |     |          |
| CBCC1                                                                       |  0  |     5     |  0  |   2.00   |
| CCFR3                                                                       |  1  |     3     |  1  |   2.40   |
| **CCMTO-MTES-DAKG**                                                         | \   |    \      | \   | **1.60** |
| **Results on different EMTO algorithms for CCMTO**                          |     |           |     |          |
| CCMTO-MaTDE                                                                 |  5  |     0     |  0  |   2.00   |
| **CCMTO-MTES-DAKG**                                                         | \   |    \      | \   | **1.00** |
| **Results on CCMTO-MTES-DAKG with different components**                    |     |           |     |          |
| wo-DA                                                                       |  1  |     4     |  0  |   4.40   |
| wo-DT-DoS                                                                   |  0  |     5     |  0  |   2.60   |
| wo-AS-SaS                                                                   |  0  |     5     |  0  |   4.00   |
| wo-SD                                                                       |  0  |     4     |  1  |   1.40   |
| **CCMTO-MTES-DAKG**                                                         | \   |    \      | \   | **2.60** |

---

## Detailed Per-Benchmark Performance Breakdown

### Results on different resource allocation strategies for CCMTO-MTES-DAKG

| Function | **CBCC1** | **CCFR3** | **CCMTO-MTES-DAKG** |
| :------- | :---: | :---: | :---: |
| **F1** | 7.61e-08±1.58e-07 | 2.77e-33±6.19e-33 | 2.07e-08±4.32e-08 |
| **F2** | 3.31e+02±5.78e+01 | 3.32e+02±5.83e+01 | 3.30e+02±5.82e+01 |
| **F4** | 9.08e+05±5.06e+05 | 2.62e+10±2.20e+10 | 1.85e+06±3.84e+06 |
| **F5** | 2.28e+06±3.73e+05 | 2.44e+06±2.39e+05 | 2.39e+06±3.57e+05 |
| **F9** | 1.87e+08±3.02e+07 | 1.82e+08±2.90e+07 | 1.57e+08±3.95e+07 |

### Results on different EMTO algorithms for CCMTO

| Function | **CCMTO-MaTDE** | **CCMTO-MTES-DAKG** |
| :------- | :---: | :---: |
| **F1** | 1.66e+11±2.67e+10 | 2.07e-08±4.32e-08 |
| **F2** | 4.50e+04±3.27e+03 | 3.30e+02±5.82e+01 |
| **F4** | 1.05e+14±4.69e+13 | 1.85e+06±3.84e+06 |
| **F5** | 1.09e+08±2.57e+07 | 2.39e+06±3.57e+05 |
| **F9** | 7.22e+08±3.48e+07 | 1.57e+08±3.95e+07 |

### Results on CCMTO-MTES-DAKG with different components

| Function | **wo-DA** | **wo-DT-DoS** | **wo-AS-SaS** | **wo-SD** | **CCMTO-MTES-DAKG** |
| :------- | :---: | :---: | :---: | :---: | :---: |
| **F1** | 4.26e-08±5.37e-08 | 3.76e-08±8.10e-08 | 1.12e-07±1.22e-07 | 1.90e-30±3.80e-30 | 2.07e-08±4.32e-08 |
| **F2** | 1.20e+03±1.78e+03 | 3.14e+02±1.34e+01 | 3.76e+02±8.84e+01 | 2.92e+02±1.56e+01 | 3.30e+02±5.82e+01 |
| **F4** | 2.95e+07±3.69e+07 | 1.84e+06±3.85e+06 | 3.83e+08±8.08e+08 | 6.28e+04±1.05e+05 | 1.85e+06±3.84e+06 |
| **F5** | 2.42e+06±7.83e+05 | 2.44e+06±3.22e+05 | 2.21e+06±3.34e+05 | 2.16e+06±1.92e+05 | 2.39e+06±3.57e+05 |
| **F9** | 1.79e+08±5.14e+07 | 1.55e+08±2.52e+07 | 1.61e+08±2.63e+07 | 1.58e+08±2.98e+07 | 1.57e+08±3.95e+07 |
