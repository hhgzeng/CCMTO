# CCMTO 复现 (Cooperative Co-Evolutionary Multitask Optimization)

本项目是论文 **《An Efficient Cooperative Co-Evolutionary Multitask Optimization Framework for Large-Scale Optimization》** 的 Python 复现与评测代码库。

---

## 📁 项目结构

```text
CCMTO/
├── src/                                  # 核心算法实现目录
│   ├── CCMTO/                            # CCMTO 协同多任务优化框架
│   │   ├── CCMTO.py                      # CCMTO 主框架 (Algorithm 1)
│   │   ├── MTOPConstruction.py           # 多任务优化问题构建策略 (Algorithm 2)
│   │   ├── ResourceAllocation.py         # 基于贡献度的资源分配策略 (Formula 2 & 3)
│   │   └── StagnantDetection.py          # 任务停滞检测机制 (Algorithm 4)
│   ├── MTES_DAKG/                        # 多任务演化策略求解器
│   │   ├── MTES_DAKG.py                  # MTES-DAKG 主优化器 (Algorithm 3)
│   │   ├── DT_DoS.py                     # 方向迁移与子空间差异引导采样 (Algorithm 5)
│   │   ├── AS_SaS.py                     # 步长自适应迁移与子空间相似度采样 (Algorithm 6)
│   │   └── CMAES.py                      # 单任务基础 CMA-ES 求解器
│   └── utils.py                          # 工具函数库
│
├── decomposition/                        # 大规模决策变量分解策略
│   ├── edg.py                            # 高效基于距离的分组策略 (EDG)
│   ├── erdg.py                           # 高效递归差分分组策略 (ERDG)
│   ├── precompute_edg.py                 # EDG 分组预计算脚本
│   └── edg_subproblems_cec2013.json      # CEC'2013 LSOPs 缓存的 EDG 分组结果
│
├── benchmarks/                           # 测试基准问题集
│   └── cec2013_LSOPs.py                  # CEC'2013 大规模全局优化基准测试函数 (F1-F15)
│
├── baselines/                            # 对比算法与消融实验基线实现
│   ├── cmaes_edg/                        # CMAES-EDG 对比算法 (EDG + CMA-ES)
│   │   └── cmaes_edg.py
│   ├── decc_erdg/                        # DECC-ERDG 对比算法 (ERDG + DECC)
│   │   └── decc_erdg.py
│   ├── gtde/                             # GTDE 对比算法
│   │   └── gtde.py
│   ├── sdlso/                            # SDLSO 对比算法
│   │   └── sdlso.py
│   ├── emto_algorithms/                  # Table IV: 多任务演化优化对比算法
│   │   ├── ccmto_gmfea.py                # CCMTO-GMFEA
│   │   ├── ccmto_matde.py                # CCMTO-MATDE
│   │   └── ccmto_mtea_ad.py              # CCMTO-MTEA-AD
│   ├── resource_allocation/              # Table V: 资源分配策略对比算法
│   │   ├── cbcc1.py / cbcc2.py / cbcc3.py# 基于贡献度的协同策略 (CBCC1~3)
│   │   └── ccfr.py / ccfr2.py / ccfr3.py # 协同资源分配策略 (CCFR1~3)
│   └── component_ablation/               # Table VI: 组件消融实验变体
│       ├── wo_sd.py                      # 移除停滞检测 (w/o SD)
│       ├── wo_da.py                      # 移除领域自适应外部采样 (w/o DA)
│       ├── wo_dt_dos.py                  # 移除方向迁移及差分引导 (w/o DT-DoS)
│       └── wo_as_sas.py                  # 移除自适应步长及相似度引导 (w/o AS-SaS)
│
├── experiments/                          # 论文实验复现运行脚本
│   ├── run_cec2013.py                    # CEC'2013 单函数评测脚本
│   ├── run_table2_experiments.py         # Table II: 与 SOTA LSGO 算法综合对比实验
│   ├── run_table3_experiments.py         # Table III: 参数敏感性实验 (N_sub 与 D_max)
│   └── run_table4_experiments.py         # Table IV: 多任务演化算法对比实验
│
├── tests/                                # 单元测试与算法验证
│   ├── test_ccmto.py                     # CCMTO 框架测试
│   ├── test_mtes_dakg.py                 # MTES-DAKG 优化器测试
│   ├── test_cmaes.py                     # CMA-ES 单任务测试
│   ├── test_edg.py                       # EDG 分组测试
│   ├── test_mtop_construction.py         # MTOP 构建测试
│   ├── test_stagnant_detection.py        # 停滞检测测试
│   ├── test_baselines.py                 # 基线算法测试
│   ├── test_table4_baselines.py          # Table IV 多任务基线测试
│   └── verify_corrected_algorithms.py    # 算法逻辑校准与验证
│
├── results/                              # 实验数据、统计摘要与图表
│   ├── TABLE II/                         # Table II 实验结果、CSV 与收敛/排名可视化图表
│   ├── TABLE III/                        # Table III 参数敏感性结果与柱状/折线图
│   └── *.json                            # 各测试函数的原始运行结果
│
├── pyproject.toml                        # 项目依赖与包配置 (uv / pip)
├── CCMTO.md                              # 论文精读与理论笔记
└── README.md                             # 项目说明文档
```

---

## 🛠️ 模块说明

1. **核心算法 (`src/`)**
   - **`CCMTO`**: 协同多任务优化框架，将 LSOP 分解后的子问题转化为多任务优化问题（MTOPs），并利用贡献度动态分配计算资源与检测停滞。
   - **`MTES_DAKG`**: 基于知识迁移的演化策略求解器，包含方向迁移（DT-DoS）与自适应步长迁移（AS-SaS），有效利用不同子任务间的相似性与互补性。
2. **问题分解 (`decomposition/`)**
   - 实现 **EDG** (Efficient Distance-based Grouping) 与 **ERDG** 分组算法，支持离线预计算与缓存加速。
3. **基准测试集 (`benchmarks/`)**
   - 适配 IEEE CEC'2013 Large-Scale Global Optimization (LSGO) 15 个基准测试函数（1000 维）。
4. **基线与消融 (`baselines/`)**
   - 包含经典 LSGO 算法（GTDE、SDLSO、CMAES-EDG、DECC-ERDG）、多任务演化算法（G-MFEA、MATDE、MTEA-AD）、经典资源分配机制（CBCC、CCFR）以及各组件消融变体。
5. **实验与测试 (`experiments/` & `tests/`)**
   - 提供与论文主要表格（Table II、III、IV 等）一一对应的自动化执行脚本与完整的测试验证套件。
