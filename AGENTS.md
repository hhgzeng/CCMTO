# CCMTO 智能体操作与项目指南 (AGENTS.md)

本文档是为在本项目中工作的 AI 智能体（以及开发人员）编写的操作指南与架构备忘录，旨在提供关于 **CCMTO (Cooperative Co-Evolutionary Multitask Optimization)** 算法库的全局理解、模块接口规范、运行指令、核心算法原理与关键避坑准则。

---

## 1. 项目概览 (Project Overview)

### 1.1 项目背景

本项目是论文 **《An Efficient Cooperative Co-Evolutionary Multitask Optimization Framework for Large-Scale Optimization》**（发表于演化计算顶级期刊）的高质量 Python 完整复现与评测基准代码库。

- **核心挑战**：大规模全局优化问题 (Large-Scale Optimization Problems, LSOPs, 通常 $D \ge 1000$) 维度灾难导致搜索空间呈指数级膨胀，传统协同演化 (CC) 单任务范式按固定顺序独立求解子问题，忽略了子问题之间的景观与最优域相似性。
- **核心思想**：
  1. **问题重构**：将大规模问题分解后的子问题（Subproblems）重新聚合成一系列多任务优化问题（Multitask Optimization Problems, MTOPs）。
  2. **知识迁移演化**：提出 **MTES-DAKG**（结合动态距离阈值与自适应精英采样的多任务演化策略），在任务间进行方向迁移（DT-DoS）与步长迁移（AS-SaS）。
  3. **贡献度资源分配与停滞检测**：动态监测各 MTOP 与子任务对全局适应度提升的贡献度（Formula 2 & 3），优先分配计算资源（FEs），并通过停滞检测机制（Algorithm 4）防止计算浪费。
- **基准测试集**：IEEE CEC'2013 Large-Scale Global Optimization (LSGO) 15 个基准函数 ($D=1000$，覆盖单峰完全可分、部分可分及重叠不可分函数)。

---

## 2. 架构与目录索引 (Repository Architecture)

```text
CCMTO/
├── AGENTS.md                             # [本文件] 智能体项目指引与操作规范
├── CCMTO.md                              # 论文精读、数学公式与理论笔记
├── README.md                             # 项目用户说明文档
├── pyproject.toml                        # 项目元数据与依赖配置 (uv / pip)
├── uv.lock                               # 依赖版本锁文件
│
├── src/                                  # 核心算法实现目录
│   ├── CCMTO/                            # CCMTO 协同多任务优化框架
│   │   ├── CCMTO.py                      # CCMTO 主框架 (Algorithm 1 / 2)
│   │   ├── MTOPConstruction.py           # MTOP 构建策略 (Algorithm 2 / 3, 维度比率约束 D_max 与子任务数 N_sub)
│   │   ├── ResourceAllocation.py         # 基于贡献度的资源分配策略 (Formula 2 & 3)
│   │   └── StagnantDetection.py          # 任务停滞检测机制 (Algorithm 4)
│   ├── MTES_DAKG/                        # 多任务演化策略求解器
│   │   ├── MTES_DAKG.py                  # MTES-DAKG 主优化器 (Algorithm 3)
│   │   ├── DT_DoS.py                     # 方向迁移与子空间差异引导采样 (Algorithm 5)
│   │   ├── AS_SaS.py                     # 步长自适应迁移与子空间相似度采样 (Algorithm 6)
│   │   └── CMAES.py                      # 单任务基础 CMA-ES 求解器
│   └── utils.py                          # 工具函数（CSV残余清理、JSON美化持久化等）
│
├── decomposition/                        # 大规模决策变量分解策略
│   ├── edg.py                            # 高效基于距离的分组策略 (EDG)
│   ├── erdg.py                           # 高效递归差分分组策略 (ERDG)
│   ├── precompute_edg.py                 # EDG 预计算与本地缓存加载器
│   └── edg_subproblems_cec2013.json      # CEC2013 F1~F15 的 EDG 分组预计算缓存
│
├── benchmarks/                           # 测试基准问题集
│   └── cec2013_LSOPs.py                  # CEC2013 LSOP 测试函数封装 (F1~F15)
│
├── baselines/                            # 对比算法与消融实验基线
│   ├── cmaes_edg/cmaes_edg.py            # CMAES-EDG (Kumar et al., 2024)
│   ├── decc_erdg/decc_erdg.py            # DECC-ERDG (Yang et al., 2021)
│   ├── gtde/gtde.py                      # GTDE (Wang et al., 2023)
│   ├── sdlso/sdlso.py                    # SDLSO (Yang et al., 2022)
│   ├── emto_algorithms/                  # Table IV 多任务对比算法 (CCMTO-GMFEA, CCMTO-MATDE, CCMTO-MTEA-AD)
│   ├── resource_allocation/              # Table IV 资源分配对比机制 (CBCC1~3, CCFR1~3)
│   └── component_ablation/               # Table IV 消融变体 (wo_sd, wo_da, wo_dt_dos, wo_as_sas)
│
├── experiments/                          # 论文实验复现与脚本套件
│   ├── run_cec2013.py                    # CEC2013 单函数通用评测入口
│   ├── run_table2_experiments.py         # Table II: 与 SOTA LSGO 算法综合对比实验
│   ├── run_table3_experiments.py         # Table III: 参数敏感性实验 (N_sub 与 D_max)
│   └── run_table4_experiments.py         # Table IV: 多任务与消融实验
│
├── tests/                                # 单元测试与验证套件
│   ├── test_ccmto.py                     # CCMTO 端到端优化测试
│   ├── test_mtes_dakg.py                 # MTES-DAKG 优化器测试
│   ├── test_cmaes.py                     # 基础 CMA-ES 求解器测试
│   ├── test_edg.py                       # EDG 分组正确性测试
│   ├── test_mtop_construction.py         # MTOP 构建逻辑测试
│   ├── test_stagnant_detection.py        # 停滞检测模块测试
│   ├── test_baselines.py                 # SOTA 基线算法冒烟测试
│   ├── test_table4_baselines.py          # Table IV 各基线算法接口与运行测试
│   └── verify_corrected_algorithms.py    # 算法逻辑校准与收敛验证脚本
│
└── results/                              # 实验评测结果存储目录 (JSON / CSV / 可视化图表)
```

---

## 3. 环境与依赖管理 (Environment & Dependencies)

- **Python 版本**：`>=3.13` (由 `.python-version` 与 `pyproject.toml` 指定)
- **包管理工具**：首选 **`uv`**（极其快速可靠），支持 `uv run <command>` 直接运行项目环境。
- **关键依赖**：
  - `cec2013lsgo>=2.2`：C++ 编译的 CEC2013 大规模基准库（通过 Cython 编译构建）。
  - `numpy>=2.5.1`, `scipy>=1.18.0`, `pandas>=3.0.5`, `matplotlib>=3.11.1`
  - `setuptools==70.3.0`

### 环境初始化命令

```bash
# 1. 使用 uv 安装并同步依赖
uv sync

# 2. 验证环境是否正常
uv run python -c "from cec2013lsgo.cec2013 import Benchmark; print('CEC2013 loaded successfully!')"
```

---

## 4. 智能体常用命令清单 (Agent Operations Cheat Sheet)

智能体在开发、修改代码或验证实验时，应当严格使用以下命令：

### 4.1 运行单元测试 (Unit Tests)

```bash
# 运行所有单元测试 (耗时 < 2秒)
uv run python -m unittest discover -s tests

# 运行特定模块测试
uv run python -m unittest tests/test_ccmto.py
uv run python -m unittest tests/test_mtes_dakg.py
uv run python -m unittest tests/test_table4_baselines.py

# 运行 SOTA 基线算法收敛性校准验证
uv run python tests/verify_corrected_algorithms.py
```

### 4.2 运行基准评测实验 (Experiments)

```bash
# 1. 运行单函数快速实验 (例如 F1, 100,000 FEs)
uv run python experiments/run_cec2013.py --func 1 --max_fes 100000 --n_sub 5 --d_max 2

# 2. 运行 Table II 复现实验 (例如测试 CCMTO 在 F1 上的 1 次独立运行)
uv run python experiments/run_table2_experiments.py --algorithms CCMTO-MTES-DAKG --funcs 1 --runs 1

# 3. 运行 Table III 参数敏感性实验 (测试 n_sub 在 F4 上的敏感性)
uv run python experiments/run_table3_experiments.py --parameters n_sub --funcs 4 --runs 1

# 4. 运行 Table IV 消融与多任务对比实验
uv run python experiments/run_table4_experiments.py --modules component_ablation --funcs 1 --runs 1
```

### 4.3 预计算 EDG 分组缓存 (EDG Caching)

```bash
# 重新计算并缓存 CEC2013 F1-F15 的 EDG 分组 (若 edg_subproblems_cec2013.json 丢失或需要重新生成)
uv run python decomposition/precompute_edg.py
```

---

## 5. 核心算法与工作流 (Core Algorithmic Principles)

### 5.1 CCMTO 整体工作流程 (Algorithm 1)

1. **变量分解 (Variable Grouping)**：
   - 默认采用 `EDG` (Efficient Distance-based Grouping) 检测变量间的不可分性，将 $D=1000$ 维分解为 $k$ 个互不重叠的子问题。
   - 为避免在每次实验中重复耗费计算评估（FEs），在已知函数上通过 `decomposition/precompute_edg.py` 加载缓存结果。
2. **多任务问题构建 (MTOP Construction - Algorithm 2)**：
   - 限制每个 MTOP 中最多包含 $N_{sub}$ 个子任务（默认 5）。
   - 限制同一 MTOP 内子任务的最大维度比率 $D_{\max} = \frac{\max D_k}{\min D_k} \le 2.0$（避免极不均衡的子空间投影造成知识负迁移）。
3. **基于贡献度的资源分配 (Resource Allocation - Formula 2 & 3)**：
   - 记录每个 MTOP 及各子任务在上一演化周期获得的适应度改进值 $\Delta f_i$。
   - 动态计算贡献度权重，优先选择高贡献度的 MTOP 进行演化，并分配合理的代数/评估预算。
4. **多任务演化求解器 (MTES-DAKG - Algorithm 3)**：
   - 核心演化算子基于 CMA-ES。
   - **DT-DoS (Algorithm 5)**：方向迁移与子空间差异引导采样。
   - **AS-SaS (Algorithm 6)**：自适应步长迁移与子空间相似度引导采样。
   - 外部迁移样本数量 $\tau=1$，采样频率 $fre=0.1 \times MaxGen$。
5. **停滞检测机制 (Stagnant Detection - Algorithm 4)**：
   - 若某任务连续若干周期适应度提升小于阈值 $\epsilon=10^{-6}$，判定为停滞任务，对其采取重组或降低资源权重的策略。

---

## 6. 代码规范与接口契约 (Code Conventions & Interface Contracts)

### 6.1 统一求解器接口 (Optimizer Interface)

无论在 `src/` 还是 `baselines/` 中，所有算法类必须遵循统一的接口范式：

```python
class OptimizerTemplate:
   def __init__(
      self,
      func: Callable[[np.ndarray], float],
      dim: int,
      lower: Union[float, np.ndarray],
      upper: Union[float, np.ndarray],
      max_fes: int,
      custom_subproblems: Optional[List[List[int]]] = None,
      verbose: bool = False,
      **kwargs
   ):
      ...

   def optimize(self) -> Dict[str, Any]:
      """
      执行优化过程并返回统计字典

      Returns:
         Dict 必须包含以下键：
         - "best_x": np.ndarray, 全局最优解向量
         - "best_f": float, 全局最优适应度值 (严格单调不增)
         - "fes": int, 实际消耗的函数评估次数 (必须 <= max_fes + 容错余量)
         - "history": List[Tuple[int, float]], (fe_count, current_best_f) 优化历史记录
      """
      ...
```

### 6.2 严格的评估次数追踪 (FE Counting)

- 大规模优化（LSGO）中，**函数评估次数 (Function Evaluations, FEs) 是最核心的成本度量**（标准上限为 $3 \times 10^6$）。
- 算法内部必须通过统一的 `_eval(x)` 包装器进行调用计数。**严禁绕过计数器私自调用 `func(x)`**。
- 每次评估后必须更新全局最优解 `best_x` 与 `best_f`。

---

## 7. 关键避坑指南与最佳实践 (Critical Gotchas & Pitfalls)

智能体在执行任务时，必须牢记以下几点：

### ⚠️ 1. CEC2013 底层 C++ 库生成的 CSV 残留文件

- **问题**：`cec2013lsgo` 在执行函数评估时，底层 C++ 代码会在当前工作目录或根目录下自动写入 `results_f*.csv` 文件。
- **应对准则**：
  - 在所有涉及评测的脚本/测试入口中，**必须**引入并注册自动清理机制：

  ```python
  from src.utils import cleanup_benchmark_csv, register_csv_cleanup
  register_csv_cleanup()
  ```

  - 并在 `try ... finally` 块中显式调用 `cleanup_benchmark_csv()`。
  - 严禁将 `results_f*.csv` 提交至版本库或遗留在磁盘上。

### ⚠️ 2. 避免无缓存的实时 EDG 分解

- **问题**：在 1000 维问题上运行完整的 EDG 变量分解算法通常需要消耗数千次甚至上万次 FEs。
- **应对准则**：
  - 在做实验评测或批量验证时，优先通过 `decomposition.precompute_edg.get_or_compute_edg_subproblems(func_id)` 加载已缓存的分组结果，传入 `custom_subproblems` 参数，避免在每次重复实验中重新消耗数十秒和数千次评估。

### ⚠️ 3. 数组形状与类型规范 (NumPy Dimensions)

- 决策变量向量维度为 1000 维（一维 `np.ndarray`，shape 为 `(1000,)`）。
- 严禁在内部高频循环中使用低效的 Python `for` 循环遍历维度，应尽量采用 NumPy 向量化操作（如矩阵广播、切片索引）。
- `func` 的返回值为标量浮点数，必须确保转换为 Python 原生 `float`，以防 NumPy 标量导致 JSON 序列化失败。

### ⚠️ 4. JSON 结果保存规范

- 实验数据保存请统一使用 `src.utils.save_json(data, out_path)`，它会自动创建父目录、按 2 空格缩进格式化，并保证浮点数和历史记录的兼容性。

---

## 8. 智能体行动协议 (Agent Operational Protocols)

当接收到修改代码、调试或运行实验的请求时，请遵循以下 SOP（标准作业流程）：

1. **查阅与确认**：
   - 检查涉及的模块位置（例如是在 `src/CCMTO` 还是 `baselines/`）。
   - 检查 `CCMTO.md` 中对应章节的公式与算法描述，确保数学逻辑一致。
2. **轻量迭代与快跑验证**：
   - 编写或修改代码后，**严禁直接启动 $3 \times 10^6$ FEs 的超长评测**。
   - 优先使用合成函数（如 Sphere 20 维，1000 FEs）或 `max_fes=1000` 快速冒烟验证。
3. **回归测试**：
   - 修改任何核心或辅助代码后，立即执行测试套件：

     ```bash
     uv run python -m unittest discover -s tests
     ```

   - 确保所有 14+ 个单元测试均全部通过且无告警。

4. **清洁工作区**：
   - 确认没有产生未清理的临时文件（如 `results_f*.csv` 或临时 `.cache`）。
