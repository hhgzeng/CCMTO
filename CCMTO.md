# An Efficient Cooperative Co-Evolutionary Multitask Optimization Framework for Large-Scale Optimization

Shiqiang Li, Jun Liu, and Yuansheng Cheng

Abstract—Cooperative co-evolution (CC) framework is a classic decomposition-based method to solve large-scale optimization problems (LSOPs) by decomposing the original problem into several subproblems. CC is a single task paradigm that sequentially solves decomposed subproblems in a specific order, and it does not fully utilize similarities among these subproblems. Evolutionary multitask optimization (EMTO) employs the potential similarities and complementarities among distinct tasks to address multiple optimization tasks simultaneously through knowledge transfer mechanism. This study integrates the CC framework with the EMTO paradigm and proposes a cooperative co-evolutionary multitask optimization (CCMTO) framework for solving LSOPs. In the CCMTO framework, the original LSOP is redefined as a set of multitask optimization problems (MTOPs), and then the EMTO algorithm is used to solve them. To improve the optimization efficiency, this study proposes a construction strategy of multitask optimization problems and a contribution-based resource allocation strategy of MTOPs and subtasks. The construction strategy of multitask optimization problems can select the appropriate subproblems to construct MTOPs. The resource allocation strategy determines the optimization order of MTOPs based on their contribution to the improvement of the best fitness value, and reasonably allocates computational resources for each subtask. A multitask evolution strategy with dynamic distance threshold and adaptive elite sampling knowledge-guided external sampling (MTES-DAKG) is proposed and used to solve these MTOPs. Empirical results show that the proposed algorithm can significantly improve the optimization performance for solving LSOPs. Moreover, the proposed algorithm is superior to 14 state-of-the-art algorithms on 29 benchmark problems and performs well in real-world applications.

Index Terms—Cooperative co-evolution, evolutionary multitask optimization, evolution strategy, knowledge transfer, large-scale optimization

## I. INTRODUCTION

Large-scale optimization problems (LSOPs) present substantial challenges in the field of optimization because they involve hundreds to thousands of design variables [1], [2], [3]. As the dimensionality increases, the volume of the design space grows exponentially, making it difficult for optimization algorithms to thoroughly explore such an expansive search space. Furthermore, the rise in dimensions also leads to a rapid increase in the complexity of the objective functions. Decomposition-based methods which employ the strategy of “divide and conquer”, have attracted widely research interest from scholars. Cooperative coevolutionary algorithms (CCEAs) [4], [5] inspired by the ecological phenomenon of mutualism are effective methods for solving LSOPs. CCEAs decompose the original LSOP into a set of lower-dimensional subproblems, each of which can be solved in an independently evolving subpopulation to alleviate the difficulties associated with high dimensionality.

The first CCEA, named cooperative co-evolutionary genetic algorithm (CCGA), was proposed by [4] in 1994. This framework decomposes an $N$-dimensional problem into $N$ one-dimensional subproblems, which are then optimized sequentially by genetic algorithm. Nevertheless, not all LSOPs are fully separable, and some of them are difficult to be solved owing to complex interaction among variables. In recent decades, there has been a rapid growth in the research on CCEAs to improve the efficiency and effect. Generally, the improvements in CCEAs mainly focus on the following aspects [5]: (1) Variable grouping strategy: Research on variable grouping strategy can be mainly categorized into dynamic variable grouping and static variable grouping. Dynamic variable grouping implies that the grouping approach will change during the process of coevolution [2], [6], [7], [8]. In contrast, static variable grouping maintains a fixed grouping scheme throughout the optimization process, and several variable interaction identification methods have been developed to improve the accuracy of static variable grouping [9], [10], [11], [12]. (2) Collaborator selection strategy: Many types of collaborator selection strategy, such as single best collaborator selection strategy [4], single worst collaborator selection [13], random collaborator selection strategy [14], elite collaborator selection strategy [15] and so on [16], [17], are employed. (3) Resource allocation strategy: Namely how to determine the order of subproblems to be optimized and how to reasonably allocate the computational resource for subproblems. The existing computational resource allocation strategies are mainly based on contribution of subproblems [18], difficulty of subproblems [19], and different subpopulation sizes [20], [21].

Furthermore, there are studies dedicated to solving complex LSOPs. Tian et al. [22] proposed a reinforcement learningbased (RLDO) framework to efficiently decompose the largescale overlapping problems, and the decomposed subproblems were subsequently integrated into the CC framework for optimization. In [23], a contribution-based cooperative coevolutionary algorithm is designed to decompose and optimize nonseparable large-scale problems with overlapping subcomponents effectively and efficiently. For the constrained LSOPs and expensive LSOPs, some efficient CCEAs have been proposed [24], [25], [26].

Although the aforementioned works have significantly improved the efficiency to solve LSOPs, these CCEAs belong to the single task paradigm algorithms. Traditional cooperative co-evolution (CC) framework sequentially optimizes decomposed subproblems in a specific order, without fully considering the similarities between these subproblems. Actually, there exist correlations and similarities of the optimal domains and function shapes among decomposed subproblems. If these properties are fully explored and utilized, there is tremendous potential to enhance both the efficiency and effect of solving LSOPs. Evolutionary multitask optimization (EMTO) can utilize the potential similarities and complementarities among distinct tasks to address multiple optimization tasks simultaneously through knowledge transfer mechanism. This emerging paradigm has attracted substantial attention as a crucial research direction within the field of evolutionary computation in recent years [27].

Inspired by the multifactor genetic model in biology, Gupta [28] innovatively proposed a new optimization problem paradigm in the field of evolutionary computation, referred to as multifactor optimization (MFO) problem, and introduced the first evolutionary multitask optimization algorithm, namely multifactor evolutionary algorithm (MFEA). The general EMTO algorithms are predicated on the assumption that tasks are closely interrelated. However, not all tasks exist inherent interconnections. The transfer of unrelated knowledge across tasks will slow down the overall optimization efficiency, resulting in the phenomenon known as "Negative Transfer" [29]. Therefore, in EMTO, the effect of knowledge transfer can be influenced by three critical factors: the methods of knowledge transfer (i.e., how to transfer knowledge), the types of knowledge being transferred (i.e., what kind of knowledge to transfer), and the frequency of knowledge transfer (i.e., when to transfer knowledge). These factors collectively impact optimization performance [30]. Consequently, extensive research efforts have been dedicated to key aspects of EMTO, including similarity measurement [31], [32], source task selection [33], [34], knowledge transfer methods [35], [36], adaptive control of the transfer process [37], [38], search space transformation [39], [40], and frequency setting of knowledge transfer [41], [42], resulting considerable and significant research progress.

However, most of the studies mentioned above are for low or medium dimensional optimization problems, typically involving decision variables with fewer than 100 dimensions.

High dimensionality is an important characteristic of many real-world optimization problems, where the dimensions of design variables can potentially reach up to thousands. Up to now, there are not many EMTO algorithms dedicated to solving LSOPs, and the methods employed in these studies belong to the non-decomposition methods. Feng et al. [43] proposed a non-decomposition approach, random embedding technique was used to reduce original high-dimensional search space. Then, some low-dimensional optimization problems obtained by dimensionality reduction were treated as auxiliary tasks, and they assisted the evolution of original LSOPs by knowledge transfer. It is worth noting that this method is difficult to ensure the preservation of the global optimal solution of the original problem in the reduced lowdimensional problem space. In [44], a new search paradigm, namely the multispace evolutionary search, is proposed to enhance the existing evolutionary search methods for solving large-scale optimization problems. The proposed paradigm is designed to conduct a search in multiple solution spaces that are derived from the given problem, each possessing a unique landscape.

It is noteworthy that when applying CC framework to solve LSOPs, the optimization efficiency can be enhanced by mining the similarities among decomposed subproblems. This idea is highly consistent with the fundamental principle of EMTO, which explores the similarities of tasks to promote effective knowledge transfer. Theoretically, integrating the CC framework with the EMTO paradigm and developing EMTO algorithms for LSOPs have significant research value and promising application prospects. Thus, this work dedicates to constructing a new framework, labeled as cooperative coevolutionary multitask optimization (CCMTO) framework and studying an efficient EMTO algorithm to facilitate the collaborative optimization of the decomposed subproblems for solving LSOPs.

In the proposed CCMTO framework, the LSOP is decomposed into several nonseparable subproblems based on the variable grouping strategy. Each subproblem is regarded as a distinct subtask, and a specific number of subtasks are selected to formulate a multitask optimization problem (MTOP). Consequently, the original LSOP is redefined as a set of MTOPs, and a multitask evolution strategy incorporating with dynamic distance threshold and adaptive elite sampling knowledge-guided external sampling (MTES-DAKG) is proposed to solve these MTOPs in the study. Finally, the paradigm for addressing LSOPs has evolved from sequentially solving each subproblem to employing EMTO algorithms for solving a series of MTOPs. Meanwhile, in order to improve the optimization efficiency, a multitask optimization problem construction strategy is proposed in this study. In addition, the resource allocation strategy of MTOPs and subtasks based on contribution is also studied. The main contributions of this study are as follows

1) In order to improve optimization efficiency for solving LSOPs by utilizing the similarities of optimal domains and function shapes among decomposed subproblems, the CCMTO framework is proposed which redefines LSOPs as a series of MTOPs and optimizes them using the EMTO algorithm.

2) The construction strategy of multitask optimization problems is proposed to determine which subproblems are selected to construct a MTOP, as well as the appropriate number of tasks. Moreover, to determine the optimization order of these MTOPs and allocate computational resources for each subtask, the contribution-based resource allocation strategy of MTOPs and subtasks is designed.

3) An efficient EMTO algorithm, namely MTES-DAKG is proposed as an optimizer within the CCMTO framework to address the constructed MTOPs.

The rest of this paper is organized as follows: Section II. briefly reviews the background knowledge. Section III. exhibits the proposed CCMTO framework, components, and details of the proposed MTES-DAKG. Numerical experimental results on large-scale test suites and a large-scale application are given in Section IV. Finally, the conclusion and future work are drawn in Section Ⅴ.

## II. BACKGROUND KNOWLEDGE

### A. Cooperative Co-Evolutionary Framework

Generally, an unconstrained optimization problem can be described as

$$
\arg\min_{\mathbf{x} \in \mathbb{R}^d} f(\mathbf{x}) \tag{1}
$$

where $f(\mathbf{x})$ is the objective function, and $\mathbf{x}$ is a $d$-dimensional vector, called the decision variable or design variable. If $d$ is large enough (usually means that $d$ is much greater than 100 in the field of evolutionary computation), it is called a LSOP. If the analytical expression and gradient of $f(\mathbf{x})$ are not available, it is called a large-scale black-box optimization problem (LSBBOP). CCEAs are one of the representative approaches adopting the “divide and conquer” strategy to address LSOPs. Fig. 1 is a general diagram of CCEA.

In Fig. 1, it can be seen that the original LSOP is decomposed into $N$ low-dimensional subproblems, and the design variables of each subproblem are only a subset of the original problem. The variable grouping strategy generates these subproblems, and the resource allocation strategy selects which subproblem to be optimized in each co-evolutionary cycle after generating subproblems. Any EA can be utilized as the optimization solver to optimize the current subproblem. The red points in Fig. 1 represent collaborators selected from other subproblems, and they are combined to obtain the complete collaborators. Since the design variable in the current subproblem is only a segment of the original problem, individuals in the current subproblem need to combine with complete collaborators to form complete solutions when evaluating their fitness. In separable problems, the collaborator is generally set to the best solution so far.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/e4396f3a-7739-44f2-a218-7a4643fa7c12/3f2a9ad82c6f1b2fd37d1da95e0628e7a6c89e09be63d30cac065cf10fa24062.jpg)

Fig. 1. Diagram of a general CCEA

Obviously, when the origin LSOP is separable and can be grouped correctly, the optimization performance of CC framework can be great improved [45]. A partially separable problem is defined as follows

$$
\arg\min_{\mathbf{x}_1, \dots, \mathbf{x}_k} f(\mathbf{x}) = \left\{ \arg\min_{\mathbf{x}_1} f(\mathbf{x}_1, \dots), \dots, \arg\min_{\mathbf{x}_k} f(\dots, \mathbf{x}_k) \right\} \tag{2}
$$

where $k$ is the number of subproblems into which the origin problem can be decomposed, and $\boldsymbol{\psi}_1, \dots, \boldsymbol{\psi}_k$ are mutually exclusive subsets of the $d$-dimensional decision variable $\mathbf{x}$. If $k$ is equal to $d$, each subproblem contains only a one-dimensional decision variable and the original problem is called a fully separable problem.

### B. Evolutionary Multitask Optimization

Generalized MTOP with $K$ minimization tasks is defined as

$$
\arg\min_{\mathbf{x}_k \in \mathbb{R}^{D_k}} f_k(\mathbf{x}_k), \quad \text{for } k = 1, \dots, K \tag{3}
$$

where $D_k$ is the dimension of decision variables $\mathbf{x}_k$ in the $k$-th task. Each task has a corresponding search space, and all of them are transformed into a unified search space $Y$. For a solution $\mathbf{y}_k$ of task $k$, its representation $\mathbf{x}_k$ in the unified search space is calculated as follows

$$
\mathbf{x}_k = \frac{\mathbf{y}_k - \mathbf{L}_k}{\mathbf{U}_k - \mathbf{L}_k} \tag{4}
$$

where $\mathbf{L}_k$ and $\mathbf{U}_k$ are the lower and upper bounds of $\mathbf{y}_k$, respectively. The dimension of unified search space $D_Y$ is set to the maximum dimension of all tasks, as shown below

$$
D_Y = \max \left\{ D_1, \dots, D_K \right\} \tag{5}
$$

Since Gupta et al. [28] first proposed the multifactorial optimization (MFO) in 2016, the research on EMTO has gradually increased in recent years. The existing EMTO algorithms are mainly based on two knowledge transfer frameworks, the first one is MFO, and the second one is multipopulation evolution (MPE). The general framework of MFO and MPE is shown in **Fig. 2**.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/e4396f3a-7739-44f2-a218-7a4643fa7c12/57e0a050795c56f395731e1d29c2bb147e063bf618e02e1b7223962747197b06.jpg)

(b) MPE

Fig. 2. General framework of MFO and MPE

MFO generates only one population to optimize all $K$ tasks, and it assigns the most suitable task to each individual by introducing the indicator called skill factors. MFO performs knowledge transfer across tasks through assortative mating. In order to achieve information transmission between parents and offspring, MFO employs the vertical cultural transmission via selective imitation to endow offspring with skill factors. Multifactorial evolutionary algorithm (MFEA) [28] is the first and most representative MFO algorithm. Afterward, several research on MFO algorithm [37], [46], such as the methods of knowledge transfer, source task selection, and similarity measure between tasks has studied.

MPE optimizes multiple tasks through multiple populations, and each population can evolve through two distinct evolution mechanisms: intra-task self-evolution and inter-task crossevolution. Self-evolution involves crossover and mutation among individuals within the same populations. However, this approach cannot facilitate information exchange among distinct tasks. Consequently, an inter-population evolutionary mechanism is essential, and the cross-evolution among tasks is performed at the information exchange nodes. In recent years, many well performing MPE algorithms have emerged [36], [47].

### C. Multitask Evolution Strategy with Knowledge-Guided External Sampling

Evolution strategy (ES) is a kind of EA that evolves through probability distribution and is widely used in black-box global optimization. Covariance matrix adaptation ES (CMA-ES) stands out from other ES in global search performance and robustness by introducing a covariance matrix. CMA-ES updates covariance matrix and step size based on the ranking and displacement vector of sampled candidate solutions to search toward the optimal solution adaptively. In recent years, several research studies have tried to introduce knowledge transfer into ES. Li et al. [48] proposed a knowledge-guided external sampling (KGxS) approach and integrated KGxS into ES to develop a multitask ES (MTES) called MTES-KG. This approach includes two types of knowledge transfer methods to transfer samples that employs optimal domain similarity and function shape similarity among tasks.

#### Algorithm 1: MTES-KG with CMA-ES

**Input:** $\tau$ (external sample number), $K$ (number of tasks), $\alpha$ (knowledge type probability)  
**Output:** $\mathbf{x}_{1:K}^*$ (optimal solution)

1: **for** $k = 1$ **to** $K$ **do**  
2: &nbsp;&nbsp;&nbsp;&nbsp;Set $\mathbf{C}_k = \mathbf{I}$, $\mathbf{p}_{\sigma,k} = \mathbf{0}$, $\mathbf{p}_{c,k} = \mathbf{0}$ ;  
3: &nbsp;&nbsp;&nbsp;&nbsp;Initialize $\mathbf{m}_k$ in the unified search space ;  
4: **end**  
5: **while** stop criterion is not met **do**  
6: &nbsp;&nbsp;&nbsp;&nbsp;**for** $k = 1$ **to** $K$ **do**  
7: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**for** $i = 1$ **to** $\lambda$ **do**  
8: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\mathbf{x}_{k,i} \leftarrow \mathbf{m}_k + \sigma_k \mathbf{y}_{k,i}, \quad \mathbf{y}_{k,i} \sim \mathcal{N}(\mathbf{0}, \mathbf{C}_k)$ ;  
9: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$d_{k,i} \leftarrow \|\mathbf{x}_{k,i} - \mathbf{m}_k\|$ ;  
10: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end**  
11: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\langle d \rangle_{M,k} \leftarrow \frac{1}{\lambda} \sum_{i=1}^{\lambda} d_{k,i}$ ;  
12: &nbsp;&nbsp;&nbsp;&nbsp;**end**  
13: &nbsp;&nbsp;&nbsp;&nbsp;**for** $k = 1$ **to** $K$ **do**  
14: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**for** $i = \lambda + 1$ **to** $\lambda + \tau$ **do**  
15: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Randomly select a source task $s$ ($s \neq k$) ;  
16: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** $\mathrm{rand}(0,1) < \alpha$ **then**  
17: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\mathbf{z} \leftarrow \mathbf{m}_s + \sigma_s \mathbf{y}_{s,i}, \quad \mathbf{y}_{s,i} \sim \mathcal{N}(\mathbf{0}, \mathbf{C}_s)$ ;  
18: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** $\|\mathbf{z} - \mathbf{m}_k\| < \langle d \rangle_{M,k}$ **then**  
19: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\mathbf{x}_{k,i} \leftarrow \mathbf{z}$ ;  
20: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**else**  
21: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\mathbf{x}_{k,i} \leftarrow \mathbf{m}_k + \langle d \rangle_{M,k} \frac{\mathbf{z} - \mathbf{m}_k}{\|\mathbf{z} - \mathbf{m}_k\|}$ ;  
22: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end**  
23: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**else**  
24: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Select $j = \mathrm{randint}(1, \mu)$ ;  
25: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\langle \mathbf{y} \rangle_S \leftarrow \sum_{t=1, t \neq j}^{\mu} \mathbf{y}_{s,t}$ ;  
26: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\mathbf{x}_{k,i} \leftarrow \mathbf{m}_k + \sigma_k \mathbf{C}_k^{\frac{1}{2}} \mathbf{C}_s^{-\frac{1}{2}} \langle \mathbf{y} \rangle_S$ ;  
27: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end**  
28: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end**  
29: &nbsp;&nbsp;&nbsp;&nbsp;**end**  
30: &nbsp;&nbsp;&nbsp;&nbsp;**for** $k = 1$ **to** $K$ **do**  
31: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update distribution parameters $\mathbf{m}_k, \mathbf{p}_{\sigma,k}, \mathbf{p}_{c,k}, \sigma_k, \mathbf{C}_k$ ;  
32: &nbsp;&nbsp;&nbsp;&nbsp;**end**  
33: **end**  
34: **Return** $\mathbf{x}_{1:K}^*$ ;

The detailed procedure of MTES-KG with CMA-ES is shown in Algorithm 1, where $\tau$ and $\alpha$ are the number of external samples per iteration and the probability of using the two types of knowledge in KGxS, respectively. The core mechanism of KGxS is to transfer a small number of knowledge-guided samples from source task to target task, thereby providing promising search directions for improving the fitness value of the target task. KGxS is divided into two types of knowledge to transfer. 1) Domain KGxS: The optimal domain knowledge guides the probability distribution of the target task to search toward the distribution position of the source task, as shown in steps 16-22 in Algorithm 1. 2) Shape KGxS: The function shape knowledge provides search preference for the target distribution by learning the distribution of success samples in the source task in steps 23-27 in Algorithm 1.

## III. THE PROPOSED METHOD

### A. Motivation

Traditional CCEAs decompose LSOPs into a series of lower-dimensional subproblems to alleviate the solving complexity. These decomposed subproblems are optimized sequentially in a specific order, which will reduce the efficiency of solving LSOPs. According to the above descriptions, it can be seen that integrating the CC framework with the EMTO paradigm is feasible theoretically, which is able to solve LSOPs more efficiently by optimizing subproblems simultaneously. Therefore, this study presents a CCMTO framework. The CCMTO framework is equipped with two strategies, one is the construction strategy of multitask optimization problems aiming to select decomposed subproblems properly to construct MTOPs, and the other is the contribution-based resource allocation strategy of MTOPs and subtasks.

Although KGxS has been successfully extended to solve MTOPs, there exists some limitations in calculating the mean sample distance of the target task, determining the pulling direction, computing the elite samples’ center position of the source task, and determining the number of elite samples.

Domain KGxS calculates the mean sample distance of the target distribution $\mathcal{N}(\mathbf{0}, \mathbf{C}_k)$ as $\langle d \rangle_M$. However, the target sample distribution is scattered during the early iteration process of the algorithm. Employing a fixed $\langle d \rangle_M$ would lead to an excessive concentration of the pulled samples’ distribution, thereby diminishing their exploratory value. In the later stages of iteration process, the target sample distribution becomes increasingly localized within regions with high fitness values (i.e., low objective function values for minimization optimization problems). A fixed $\langle d \rangle_M$ increases the probability that the pulled samples fall near the distribution boundary, where the corresponding fitness values are relatively lower. Moreover, Domain KGxS pulls the sample $\mathbf{z}$ toward the direction from $\mathbf{m}$ to $\mathbf{z}$, regardless of whether this direction points towards the direction that increases fitness value of target task. If the pulling direction aligns with the direction of decreasing fitness value, the pulled sample will still fall within regions with lower fitness values, resulting in inefficient utilization of samples.

Shape KGxS calculates $\langle \mathbf{y} \rangle_S$ by applying equal weighting to the top $\mu$ elite samples. However, the fitness values among these elite samples can exhibit significant variance. The equal weighting approach diminishes the shape preference of samples with high fitness value, leading to a bias of $\langle \mathbf{y} \rangle_S$ towards samples with low fitness value and conveying shape knowledge with low accuracy. Furthermore, the number of elite samples is fixed and equal to $\mu$, whereas the characteristics of source sample distribution can vary significantly during different periods of iteration process. In the initial phase of iteration process, the variance of the sample fitness values is large. Using a large $\mu$ introduces elite samples with low fitness values, which can have a negative impact on the acquisition of shape knowledge. Conversely, in the later stages of iteration process, using a small $\mu$ will lose the detailed shape knowledge of elite samples, such as local search preferences.

To address the above issues, a MTES incorporating with dynamic distance threshold and adaptive elite sampling KGxS (MTES-DAKG) is proposed in this paper. It consists of a dynamic distance threshold domain KGxS with gradient correction and an adaptive elite sampling shape KGxS. The first domain KGxS approach increases the probability of samples falling within the region of high fitness values and ensures that the pulled sample always points towards the direction of increasing fitness value of the target distribution. The shape KGxS approach improves the accuracy of transferred shape knowledge.

It should be noted that the proposed CCMTO framework can only solve fully separable and partially separable LSOPs. When solving a MTOP, the fitness evaluation of a subtask’s individuals requires the collaboration of other subtasks’ individuals to complete the problem solutions. However, all design variables are interacted in nonseparable LSOPs, which implies that the objective function of a subtask will vary with changes in the collaborators of other tasks. If the objective function keeps changing during the optimization process, it will be difficult for the algorithm to find the optimal solution.

In real-world LSOPs, only a negligible number of them have strong correlations among all variables. For problems characterized by weak correlations, variables with weak correlations can be approximately decomposed by applying a correlation identification threshold, which transforms the original problem into a separable problem. Therefore, applying the CCMTO framework presents a novel and promising approach to address LSOPs.

### B. Overall Framework

The proposed CCMTO overall framework is given in Algorithm 2, and Fig. 3 is the flowchart of CCMTO. The original LSOP is decomposed into a series of low-dimensional subproblems by the variable grouping strategy in step 1 of Algorithm 2, then these decomposed subproblems are selected to construct several MTOPs by the construction strategy of multitask optimization problems in step 2. As shown in step 7 of Algorithm 2, the EMTO algorithm is utilized as the optimization solver to optimize the current MTOP, and the contribution of this MTOP is calculated in step 13. After undergoing a whole co-evolutionary cycle, the optimization order is determined by the proposed resource allocation strategy in step 16 of Algorithm 2.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/e4396f3a-7739-44f2-a218-7a4643fa7c12/e40546ceefca9246d3209659c1d9717a5a867449f3bbc5c29f44dbe544800453.jpg)
Fig. 3. Flowchart of CCMTO

#### Algorithm 2: The Proposed CCMTO Framework

**Input:** $f(\mathbf{x})$ (objective function)  
**Output:** $\mathbf{x}^{\mathrm{best}}$ (final optimal global solution)

1: $\{s_1, \dots, s_m\} \leftarrow \mathrm{Grouping}(f(\mathbf{x}))$ via variable grouping strategy ;  
2: $\{T_1, \dots, T_k\} \leftarrow \mathrm{MTOPs\_Construction}(\{s_1, \dots, s_m\})$ (Algorithm 3) ;  
3: Initialize contributions $\Delta F_i = 0$ ($i = 1, \dots, k$) and global best solution $\mathbf{x}^{\mathrm{best}}$ ;  
4: **while** stop criterion is not met **do**  
5: &nbsp;&nbsp;&nbsp;&nbsp;**for** $i = 1$ **to** $k$ **do**  
6: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Set last best solution $\mathbf{x}_{\mathrm{last}}^{\mathrm{best}} \leftarrow \mathbf{x}^{\mathrm{best}}$ ;  
7: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Optimize current MTOP $T_i = \{t_{i,1}, \dots, t_{i,n_i}\}$ ;  
8: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Obtain optimal solution of each task: $\mathbf{x}_{e_{i,j}}^{\mathrm{best}} = \arg\min f(\mathbf{x}_{e_{i,j}}; \mathbf{x}_{\notin e_{i,j}}^{\mathrm{best}}), \, j = 1, \dots, n_i$ ;  
9: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update optimal solution of $T_i$: $\mathbf{x}_{e_i}^{\mathrm{best}} \leftarrow \{\mathbf{x}_{e_{i,1}}^{\mathrm{best}}, \dots, \mathbf{x}_{e_{i,n_i}}^{\mathrm{best}}\}$ ;  
10: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** $f(\mathbf{x}_{e_i}^{\mathrm{best}}; \mathbf{x}_{\notin e_i}^{\mathrm{best}}) < f(\mathbf{x}^{\mathrm{best}})$ **then**  
11: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\mathbf{x}^{\mathrm{best}} \leftarrow \{\mathbf{x}_{e_i}^{\mathrm{best}}; \mathbf{x}_{\notin e_i}^{\mathrm{best}}\}$ ;  
12: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end**  
13: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update contribution $\Delta F_i$ of current MTOP ;  
14: &nbsp;&nbsp;&nbsp;&nbsp;**end**  
15: &nbsp;&nbsp;&nbsp;&nbsp;**while** $\min_{i=1,\dots,k}(\Delta F_i) \neq \max_{i=1,\dots,k}(\Delta F_i)$ **do**  
16: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Determine index $i$ of the MTOP to be optimized ;  
17: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Repeat steps 6 to 13 ;  
18: &nbsp;&nbsp;&nbsp;&nbsp;**end**  
19: **end**  
20: **Return** $\mathbf{x}^{\mathrm{best}}$ ;

### C. Construction Strategy of Multitask Optimization Problems

The construction strategy of multitask optimization problems is presented in Algorithm 3. The decomposed subproblems are sorted by the dimension in ascending order, and subproblems with the same dimensions are grouped into a group in step 1 of Algorithm 3. The number of subtasks in a MTOP is set to $n_{\mathrm{sub}}$.

As is well known, the dimension of unified search space is set to the maximum dimension of all tasks in EMTO algorithms, so the dimensional disparity among tasks is an important factor affecting the efficiency of optimization. Excessive dimensional differences between tasks can lead to poor optimal solution performance and waste of computational resources. The maximum dimension ratio $d_{\max}$ is introduced in the strategy. When there is only one subproblem in a group, if the dimensional difference between this subproblem and the subproblem in the nearest group is too large, this subproblem will be optimized separately. If the dimension ratio of this subproblem and the subproblem in the nearest group is within $[1 / d_{\max}, d_{\max}]$, this subproblem is put into the nearest group. For subproblems in the same group, if their number is greater than 1 and not greater than $n_{\mathrm{sub}}$, these subproblems are constructed to a MTOP in step 17 of Algorithm 3. Otherwise, every $n_{\mathrm{sub}}$ subproblems are selected randomly as a MTOP to construct a series of MTOPs in step 19 of Algorithm 3. The number of subproblems to construct each MTOP is also a significant factor, as too many or too few subproblems can affect the optimization efficiency. The parameter sensitivity discussions on $d_{\max}$ and $n_{\mathrm{sub}}$ are in Section IV.

#### Algorithm 3: Construction Strategy of Multitask Optimization Problems

**Input:** $\{s_{1}, \dots, s_{m}\}$ (decomposed subproblems), $\{d_{1}, \dots, d_{m}\}$ (subproblem dimensions), $n_{\mathrm{sub}}$ (number of subtasks in a MTOP), $d_{\max}$ (maximum dimension ratio)  
**Output:** $\{T_{1}, \dots, T_{k}\}$ (constructed MTOPs)

1: Sort subproblems by dimension in ascending order, and group subproblems with identical dimensions: $Group_1 = \{s_{1,1}, \dots, s_{1,n_1}\}, \dots, Group_j = \{s_{j,1}, \dots, s_{j,n_j}\}$, with dimensions $D_1, \dots, D_j$ ;  
2: **for** $i = 1$ **to** $j$ **do**  
3: &nbsp;&nbsp;&nbsp;&nbsp;**if** $\mathrm{card}(Group_i) == 1$ **then**  
4: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** $D_i / D_{i-1} > d_{\max}$ **and** $D_{i+1} / D_i > d_{\max}$ **then**  
5: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Treat subproblem $s_{i,1}$ as a single task and optimize separately ;  
6: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**elseif** $D_{i+1} / D_i \le d_{\max}$ **then**  
7: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Add subproblem $s_{i,1}$ into $Group_{i+1}$ ;  
8: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**else**  
9: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Add subproblem $s_{i,1}$ into $Group_{i-1}$ ;  
10: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end**  
11: &nbsp;&nbsp;&nbsp;&nbsp;**end**  
12: **end**  
13: **for** $i = 1$ **to** $\mathrm{num\_groups}$ **do**  
14: &nbsp;&nbsp;&nbsp;&nbsp;**if** $\mathrm{card}(Group_i) == 1$ **then**  
15: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Treat subproblem $s_{i,1}$ as a single task and optimize separately ;  
16: &nbsp;&nbsp;&nbsp;&nbsp;**elseif** $1 < \mathrm{card}(Group_i) \le n_{\mathrm{sub}}$ **then**  
17: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Select all subproblems in $Group_i$ to construct a MTOP $T_{i,1}$ ;  
18: &nbsp;&nbsp;&nbsp;&nbsp;**else**  
19: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Partition $Group_i$ by randomly selecting every $n_{\mathrm{sub}}$ subproblems to construct MTOPs $\{T_{i,1}, \dots, T_{i,m}\}$ ;  
20: &nbsp;&nbsp;&nbsp;&nbsp;**end**  
21: **end**  
22: **Return** $\{T_{1}, \dots, T_{k}\}$ ;

### D. MTES with Dynamic Distance Threshold and Adaptive Elite Sampling KGxS

1) Dynamic Distance Threshold Domain KGxS with Gradient Correction: Dynamic distance threshold domain KGxS with gradient correction divides the samples from the target distribution into different regions according to their fitness values, and calculates dynamic distance threshold for each region to replace the fixed $\langle d \rangle_M$. The sample $\mathbf{z}$ generated by the source distribution is assigned to the nearest region by calculating the mean distance between $\mathbf{z}$ and samples in each region. The gradient correction strategy estimates the gradient at the expectation of the target distribution to determine whether the vector direction from the sample generated by source distribution to the expectation of the target distribution is towards the direction of increasing fitness value. This strategy fine tunes the vector direction of the sample if it is towards the direction of decreasing fitness value.

Suppose that all tasks are minimization optimization problems, the general framework of the proposed dynamic distance threshold domain KGxS with gradient correction is given in Algorithm 4. In the first step, the samples of target distribution are sort by fitness values in descending order into different regions, the detailed regions are defined as follows

$$
\begin{aligned}
S_{\mathrm{high}} &= \left\{ \mathbf{x}_{1:\mu_1 \lambda}^{\mathrm{order}} \mid \mathbf{x}^{\mathrm{order}} \sim \mathcal{N}(\mathbf{m}_t, \mathbf{C}_t) \right\} \\
S_{\mathrm{mid}} &= \left\{ \mathbf{x}_{\mu_1 \lambda : (\mu_1 + \mu_2) \lambda}^{\mathrm{order}} \mid \mathbf{x}^{\mathrm{order}} \sim \mathcal{N}(\mathbf{m}_t, \mathbf{C}_t) \right\} \\
S_{\mathrm{low}} &= \left\{ \mathbf{x}_{(\mu_1 + \mu_2) \lambda : \lambda}^{\mathrm{order}} \mid \mathbf{x}^{\mathrm{order}} \sim \mathcal{N}(\mathbf{m}_t, \mathbf{C}_t) \right\}
\end{aligned} \tag{6}
$$

where $\lambda$ is the number of samples of target distribution, $\mathbf{x}^{\mathrm{order}}$ is target distribution samples sorted by their fitness values in the descending order, $\mu_1$ and $\mu_2$ are the proportion coefficients of samples in the first and second regions, respectively.

#### Algorithm 4: Dynamic Distance Threshold Domain KGxS with Gradient Correction

**Input:** $X_t = \{\mathbf{x}_{t,1}, \dots, \mathbf{x}_{t,\lambda}\} \sim \mathcal{N}(\mathbf{m}_t, \mathbf{C}_t)$ (target task samples), $n$ (dimension), $k$ (KNN count), $\beta$ (gradient step), $\mathcal{N}(\mathbf{m}_s, \mathbf{C}_s)$ (source distribution)  
**Output:** $\hat{\mathbf{x}}$ (knowledge-guided external sample)

1: Sort $X_t$ by fitness value in descending order into regions $\{S_{\mathrm{high}}, S_{\mathrm{mid}}, S_{\mathrm{low}}\}$ according to Eq. (6) ;  
2: Calculate dynamic distance thresholds $\langle d \rangle_{\mathrm{high}}, \langle d \rangle_{\mathrm{mid}}, \langle d \rangle_{\mathrm{low}}$ ;  
3: Sample $\mathbf{z} \sim \mathcal{N}(\mathbf{m}_s, \mathbf{C}_s)$ ;  
4: **for** each region $S_g \in \{S_{\mathrm{high}}, S_{\mathrm{mid}}, S_{\mathrm{low}}\}$ ($g \in \{\mathrm{high}, \mathrm{mid}, \mathrm{low}\}$) **do**  
5: &nbsp;&nbsp;&nbsp;&nbsp;Calculate $\mathrm{dist}(\mathbf{z}, \mathbf{x}) = \|\mathbf{z} - \mathbf{x}\|$ for all $\mathbf{x} \in S_g$ ;  
6: &nbsp;&nbsp;&nbsp;&nbsp;Sort $\mathrm{dist}(\mathbf{z}, \mathbf{x})$ in ascending order, take top $k$ as $k\text{-}\mathrm{NN}_g$ ;  
7: &nbsp;&nbsp;&nbsp;&nbsp;Calculate average distance: $\mathrm{avg\_dist}_g = \frac{1}{k} \sum_{\mathbf{x} \in k\text{-}\mathrm{NN}_g} \mathrm{dist}(\mathbf{z}, \mathbf{x})$ ;  
8: **end**  
9: Determine region affiliation: $S_{g^*} = \arg\min_{g} (\mathrm{avg\_dist}_{\mathrm{high}}, \mathrm{avg\_dist}_{\mathrm{mid}}, \mathrm{avg\_dist}_{\mathrm{low}})$ ;  
10: Generate standard basis unit vectors $\mathbf{e}_1, \dots, \mathbf{e}_n$ ;  
11: **for** $i = 1$ **to** $n$ **do**  
12: &nbsp;&nbsp;&nbsp;&nbsp;Calculate gradient component: $g_i = \frac{f(\mathbf{m}_t + \beta \mathbf{e}_i) - f(\mathbf{m}_t - \beta \mathbf{e}_i)}{2\beta}$ ;  
13: &nbsp;&nbsp;&nbsp;&nbsp;Calculate optimal direction component: $g_i^{\mathrm{opt}} = \begin{cases} -g_i, & \text{if } g_i > 0 \\ g_i, & \text{if } g_i \le 0 \end{cases}$ ;  
14: **end**  
15: Normalize optimal direction vector: $\mathbf{g}_{\mathrm{grad}}^{\mathrm{opt}} = \frac{[g_1^{\mathrm{opt}}, \dots, g_n^{\mathrm{opt}}]^T}{\|[g_1^{\mathrm{opt}}, \dots, g_n^{\mathrm{opt}}]^T\|}$ ;  
16: Calculate direction vector: $\mathbf{v} = \frac{\mathbf{z} - \mathbf{m}_t}{\|\mathbf{z} - \mathbf{m}_t\|}$ ;  
17: Calculate angle between $\mathbf{v}$ and $\mathbf{g}_{\mathrm{grad}}^{\mathrm{opt}}$: $\theta = \arccos\left(\frac{\mathbf{v} \cdot \mathbf{g}_{\mathrm{grad}}^{\mathrm{opt}}}{\|\mathbf{v}\| \cdot \|\mathbf{g}_{\mathrm{grad}}^{\mathrm{opt}}\|}\right)$ ;  
18: **if** $\theta < 90^\circ$ **then**  
19: &nbsp;&nbsp;&nbsp;&nbsp;**if** $\|\mathbf{z} - \mathbf{m}_t\| < \langle d \rangle_{g^*}$ **then**  
20: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\hat{\mathbf{x}} \leftarrow \mathbf{z}$ ;  
21: &nbsp;&nbsp;&nbsp;&nbsp;**else**  
22: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\hat{\mathbf{x}} \leftarrow \mathbf{m}_t + \langle d \rangle_{g^*} \cdot \mathbf{v}$ ;  
23: &nbsp;&nbsp;&nbsp;&nbsp;**end**  
24: **else**  
25: &nbsp;&nbsp;&nbsp;&nbsp;Calculate correction vector $\mathbf{v}'$ according to Eq. (7) ;  
26: &nbsp;&nbsp;&nbsp;&nbsp;$\hat{\mathbf{x}} \leftarrow \mathbf{m}_t + \langle d \rangle_{g^*} \cdot \mathbf{v}'$ ;  
27: **end**  
28: **Return** $\hat{\mathbf{x}}$ ;

The dynamic distance threshold $\langle d \rangle_g$ is calculated to obtain the mean distance of samples from the expectation $\mathbf{m}_t$ of the target distribution in each region in step 2 of Algorithm 4. Next, the sample $\mathbf{z}$ is generated by the source distribution in step 3. The region affiliation of $\mathbf{z}$ is determined in steps 4-9 of Algorithm 4. First, for each region, the distances from $\mathbf{z}$ to all samples within the region are calculated and sorted in ascending order. Then, by using K-Nearest Neighbor algorithm, the average of the first $k$ distances is computed, and the sample $\mathbf{z}$ is assigned to the region with the smallest average distance. The normalized optimal direction vector at the target expectation $\mathbf{m}_t$ is estimated in steps 10-15 of Algorithm 4. The direction vector from $\mathbf{m}_t$ to $\mathbf{z}$, as well as the angle between direction vector and optimal direction vector, is calculated in steps 16-17 of Algorithm 4. If the angle is less than $90^\circ$ and the distance from $\mathbf{m}_t$ to $\mathbf{z}$ is less than $\langle d \rangle_{g^*}$, i.e., $\mathbf{z}$ is in the domain of the target distribution and the direction vector is towards the direction that increases fitness value of target task, the sample $\mathbf{z}$ is directly received as an external sample. If the angle is less than $90^\circ$ but this distance is more than $\langle d \rangle_{g^*}$, $\mathbf{z}$ is pulled toward $\mathbf{m}_t$ as length as $\langle d \rangle_{g^*}$. When the angle is more than $90^\circ$, $\mathbf{z}$ is pulled toward the correction vector $\mathbf{v}'$ as length as $\langle d \rangle_g$. The correction vector $\mathbf{v}'$ is calculated as follows

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/e4396f3a-7739-44f2-a218-7a4643fa7c12/aad30f8b4f545285878c008b4347612bdee648db8deedb5e8102fde4b9b71ec8.jpg)
![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/e4396f3a-7739-44f2-a218-7a4643fa7c12/8b82383b6d7e7b501a2d99eaf7cb141f03694a33087bd0cc839017fa0d4b4dc3.jpg)
![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/e4396f3a-7739-44f2-a218-7a4643fa7c12/fbe99a5bb45d19c6f6478e8bf1d8afb18254cf268dfbf48bcae8ef4447fcfc94.jpg)

(c) KGxS with gradient correction

Fig. 4. Dynamic distance threshold domain KGxS with gradient correction

$$
\mathbf{v}' = \frac{\mathbf{v} + \varphi \cdot \mathbf{g}_{\mathrm{grad}}^{\mathrm{opt}}}{\left\| \mathbf{v} + \varphi \cdot \mathbf{g}_{\mathrm{grad}}^{\mathrm{opt}} \right\|} \tag{7}
$$

where $\mathbf{v}$ is the normalized direction vector from $\mathbf{m}_t$ to $\mathbf{z}$, $\mathbf{g}_{\mathrm{grad}}^{\mathrm{opt}}$ is the normalized optimal direction vector, and $\varphi$ is the gradient correction coefficient to ensure that the angle between correction vector and optimal direction vector is less than $90^\circ$.

Compared with domain KGxS approach in [48], the dynamic distance threshold ensures that the external samples accurately fall within the distribution of their regions respectively, and convey information about the source distribution more effectively. This KGxS strategy increases the probability of samples falling within the regions with high fitness values and avoids wasting computational resources in the regions with low fitness values. Meanwhile, the gradient correction strategy can ensure that the pulled sample always points towards the direction of increasing fitness value of the target distribution and increase the possibility of improving the target fitness value through external sampling.

In order to visually display the details of the proposed dynamic distance threshold domain KGxS with gradient correction, Fig. 4 takes three subfigures as the example to show three types of the external sampling. In subfigure (a), the sample (red dot) generated by the source distribution is in target distribution’s high-potential region, i.e., region with high fitness values, and the angle between direction vector and optimal direction vector is an acute angle. Therefore, the sample is accepted directly as an external sample of the target distribution. As shown in subfigure (b), the red diamond indicates a sample generated by the source distribution, which is classified in target distribution’s high-potential region. This sample exceeds the dynamic distance threshold, and the angle between direction vector and optimal direction vector is an acute angle. Therefore, the red diamond is pulled towards the direction vector and then located at the high-potential boundary as the external sample (red dot). In subfigure (c), the red diamond indicates a sample generated by the source distribution, which is classified in target distribution’s low-potential region. The angle between direction vector and optimal direction vector is an obtuse angle, which means that the direction vector will be towards the direction of decreasing fitness value. The red diamond is pulled toward the correction vector and then located at the low-potential boundary as the external sample (red dot).

1) Adaptive Elite Sampling Shape KGxS: Adaptive elite sampling shape KGxS assigns weights based on fitness values of samples and enhances the contribution of elite samples with high fitness values. Meanwhile, adaptive elite sampling shape KGxS dynamically adjusts the number of elite samples to improve accuracy of shape knowledge. The detailed procedure of the proposed adaptive elite sampling shape KGxS is shown in Algorithm 5. The number of elite samples $n$ increases with the number of iterations increases, and is calculated according to the following equation

$$
n = \begin{cases} \left( \frac{\mathrm{gen}}{M} \right)^a (n_{\max} - n_{\min}) + n_{\min}, & \text{if } \mathrm{gen} \le M \\ n_{\max}, & \text{otherwise} \end{cases} \tag{8}
$$

where $\mathrm{gen}$ is the current number of generations, $M = 0.6 \mathrm{Maxgen}$, $\mathrm{Maxgen}$ is the maximum number of generations. $n_{\max}$ and $n_{\min}$ represent the maximum elite sample number and the minimum elite sample number, respectively. $a$ is a parameter that controls the increasing rate of the elite sample number. In step 2 of Algorithm 5, the samples of the source distribution are sort by fitness values in descending order, and the top $n$ samples are selected as the elite set. Then, the weight of each elite sample is computed as follows

$$
w_{si} = \frac{\exp\left(\gamma \cdot \frac{n - i}{n}\right)}{\sum_{k=1, k \neq j}^n \exp\left(\gamma \cdot \frac{n - k}{n}\right)}, \quad i = 1, \dots, n, \, i \neq j, \, j = \mathrm{randint}(1, n) \tag{9}
$$

where $w_{si}$ represents the weight of the $i$-th elite sample, and $\gamma$ is the weight coefficient. The random number $j$ is randomly excluded from the computation to ensure that the $w_{si}$ generated by each execution is different. In step 4 of Algorithm 5, the center position $\langle \mathbf{y} \rangle_S$ of elite samples $\mathbf{z}_{s,1:n}$ relative to the expectation $\mathbf{m}_s$ of the source distribution is computed. In the end, $\langle \mathbf{y} \rangle_S$ is transformed into an external sample of the target distribution by applying the domain alignment approach.

#### Algorithm 5: Adaptive Elite Sampling Shape KGxS

**Input:** $X_t = \{\mathbf{x}_{t,1}, \dots, \mathbf{x}_{t,\lambda}\} \sim \mathcal{N}(\mathbf{m}_t, \mathbf{C}_t)$ (target task samples), $X_s = \{\mathbf{x}_{s,1}, \dots, \mathbf{x}_{s,\lambda}\} \sim \mathcal{N}(\mathbf{m}_s, \mathbf{C}_s)$ (source task samples)  
**Output:** $\hat{\mathbf{x}}$ (knowledge-guided external sample)

1: Calculate dynamic elite sample count $n$ according to Eq. (8) ;  
2: Sort $X_s$ by fitness value in ascending order, take top $n$ as elite set $Z_s = \{\mathbf{z}_{s,1}, \dots, \mathbf{z}_{s,n}\}$ ;  
3: Calculate weights $w_{si}$ ($i = 1, \dots, n, i \neq j$) according to Eq. (9) with $j = \mathrm{randint}(1, n)$ ;  
4: Calculate weighted center position relative to source mean: $\langle \mathbf{y} \rangle_S \leftarrow \sum_{i=1, i \neq j}^{n} w_{si} \cdot (\mathbf{z}_{s,i} - \mathbf{m}_s)$ ;  
5: Transform center position to target space: $\hat{\mathbf{x}} \leftarrow \mathbf{m}_t + \mathbf{C}_t^{\frac{1}{2}} \mathbf{C}_s^{-\frac{1}{2}} \langle \mathbf{y} \rangle_S$ ;  
6: **Return** $\hat{\mathbf{x}}$ ;

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/e4396f3a-7739-44f2-a218-7a4643fa7c12/b0f6d3a32c8ba9cd11c89e03e24a22a16316bbd400ec83043205ebe6fc528eca.jpg)

Fig. 5. Adaptive elite sampling shape KGxS

Compared with shape KGxS approach in [48], the adaptive elite sampling shape KGxS calculates $\langle \mathbf{y} \rangle_S$ by assigning weights to elite samples in descending order of fitness, and highlights the contributions of elite samples with high fitness values. The shape preference of these samples can dominate $\langle \mathbf{y} \rangle_S$ to improve the accuracy of transferred shape knowledge. The adaptive adjustment of elite sample number can utilize more shape details of elite samples and improve shape knowledge accuracy.

Fig. 5 shows two types of adaptive elite sampling shape KGxS. In subfigure (a), elite samples of the source distribution are denoted as red diamonds within black circle, showing a trend of searching to the right side of the source distribution. The external sample in the target distribution generated through adaptive elite sampling shape KGxS also tends to search towards the right. In subfigure (b), elite samples are searching towards the center of the source distribution, which guide the external sample to search towards the target distribution center.

It should be noted that the number $\lambda$ of generated samples in MTES-DAKG is less than 100 set in [48]. So, the number $\tau$ of external samples should be set to a smaller value, which is discussed in Section IV. Numerical Experiments and Discussions. In addition, using external sampling every generation may affect the task distribution and optimization efficiency owing to the small $\lambda$, the frequency of external sampling $fre$ is introduced in MTES-DAKG, and its value is also discussed in Section IV. Numerical Experiments and Discussions.

### E. Contribution-Based Resource Allocation Strategy of MTOPs and Subtasks

1) Stagnant Subtask Detection: In the proposed CCMTO framework, the original LSOP is decomposed into a series of MTOPs, and each MTOP contains multiple subtasks. When using EMTO algorithms for solving a MTOP, all subtasks in this MTOP are optimized simultaneously. However, not all subtasks are equally difficult to solve, and for the subtasks that are easy to be optimized, a small number of computational resources are sufficient to obtain their optimal solutions. At this point, continuing to allocate computing resources to these subtasks does not make contributions to the improvement of the best overall objective function value, as the corresponding subpopulations are in a stagnant stage. Therefore, a stagnant subtask detection method is proposed, in which computational resources are no longer allocated to these stagnant subpopulations. This mechanism can save some computational cost on stagnant subtasks to improve the efficiency of the proposed CCMTO framework.

In order to check whether the subtask is stagnant, the proposed stagnant subtask detection method employs both fitness value improvement and population diversity as detection indicators by calculating the relative variation of objective function values, and the relative variation of the mean and standard deviation of individuals’ design variable values in dimension.

Suppose $T_i$ denotes the $i$-th constructed MTOP through the construction strategy of multitask optimization problems, and $t_{i,j}$ denotes the $j$-th subtask in the MTOP. For the subpopulation corresponding to $t_{i,j}$ at the $G$-th generation, the relative variation of the best objective function values, and the relative variation of the mean and standard deviation of individuals design variable values in dimension are calculated as follows

$$
\Delta f_G = \left| \frac{f(\mathbf{x}_{\in t_{i,j}, G-1}^{\mathrm{best}}; \mathbf{x}_{\notin t_{i,j}}) - f(\mathbf{x}_{\in t_{i,j}, G}^{\mathrm{best}}; \mathbf{x}_{\notin t_{i,j}})}{f(\mathbf{x}_{\in t_{i,j}, G-1}^{\mathrm{best}}; \mathbf{x}_{\notin t_{i,j}})} \right| \tag{10}
$$

$$
\Delta m_{d,G} = \left| \frac{m_{d,G-1} - m_{d,G}}{m_{d,G-1}} \right|, \quad m_{d,G} = \frac{1}{N} \sum_{n=1}^N x_{\in t_{i,j}, d, G}^n \tag{11}
$$

$$
\Delta \mathrm{std}_{d,G} = \left| \frac{\mathrm{std}_{d,G-1} - \mathrm{std}_{d,G}}{\mathrm{std}_{d,G-1}} \right|, \quad \mathrm{std}_{d,G} = \sqrt{\frac{1}{N} \sum_{n=1}^N (x_{\in t_{i,j}, d, G}^n - m_{d,G})^2} \tag{12}
$$

where $\mathbf{x}_{\in t_{i,j}, G}^{\mathrm{best}}$ is the best solution at the $G$-th generation, and the collaborator $\mathbf{x}_{\notin t_{i,j}}$ is set to be fixed. $N$ is the subpopulation size, $\mathbf{x}_{\in t_{i,j}, G}^n = (\mathbf{x}_{\in t_{i,j}, 1, G}^n, \dots, \mathbf{x}_{\in t_{i,j}, D, G}^n)$ is $n$-th individual, $D$ is the dimension of decision variables, and $x_{\in t_{i,j}, d, G}^n$ is the $d$-th variable value of $\mathbf{x}_{\in t_{i,j}, G}^n$. If the relative variation of the best objective function values remains unchanged for several successive generations, this subpopulation is considered to be stagnant in fitness value. When the relative variation of both the mean and standard deviation of individuals’ $d$-th design variable value remains unchanged over several successive generations, this subpopulation can be considered to be stagnant in this dimension [18]. Only when a subpopulation is stagnant in fitness value and in all dimensions, the subtask can be considered to be in a stagnant state. The method to check whether a subpopulation is stagnant in fitness value is as shown in follows

$$
v_G = \begin{cases} v_{G-1} + 1, & \text{if } \Delta f_G < \varepsilon \\ 0, & \text{otherwise} \end{cases} \tag{13}
$$

where $v_G$ denotes the number of successive generations where the value best objective function remains unchanged, and note that $v_0 = 0$. $\varepsilon$ is the threshold of objective function value variation, and the value is $1\text{e-}6$ in this study.

If a subpopulation is stagnant in the $d$-th dimension, the indicator $\varphi_{d,G}$ is defined as follows

$$
\varphi_{d,G} = \begin{cases} 1, & \text{if } \Delta m_{d,G} < \varepsilon \text{ and } \Delta \mathrm{std}_{d,G} < \varepsilon \\ 0, & \text{otherwise} \end{cases} \tag{14}
$$

where $\varphi_{d,G}$ denotes whether the mean and standard deviation of individuals’ design variable values in dimension $d$ remain unchanged from the last generation, and note that $\varphi_{d,0} = 0$. Then $\sigma_G$ denotes the number of dimensions where $\varphi_{d,G} = 1$:

$$
\sigma_G = \sum_{d=1}^D \varphi_{d,G} \tag{15}
$$

If the subpopulation is stagnant in all dimensions, $\sigma_G = D$, $\eta_G$ denotes the number of successive generations where $\sigma_G = D$, and note that $\eta_0 = 0$:

$$
\eta_G = \begin{cases} \eta_{G-1} + 1, & \text{if } \sigma_G = D \\ 0, & \text{otherwise} \end{cases} \tag{16}
$$

When the subpopulation is stagnant in fitness value and in all dimensions for successive generations, the subtask is in a stagnant state, and the detection flag $\rho_G$ is calculated as follows

$$
\rho_G = \begin{cases} 1, & \text{if } v_G \ge U \text{ and } \eta_G \ge U \\ 0, & \text{otherwise} \end{cases} \tag{17}
$$

where $U$ is a parameter and is defined as

$$
U = \min(D, \mathrm{Maxgen}) \tag{18}
$$

Once $\rho_G = 1$ for a subpopulation, computational resources are immediately no longer allocated to this stagnant subtask. This subtask is excluded from the optimization of the MTOP it belongs to, which means that it will not undergo evolution in the CCMTO framework.

1) Resource Allocation Strategy: For a MTOP $T_i$, after finishing optimization in a cycle, its contribution is calculated as follows

$$
\Delta F_i = |f(\mathbf{x}_{\mathrm{last}}^{\mathrm{best}}) - f(\mathbf{x}^{\mathrm{best}})| \tag{19}
$$

where $f(\mathbf{x}_{\mathrm{last}}^{\mathrm{best}})$ and $f(\mathbf{x}^{\mathrm{best}})$ are the best overall objective values before and after $T_i$ undergoes optimization, respectively. The contribution-based resource allocation strategy of MTOPs and subtasks is shown in Algorithm 6.

#### Algorithm 6: Contribution-Based Resource Allocation Strategy of MTOPs and Subtasks

**Input:** $\{T_{1}, \dots, T_{k}\}$ (constructed MTOPs), subtasks in each MTOP $T_i = \{t_{i,1}, \dots, t_{i,n_i}\}$  
**Output:** $\mathbf{x}^{\mathrm{best}}$ (final optimal global solution)

1: Set contributions $\Delta F_i = 0$, stagnant subtask sets $S_i = \varnothing$ ($i = 1, \dots, k$), initialize global best solution $\mathbf{x}^{\mathrm{best}}$ ;  
2: **while** stop criterion is not met **do**  
3: &nbsp;&nbsp;&nbsp;&nbsp;**for** each MTOP $T_i$ ($i = 1, \dots, k$) **do** reset $S_i = \varnothing, v^{i,j} = 0, \eta^{i,j} = 0$ ($j = 1, \dots, n_i$) ;  
4: &nbsp;&nbsp;&nbsp;&nbsp;**for** $i = 1$ **to** $k$ **do**  
5: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\mathbf{x}_{\mathrm{last}}^{\mathrm{best}} \leftarrow \mathbf{x}^{\mathrm{best}}$ ;  
6: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**for** $j = 1$ **to** $n_i$ **do**  
7: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\mathbf{x}_{e_{i,j}}^{\mathrm{best}} \leftarrow$ Optimize subtask $t_{i,j}$ by EMTO optimizer ;  
8: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** $\rho_G^{i,j} == 1$ **then**  
9: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$S_i \leftarrow S_i \cup \{j\}$ ;  
10: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end**  
11: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end**  
12: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update MTOP solution $\mathbf{x}_{e_i}^{\mathrm{best}} \leftarrow \{\mathbf{x}_{e_{i,1}}^{\mathrm{best}}, \dots, \mathbf{x}_{e_{i,n_i}}^{\mathrm{best}}\}$ and global solution $\mathbf{x}^{\mathrm{best}} \leftarrow \{\mathbf{x}_{e_i}^{\mathrm{best}}; \mathbf{x}_{\notin e_i}^{\mathrm{best}}\}$ ;  
13: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\Delta F_i \leftarrow |f(\mathbf{x}_{\mathrm{last}}^{\mathrm{best}}) - f(\mathbf{x}^{\mathrm{best}})|$ ;  
14: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** all $\rho_G^{i,j} == 1$ ($j = 1, \dots, n_i$) **then**  
15: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\Delta F_i \leftarrow 0$ ;  
16: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end**  
17: &nbsp;&nbsp;&nbsp;&nbsp;**end**  
18: &nbsp;&nbsp;&nbsp;&nbsp;**while** $\min_{i=1,\dots,k}(\Delta F_i) \neq \max_{i=1,\dots,k}(\Delta F_i)$ **do**  
19: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$i \leftarrow \arg\max_{l \in \{1,\dots,k\}} (\Delta F_l)$ ;  
20: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\mathbf{x}_{\mathrm{last}}^{\mathrm{best}} \leftarrow \mathbf{x}^{\mathrm{best}}$ ;  
21: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**for** $j = 1$ **to** $n_i$ **do**  
22: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** $j \notin S_i$ **then**  
23: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\mathbf{x}_{e_{i,j}}^{\mathrm{best}} \leftarrow$ Optimize subtask $t_{i,j}$ by EMTO optimizer ;  
24: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** $\rho_G^{i,j} == 1$ **then**  
25: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$S_i \leftarrow S_i \cup \{j\}$ ;  
26: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end**  
27: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end**  
28: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end**  
29: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update MTOP solution $\mathbf{x}_{e_i}^{\mathrm{best}} \leftarrow \{\mathbf{x}_{e_{i,1}}^{\mathrm{best}}, \dots, \mathbf{x}_{e_{i,n_i}}^{\mathrm{best}}\}$ and global solution $\mathbf{x}^{\mathrm{best}} \leftarrow \{\mathbf{x}_{e_i}^{\mathrm{best}}; \mathbf{x}_{\notin e_i}^{\mathrm{best}}\}$ ;  
30: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\Delta F_i \leftarrow |f(\mathbf{x}_{\mathrm{last}}^{\mathrm{best}}) - f(\mathbf{x}^{\mathrm{best}})|$ ;  
31: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** all $\rho_G^{i,j} == 1$ ($j = 1, \dots, n_i$) **then**  
32: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\Delta F_i \leftarrow 0$ ;  
33: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end**  
34: &nbsp;&nbsp;&nbsp;&nbsp;**end**  
35: **end**  
36: **Return** $\mathbf{x}^{\mathrm{best}}$ ;

In the first, the contribution of every MTOP is set to zero, and the stagnant subtask set of every MTOP is set to the empty set. During the first co-evolutionary cycle, all MTOPs undergo optimization one by one in steps 4-17 of Algorithm 6. If a subtask $t_{i,j}$ is in a stagnant state, i.e. $\rho_G^{i,j} = 1$ when optimizing the MTOP $T_i$, the index $j$ of this subtask is put into the set $S_i$, and no more computational resources will be allocated to this subtask in the subsequent co-evolutionary cycles. The value of $\Delta F_i$ for each MTOP is computed according to Eq. (19) in step 13 of Algorithm 6. If all subtasks in a MTOP are all in a stagnant state, i.e. all $\rho_G^{i,j} = 1$, this MTOP can be considered to be in a stagnant state, and the value of $\Delta F_i$ is set to zero. In the subsequent co-evolutionary cycles, the MTOP with the largest value of $\Delta F_i$ is selected to undergo optimization in step 19 of Algorithm 6. In steps 21-28 of Algorithm 6, the subtasks that have been considered to be stagnant will be excluded from the MTOP optimization cycle, and the index of stagnant subtasks is also put into the set $S_i$ during optimization. Then, the contribution value of this MTOP is updated according to Eq. (19) at the end of the optimization. When the values of $\Delta F_i$ are same for all the MTOPs, the algorithm will go to steps 3-17 to reset the contribution and the stagnant subtask set of each MTOP, then initiates a new co-evolutionary cycle. The above process is repeated until the stop criterion is met.

The proposed stagnant subtask detection can accurately identify a stagnant subtask according to the mean and standard deviation of individuals’ gene values, as well as the best objective function value in the subpopulation. Meanwhile, this resource allocation strategy can save computational cost on stagnant subtasks, and make a more efficient computational resource allocation among subtasks.

### F. Computational Complexity

In the proposed MTES-DAKG, assuming that $K$ is the number of subtasks in a MTOP, $\lambda$ is the sample number for each task, and $n$ is the maximum dimension of decision variables. The computational complexity of MTES-DAKG in one evolutionary loop is analyzed as follows. The complexity of sample new solutions is $O(n^2 \cdot K)$. In the dynamic distance threshold domain KGxS with gradient correction as shown in Algorithm 4, calculating dynamic distance threshold takes $O(\lambda \cdot n \cdot K)$ in step 2 of Algorithm 4. Then, calculating the average distance takes $O(n \cdot K)$ in step 4-8 of Algorithm 4. Subsequently, $O(n^2 \cdot K)$ is used to calculate the optimal direction in step 11-16 of Algorithm 4. At last, calculating angle and conducting gradient correction takes $O(n \cdot K)$ in step 17-27 of Algorithm 4. Therefore, the total computational complexity is $O(n^2 \cdot K)$, which is the same as the original domain KGxS. In the adaptive elite sampling shape KGxS as shown in Algorithm 5, it needs $O(\lambda \cdot \log(\lambda) \cdot K)$ to sort the elite samples in step 2 of Algorithm 5. Then, calculating the center position takes $O(n \cdot K)$ in step 4 of Algorithm 5. At last, $O(n^2 \cdot K)$ is used to transform the external sample in step 5 of Algorithm 5. Overall, the total computational complexity is $\max(O(\lambda \cdot \log(\lambda) \cdot K), O(n^2 \cdot K))$. In summary, the computational complexity of MTES-DAKG is $\max(O(\lambda \cdot \log(\lambda) \cdot K), O(n^2 \cdot K))$, which is equal to MTES-KG. The proposed MTES-DAKG can improve the optimization performance through finer external sample transfer without increasing the computational complexity.

Actually, the proposed CCMTO framework is just transformed the paradigm for addressing LSOPs from sequentially solving each subproblem to employing EMTO algorithms for solving a series of MTOPs. Therefore, compared to traditional CC framework, the CCMTO framework does not increase additional computational complexity.

## IV. NUMERICAL EXPERIMENTS AND DISCUSSIONS

In this section, the proposed CCMTO using MTES-DAKG is tested and compared with several state-of-the-art large-scale algorithms developed in recent years. Then, the numerical experiments are conducted to analyze the parameter sensitivity, as well as the effectiveness of components within the proposed algorithm. At last, the proposed algorithm is employed in a real-world large-scale problem to verify its applicability.

### A. Benchmark Test Suite and Parameter Settings

The proposed CCMTO with MTES-DAKG (CCMTO-MTES-DAKG) is tested on CEC2010 and CEC2013 largescale benchmark test suites [45] with dimensions up to 1000. The CEC2010 and CEC2013 test suites are widely used in large-scale problem algorithm testing, consisting of 20 and 15 LSOPs, respectively. According to the separability of the problems, the benchmark suites could be divided into three categories, fully separable problems, partially separable problems, and nonseparable problems [25]. Because CCMTO-MTES-DAKG can only solve fully separable and partially separable LSOPs, the first 18 benchmark test problems on CEC2010 test suite, and the first 11 benchmark test problems on CEC2013 test suite, are used to test the performance of CCMTO-MTES-DAKG. The stop criterion for all compared algorithms in this study is that the maximum number of fitness evaluations MaxFEs reaches to $3\text{e}6$. EDG [12] is an efficient differential grouping method, and it is employed in CCMTO-MTES-DAKG in this study. The FEs used by EDG are included in the total number of FEs. The parameter settings for CCMTO and MTES-DAKG are given in TABLE I, and the parameters in bold fonts will be discussed further in a later section.

All algorithms were run independently 25 times in experiments. To better assess the comparison results, the Welch’s t-test was conducted for all comparison algorithms, and Wilcoxon’s rank-sum test with the Holm–Bonferroni correction was utilized for each parameter setting. The significance level was set at $\alpha = 0.05$. “+/≈/-” indicates that the proposed method is significant better/equal/worse than the comparison algorithm, respectively.

TABLE I
THE PARAMETER SETTINGS FOR CCMTO-MTES-DAKG

| Parameter Category / Description | Value / Setting |
| :--- | :--- |
| **The parameter settings for CCMTO framework:** | |
| Number of subtasks in a MTOP | $n_{\mathrm{sub}} = 5$ |
| Maximum dimension ratio | $d_{\max} = 2$ |
| Threshold of objective function value variation | $\varepsilon = 1\text{e-}6$ |
| **The parameter settings for MTES-DAKG:** | |
| Proportion coefficient of samples in the first region | $\mu_1 = \begin{cases} 0.4 & \text{if } \lambda \le 6 \\ 0.3 & \text{otherwise} \end{cases}$ |
| Proportion coefficient of samples in the second region | $\mu_2 = \begin{cases} 0.6 & \text{if } \lambda \le 6 \\ 0.4 & \text{otherwise} \end{cases}$ |
| KNN neighbor count | $k = 5$ |
| Gradient perturbation step | $\beta = 1\text{e-}5$ |
| Gradient correction coefficient | $\varphi = 1$ |
| Minimum elite sample number | $n_{\min} = 0.3\lambda$ |
| Maximum elite sample number | $n_{\max} = 0.8\lambda$ |
| Elite coefficient | $a = 2$ |
| Weight coefficient | $\gamma = 2$ |
| Number of external samples | $\tau = 1$ |
| Frequency of external sampling | $fre = 0.1 \mathrm{Maxgen}$ |

TABLE II
THE AVERAGE RANKINGS OF EACH ALGORITHM

| Algorithm | CEC2010 (+) | CEC2010 ($\approx$) | CEC2010 (-) | CEC2010 Ranking | CEC2013 (+) | CEC2013 ($\approx$) | CEC2013 (-) | CEC2013 Ranking |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EDGCC | 16 | 0 | 2 | 8.94 | 8 | 0 | 3 | 10.09 |
| CMAES-EDG | 9 | 3 | 6 | 5.72 | 9 | 1 | 1 | 6.18 |
| DCCC | 13 | 0 | 5 | 7.06 | 7 | 0 | 4 | 4.64 |
| EDCC-ERDG | 18 | 0 | 0 | 10.89 | 10 | 0 | 1 | 9.64 |
| MMO-CC | 15 | 2 | 1 | 8.78 | 9 | 1 | 1 | 11.27 |
| SSLPSO | 13 | 2 | 3 | 7.00 | 8 | 0 | 3 | 6.45 |
| DCBA | 14 | 1 | 3 | 7.39 | 9 | 0 | 2 | 5.64 |
| RCI-PSO | 14 | 0 | 4 | 7.00 | 8 | 0 | 3 | 6.82 |
| MSORL | 13 | 1 | 4 | 8.89 | 8 | 0 | 3 | 8.91 |
| GTDE | 14 | 1 | 3 | 8.33 | 8 | 0 | 3 | 6.55 |
| SRTP | 14 | 0 | 4 | 8.06 | 9 | 0 | 2 | 5.82 |
| SDLSO | 12 | 0 | 6 | 7.61 | 8 | 0 | 3 | 9.00 |
| RLLPSO | 12 | 0 | 6 | 9.50 | 8 | 0 | 3 | 9.55 |
| AGLDPSO | 15 | 0 | 3 | 10.33 | 8 | 1 | 2 | 9.64 |
| CCMTO-MTES-DAKG | \ | \ | \ | 4.22 | \ | \ | \ | 4.18 |

### B. Compared with State-of-the-art Algorithms

The proposed CCMTO-MTES-DAKG is tested and compared with fourteen state-of-the-art large-scale algorithms in recent years, including 5 CC-based algorithms, namely EDGCC [10], CMAES-EDG [12], DCCC [19], DECC-ERDG [9], MMO-CC [49], and 9 non-decomposition algorithms, including SSLPSO [50], DCBA [51], RCI-PSO [52], MSORL [53], GTDE [54], SRTP [55], SDLSO [56], RLLPSO [57], AGLDPSO [58]. The parameters of these algorithms were set as recommended in the original papers.

TABLE III
RESULTS OF PARAMETER SENSITIVITY ANALYSIS

| Parameter | Setting 1 | Setting 2 | Setting 3 | Setting 4 | Setting 5 | Setting 6 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$n_{\mathrm{sub}}$** | **2** | **3** | **5** | **7** | **10** | **20** |
| + / $\approx$ / - | 7/3/0 | 7/3/0 | \ | 6/4/0 | 6/4/0 | 7/3/0 |
| Ranking | 3.50 | 4.70 | 1.20 | 3.70 | 3.10 | 3.90 |
| **$d_{\max}$** | **1** | **1** | **2** | **4** | **4** | **limitless** |
| + / $\approx$ / - | 3/1/0 | 3/1/0 | \ | 3/1/0 | 3/1/0 | 4/0/0 |
| Ranking | 2.00 | 2.00 | 1.00 | 2.75 | 2.75 | 3.75 |
| **$\tau$** | **0** | **1** | **2** | **3** | **4** | **5** |
| + / $\approx$ / - | 7/3/0 | \ | 8/2/0 | 8/2/0 | 8/1/1 | 7/2/1 |
| Ranking | 3.70 | 1.20 | 3.60 | 4.00 | 3.60 | 3.60 |
| **$fre$** | **everygen** | **0.05Maxgen** | **0.1Maxgen** | **0.2Maxgen** | **0.3Maxgen** | **0.5Maxgen** |
| + / $\approx$ / - | 8/1/1 | 7/2/1 | \ | 5/5/0 | 6/3/1 | 6/4/0 |
| Ranking | 4.60 | 3.60 | 1.90 | 3.50 | 3.30 | 3.60 |

The comparison results of different algorithms on the CEC2010 and CEC2013 benchmark test suites are presented in Table SⅠ and Table SⅡ in the supplementary material, respectively. The average rankings of each algorithm are given in TABLE II.

CCMTO-MTES-DAKG achieved the best average rankings on both the CEC2010 and CEC2013 benchmark test suites, and it performed best among all algorithms. As seen from results of CEC2010 test suite, CCMTO-MTES-DAKG achieved the best results on 5 benchmark problems, which is most in all algorithms. Besides, it can be seen that CCMTO-MTES-DAKG performed at a moderate level on fully separable problems except f3. This is because the grouping method EDG [12] incorrectly identified the completely separable f3 as a completely nonseparable problem, resulting in poor performance of CCMTO-MTES-DAKG on f3. For partially separable problems, even on the problems where CCMTO-MTES-DAKG does not achieve the best result, it still obtains well performance.

From the results of CEC2013 test suite, although CCMTO-MTES-DAKG did not demonstrate the best performance among fully separable problems f1-f3, it still obtained the highly competitive near-optimal results for f1 and f3. Among partially separable problems f4-f11, CCMTO-MTES-DAKG achieved the best results on 6 problems except f5 and f9, and it performed significantly better than its competitors on f4, f7, f8 by several orders of magnitude.

From the comparison results, it is obvious that the proposed CCMTO-MTES-DAKG is effective in solving both the largescale fully separable and partially separable problems, and it is a competitive solver for large-scale problems by employing the novel CCMTO framework.

### C. Parameter Sensitivity Discussion

In this section, the effects of these parameters (as listed in bold fonts in TABLE I) are discussed on the CEC2013 test suite. Because the test problems in CEC2013 test suite are more complex than those in CEC2010 test suite, and the results of parameter sensitivity are representative and universal. In the CCMTO framework, $n_{\mathrm{sub}}$ and $d_{\max}$ are necessary to analyze their effects, because they can affect the optimization performance of EMTO algorithm. In the proposed MTES-DAKG, $\tau$ and $fre$ are significant to external sampling. The detailed settings for these four parameters are given in Section S-Ⅰ of the supplementary material. It should be noted that the value of $d_{\max}$ does not affect the performance of the algorithm on fully separable problems, because the dimensions of all subproblems are equal to 1. Therefore, partially separable problems f4-f7 of CEC2013 test suite as representative problems are tested.

The comparison results and convergence curves are provided in Table SⅢ-Table SⅥ and Fig. S1-Fig. S4, respectively. The rankings for each parameter sensitivity analysis are provided in TABLE III.

The results of each setting for f3 are identical, because f3 is a fully separable function, the grouping method EDG cannot recognize its separability, and it is classified as a nonseparable function. Thus, the number of tasks is only one and it is not affected by these parameters. The result for f3 is not included in the calculation of average rankings.

*1) Parameter $n_{\mathrm{sub}}$*: The results of the significance tests show that $n_{\mathrm{sub}} = 5$ performed best among all settings of $n_{\mathrm{sub}}$, and it achieved the best average ranking. Because MTES-DAKG adopts a random source task selection strategy, without considering the similarities between subtasks. When $n_{\mathrm{sub}}$ is small, there is a relatively high probability of assigning subtasks with low similarity to the same MTOP and selecting tasks with low similarity as source tasks. Consequently, this can negatively affect the optimization efficiency, which is evident in the result. When $n_{\mathrm{sub}}$ is excessively large, the probability of negative transfer increases. Meanwhile, the optimization counts of subtasks with large contributions to fitness values will be reduced in comparison to those with smaller $n_{\mathrm{sub}}$ due to limited computational resources, and the optimization performance can be affected. From the comparison results, this study suggests that the value of $n_{\mathrm{sub}}$ is set to 5.

*2) Parameter $d_{\max}$*: The statistical test results show that $d_{\max} = 2$ performed best among all $d_{\max}$ settings. It can be found that the optimization performance is poor when $d_{\max}$ is set to 4 or bigger. From the convergence curves, it can be seen that the algorithm can find a well solution and also exhibit the fastest convergence speed when $d_{\max} = 2$. The obtained optimal solutions are poor for other settings of $d_{\max}$, and the optimization performances are inferior to $d_{\max} = 2$. Therefore, it is recommended that $d_{\max} = 2$ in this study.

*3) Parameter $\tau$*: From the results, it can be seen that $\tau = 1$ performs best among all $\tau$ settings. Theoretically, when $\tau$ is set to 0, no external samples are transferred to target distribution, which means that there is no knowledge transfer between source task and target task. Consequently, MTES-DAKG becomes equivalent to CMA-ES due to the absence of knowledge transfer, resulting in a decrease in optimization performance. Because the number $\lambda$ of generated samples in MTES-DAKG is less than 100 set in [48], if $\tau$ is set too large, the phenomenon of negative transfer can affect the performance of MTES-DAKG. In summary, it is recommended that $\tau$ be set to 1 in this study.

*4) Parameter $fre$*: These results show that setting $fre$ to $0.1 \mathrm{Maxgen}$ leads to the best performance on these test problems. If employing external sampling every generation, the algorithm performance will decrease due to excessive knowledge transfer, leading to negative transfer and trapping in local optimal solution. Low frequency of external sampling also decreases the optimization performance due to insufficient knowledge transfer. It suggests that $fre$ be set to $0.1 \mathrm{Maxgen}$ in this study.

### D. Discussion of Components in CCMTO-MTES-DAKG

In order to demonstrate the effectiveness and performance of the proposed contribution-based resource allocation strategy, the proposed MTES-DAKG for solving constructed MTOPs, and components in CCMTO-MTES-DAKG, this section conducts component analysis experiments. Six resource allocation strategies are tested to compare with the proposed one, they are CBCC1 [59], CBCC2 [59], CBCC3 [60], CCFR [18], CCFR2 [20], and CCFR3 [21]. To show the performance of the proposed MTES-DAKG for solving constructed MTOPs, this study also incorporates 3 EMTO algorithms with CCMTO, respectively. They are MaTDE [61], G-MFEA [40], and MTEA-AD [62]. This section conducts component analysis experiments to investigate the performance improvement of the proposed dynamic distance threshold domain KGxS with gradient correction (DT-DoS) and adaptive elite sampling shape KGxS (AS-SaS). Meanwhile, the effectiveness of stagnant subtask detection mechanism on the CCMTO framework is also discussed. Four types of variants are employed for comparison, which are detailed below.

1) wo-DA: equivalent to MTES-KG [48].  
2) wo-DT-DoS: Without DT-DoS, only AS-SaS.  
3) wo-AS-SaS: Without AS-SaS, only DT-DoS.  
4) wo-SD: Without stagnant subtask detection.  

TABLE IV
RESULTS OF COMPONENT ANALYSIS IN CCMTO-MTES-DAKG

| Algorithm | + | $\approx$ | - | Ranking |
| :--- | :---: | :---: | :---: | :---: |
| **Results on different resource allocation strategies for CCMTO-MTES-DAKG** | | | | |
| CBCC1 | 8 | 3 | 0 | 3.82 |
| CBCC2 | 8 | 3 | 0 | 4.73 |
| CBCC3 | 8 | 2 | 1 | 3.73 |
| CCFR | 6 | 4 | 1 | 3.73 |
| CCFR2 | 7 | 3 | 1 | 3.18 |
| CCFR3 | 7 | 3 | 1 | 3.00 |
| CCMTO-MTES-DAKG | \ | \ | \ | 1.36 |
| **Results on different EMTO algorithms for CCMTO** | | | | |
| CCMTO-MaTDE | 10 | 0 | 1 | 2.55 |
| CCMTO-G-MFEA | 10 | 0 | 1 | 3.00 |
| CCMTO- MTEA-AD | 10 | 1 | 0 | 3.18 |
| CCMTO-MTES-DAKG | \ | \ | \ | 1.18 |
| **Results on CCMTO-MTES-DAKG with different components** | | | | |
| wo-DA | 7 | 3 | 0 | 3.30 |
| wo-DT-DoS | 8 | 2 | 0 | 3.40 |
| wo-AS-SaS | 7 | 3 | 0 | 2.70 |
| wo-SD | 6 | 4 | 0 | 3.30 |
| CCMTO-MTES-DAKG | \ | \ | \ | 1.10 |

These ablation experiments are conducted on the 11 test problems of CEC2013 test suite. The detailed experimental results are shown in Table SⅦ-Table SⅨ in the supplementary material, and the convergence curves are given in Fig. S5- Fig. S7. The results of nonparametric tests are presented in TABLE IV.

*1) Resource allocation strategies*: It is obvious that the proposed resource allocation strategy achieved the best average ranking on both the strategies, and it achieved the best results on 10 benchmark problems except f8. From the experimental results and convergence curves, it can be seen that the proposed strategy can find the best solution and exhibits fast convergence speed for each problem compared with other strategies. Therefore, the proposed resource allocation strategy is suitable for CCMTO and effective.

*2) EMTO algorithms*: The results show that MTES-DAKG achieved the best average ranking on both the EMTO algorithms, and it performed significantly better than other EMTO algorithms. Therefore, MTES-DAKG is an efficient EMTO solver.

*3) Components in CCMTO-MTES-DAKG*: The statistical test results show that CCMTO-MTES-DAKG performed significantly better on 8 problems, and the optimal solution mean of CCMTO-MTES-DAKG for each problem in 25 independent runs was better than wo-DT-DoS, except f3. This reveals that DT-DoS plays an important role in knowledge transfer, and it can improve the convergence of the distribution and performance of optimal solution, especially in MTOPs where optimal domains of different subtasks are similar. From the statistical result, the validity of AS-SaS can be verified. AS-SaS provides search preference for the target distribution by learning the distribution of the elite samples in the source distribution, and the optimization performance can be improved.

It can be seen obviously that the proposed strategy of DT-DoS and AS-SaS performs significantly better than incorporating KGxS into ESs by comparing CCMTO-MTES-DAKG and wo-DA. The dynamic distance threshold and gradient correction strategy ensure that the external samples can accurately fall within the distribution of their regions respectively, and improve the target fitness value. Meanwhile, adaptive elite sampling can enhance the contribution of elite samples, leading to better optimization performance.

The statistical test results reveal that CCMTO-MTES-DAKG is superior to wo-SD, and the effectiveness of stagnant subtask detection mechanism can be verified. It can be demonstrated that the proposed stagnant subtask detection mechanism can save computational resources on stagnant subtasks and improve the efficiency of algorithm.

### E. REAL-WORLD APPLICATION

TABLE V
RESULTS OF PERFORMANCE COMPARISON FOR WSNs DEPLOYMENT

| Algorithm | Sensor Nodes = 500 (Average) | Sensor Nodes = 500 (Std) | Sensor Nodes = 500 (Best) | Sensor Nodes = 1000 (Average) | Sensor Nodes = 1000 (Std) | Sensor Nodes = 1000 (Best) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| CCMTO-MTES-DAKG | 48.10% | 0.0286 | 48.65% | 80.02% | 0.0233 | 80.18% |
| CCFR-CMAES | 45.57% | 0.0942 | 46.69% | 75.98% | 0.0545 | 76.26% |
| CCPSO2 | 46.28% | 0.0541 | 46.80% | 74.86% | 0.0688 | 75.30% |
| TPLSO | 47.16% | 0.0852 | 47.62% | 77.02% | 0.0259 | 77.69% |
| CMAES | 40.67% | 0.0178 | 41.01% | 64.10% | 0.0245 | 64.83% |

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/e4396f3a-7739-44f2-a218-7a4643fa7c12/388df1e3a848063194e34719196a151bbcaa7f22a484bd38827c83a745e07bda.jpg)
(a) 1000D deployment

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/e4396f3a-7739-44f2-a218-7a4643fa7c12/d0c1ad3e6a6a352eeadf9c18f9a0ac2d836cdc4ac65429d58879496686eb88ea.jpg)
(b) 2000D deployment

Fig. 6. The coverage curves of different algorithms for WSNs deployment problems

The real-world application is the wireless sensor networks (WSNs) deployment, its detailed descriptions and problem settings are provided in Section S-Ⅱ of the supplementary material. The initial and final WSNs deployments of different algorithms are provided in Fig. S8-Fig. S11 of the supplementary material. TABLE V gives the average and standard deviation of the final effective coverage, the best situation is also listed in the table. The coverage curves of different algorithms are shown in Fig. 6.

From the statistical results, it can be found that CCMTO-MTES-DAKG outperformed the other algorithms in 1000D and 2000D problems, and it obtained the best average effective coverage in the 25 independent runs. As seen from the coverage curves, CCMTO-MTES-DAKG performed well in terms of the coverage and the convergence speed. CCMTO-MTES-DAKG can find a highly competitive solution with a small amount of computing resources, and it can fast converge to the optimal solution. Compared to the traditional decomposition-based and non-decomposition algorithms, the results illustrate the outstanding performance of the proposed CCMTO-MTES-DAKG for large scale WSNs deployment problems.

## V. CONCLUSIONS AND FUTURE WORK

This study proposes a novel cooperative co-evolutionary multitask optimization (CCMTO) framework for solving large-scale optimization problems. In this framework, each subproblem is regarded as a distinct subtask, and these subtasks are constructed a series of MTOPs. CCMTO employs the EMTO algorithm to solve these MTOPs. To further improve the efficiency of the CCMTO framework, a construction strategy of multitask optimization problems and a contribution-based resource allocation strategy of MTOPs and subtasks are proposed. The first strategy considers both the number of subtasks and the dimensional disparity among tasks in a MTOP, which can select the appropriate number of subproblems to construct each MTOP. The second strategy determines the optimization order of MTOPs and allocates computational resources for each subtask by calculating the contribution of each MTOP. The mechanism of stagnant subtask detection can save computational resources on stagnant subtasks. To improve optimization performance of EMTO algorithm, a MTES with dynamic distance threshold and adaptive elite sampling KGxS (MTES-DAKG) is proposed, and it is incorporated into CCMTO framework to form CCMTO-MTES-DAKG.

The experimental studies demonstrate that the proposed CCMTO-MTES-DAKG outperforms 14 state-of-the-art LSOP algorithms on the 18 test problems of CEC2010 test suite and 11 test problems of CEC2013 test suite. CCMTO-MTES-DAKG achieves the best average ranking among all algorithms. CCMTO-MTES-DAKG is compared with 6 resource allocation strategies and 3 EMTO algorithms, to show the robustness and effectiveness of the proposed resource allocation strategy and MTES-DAKG, respectively.

The component analysis experiments show the effectiveness and flexibility of the proposed DT-DoS, AS-SaS and the stagnant subtask detection mechanism, respectively. At last, the results of applications to large-scale WSNs deployment problems demonstrate the effectiveness and applicability of the proposed CCMTO-MTES-DAKG.

Although the CCMTO framework can only solve fully separable and partially separable LSOPs theoretically, there are almost no optimization problems with strong correlations among all variables in real-world applications. Variables with weak correlations can be approximately decomposed by applying a correlation identification threshold.

Therefore, CCMTO also shows promises in solving realworld LSOPs. Based on the CCMTO framework, some improvement methods, such as the source task selection strategy, the resource allocation strategy based on the combined effect of task similarity and contribution, and the knowledge transfer strategy combined constraint handling technique can be employed to solve complicated and constrained LSOPs in the future.

Besides, in the field of engineering optimization, the optimization may involve multiple stages, where at each stage, the design models are incrementally modified and optimized. This type of problem is called incremental optimization problem (IOP) [63], and current research on IOPs is relatively limited. The proposed CCMTO framework has the potential for solving IOPs. Based on the proposed CCMTO framework, the original subproblems and new ones due to incremental decision variables can be regarded as different tasks. These all subproblems can be constructed into several new MTOPs by the proposed construct strategy of multitask optimization problems and solved through EMTO algorithms.

## REFERENCES

[1] S. Mahdavi, M. E. Shiri, and S. Rahnamayan, “Metaheuristics in largescale global continues optimization: A survey,” Inf. Sci., vol. 295, pp. 407-428, Feb. 2015.
[2] Z. Yang, K. Tang, and X. Yao, “Large scale evolutionary optimization using cooperative coevolution,” Inf. Sci., vol. 178, no. 15, pp. 2985-2999, Aug. 2008.
[3] R. Cheng, and Y. Jin, “A Competitive Swarm Optimizer for Large Scale Optimization,” IEEE Trans. Cybern., vol. 45, no. 2, pp. 191-204, Feb. 2015.
[4] M. A. Potter, and K. A. De Jong, "A cooperative coevolutionary approach to function optimization," in Proc. Parallel Problem Solving from Nature — PPSNIII, 1994, pp. 249-257.
[5] X. Ma, X. Li, Q. Zhang, K. Tang, Z. Liang, W. Xie, and Z. Zhu, “A Survey on Cooperative Co-Evolutionary Algorithms,” IEEE Trans. Evol. Comput., vol. 23, no. 3, pp. 421-441, Jun. 2019.
[6] Y. Zhenyu, T. Ke, and Y. Xin, "Multilevel cooperative coevolution for large scale optimization," in Proc. IEEE Congr. Evol. Computat. (CEC), 2008, pp. 1663-1670.
[7] M. N. Omidvar, X. Li, Z. Yang, and X. Yao, "Cooperative Co-evolution for large scale optimization through more frequent random grouping," in Proc. IEEE Congr. Evol. Computat. (CEC), 2010, pp. 1-8.
[8] M. N. Omidvar, X. Li, and X. Yao, "Cooperative Co-evolution with delta grouping for large scale non-separable function optimization," in Proc. IEEE Congr. Evol. Computat. (CEC), 2010, pp. 1-8.
[9] M. Yang, A. Zhou, C. Li, and X. Yao, “An Efficient Recursive Differential Grouping for Large-Scale Continuous Problems, ” IEEE Trans. Evol. Comput., vol. 25, no. 1, pp. 159-171, Feb. 2021.
[10] W. Yang, J. Liu, S. Tan, W. Zhang, and Y. Liu, “Evolutionary dynamic grouping based cooperative co-evolution algorithm for large-scale optimization,” Appl. Intell., vol. 54, no. 6, pp. 4585-4601, Mar. 2024.
[11] Q. Liang, J.-S. Pan, S.-C. Chu, L. Kong, and W. Li, “A decomposition framework based on memorized binary search for large-scale optimization problems,” Inf. Sci., vol. 679, pp. 121063, Sep. 2024.
[12] A. Kumar, S. Das, and R. Mallipeddi, “An Efficient Differential Grouping Algorithm for Large-Scale Global Optimization,” IEEE Trans. Evol. Comput., vol. 28, no. 1, pp. 32-46, Feb. 2024.
[13] R. P. Wiegand, W. C. Liles, and K. A. De Jong, "An empirical analysis of collaboration methods in cooperative coevolutionary algorithms," in Proc. Proceedings of the genetic and evolutionary computation conference (GECCO), 2001, pp. 1235-1245.
[14] F. B. de Oliveira, R. Enayatifar, H. J. Sadaei, F. G. Guimarães, and J.-Y. Potvin, “A cooperative coevolutionary algorithm for the Multi-Depot Vehicle Routing Problem,” Expert Syst. Appl., vol. 43, pp. 117-130, Jan. 2016.
[15] E. Glorieux, B. Svensson, F. Danielsson, and B. Lennartson, "Improved Constructive Cooperative Coevolutionary Differential Evolution for Large-Scale Optimisation," in Proc. 2015 IEEE Symposium Series on Computational Intelligence, 2015, pp. 1703-1710.
[16] L. Panait, S. Luke, and J. F. Harrison, "Archive-based cooperative coevolutionary algorithms," in Proc. the 8th annual conference on Genetic and evolutionary computation, 2006, pp. 345–352.
[17] L. Panait, and S. Luke, "Selecting informative actions improves cooperative multiagent learning," in Proc. the fifth international joint conference on Autonomous agents and multiagent systems, 2006, pp. 760–766.
[18] M. Yang, M. N. Omidvar, C. Li, X. Li, Z. Cai, B. Kazimipour, and X. Yao, “Efficient Resource Allocation in Cooperative Co-Evolution for Large-Scale Global Optimization,” IEEE Trans. Evol. Comput., vol. 21, no. 4, pp. 493-505, Aug. 2017.
[19] P. Xu, W. Luo, X. Lin, Y. Chang, and K. Tang, “Difficulty and Contribution-Based Cooperative Coevolution for Large-Scale Optimization,” IEEE Trans. Evol. Comput., vol. 27, no. 5, pp. 1355- 1369, Oct. 2023.
[20] M. Yang, A. Zhou, C. Li, J. Guan, and X. Yan, “CCFR2: A more efficient cooperative co-evolutionary framework for large-scale global optimization,” Inf. Sci., vol. 512, pp. 64-79, Feb. 2020.
[21] M. Yang, A. Zhou, X. Lu, Z. Cai, C. Li, and J. Guan, “CCFR3: A cooperative co-evolution with efficient resource allocation for largescale global optimization,” Expert Syst. Appl., vol. 203, pp. 117397, Oct. 2022.
[22] M. Tian, W. Du, W. Fang, Y. Tang, and Y. Jin, “Learning to Decompose and Optimize for Large-Scale Overlapping Problems,” IEEE Trans. Evol. Comput., pp. 1-1, Oct. 2025.
[23] Y. H. Jia, Y. Mei, and M. Zhang, “Contribution-Based Cooperative Co-Evolution for Nonseparable Large-Scale Problems With Overlapping Subcomponents,” IEEE Trans. Cybern., vol. 52, no. 6, pp. 4246-4259, Oct. 2022.
[24] P. Jiang, J. Liu, and Y. Cheng, “Bi-Population-Enhanced Cooperative Differential Evolution for Constrained Large-Scale Optimization Problems,” IEEE Trans. Evol. Comput., vol. 28, no. 6, pp. 1620-1632, Dec. 2024.
[25] P. Jiang, Y. Cheng, and J. Liu, “Cooperative Bayesian optimization with hybrid grouping strategy and sample transfer for expensive large-scale black-box problems,” Knowl. Based. Syst., vol. 254, pp. 109633, Oct. 2022.
[26] P. Jiang, J. Liu, Q. Luo, and Y. Cheng, “Domain knowledge-driven decomposition-based large-scale optimization for ship cabin structures,” Chinese Journal ofShip Research, vol. 20, no. 3, pp. 108-117, Jun. 2025.
[27] T. Wei, S. Wang, J. Zhong, D. Liu, and J. Zhang, “A Review on Evolutionary Multitask Optimization: Trends and Challenges,” IEEE Trans. Evol. Comput., vol. 26, no. 5, pp. 941-960, Oct. 2022.
[28] A. Gupta, Y. S. Ong, and L. Feng, “Multifactorial Evolution: Toward Evolutionary Multitasking,” IEEE Trans. Evol. Comput., vol. 20, no. 3, pp. 343-357, Jun. 2016.
[29] B. Da, A. Gupta, Y. S. Ong, and L. Feng, "The Boon of Gene-Culture Interaction for Effective Evolutionary Multitasking," in Proc. Artificial Life and Computational Intelligence, 2016, pp. 54-65.
[30] K. C. Tan, L. Feng, and M. Jiang, “Evolutionary Transfer Optimization - A New Frontier in Evolutionary Computation Research,” IEEE Comput. Intell. Mag., vol. 16, no. 1, pp. 22-33, Jan. 2021.
[31] X. Ma, Y. Zheng, Z. Zhu, X. Li, L. Wang, Y. Qi, and J. Yang, “Improving Evolutionary Multitasking Optimization by Leveraging Inter-Task Gene Similarity and Mirror Transformation,” IEEE Comput. Intell. Mag., vol. 16, no. 4, pp. 38-53, Oct. 2021.
[32] Y. Cai, D. Peng, P. Liu, and J.-M. Guo, “Evolutionary multi-task optimization with hybrid knowledge transfer strategy,” Inf. Sci., vol. 580, pp. 874-896, Nov. 2021.
[33] Y. Zhang, Y. Qian, G. Ma, X. Liang, G. Liu, Q. Zhang, and K. Tang, “ESSR: Evolving Sparse Sharing Representation for Multitask Learning,” IEEE Trans. Evol. Comput., vol. 28, no. 3, pp. 748-762, Jun. 2024.
[34] Z. Cui, B. Zhao, T. Zhao, X. Cai, and J. Chen, “Adaptive multi-task evolutionary algorithm based on knowledge reuse,” Inf. Sci., vol. 648, pp. 119568, Nov. 2023.
[35] Z. Tang, M. Gong, Y. Wu, W. Liu, and Y. Xie, “Regularized Evolutionary Multitask Optimization: Learning to Intertask Transfer in Aligned Subspace,” IEEE Trans. Evol. Comput., vol. 25, no. 2, pp. 262- 276, Apr. 2021.
[36] Z. Wang, L. Cao, L. Feng, M. Jiang, and K. C. Tan, “Evolutionary Multitask Optimization With Lower Confidence Bound-Based Solution Selection Strategy,” IEEE Trans. Evol. Comput., vol. 29, no. 1, pp. 132- 144, Feb. 2025.
[37] C. Wang, Z. Wang, and Z. Kou, “Adaptive Bi-Operator Evolution for Multitasking Optimization Problems,” Biomimetics, vol. 9, no. 10, pp. 604, Oct. 2024.
[38] Y.-L. Li, Y.-Y. Cheng, Z.-Y. Chai, X. Liu, H.-L. Hou, and G. Chen, “Evolutionary multitasking for multiobjective optimization based on hybrid differential evolution and multiple search strategy,” Futur. Gener. Comp. Syst., vol. 158, pp. 230-241, Sep. 2024.
[39] K. K. Bali, A. Gupta, L. Feng, Y. S. Ong, and S. Tan Puay, "Linearized domain adaptation in evolutionary multitasking," in Proc. IEEE Congr. Evol. Computat. (CEC), 2017, pp. 1295-1302.
[40] J. Ding, C. Yang, Y. Jin, and T. Chai, “Generalized Multitasking for Evolutionary Optimization of Expensive Problems,” IEEE Trans. Evol. Comput., vol. 23, no. 1, pp. 44-58, Feb. 2019.
[41] K. K. Bali, Y. S. Ong, A. Gupta, and P. S. Tan, “Multifactorial Evolutionary Algorithm With Online Transfer Parameter Estimation: MFEA-II,” IEEE Trans. Evol. Comput., vol. 24, no. 1, pp. 69-83, Feb. 2020.
[42] Z. Yang, Y. Zhu, Y. Jiang, Y. Jin, F. Ju, and Y. Feng, “An adaptive multitask optimization algorithm based on competitive scoring,” Swarm Evol. Comput., vol. 92, pp. 101798, Feb. 2025.
[43] Y. Feng, L. Feng, Y. Hou, and K. C. Tan, "Large-Scale optimization via Evolutionary Multitasking assisted Random Embedding," in Proc. 2020 IEEE Congress on Evolutionary Computation (CEC), 2020, pp. 1-8.
[44] L. Feng, Q. Shang, Y. Hou, K. C. Tan, and Y. S. Ong, “Multispace Evolutionary Search for Large-Scale Optimization With Applications to Recommender Systems,” IEEE Transactions on Artificial Intelligence, vol. 4, no. 1, pp. 107-120, Mar. 2023.
[45] M. N. Omidvar, X. Li, and K. Tang, “Designing benchmark problems for large-scale continuous optimization,” Inf. Sci., vol. 316, pp. 419-436, Sep. 2015.
[46] L. Feng, L. Zhou, J. Zhong, A. Gupta, Y. S. Ong, K. C. Tan, and A. K. Qin, “Evolutionary Multitasking via Explicit Autoencoding,” IEEE Trans. Cybern., vol. 49, no. 9, pp. 3457-3470, Sep. 2019.
[47] X. Ban, J. Liang, K. Yu, Y. Wang, K. Qiao, J. Peng, D. Gong, and C. Dai, “A Local Knowledge Transfer-Based Evolutionary Algorithm for Constrained Multitask Optimization,” IEEE Trans. Syst., Man, Cybern., Syst., vol. 55, no. 3, pp. 2183-2195, Mar. 2025.
[48] Y. Li, W. Gong, and S. Li, “Multitask Evolution Strategy With Knowledge-Guided External Sampling,” IEEE Trans. Evol. Comput., vol. 28, no. 6, pp. 1733-1745, Dec. 2024.
[49] X. Peng, Y. Jin, and H. Wang, “Multimodal Optimization Enhanced Cooperative Coevolution for Large-Scale Optimization,” IEEE Trans. Cybern., vol. 49, no. 9, pp. 3507-3520, Sep. 2019.
[50] S. Liu, Z. J. Wang, Z. Kou, Z. H. Zhan, S. Kwong, and J. Zhang, “Less Is More: A Small-Scale Learning Particle Swarm Optimization for Large-Scale Optimization,” IEEE Trans. Cybern., vol. 56, no. 1, pp. 523-536, Sep. 2026.
[51] H. Liu, W. Song, Y. Cheng, S. Tuo, and Y. Wang, “A large-scale optimization algorithm based on variable decomposition and space compression,” Swarm Evol. Comput., vol. 94, pp. 101863, Apr. 2025.
[52] Q. Yang, G. W. Song, W. N. Chen, Y. H. Jia, X. D. Gao, Z. Y. Lu, S. W. Jeon, and J. Zhang, “Random Contrastive Interaction for Particle Swarm Optimization in High-Dimensional Environment,” IEEE Trans. Evol. Comput., vol. 28, no. 4, pp. 933-949, May. 2024.
[53] X. Wang, F. Wang, Q. He, and Y. Guo, “A multi-swarm optimizer with a reinforcement learning mechanism for large-scale optimization,” Swarm Evol. Comput., vol. 86, pp. 101486, Apr. 2024.
[54] Z. J. Wang, J. R. Jian, Z. H. Zhan, Y. Li, S. Kwong, and J. Zhang, “Gene Targeting Differential Evolution: A Simple and Efficient Method for Large-Scale Optimization,” IEEE Trans. Evol. Comput., vol. 27, no. 4, pp. 964-979, Aug. 2023.
[55] H. Liu, Y. Cheng, S. Xue, and S. Tuo, “A space-reduction based threephase approach for large-scale optimization,” Appl. Soft. Comput., vol. 144, pp. 110517, Sep. 2023.
[56] Q. Yang, W. N. Chen, T. Gu, H. Jin, W. Mao, and J. Zhang, “An Adaptive Stochastic Dominant Learning Swarm Optimizer for High-Dimensional Optimization,” IEEE Trans. Cybern., vol. 52, no. 3, pp. 1960-1976, Dec. 2022.
[57] F. Wang, X. Wang, and S. Sun, “A reinforcement learning level-based particle swarm optimization algorithm for large-scale optimization,” Inf. Sci., vol. 602, pp. 298-312, Jul. 2022.
[58] Z. J. Wang, Z. H. Zhan, S. Kwong, H. Jin, and J. Zhang, “Adaptive Granularity Learning Distributed Particle Swarm Optimization for Large-Scale Optimization,” IEEE Trans. Cybern., vol. 51, no. 3, pp. 1175-1188, Mar. 2021.
[59] M. N. Omidvar, X. Li, and X. Yao, "Smart use of computational resources based on contribution for cooperative co-evolutionary algorithms," in Proc. the 13th annual conference on Genetic and evolutionary computation, 2011, pp. 1115–1122.
[60] M. N. Omidvar, B. Kazimipour, X. Li, and X. Yao, "CBCC3 — A contribution-based cooperative co-evolutionary algorithm with improved exploration/exploitation balance," in Proc. 2016 IEEE Congress on Evolutionary Computation (CEC), 2016, pp. 3541-3548.
[61] Y. Chen, J. Zhong, L. Feng, and J. Zhang, “An Adaptive Archive-Based Evolutionary Framework for Many-Task Optimization,” IEEE Trans. Emerg. Topics. Comput. Intell., vol. 4, no. 3, pp. 369-384, Jun. 2020.
[62] C. Wang, J. Liu, K. Wu, and Z. Wu, “Solving Multitask Optimization Problems With Adaptive Knowledge Transfer via Anomaly Detection,” IEEE Trans. Evol. Comput., vol. 26, no. 2, pp. 304-318, Mar. 2022.
[63] R. Cheng, M. N. Omidvar, A. H. Gandomi, B. Sendhoff, S. Menzel, and X. Yao, “Solving Incremental Optimization Problems via Cooperative Coevolution,” IEEE Trans. Evol. Comput., vol. 23, no. 5, pp. 762-775, Oct. 2019.
