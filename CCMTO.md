# An Efficient Cooperative Co-Evolutionary Multitask Optimization Framework for Large-Scale Optimization

Shiqiang Li, Jun Liu, and Yuansheng Cheng 

Abstract—Cooperative co-evolution (CC) framework is a classic decomposition-based method to solve large-scale optimization problems (LSOPs) by decomposing the original problem into several subproblems. CC is a single task paradigm that sequentially solves decomposed subproblems in a specific order, and it does not fully utilize similarities among these subproblems. Evolutionary multitask optimization (EMTO) employs the potential similarities and complementarities among distinct tasks to address multiple optimization tasks simultaneously through knowledge transfer mechanism. This study integrates the CC framework with the EMTO paradigm and proposes a cooperative co-evolutionary multitask optimization (CCMTO) framework for solving LSOPs. In the CCMTO framework, the original LSOP is redefined as a set of multitask optimization problems (MTOPs), and then the EMTO algorithm is used to solve them. To improve the optimization efficiency, this study proposes a construction strategy of multitask optimization problems and a contribution-based resource allocation strategy of MTOPs and subtasks. The construction strategy of multitask optimization problems can select the appropriate subproblems to construct MTOPs. The resource allocation strategy determines the optimization order of MTOPs based on their contribution to the improvement of the best fitness value, and reasonably allocates computational resources for each subtask. A multitask evolution strategy with dynamic distance threshold and adaptive elite sampling knowledge-guided external sampling (MTES-DAKG) is proposed and used to solve these MTOPs. Empirical results show that the proposed algorithm can significantly improve the optimization performance for solving LSOPs. Moreover, the proposed algorithm is superior to 14 state-of-the-art algorithms on 29 benchmark problems and performs well in real-world applications. 

Index Terms—Cooperative co-evolution, evolutionary multitask optimization, evolution strategy, knowledge transfer, large-scale optimization 

## I. INTRODUCTION

Large-scale optimization problems (LSOPs) present substantial challenges in the field of optimization because they involve hundreds to thousands of design variables [1], [2], [3]. As the dimensionality increases, the volume of the design space grows exponentially, making it difficult for optimization algorithms to thoroughly explore such an expansive search space. Furthermore, the rise in dimensions also leads to a rapid increase in the complexity of the objective functions. Decomposition-based methods which employ the strategy of “divide and conquer”, have attracted widely research interest from scholars. Cooperative coevolutionary algorithms (CCEAs) [4], [5] inspired by the ecological phenomenon of mutualism are effective methods for solving LSOPs. CCEAs decompose the original LSOP into a set of lower-dimensional subproblems, each of which can be solved in an independently evolving subpopulation to alleviate the difficulties associated with high dimensionality. 

The first CCEA, named cooperative co-evolutionary genetic algorithm (CCGA), was proposed by [4] in 1994. This framework decomposes an N-dimensional problem into N one-dimensional subproblems, which are then optimized sequentially by genetic algorithm. Nevertheless, not all LSOPs are fully separable, and some of them are difficult to be solved owing to complex interaction among variables. In recent decades, there has been a rapid growth in the research on CCEAs to improve the efficiency and effect. Generally, the improvements in CCEAs mainly focus on the following aspects [5]: (1) Variable grouping strategy: Research on variable grouping strategy can be mainly categorized into dynamic variable grouping and static variable grouping. Dynamic variable grouping implies that the grouping approach will change during the process of coevolution [2], [6], [7], [8]. In contrast, static variable grouping maintains a fixed grouping scheme throughout the optimization process, and several variable interaction identification methods have been developed to improve the accuracy of static variable grouping [9], [10], [11], [12]. (2) Collaborator selection strategy: Many types of collaborator selection strategy, such as single best collaborator selection strategy [4], single worst collaborator selection [13], random collaborator selection strategy [14], elite collaborator selection strategy [15] and so on [16], [17], are employed. (3) Resource allocation strategy: Namely how to determine the order of subproblems to be optimized and how to reasonably allocate the computational resource for subproblems. The existing computational resource allocation strategies are mainly based on contribution of subproblems [18], difficulty of subproblems [19], and different subpopulation sizes [20], [21]. 

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
\arg \min _ {\boldsymbol {x} \in R ^ {d}} f (\boldsymbol {x})\tag{1}
$$

where $f ( x )$ is the objective function, and x is a ddimensional vector, called the decision variable or design variable. If d is large enough (usually means that d is much greater than 100 in the field of evolutionary computation), it is called a LSOP. If the analytical expression and gradient of f (x) are not available, it is called a large-scale black-box optimization problem (LSBBOP). CCEAs are one of the representative approaches adopting the “divide and conquer” strategy to address LSOPs. Fig. 1 is a general diagram of CCEA. 

In Fig. 1, it can be seen that the original LSOP is decomposed into N low-dimensional subproblems, and the design variables of each subproblem are only a subset of the original problem. The variable grouping strategy generates these subproblems, and the resource allocation strategy selects which subproblem to be optimized in each co-evolutionary cycle after generating subproblems. Any EA can be utilized as the optimization solver to optimize the current subproblem. The red points in Fig. 1 represent collaborators selected from other subproblems, and they are combined to obtain the complete collaborators. Since the design variable in the current subproblem is only a segment of the original problem, individuals in the current subproblem need to combine with complete collaborators to form complete solutions when evaluating their fitness. In separable problems, the collaborator is generally set to the best solution so far. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/e4396f3a-7739-44f2-a218-7a4643fa7c12/3f2a9ad82c6f1b2fd37d1da95e0628e7a6c89e09be63d30cac065cf10fa24062.jpg)
Fig. 1. Diagram of a general CCEA

Obviously, when the origin LSOP is separable and can be grouped correctly, the optimization performance of CC framework can be great improved [45]. A partially separable problem is defined as follows 

$$
\underset {x _ {1}, \dots , x _ {k}} {\arg \min} f (\boldsymbol {x}) = \left\{\underset {x _ {1}} {\arg \min} f \left(\boldsymbol {x} _ {1}, \dots\right), \dots , \underset {x _ {k}} {\arg \min} f \left(\dots , \boldsymbol {x} _ {k}\right) \right\}\tag{2}
$$

where k is the number of subproblems into which the origin problem can be decomposed, and $\mathbf { \psi } _ { \mathbf { 1 } } \ldots , \mathbf { \psi } _ { \mathbf { k } }$ are mutually exclusive subsets of the d-dimensional decision variable $\boldsymbol { x }$ . if k is equal to $d ,$ each subproblem contains only a onedimensional decision variable and the original problem is called a fully separable problem. 

### B. Evolutionary Multitask Optimization

Generalized MTOP with K minimization tasks is defined as 

$$
\arg \min f _ {k} (\boldsymbol {x} _ {k}) \boldsymbol {x} _ {k} \in \mathrm{R} ^ {D _ {k}}, \text {   for   } k = 1,..., K\tag{3}
$$

where $D _ { k }$ is the dimension of decision variables $\boldsymbol { x } _ { k }$ in the kth task. Each task has a corresponding search space, and all of them are transformed into a unified search space Y. For a solution $\boldsymbol { y } _ { k }$ of task $k ,$ its representation $\boldsymbol { x } _ { k }$ in the unified search space is calculated as follows 

$$
\boldsymbol {x} _ {k} = \frac {\boldsymbol {y} _ {k} - \boldsymbol {L} _ {k}}{\boldsymbol {U} _ {k} - \boldsymbol {L} _ {k}}\tag{4}
$$

where $\pmb { L } _ { k }$ and $\pmb { U } _ { k }$ are the lower and upper bounds of $\boldsymbol { y } _ { k }$ , respectively. The dimension of unified search space $D _ { \scriptscriptstyle { Y } }$ is set to the maximum dimension of all tasks, as shown below 

$$
D _ {Y} = \max \left\{D _ {1}, \dots , D _ {K} \right\}\tag{5}
$$

Since Gupta et al. [28] first proposed the multifactorial optimization (MFO) in 2016, the research on EMTO has gradually increased in recent years. The existing EMTO algorithms are mainly based on two knowledge transfer frameworks, the first one is MFO, and the second one is multipopulation evolution (MPE). The general framework of MFO and MPE is shown in **Fig. 2**. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/e4396f3a-7739-44f2-a218-7a4643fa7c12/57e0a050795c56f395731e1d29c2bb147e063bf618e02e1b7223962747197b06.jpg)
(b) MPE
Fig. 2. General framework of MFO and MPE

MFO generates only one population to optimize all K tasks, and it assigns the most suitable task to each individual by introducing the indicator called skill factors. MFO performs knowledge transfer across tasks through assortative mating. In order to achieve information transmission between parents and offspring, MFO employs the vertical cultural transmission via selective imitation to endow offspring with skill factors. Multifactorial evolutionary algorithm (MFEA) [28] is the first and most representative MFO algorithm. Afterward, several research on MFO algorithm [37], [46], such as the methods of knowledge transfer, source task selection, and similarity measure between tasks has studied. 

MPE optimizes multiple tasks through multiple populations, and each population can evolve through two distinct evolution mechanisms: intra-task self-evolution and inter-task crossevolution. Self-evolution involves crossover and mutation among individuals within the same populations. However, this approach cannot facilitate information exchange among distinct tasks. Consequently, an inter-population evolutionary mechanism is essential, and the cross-evolution among tasks is performed at the information exchange nodes. In recent years, many well performing MPE algorithms have emerged [36], [47]. 

### C. Multitask Evolution Strategy with Knowledge-Guided External Sampling

Evolution strategy (ES) is a kind of EA that evolves through probability distribution and is widely used in black-box global optimization. Covariance matrix adaptation ES (CMA-ES) stands out from other ES in global search performance and robustness by introducing a covariance matrix. CMA-ES updates covariance matrix and step size based on the ranking and displacement vector of sampled candidate solutions to search toward the optimal solution adaptively. In recent years, several research studies have tried to introduce knowledge transfer into ES. Li et al. [48] proposed a knowledge-guided external sampling (KGxS) approach and integrated KGxS into ES to develop a multitask ES (MTES) called MTES-KG. This approach includes two types of knowledge transfer methods to transfer samples that employs optimal domain similarity and function shape similarity among tasks. 

#### Algorithm 1: MTES-KG with CMA-ES
Input: The external sample number: $\tau$ ; tasks number: $K$ ; the knowledge type probability: $\alpha$
Output: The optimal solution: $x_{1;K}^{*}$ for $k = 1: K$ do
Set $C_k = I$ , $p_{\sigma,k} = 0$ , $p_{c,k} = 0$ ;
Initialize $m_k$ in the unified search space;
end
while the stop criterion is not met do
for $k = 1: K$ do
for $i = 1: \lambda$ do $x_{k,i} \leftarrow m_k + \sigma_k y_{k,i}, y_{k,i} \sim N(0, C_k)$ ; $d_{k,i} \leftarrow \left\| x_{k,i} - m_k \right\|$ ;
end $\langle d \rangle_{M,k} \leftarrow \frac{1}{\lambda} \sum_{i=1}^{\lambda} d_{k,i}$ ;
end
for $k = 1: K$ do
for $i = \lambda + 1: \lambda + \tau$ do
Randomly select a task $s (s \neq k)$ as the source task;
if rand (0,1) < $\alpha$ then $z \leftarrow m_s + \sigma_s y_{s,i}, y_{s,i} \sim N(0, C_s)$ ;
if $\| z - m_k \| < \langle d \rangle_{M,k}$ then $x_{k,i} \leftarrow z$ ;
else $x_{k,i} \leftarrow m_k + \langle d \rangle_{M,k} \frac{z - m_k}{\|z - m_k\|}$ ;
end
else $\langle y \rangle_S \leftarrow \sum_{i=1, t \neq j}^{\mu} y_{s,t,\mu}, j = rand\_int(1, \mu)$ ; $x_{k,i} \leftarrow m_k + \sigma_k C_k^{\frac{1}{2}} C_s^{-\frac{1}{2}} \langle y \rangle_S$ ;
end
end
for $k = 1: K$ do
Update distribution
parameters $m_k; p_{\sigma,k}; p_{c,k}; \sigma_k; C_k$ ;
end 
The detailed procedure of MTES-KG with CMA-ES is shown in Algorithm 1. where τ and α are the number of external samples per iteration, and the probability of using the two types of knowledge in KGxS, respectively. The core mechanism of KGxS is to transfer a small number of knowledge-guided samples from source task to target task, thereby providing promising search directions for improving the fitness value of the target task. KGxS is divided into two types of knowledge to transfer. 1) Domian KGxS: The optimal domain knowledge guides the probability distribution of the target task to search toward the distribution position of the source task, as shown in steps 17-22 in Algorithm 1. 2) Shape KGxS: The function shape knowledge provides search preference for the target distribution by learning the distribution of success samples in the source task in steps 24- 25 in Algorithm 1. 

## III. THE PROPOSED METHOD

### A. Motivation

Traditional CCEAs decompose LSOPs into a series of lower-dimensional subproblems to alleviate the solving complexity. These decomposed subproblems are optimized sequentially in a specific order, which will reduce the efficiency of solving LSOPs. According to the above descriptions, it can be seen that integrating the CC framework with the EMTO paradigm is feasible theoretically, which is able to solve LSOPs more efficiently by optimizing subproblems simultaneously. Therefore, this study presents a CCMTO framework. The CCMTO framework is equipped with two strategies, one is the construction strategy of multitask optimization problems aiming to select decomposed subproblems properly to construct MTOPs, and the other is the contribution-based resource allocation strategy of MTOPs and subtasks. 

Although KGxS has been successfully extended to solve MTOPs, there exists some limitations in calculating the mean sample distance of the target task, determining the pulling direction, computing the elite samples’ center position of the source task, and determining the number of elite samples. 

Domain KGxS calculates the mean sample distance of the target distribution N (0, C<sub>k</sub>) as $\left. d \right. _ { M }$ . However, the target sample distribution is scattered during the early iteration process of the algorithm. Employing a fixed $\left. d \right. _ { M }$ would lead to an excessive concentration of the pulled samples’ distribution, thereby diminishing their exploratory value. In the later stages of iteration process, the target sample distribution becomes increasingly localized within regions with high fitness values (i.e., low objective function values for minimization optimization problems). A fixed $\left. d \right. _ { M }$ increases the probability that the pulled samples fall near the distribution boundary, where the corresponding fitness values are relatively lower. Moreover, Domain KGxS pulls the sample z toward the direction from m to z, regardless of whether this direction points towards the direction that increases fitness value of target task. If the pulling direction aligns with the direction of decreasing fitness value, the pulled sample will still fall within regions with lower fitness values, resulting in inefficient utilization of samples. 

Shape KGxS calculates $\langle \boldsymbol { y } \rangle _ { s }$ by applying equal weighting to the top μ elite samples. However, the fitness values among these elite samples can exhibit significant variance. The equal weighting approach diminishes the shape preference of samples with high fitness value, leading to a bias of $\langle \boldsymbol { y } \rangle _ { s }$ towards samples with low fitness value and conveying shape knowledge with low accuracy. Furthermore, the number of elite samples is fixed and equal to μ, whereas the characteristics of source sample distribution can vary significantly during different periods of iteration process. In the initial phase of iteration process, the variance of the sample fitness values is large. Using a large μ introduces elite samples with low fitness values, which can have a negative impact on the acquisition of shape knowledge. Conversely, in the later stages of iteration process, using a small $\mu$ will lose the detailed shape knowledge of elite samples, such as local search preferences. 

To address the above issues, a MTES incorporating with dynamic distance threshold and adaptive elite sampling KGxS (MTES-DAKG) is proposed in this paper. It consists of a dynamic distance threshold domain KGxS with gradient correction and an adaptive elite sampling shape KGxS. The first domain KGxS approach increases the probability of samples falling within the region of high fitness values and ensures that the pulled sample always points towards the direction of increasing fitness value of the target distribution. The shape KGxS approach improves the accuracy of transferred shape knowledge. 

It should be noted that the proposed CCMTO framework can only solve fully separable and partially separable LSOPs. When solving a MTOP, the fitness evaluation of a subtask’s individuals requires the collaboration of other subtasks’ individuals to complete the problem solutions. However, all design variables are interacted in nonseparable LSOPs, which implies that the objective function of a subtask will vary with changes in the collaborators of other tasks. If the objective function keeps changing during the optimization process, it will be difficult for the algorithm to find the optimal solution. 

In real-world LSOPs, only a negligible number of them have strong correlations among all variables. For problems characterized by weak correlations, variables with weak correlations can be approximately decomposed by applying a correlation identification threshold, which transforms the original problem into a separable problem. Therefore, applying the CCMTO framework presents a novel and promising approach to address LSOPs. 

### B. Overall Framework

The proposed CCMTO overall framework is given in Algorithm 2, and Fig. 3 is the flowchart of CCMTO. The original LSOP is decomposed into a series of low-dimensional subproblems by the variable grouping strategy in step 1 of Algorithm 2, then these decomposed subproblems are selected to construct several MTOPs by the construction strategy of multitask optimization problems in step 2. As shown in step 7 of Algorithm 2, the EMTO algorithm is utilized as the optimization solver to optimize the current MTOP, and the contribution of this MTOP is calculated in step 13. After undergoing a whole co-evolutionary cycle, the optimization order is determined by the proposed resource allocation strategy in step 16 of Algorithm 2. 

### C. Construction Strategy of Multitask Optimization Problems

The construction strategy of multitask optimization problems is presented in Algorithm 3. The decomposed subproblems are sorted by the dimension in ascending order, and subproblems with the same dimensions are grouped into a group in step 1 of Algorithm 3. The number of subtasks in a MTOP is set to $n _ { s u b } .$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/e4396f3a-7739-44f2-a218-7a4643fa7c12/e40546ceefca9246d3209659c1d9717a5a867449f3bbc5c29f44dbe544800453.jpg)

Fig. 3. Flowchart of CCMTO

#### Algorithm 2: The proposed CCMTO Framework
Input: The objective function: $f(x)$
Output: The final optimal global solution: $x^{best}$
1 Decompose the original LSPO into several subproblems $\{s_1, ..., s_m\} = \text{grouping}(f(x))$ by the variable grouping strategy;
2 Select subproblems to construct MTOPs $\{T_1, ..., T_k\} = \text{MTOPs\_construction}(\{s_1, ..., s_m\})$ ;
3 Set contributions $\Delta F_i = 0 \quad i = 1, ..., k$ , best solution so far: $x^{best}$ ;
4 while the stop criterion is not met do
5 for $i = 1$ to $k$ do
6 Set last best solution $x_{last}^{best} = x^{best}$ ;
7 Optimize the current MTOP $T_i = \{t_{i,1}, ..., t_{i,n_i}\}$ ;
8 Obtain the optimal solution of each task: $x_{e_{i,j}}^{best} = \arg \min f(x_{e_{i,j}}; x_{e_{i,j}}^{best}) \quad j = 1, ...n_i$ ;
9 Update the optimal solution of current MTOP $T_i$ : $x_{e_{i}}^{best} = \{x_{e_{i,1}}^{best}, ..., x_{e_{i,n_i}}^{best}\}$ ;
10 if $f(x_{e_{i}}^{best}; x_{e_{i}}^{best}) < f(x^{best})$ then
11 Set $x^{best} = \{x_{e_{i}}^{best}; x_{e_{i}}^{best}\}$ ;
12 end
13 Update the contribution $\Delta F_i$ of current MTOP;
14 end
15 while $\min(\Delta F_i | i = 1, ...k) \neq \max(\Delta F_i | i = 1, ...k)$ do
16 Determine index $i$ of the MTOP to be optimized;
17 Repeat steps 6 to 13;
18 end
19 end 
As is well known, the dimension of unified search space is set to the maximum dimension of all tasks in EMTO algorithms, so the dimensional disparity among tasks is an important factor affecting the efficiency of optimization. Excessive dimensional differences between tasks can lead to poor optimal solution performance and waste of computational resources. The maximum dimension ratio $d _ { m a x }$ is introduced in the strategy. When there is only one subproblem in a group, if the dimensional difference between this subproblem and the subproblem in the nearest group is too large, this subproblem will be optimized separately. If the dimension ratio of this subproblem and the subproblem in the nearest group is within $[ 1 / d _ { m a x } , ~ d _ { m a x } ]$ , this subproblem is put into the nearest group. For subproblems in the same group, if their number is greater than 1 and not greater than $n _ { s u b } ,$ these subproblems are constructed to a MTOP in step 19 of Algorithm 3. Otherwise, every $n _ { s u b }$ subproblems are selected randomly as a MTOP to construct a series of MTOPs in step 21 of Algorithm 3. The number of subproblems to construct each MTOP is also a significant factor, as too many or too few subproblems can affect the optimization efficiency. The parameter sensitivity discussions on the $d _ { m a x }$ and $n _ { s u b }$ are in Section IV. 

#### Algorithm 3: Construction Strategy of Multitask Optimization Problems
Input: The decomposed subproblems: $\{s_{1},...,s_{m}\}$ ; the corresponding dimension of each subproblem: $\{d_{1},...,d_{m}\}$ ; the number of subtasks in a MTOP: $n_{sub}$ ; the maximum dimension ratio: $d_{max}$
Output: The constructed MTOPs: $\{T_{1},...,T_{k}\}$
1 Sort subproblems by the dimension in ascending order, and group subproblems with the same dimensions into one group: $Group_{i}=\{s_{1,1},...,s_{1,n_{i}}\},...,Group_{j}=\{s_{j,1},...,s_{j,n_{j}}\}$ , and the dimension of subproblems in each group is $D_{1},...,D_{j}$ ;
2 for i=1 to j do
3 if card( $Group_{i}$ )=1 then
4 if $D_{i}/D_{i-1}>d_{max}$ and $D_{i+1}/D_{i}>d_{max}$ then
5 See the subproblem $s_{i,1}$ as a single task and optimize it separately;
6 elseif $D_{i+1}/D_{i}<d_{max}$ then
7 Put the subproblem $s_{i,1}$ into $Group_{i+1}$ ;
8 else
9 Put the subproblem $s_{i,1}$ into $Group_{i-1}$ ;
10 end
11 end
12 end
13 for i=1 to sum(Group) do
14 if card( $Group_{i}$ )=1 then
15 See the subproblem $s_{i,1}$ as a single task and optimize it separately;
16 elseif $1<card(Group_{i})\leq n_{sub}$ then
17 Select all subproblems to construct a MTOP $T_{i,1}$ ;
18 else
19 Randomly select each $n_{sub}$ subproblems as a MTOP to construct m
20 MTOPs: $\left\{T_{i,1}=\{t_{i,1},...,t_{i,n_{sub}}\},...,T_{m,1}=\{t_{m,1},...,t_{m,n_{sub}}\}\right\}$ ;
21 end
22 end 

### D. MTES with Dynamic Distance Threshold and Adaptive Elite Sampling KGxS

1) Dynamic Distance Threshold Domain KGxS with Gradient Correction: Dynamic distance threshold domain KGxS with gradient correction divides the samples from the target distribution into different regions according to their fitness values, and calculates dynamic distance threshold for each region to replace the fixed $\left. d \right. _ { M }$ . The sample z generated by the source distribution is assigned to the nearest region by calculating the mean distance between z and samples in each region. The gradient correction strategy estimates the gradient at the expectation of the target distribution to determine whether the vector direction from the sample generated by source distribution to the expectation of the target distribution is towards the direction of increasing fitness value. This strategy fine tunes the vector direction of the sample if it is towards the direction of decreasing fitness value. 

Suppose that all tasks are minimization optimization problems, the general framework of the proposed dynamic distance threshold domain KGxS with gradient correction is given in Algorithm 4. In the first step, the samples of target distribution are sort by fitness values in descending order into different regions, the detailed regions are defined as follows 

$$
\begin{array}{l} S _ {h i g h} = \left\{\boldsymbol {x} _ {1: \mu_ {1} * \lambda} ^ {\text { order }}, \boldsymbol {x} ^ {\text { order }} \sim N (\boldsymbol {m} _ {t}, \boldsymbol {C} _ {t}) \right\} \\ S _ {m i d} = \left\{\boldsymbol {x} _ {\mu_ {1} * \lambda : (\mu_ {1} + \mu_ {2}) * \lambda} ^ {\text { order }}, \boldsymbol {x} ^ {\text { order }} \sim N (\boldsymbol {m} _ {t}, \boldsymbol {C} _ {t}) \right\} \\ S _ {l o w} = \left\{\boldsymbol {x} _ {(\mu_ {1} + \mu_ {2}) * \lambda : \lambda} ^ {\text { order }}, \boldsymbol {x} ^ {\text { order }} \sim N (\boldsymbol {m} _ {t}, \boldsymbol {C} _ {t}) \right\} \end{array}\tag{6}
$$

where $\lambda$ is the number of samples of target distribution, $x ^ { \mathrm { o r d e r } }$ is target distribution samples sorted by their fitness values in the 

#### Algorithm 4: Dynamic Distance Threshold Domain KGxS with Gradient Correction
Input: The target task samples: $X_t = \{x_{t,1}, ..., x_{t,\lambda}\} \sim N(m_t, C_t)$ ; the dimension of decision variable: n; KNN neighbor count: k; gradient perturbation step: β; the source task distribution: $N(m_s, C_s)$
Output: Knowledge-guided external sample: $\hat{x}$
1 Sort $X_t$ by the fitness value in descending order into different regions $\{S_{high}; S_{mid}; S_{low}\}$ ;
2 Calculate dynamic distance threshold $\langle d \rangle_{high}, \langle d \rangle_{mid}, \langle d \rangle_{low}$ ;
3 Generate sample $z \sim N(m_s, C_s)$ ;
4 for each region $S_g \in \{S_{high}; S_{mid}; S_{low}\}$ , g = high, mid, low do
5 Calculate $\text{dist}(z, x) = \|z - x\|$ for all $x \in S_g$ ;
6 Sort $\text{dist}(z, x)$ in ascending order, and take the top k as $K - NN_g$ ;
7 Calculate the average distance: $\text{avg\_dist}_g = \frac{1}{k} \sum_{x \in K - NN_g} \text{dist}(z, x)$ ;
8 end
9 Determine region affiliation: $S_{g^*} = \arg\min(\text{avg\_dist}_{high}, \text{avg\_dist}_{mid}, \text{avg\_dist}_{low})$ ;
10 Generate n unit vectors $e_1, ...e_n$ ;
11 for i = 1: n do
12 Calculate gradient component: $g_i = \frac{f(m_t + \beta \cdot e_i) - f(m_t - \beta \cdot e_i)}{2\beta}$ ;
13 Calculate optimal direction vector component: $g_i^{opt} = \begin{cases} -g_i, & \text{if } g_i > 0 \\ g_i, & \text{if } g_i \leq 0 \end{cases}$ ;
14 end
15 Normalized optimal direction vector is represented as: $g_{grad}^{opt} = \frac{[g_1^{opt}, ..., g_n^{opt}]^T}{\left\|\left[g_1^{opt}, ..., g_n^{opt}\right]^T\right\|}$ ;
16 Calculate direction vector: $v = \frac{z - m_t}{\|z - m_t\|}$ ;
17 Calculate angle between v and $g_{grad}^{opt}$ : $\theta = \arccos\left(\frac{v \cdot g_{grad}^{opt}}{\|v\| \cdot \|g_{grad}^{opt}\|}\right)$ ;
18 if $\theta < 90^\circ$ then
19 if $\|z - m_t\| < \langle d \rangle_g^*$ then
20 $\hat{x} \gets z$ ;
21 else
22 $\hat{x} \gets m_t + \langle d \rangle_g^* \cdot v$ ;
23 end
24 else
25 Calculate the correction vector $v'$ ;
26 $\hat{x} \gets m_t + \langle d \rangle_g^* \cdot v'$ ;
27 end 
descending order, $\mu _ { 1 }$ and $\mu _ { 2 }$ are the proportion coefficients of samples in the first and second regions, respectively. 
The dynamic distance threshold $\left. d \right. _ { g } .$ is calculated to obtain the mean distance of samples from the expectation ${ \pmb { m } } _ { t }$ of the target distribution in each region in step 2 of Algorithm 4. Next, the sample z is generated by the source distribution in step 3. The region affiliation of z is determined in steps 4-9 of Algorithm 4. First, for each region, the distances from z to all samples within the region are calculated and sorted in ascending order. Then, by using K-Nearest Neighbor algorithm, the average of the first k distances is computed, and the sample z is assigned to the region with the smallest average distance. The normalized optimal direction vector at the target expectation m<sub>t</sub> is estimated in steps 10-15 of Algorithm 4. The direction vector from m<sub>t</sub> to z, as well as the angle between direction vector and optimal direction vector, is calculated in steps 16-17 of Algorithm 4. If the angle is less than 90°and the distance from m<sub>t</sub> to z is less than the $\left. d \right. _ { g ^ { * } }$ , i.e., 
z is in the domain of the target distribution and the direction vector is towards the direction that increases fitness value of target task, the sample z is directly received as an external sample. If the angle is less than $9 0 ^ { \circ }$ but this distance is more than $\left. d \right. _ { g ^ { * } }$ , z is pulled toward ${ \pmb { m } } _ { t }$ as length as $\left. d \right. _ { g ^ { * } }$ . When the angle is more than 90°, z is pulled toward the correction vector $\nu ^ { \ast }$ as length as $\left. d \right. _ { g } .$ . The correction vector $\nu ^ { \ast }$ is calculated as follows 
![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/e4396f3a-7739-44f2-a218-7a4643fa7c12/aad30f8b4f545285878c008b4347612bdee648db8deedb5e8102fde4b9b71ec8.jpg)
![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/e4396f3a-7739-44f2-a218-7a4643fa7c12/8b82383b6d7e7b501a2d99eaf7cb141f03694a33087bd0cc839017fa0d4b4dc3.jpg)
![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/e4396f3a-7739-44f2-a218-7a4643fa7c12/fbe99a5bb45d19c6f6478e8bf1d8afb18254cf268dfbf48bcae8ef4447fcfc94.jpg)
(c) KGxS with gradient correction

Fig. 4. Dynamic distance threshold domain KGxS with gradient correction

$$
\boldsymbol {v} ^ {\prime} = \frac {\boldsymbol {v} + \boldsymbol {\varphi} \cdot \boldsymbol {g} _ {g a r d} ^ {o p t}}{\left\| \boldsymbol {v} + \boldsymbol {\varphi} \cdot \boldsymbol {g} _ {g a r d} ^ {o p t} \right\|}\tag{7}
$$

where v is the normalized direction vector from ${ \pmb { m } } _ { t }$ to $z ,$ $\pmb { g } _ { g a r d } ^ { o p t }$ is the normalized optimal direction vector, and $\varphi$ is the gradient correction coefficient to ensure that the angle between correction vector and optimal direction vector is less than 90°. 

Compared with domain KGxS approach in [48], the dynamic distance threshold ensures that the external samples accurately fall within the distribution of their regions respectively, and convey information about the source distribution more effectively. This KGxS strategy increases the probability of samples falling within the regions with high fitness values and avoids wasting computational resources in the regions with low fitness values. Meanwhile, the gradient correction strategy can ensure that the pulled sample always points towards the direction of increasing fitness value of the target distribution and increase the possibility of improving the target fitness value through external sampling. 

In order to visually display the details of the proposed dynamic distance threshold domain KGxS with gradient correction, Fig. 4 takes three subfigures as the example to show three types of the external sampling. In subfigure (a), the sample (red dot) generated by the source distribution is in target distribution’s high-potential region, i.e., region with high fitness values, and the angle between direction vector and optimal direction vector is an acute angle. Therefore, the sample is accepted directly as an external sample of the target distribution. As shown in subfigure (b), the red diamond 

#### Algorithm 5: Adaptive Elite Sampling Shape KGxS
Input: The target task samples: $X_t = \{ x_{t,1}, ..., x_{t,\lambda} \} \sim N(\boldsymbol{m}_t, C_t)$ ; the source task distribution: $X_s = \{ x_{s,1}, ..., x_{s,\lambda} \} \sim N(\boldsymbol{m}_s, C_s)$
Output: Knowledge-guided external sample: $\hat{x}$
1 Calculate dynamic elite sample count n;
2 Sort $X_s$ by the fitness value in ascending order, take top n as elite set $Z_s = \{ z_{s,1}, ..., z_{s,n} \}$ ;
3 Calculate the weight of sample $w_{si}$ ;
4 $\langle y \rangle_S \leftarrow \sum_{i=1,i \neq j}^{n} w_{si} \cdot (z_{s,i} - \boldsymbol{m}_s), j = rand\_int(1,n)$ ;
5 $\hat{x} \leftarrow \boldsymbol{m}_t + C_i^{\frac{1}{2}} C_s^{-\frac{1}{2}} \langle y \rangle_S$ 
indicates a sample generated by the source distribution, which is classified in target distribution’s high-potential region. This sample exceeds the dynamic distance threshold, and the angle between direction vector and optimal direction vector is an acute angle. Therefore, the red diamond is pulled towards the direction vector and then located at the high-potential boundary as the external sample (red dot). In subfigure (c), the red diamond indicates a sample generated by the source distribution, which is classified in target distribution’s lowpotential region. The angle between direction vector and optimal direction vector is an obtuse angle, which means that the direction vector will be towards the direction of decreasing fitness value. The red diamond is pulled toward the correction vector and then located at the low-potential boundary as the external sample (red dot). 
2) Adaptive Elite Sampling Shape KGxS: Adaptive elite sampling shape KGxS assigns weights based on fitness values of samples and enhances the contribution of elite samples with high fitness values. Meanwhile, adaptive elite sampling shape KGxS dynamically adjusts the number of elite samples to improve accuracy of shape knowledge. The detailed procedure of the proposed adaptive elite sampling shape KGxS is shown in Algorithm 5. The number of elite samples n increases with the number of iterations increases, and is calculated according to the following equation 
$$
n = \left\{ \begin{array}{l l} \left(\frac {\text { gen }}{M}\right) ^ {a} \left(n _ {\max} - n _ {\min}\right) + n _ {\min} & \text { if   } \text { gen } \leq M \\ n _ {\max} & \text { otherwise } \end{array} \right.\tag{8}
$$
where gen is the current number of generations, $M \ =$ 0.6Maxgen, Maxgen is the maximum number of generations. $n _ { m a x }$ and $n _ { m i n }$ represent the minimum elite sample number and the maximum elite sample number, respectively. a is a parameter that controls the increasing rate of the elite sample number. In step 2 of Algorithm 5, the samples of the source distribution are sort by fitness values in descending order, and the top n samples are selected as the elite set. Then, the weight of each elite sample is computed as follows 
$$
w _ {s i} = \frac {\exp \left(\gamma \cdot \frac {n - i}{n}\right)}{\sum_ {k = 1 , k \neq j} ^ {n} \exp \left(\gamma \cdot \frac {n - k}{n}\right)}, i = 1, \dots , n, i \neq j, j = r a n d \_ i n t (1, n)\tag{9}
$$
where $w _ { s i }$ represents the weight of the ith elite sample, and $\gamma$ is the weight coefficient. The random number $j$ is randomly excluded from the computation to ensure that the $w _ { s i }$ generated by each execution is different. In step 4 of Algorithm $^ { 5 , }$ the center position $\langle \boldsymbol { y } \rangle _ { s }$ of elite samples $z _ { s , 1 : n }$ relative to the expectation $\pmb { m } _ { s }$ of the source distribution is computed. In the end, the $\langle \boldsymbol { y } \rangle _ { s }$ is transformed into an external sample of the target distribution by applying the domain alignment approach. 
![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/e4396f3a-7739-44f2-a218-7a4643fa7c12/b0f6d3a32c8ba9cd11c89e03e24a22a16316bbd400ec83043205ebe6fc528eca.jpg)

Fig. 5. Adaptive elite sampling shape KGxS

Compared with shape KGxS approach in [48], the adaptive elite sampling shape KGxS calculates $\langle \boldsymbol { y } \rangle _ { s }$ by assigning weights to elite samples in descending order of fitness, and highlights the contributions of elite samples with high fitness values. The shape preference of these samples can dominate $\langle \boldsymbol { y } \rangle _ { s }$ to improve the accuracy of transferred shape knowledge. The adaptive adjustment of elite sample number can utilize more shape details of elite samples and improve shape knowledge accuracy. 

Fig. 5 shows two types of adaptive elite sampling shape KGxS. In subfigure (a), elite samples of the source distribution are denoted as red diamonds within black circle, showing a trend of searching to the right side of the source distribution. The external sample in the target distribution generated through adaptive elite sampling shape KGxS also tends to search towards the right. In subfigure (b), elite samples are searching towards the center of the source distribution, which guide the external sample to search towards the target distribution center. 

It should be noted that the number λ of generated samples in MTES-DAKG is less than 100 set in [48]. So, the number τ of external samples should be set to a smaller value, which is discussed in Section IV. Numerical Experiments and Discussions. In addition, using external sampling every generation may affect the task distribution and optimization efficiency owing to the small λ, the frequency of external sampling $f r e$ is introduced in MTES-DAKG, and its value is also discussed in Section IV. Numerical Experiments and Discussions. 

### E. Contribution-Based Resource Allocation Strategy of MTOPs and Subtasks

1) Stagnant Subtask Detection: In the proposed CCMTO framework, the original LSOP is decomposed into a series of MTOPs, and each MTOP contains multiple subtasks. When using EMTO algorithms for solving a MTOP, all subtasks in this MTOP are optimized simultaneously. However, not all subtasks are equally difficult to solve, and for the subtasks that 

are easy to be optimized, a small number of computational resources are sufficient to obtain their optimal solutions. At this point, continuing to allocate computing resources to these subtasks does not make contributions to the improvement of the best overall objective function value, as the corresponding subpopulations are in a stagnant stage. Therefore, a stagnant subtask detection method is proposed, in which computational resources are no longer allocated to these stagnant subpopulations. This mechanism can save some computational cost on stagnant subtasks to improve the efficiency of the proposed CCMTO framework. 

In order to check whether the subtask is stagnant, the proposed stagnant subtask detection method employs both fitness value improvement and population diversity as detection indicators by calculating the relative variation of objective function values, and the relative variation of the mean and standard deviation of individuals’ design variable values in dimension. 

Suppose T<sub>i</sub> denotes the ith constructed MTOP through the construction strategy of multitask optimization problems, and $t _ { i , j }$ denotes thejth subtask in the MTOP. For the subpopulation corresponding to $t _ { i , j }$ at the Gth generation, the relative variation of the best objective function values, and the relative variation of the mean and standard deviation of individuals design variable values in dimension are calculated as follows 

$$
\Delta f _ {G} = \left\| \frac {f (\boldsymbol {x} _ {\in t _ {i , j} , G - 1} ^ {b e s t} ; \boldsymbol {x} _ {\notin t _ {i , j}}) - f (\boldsymbol {x} _ {\in t _ {i , j} , G} ^ {b e s t} ; \boldsymbol {x} _ {\notin t _ {i , j}})}{f (\boldsymbol {x} _ {\in t _ {i , j} , G - 1} ^ {b e s t} ; \boldsymbol {x} _ {\notin t _ {i , j}})} \right\|\tag{10}
$$

$$
\Delta m _ {d, G} = \left\| \frac {m _ {d , G - 1} - m _ {d , G}}{m _ {d , G - 1}} \right\|, \quad m _ {d, G} = \frac {1}{N} \sum_ {n = 1} ^ {N} x _ {\in t _ {i, j}, d, G} ^ {n}\tag{11}
$$

$$
\Delta \mathrm{std} _ {d, G} = \left\| \frac {\mathrm{std} _ {d , G - 1} - \mathrm{std} _ {d , G}}{\mathrm{std} _ {d , G - 1}} \right\|, \quad \mathrm{std} _ {d, G} = \sqrt {\frac {1}{N} \sum_ {n = 1} ^ {N} (x _ {\in t _ {i , j} , d , G} ^ {n} - m _ {d , G}) ^ {2}}\tag{12}
$$

where $\pmb { x } _ { \in t _ { i , j } , G } ^ { b e s t }$ is the best solution at the Gth generation, and the collaborator $\boldsymbol { x } _ { \ u { \notin t _ { i , j } } }$ is set to be fixed. N is the subpopulation size, $\pmb { x } _ { \in t _ { i , j } , G } ^ { n } = ( \pmb { x } _ { \in t _ { i , j } , 1 , G } ^ { n } , . . . , \pmb { x } _ { \in t _ { i , j } , D , G } ^ { n } )$ is nth individual, D is the dimension of decision variables, and $x _ { \in t _ { i , j } , d , G } ^ { n }$ is the dth variable value of the $\pmb { x } _ { \in t _ { i , j } , G } ^ { n }$ . If the relative variation of the best objective function values remains unchanged for several successive generations, this subpopulation is considered to be stagnant in fitness value. When the relative variation of both the mean and standard deviation of individuals’ dth design variable value remains unchanged over several successive generations, this subpopulation can be considered to be stagnant in this dimension [18]. Only when a subpopulation is stagnant in fitness value and in all dimensions, the subtask can be considered to be in a stagnant state. The method to check whether a subpopulation is stagnant in fitness value is as shown in follows 

$$
\nu_ {G} = \left\{ \begin{array}{l l} \nu_ {G - 1} + 1 & \text { if } \Delta f _ {G} <   \varepsilon \\ 0 & \text { otherwise } \end{array} \right.\tag{13}
$$

where $\upsilon _ { G }$ denotes the number of successive generations where the value best objective function remains unchanged, and note that $\upsilon _ { 0 } = 0$ . ε is the threshold of objective function value variation, and the value is 1e-6 in this study. 

#### Algorithm 6: Contribution-Based Resource Allocation Strategy of MTOPs and Subtasks
Input: The constructed MTOPs: $\{T_{1},...,T_{k}\}$ ; The subtasks in each MTOP: $T_{i}=\{t_{i,1},...,t_{i,n_{i}}\}$
Output: The final optimal global solution: $x^{best}$
1 Set contributions of MTOPs $\Delta F_{i}=0$ , stagnant subtask set $S_{i}=\varnothing$ , $i=1,\ldots,k$ , best solution so far: $x^{best}$ ;
2 while the stop criterion is not met do
3 For each MTOP, reset $S_{i}=\varnothing$ , $v^{i,j}=0$ , $\eta^{i,j}=0$ , $j=1,\ldots,n_{i}$ ;
4 for i=1 to k do
5 $x_{last}^{best} \leftarrow x^{best}$ ;
6 for j=1 to $n_{i}$ do
7 $x_{e_{i,t_{j}}}^{best} \leftarrow$ Optimize $t_{i,j}$ by EMTO optimizer;
8 if $\rho_{G}^{i,j}=1$ then
9 $|S_{i}\leftarrow j;$ 10 end
11 end
12 $x_{e_{i,t_{i}}}^{best} \leftarrow \{x_{e_{i,t_{i}}}^{best},...,x_{e_{i,n_{i}}}^{best}\}$ , $x^{best} \leftarrow \{x_{e_{i,t_{i}}}^{best};x_{e_{i,t_{i}}}^{best}\}$ ;
13 $\Delta F_{i}=\left|f(x_{last}^{best})-f(x^{best})\right|$ ;
14 if all $\rho_{G}^{i,j}=1$ , $j=1,\ldots,n_{i}$ then
15 $|\Delta F_{i}=0;$ 16 end
17 end
18 while $\min(\Delta F_{i}|i=1,\ldots,k)\neq\max(\Delta F_{i}|i=1,\ldots,k)$ do
19 $i\leftarrow$ the index of the maximum $\Delta F_{i}$ ;
20 $x_{last}^{best} \leftarrow x^{best}$ ;
21 for $j=1,\ldots,n_{i}, j\notin S$ do
22 $x_{e_{i,t_{j}}}^{best} \leftarrow$ Optimize $t_{i,j}$ by EMTO optimizer;
23 if $\rho_{G}^{i,j}=1$ then
24 $|S_{i}\leftarrow j;$ 25 end
26 end
27 $x_{e_{i,t_{i}}}^{best} \leftarrow \{x_{e_{i,t_{i}}}^{best},...,x_{e_{i,n_{i}}}^{best}\}$ , $x^{best} \leftarrow \{x_{e_{i,t_{i}}}^{best};x_{e_{i,t_{i}}}^{best}\}$ ;
28 $\Delta F_{i}=\left|f(x_{last}^{best})-f(x^{best})\right|$ ;
29 if all $\rho_{G}^{i,j}=1$ , $j=1,\ldots,n_{i}$ then
30 $|\Delta F_{i}=0;$ 31 end
32 end
33 end 
If a subpopulation is stagnant in the dth dimension, the indicator $\varphi _ { d , G }$ is defined as follows 
$$
\varphi_ {d, G} = \left\{ \begin{array}{l l} 1 & \text { if } \Delta m _ {d, G} <   \varepsilon \text { and } \Delta \mathrm{std} _ {d, G} <   \varepsilon \\ 0 & \text { otherwise } \end{array} \right.\tag{14}
$$
where $\varphi _ { d , G }$ denotes whether the mean and standard deviation of individuals’ design variable values in dimension d remain unchanged from the last generation, and note that $\varphi _ { d , 0 } = 0$ Then $\sigma _ { G }$ denotes the number of dimensions where $\varphi _ { d , G } = 1$ 
$$
\sigma_ {G} = \sum \varphi_ {d, G}\tag{15}
$$
If the subpopulation is stagnant in all dimensions, $\sigma _ { \it G } = D \it { \cdot } \eta _ { \it G }$ denotes the number of successive generations where $\sigma _ { G } = D$ , and note that $\eta _ { \mathrm { 0 } } = 0$ 
$$
\eta_ {G} = \left\{ \begin{array}{l l} \eta_ {G - 1} + 1 & \text { if } \sigma_ {G} = D \\ 0 & \text { otherwise } \end{array} \right.\tag{16}
$$
When the subpopulation is stagnant in fitness value and in all dimensions for successive generations, the subtask is in a stagnant state, and the detection flag $\rho _ { G }$ is calculated as follows 
$$
\rho_ {G} = \left\{ \begin{array}{l l} 1 & \text { if } \nu_ {G} \geq U \text { and } \eta_ {G} \geq U \\ 0 & \text { otherwise } \end{array} \right.\tag{17}
$$
where U is a parameter and is defined as 
$$
U = \min (D, \text { Maxgen })\tag{18}
$$
Once $\rho _ { \scriptscriptstyle G } = 1$ for a subpopulation, computational resources are immediately no longer allocated to this stagnant subtask. This subtask is excluded from the optimization of the MTOP it belongs to, which means that it will not undergo evolution in the CCMTO framework. 
2) Resource Allocation Strategy: For a MTOP $T _ { i , }$ after finishing optimization in a cycle, its contribution is calculated as follows 
$$
\Delta F _ {i} = \left| f (\boldsymbol {x} _ {l a s t} ^ {b e s t}) - f (\boldsymbol {x} ^ {b e s t}) \right|\tag{19}
$$
where $f ( \boldsymbol { x } _ { l a s t } ^ { b e s t } )$ and $f ( \boldsymbol { x } ^ { b e s t } )$ are the best overall objective values before and after $T _ { i }$ undergoes optimization, respectively. The contribution-based resource allocation strategy of MTOPs and subtasks is shown in Algorithm 6. 
In the first, the contribution of every MTOP is set to zero, and the stagnant subtask set of every MTOP is set to the empty set. During the first co-evolutionary cycle, all MTOPs undergo optimization one by one in steps 4-17 of Algorithm 6. If a subtask $t _ { i , j }$ is in a stagnant state, i.e. $\rho _ { G } ^ { i , j } = 1$ when optimizing the MTOP $T _ { i , }$ the index j of this subtask is put into the set $S _ { i , }$ and no more computational resources will be allocated to this subtask in the subsequent co-evolutionary cycles. The value of $\Delta F _ { i }$ for each MTOP is computed according to (19) in step 13 of Algorithm 6. If all subtasks in a MTOP are all in a stagnant state, i.e. all $\rho _ { G } ^ { i , j } = 1$ , this MTOP can be considered to be in a stagnant state, and the value of $\Delta F _ { i }$ is set to zero. In the subsequent co-evolutionary cycles, the MTOP with the largest value of $\Delta F _ { i }$ is selected to undergo optimization in step 19 of Algorithm 6. In steps 21-26 of Algorithm 6, the subtasks that have been considered to be stagnant will be excluded from the MTOP optimization cycle, and the index of stagnant subtasks is also put into the set S<sub>i</sub> during optimization. Then, the contribution value of this MTOP is updated according to (19) at the end of the optimization. When the values of $\Delta F _ { i }$ are same for all the MTOPs, the algorithm will go to steps 3-17 to reset the contribution and the stagnant subtask set of each MTOP, then initiates a new co-evolutionary cycle. The above process is repeated until the stop criterion is met. 
The proposed stagnant subtask detection can accurately identify a stagnant subtask according to the mean and standard deviation of individuals’ gene values, as well as the best objective function value in the subpopulation. Meanwhile, this resource allocation strategy can save computational cost on stagnant subtasks, and make a more efficient computational resource allocation among subtasks. 

### F. Computational Complexity

In the proposed MTES-DAKG, assuming that K is the number of subtasks in a MTOP, λ is the sample number for each task, and n is the maximum dimension of decision variables. The computational complexity of MTES-DAKG in one evolutionary loop is analyzed as follows. The complexity of sample new solutions is $O \left( n ^ { 2 } \cdot K \right)$ . In the dynamic distance threshold domain KGxS with gradient correction as shown in Algorithm 4, calculating dynamic distance threshold takes O $( \lambda \cdot n \cdot K )$ in step 2 of Algorithm 4. Then, calculating the average distance takes $O \left( n \cdot K \right)$ in step 4-8 of Algorithm 4. Subsequently, $O \ ( n ^ { 2 } \cdot K )$ is used to calculate the optimal direction in step 11-16 of Algorithm 4. At last, calculating angle and conducting gradient correction takes $O \ ( n \cdot K )$ in step 17-27 of Algorithm 4. Therefore, the total computational complexity is $O \ ( n ^ { 2 } \cdot K )$ , which is the same as the original domain KGxS. In the adaptive elite sampling shape KGxS as shown in Algorithm 5, it needs $O \left( \lambda \cdot \log ( \lambda ) \cdot K \right)$ to sort the elite samples in step 2 of Algorithm 5. Then, calculating the center position takes $O \ ( n \cdot K )$ in step 4 of Algorithm 5. At last, $O \left( n ^ { 2 } \cdot K \right)$ is used to transform the external sample in step 5 of Algorithm 5. Overall, the total computational complexity is max $( O \ ( \lambda \ \cdot \ \log ( \lambda ) \ \cdot \ K ) , \ O \ ( n ^ { 2 } \ \cdot \ K ) )$ . In summary, the computational complexity of MTES-DAKG is max (O $( \lambda \cdot \log ( \lambda ) \cdot K ) , O \left( n ^ { 2 } \cdot K \right) )$ , which is equal to MTES-KG. The proposed MTES-DAKG can improve the optimization performance through finer external sample transfer without increasing the computational complexity. 

Actually, the proposed CCMTO framework is just transformed the paradigm for addressing LSOPs from sequentially solving each subproblem to employing EMTO algorithms for solving a series of MTOPs. Therefore, compared to traditional CC framework, the CCMTO framework does not increase additional computational complexity. 

## IV. NUMERICAL EXPERIMENTS AND DISCUSSIONS

In this section, the proposed CCMTO using MTES-DAKG is tested and compared with several state-of-the-art large-scale algorithms developed in recent years. Then, the numerical experiments are conducted to analyze the parameter sensitivity, as well as the effectiveness of components within the proposed algorithm. At last, the proposed algorithm is employed in a real-world large-scale problem to verify its applicability. 

### A. Benchmark Test Suite and Parameter Settings

The proposed CCMTO with MTES-DAKG (CCMTO-MTES-DAKG) is tested on CEC2010 and CEC2013 largescale benchmark test suites [45] with dimensions up to 1000. The CEC2010 and CEC2013 test suites are widely used in large-scale problem algorithm testing, consisting of 20 and 15 LSOPs, respectively. According to the separability of the problems, the benchmark suites could be divided into three categories, fully separable problems, partially separable problems, and nonseparable problems [25]. Because CCMTO-MTES-DAKG can only solve fully separable and partially separable LSOPs, the first 18 benchmark test problems on CEC2010 test suite, and the first 11 benchmark test problems on CEC2013 test suite, are used to test the performance of CCMTO-MTES-DAKG. The stop criterion for all compared algorithms in this study is that the maximum number of fitness 

TABLE I

THE PARAMETER SETTINGS FOR CCMTO-MTES-DAKG

<table><tr><td colspan="9">The parameter settings for CCMTO framework:</td></tr><tr><td colspan="9">Number of subtasks in a MTOP:<eq>n_{sub} = 5</eq>Maximum dimension ratio:<eq>d_{max} = 2</eq>Threshold of objective function value variation: <eq>\varepsilon = 1e-6</eq></td></tr><tr><td colspan="9">The parameter settings for MTES-DAKG:</td></tr><tr><td colspan="9">Proportion coefficient of samples in the first region: <eq>\mu_1 = \begin{cases} 0.4 &amp; \text{if } \lambda \leq 6 \\ 0.3 &amp; \text{otherwise} \end{cases}</eq></td></tr><tr><td colspan="9">Proportion coefficient of samples in the second region: <eq>\mu_2 = \begin{cases} 0.6 &amp; \text{if } \lambda \leq 6 \\ 0.4 &amp; \text{otherwise} \end{cases}</eq></td></tr><tr><td colspan="9">KNN neighbor count: <eq>k = 5</eq>Gradient perturbation step: <eq>\beta = 1e-5</eq>Gradient correction coefficient: <eq>\varphi = 1</eq>Minimum elite sample number: <eq>n_{\min} = 0.3\lambda</eq>Maximum elite sample number: <eq>n_{\max} = 0.8\lambda</eq>Elite coefficient: <eq>a = 2</eq>Weight coefficient: <eq>\gamma = 2</eq>Number of external samples: <eq>\tau = 1</eq>Frequency of external sampling: <eq>fre = 0.1Maxgen</eq></td></tr><tr><td colspan="9">TABLE IITHE AVERAGE RANKINGS OF EACH ALGORITHM</td></tr><tr><td rowspan="2">Algorithm</td><td colspan="4">CEC2010</td><td colspan="4">CEC2013</td></tr><tr><td>+</td><td><eq>\approx</eq></td><td>-</td><td>Ranking</td><td>+</td><td><eq>\approx</eq></td><td>-</td><td>Ranking</td></tr><tr><td>EDGCC</td><td>16</td><td>0</td><td>2</td><td>8.94</td><td>8</td><td>0</td><td>3</td><td>10.09</td></tr><tr><td>CMAES-EDG</td><td>9</td><td>3</td><td>6</td><td>5.72</td><td>9</td><td>1</td><td>1</td><td>6.18</td></tr><tr><td>DCCC</td><td>13</td><td>0</td><td>5</td><td>7.06</td><td>7</td><td>0</td><td>4</td><td>4.64</td></tr><tr><td>EDCC-ERDG</td><td>18</td><td>0</td><td>0</td><td>10.89</td><td>10</td><td>0</td><td>1</td><td>9.64</td></tr><tr><td>MMO-CC</td><td>15</td><td>2</td><td>1</td><td>8.78</td><td>9</td><td>1</td><td>1</td><td>11.27</td></tr><tr><td>SSLPSO</td><td>13</td><td>2</td><td>3</td><td>7.00</td><td>8</td><td>0</td><td>3</td><td>6.45</td></tr><tr><td>DCBA</td><td>14</td><td>1</td><td>3</td><td>7.39</td><td>9</td><td>0</td><td>2</td><td>5.64</td></tr><tr><td>RCI-PSO</td><td>14</td><td>0</td><td>4</td><td>7.00</td><td>8</td><td>0</td><td>3</td><td>6.82</td></tr><tr><td>MSORL</td><td>13</td><td>1</td><td>4</td><td>8.89</td><td>8</td><td>0</td><td>3</td><td>8.91</td></tr><tr><td>GTDE</td><td>14</td><td>1</td><td>3</td><td>8.33</td><td>8</td><td>0</td><td>3</td><td>6.55</td></tr><tr><td>SRTP</td><td>14</td><td>0</td><td>4</td><td>8.06</td><td>9</td><td>0</td><td>2</td><td>5.82</td></tr><tr><td>SDLSO</td><td>12</td><td>0</td><td>6</td><td>7.61</td><td>8</td><td>0</td><td>3</td><td>9.00</td></tr><tr><td>RLLPSO</td><td>12</td><td>0</td><td>6</td><td>9.50</td><td>8</td><td>0</td><td>3</td><td>9.55</td></tr><tr><td>AGLDPSO</td><td>15</td><td>0</td><td>3</td><td>10.33</td><td>8</td><td>1</td><td>2</td><td>9.64</td></tr><tr><td>CCMTO-MTES-DAKG</td><td>\</td><td>\</td><td>\</td><td>4.22</td><td>\</td><td>\</td><td>\</td><td>4.18</td></tr></table>

evaluations MaxFEs reaches to 3e6. EDG [12] is an efficient differential grouping method, and it is employed in CCMTO-MTES-DAKG in this study. The FEs used by EDG are included in the total number of FEs. The parameter settings for CCMTO and MTES-DAKG are given in TABLE I, and the parameters in bold fonts will be discussed further in a later section. 

All algorithms were run independently 25 times in experiments. To better assess the comparison results, the Welch’s t-test was conducted for all comparison algorithms, and Wilcoxon’s rank-sum test with the Holm–Bonferroni correction was utilized for each parameter setting. The significance level was set at $\begin{array} { r } { \alpha = 0 . 0 5 . } \end{array}$ . “+/≈/-” indicates that the proposed method is significant better/equal/worse than the comparison algorithm, respectively. 

### B. Compared with State-of-the-art Algorithms

The proposed CCMTO-MTES-DAKG is tested and compared with fourteen state-of-the-art large-scale algorithms in recent years, including 5 CC-based algorithms, namely EDGCC [10], CMAES-EDG [12], DCCC [19], DECC-ERDG [9], MMO-CC [49], and 9 non-decomposition algorithms, including SSLPSO [50], DCBA [51], RCI-PSO [52], MSORL [53], GTDE [54], SRTP [55], SDLSO [56], RLLPSO [57], AGLDPSO [58]. The parameters of these algorithms were set as recommended in the original papers. 

TABLE III

RESULTS OF PARAMETER SENSITIVITY ANALYSIS

<table><tr><td><eq>n_{sub}</eq></td><td>2</td><td>3</td><td>5</td><td>7</td><td>10</td><td>20</td></tr><tr><td>+/-/-</td><td>7/3/0</td><td>7/3/0</td><td>\</td><td>6/4/0</td><td>6/4/0</td><td>7/3/0</td></tr><tr><td>Ranking</td><td>3.50</td><td>4.70</td><td>1.20</td><td>3.70</td><td>3.10</td><td>3.90</td></tr><tr><td><eq>d_{max}</eq></td><td colspan="2">1</td><td>2</td><td colspan="2">4</td><td>limitless</td></tr><tr><td>+/-/-</td><td colspan="2">3/1/0</td><td>\</td><td colspan="2">3/1/0</td><td>4/0/0</td></tr><tr><td>Ranking</td><td colspan="2">2.00</td><td>1.00</td><td colspan="2">2.75</td><td>3.75</td></tr><tr><td>τ</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>+/-/-</td><td>7/3/0</td><td>\</td><td>8/2/0</td><td>8/2/0</td><td>8/1/1</td><td>7/2/1</td></tr><tr><td>Ranking</td><td>3.70</td><td>1.20</td><td>3.60</td><td>4.00</td><td>3.60</td><td>3.60</td></tr><tr><td>fre</td><td>everygen</td><td>0.05Maxgen</td><td>0.1Maxgen</td><td>0.2Maxgen</td><td>0.3Maxgen</td><td>0.5Maxgen</td></tr><tr><td>+/-/-</td><td>8/1/1</td><td>7/2/1</td><td>\</td><td>5/5/0</td><td>6/3/1</td><td>6/4/0</td></tr><tr><td>Ranking</td><td>4.60</td><td>3.60</td><td>1.90</td><td>3.50</td><td>3.30</td><td>3.60</td></tr></table>

The comparison results of different algorithms on the CEC2010 and CEC2013 benchmark test suites are presented in Table SⅠ and Table SⅡ in the supplementary material, respectively. The average rankings of each algorithm are given in TABLE II. 

CCMTO-MTES-DAKG achieved the best average rankings on both the CEC2010 and CEC2013 benchmark test suites, and it performed best among all algorithms. As seen from results of CEC2010 test suite, CCMTO-MTES-DAKG achieved the best results on 5 benchmark problems, which is most in all algorithms. Besides, it can be seen that CCMTO-MTES-DAKG performed at a moderate level on fully separable problems except f3. This is because the grouping method EDG [12] incorrectly identified the completely separable f3 as a completely nonseparable problem, resulting in poor performance of CCMTO-MTES-DAKG on f3. For partially separable problems, even on the problems where CCMTO-MTES-DAKG does not achieve the best result, it still obtains well performance. 

From the results of CEC2013 test suite, although CCMTO-MTES-DAKG did not demonstrate the best performance among fully separable problems f1-f3, it still obtained the highly competitive near-optimal results for f1 and f3. Among partially separable problems f4-f11, CCMTO-MTES-DAKG achieved the best results on 6 problems except f5 and f9, and it performed significantly better than its competitors on f4, f7, f8 by several orders of magnitude. 

From the comparison results, it is obvious that the proposed CCMTO-MTES-DAKG is effective in solving both the largescale fully separable and partially separable problems, and it is a competitive solver for large-scale problems by employing the novel CCMTO framework. 

### C. Parameter Sensitivity Discussion

In this section, the effects of these parameters (as listed in bold fonts in TABLE I) are discussed on the CEC2013 test suite. Because the test problems in CEC2013 test suite are more complex than those in CEC2010 test suite, and the results of parameter sensitivity are representative and universal. In the CCMTO framework, $n _ { s u b }$ and $d _ { m a x }$ are necessary to analyze their effects, because they can affect the optimization performance of EMTO algorithm. In the proposed MTES-DAKG, τ and fre are significant to external sampling. The detailed settings for these four parameters are given in Section S-Ⅰ of the supplementary material. It should be noted that the value of $d _ { m a x }$ does not affect the performance of the algorithm on fully separable problems, because the dimensions of all subproblems are equal 

TABLE IV RESULTS OF COMPONENT ANALYSIS IN CCMTO-MTES-DAKG

<table><tr><td colspan="5">Results on different resource allocation strategies for CCMTO-MTES-DAKG</td></tr><tr><td>Algorithm</td><td>+</td><td>≈</td><td>-</td><td>Ranking</td></tr><tr><td>CBCC1</td><td>8</td><td>3</td><td>0</td><td>3.82</td></tr><tr><td>CBCC2</td><td>8</td><td>3</td><td>0</td><td>4.73</td></tr><tr><td>CBCC3</td><td>8</td><td>2</td><td>1</td><td>3.73</td></tr><tr><td>CCFR</td><td>6</td><td>4</td><td>1</td><td>3.73</td></tr><tr><td>CCFR2</td><td>7</td><td>3</td><td>1</td><td>3.18</td></tr><tr><td>CCFR3</td><td>7</td><td>3</td><td>1</td><td>3.00</td></tr><tr><td>CCMTO-MTES-DAKG</td><td>\</td><td>\</td><td>\</td><td>1.36</td></tr><tr><td colspan="5">Results on different EMTO algorithms for CCMTO</td></tr><tr><td>Algorithm</td><td>+</td><td>≈</td><td>-</td><td>Ranking</td></tr><tr><td>CCMTO-MaTDE</td><td>10</td><td>0</td><td>1</td><td>2.55</td></tr><tr><td>CCMTO-G-MFEA</td><td>10</td><td>0</td><td>1</td><td>3.00</td></tr><tr><td>CCMTO- MTEA-AD</td><td>10</td><td>1</td><td>0</td><td>3.18</td></tr><tr><td>CCMTO-MTES-DAKG</td><td>\</td><td>\</td><td>\</td><td>1.18</td></tr><tr><td colspan="5">Results on CCMTO-MTES-DAKG with different components</td></tr><tr><td>Algorithm</td><td>+</td><td>≈</td><td>-</td><td>Ranking</td></tr><tr><td>wo-DA</td><td>7</td><td>3</td><td>0</td><td>3.30</td></tr><tr><td>wo-DT-DoS</td><td>8</td><td>2</td><td>0</td><td>3.40</td></tr><tr><td>wo-AS-SaS</td><td>7</td><td>3</td><td>0</td><td>2.70</td></tr><tr><td>wo-SD</td><td>6</td><td>4</td><td>0</td><td>3.30</td></tr><tr><td>CCMTO-MTES-DAKG</td><td>\</td><td>\</td><td>\</td><td>1.10</td></tr></table>

to 1. Therefore, partially separable problems f4-f7 of CEC2013 test suite as representative problems are tested. 

The comparison results and convergence curves are provided in Table SⅢ-Table SⅥ and Fig. S1-Fig. S4, respectively. The rankings for each parameter sensitivity analysis are provided in TABLE III. 

The results of each setting for f3 are identical, because f3 is a fully separable function, the grouping method EDG cannot recognize its separability, and it is classified as a nonseparable function. Thus, the number of tasks is only one and it is not affected by these parameters. The result for f3 is not included in the calculation ofaverage rankings. 

*1) Parameter $n_{sub}$*: The results of the significance tests show that $n _{sub} = 5$ performed best among all settings of $n _ { s u b } ,$ and it achieved the best average ranking. Because MTES-DAKG adopts a random source task selection strategy, without considering the similarities between subtasks. When the $n _ { s u b }$ is small, there is a relatively high probability of assigning subtasks with low similarity to the same MTOP and selecting tasks with low similarity as source tasks. Consequently, this can negatively affect the optimization efficiency, which is evident in the result. When the $n _ { s u b }$ is excessively large, the probability of negative transfer increases. Meanwhile, the optimization counts of subtasks with large contributions to fitness values will be reduced in comparison to those with smaller $n _ { s u b }$ due to limited computational resources, and the optimization performance can be affected. From the comparison results, this study suggests that the value of $n _ { s u b }$ is set to 5. 

*2) Parameter $d_{max}$*: The statistical test results show that $d_{max}$ = 2 performed best among all $d_{max}$ settings. It can be found that the optimization performance is poor when $d_{ m a x }$ is set to 4 or bigger. From the convergence curves, it can be seen that the algorithm can find a well solution and also exhibit the fastest convergence speed when $d _ { m a x } \ = \ 2$ . The obtained optimal solutions are poor for other settings of $d _ { m a x } ,$ and the optimization performances are inferior to $d _ { m a x } = 2$ . Therefore, it is recommended that $d _ { m a x } = 2$ in this study. 

*3) Parameter $\tau$*: From the results, it can be seen that $\tau = 1$ performs best among all τ settings. Theoretically, when τ is set to 0, no external samples are transferred to target distribution, which means that there is no knowledge transfer between source task and target task. Consequently, MTES-DAKG becomes equivalent to CMA-ES due to the absence ofknowledge transfer, resulting in a decrease in optimization performance. Because the number λ of generated samples in MTES-DAKG is less than 100 set in [48], if τ is set too large, the phenomenon of negative transfer can affect the performance of MTES-DAKG. In summary, it is recommended that τ be set to 1 in this study. 

*4) Parameter $fre$*: These results show that setting fre to 0.1Maxgen leads to the best performance on these test problems. If employing external sampling every generation, the algorithm performance will decrease due to excessive knowledge transfer, leading to negative transfer and trapping in local optimal solution. Low frequency of external sampling also decreases the optimization performance due to insufficient knowledge transfer. It suggests that fre be set to 0.1Maxgen in this study. 

### D. Discussion of Components in CCMTO-MTES-DAKG

In order to demonstrate the effectiveness and performance of the proposed contribution-based resource allocation strategy, the proposed MTES-DAKG for solving constructed MTOPs, and components in CCMTO-MTES-DAKG, this section conducts component analysis experiments. Six resource allocation strategies are tested to compare with the proposed one, they are CBCC1 [59], CBCC2 [59], CBCC3 [60], CCFR [18], CCFR2 [20], and CCFR3 [21]. To show the performance of the proposed MTES-DAKG for solving constructed MTOPs, this study also incorporates 3 EMTO algorithms with CCMTO, respectively. They are MaTDE [61], G-MFEA [40], and MTEA-AD [62]. This section conducts component analysis experiments to investigate the performance improvement of the proposed dynamic distance threshold domain KGxS with gradient correction (DT-DoS) and adaptive elite sampling shape KGxS (AS-SaS). Meanwhile, the effectiveness of stagnant subtask detection mechanism on the CCMTO framework is also discussed. Four types of variants are employed for comparison, which are detailed below. 
1) wo-DA: equivalent to MTES-KG [48]. 
2) wo-DT-DoS: Without DT-DoS, only AS-SaS. 
3) wo-AS-SaS: Without AS-SaS, only DT-DoS. 
4) wo-SD: Without stagnant subtask detection. 

These ablation experiments are conducted on the 11 test problems of CEC2013 test suite. The detailed experimental results are shown in Table SⅦ-Table SⅨ in the supplementary material, and the convergence curves are given in Fig. S5- Fig. S7. The results of nonparametric tests are presented in TABLE IV. 

*1) Resource allocation strategies*: It is obvious that the proposed resource allocation strategy achieved the best average ranking on both the strategies, and it achieved the best results on 10 benchmark problems except f8. From the experimental results and convergence curves, it can be seen that the proposed strategy can find the best solution and exhibits fast convergence speed for each problem compared with other strategies. Therefore, the proposed resource allocation strategy is suitable for CCMTO and effective. 

*2) EMTO algorithms*: The results show that MTES-DAKG achieved the best average ranking on both the EMTO algorithms, and it performed significantly better than other EMTO algorithms. Therefore, MTES-DAKG is an efficient EMTO solver. 

*3) Components in CCMTO-MTES-DAKG*: The statistical test results show that CCMTO-MTES-DAKG performed significantly better on 8 problems, and the optimal solution mean of CCMTO-MTES-DAKG for each problem in 25 independent runs was better than wo-DT-DoS, except f3. This reveals that DT-DoS plays an important role in knowledge transfer, and it can improve the convergence of the distribution and performance of optimal solution, especially in MTOPs where optimal domains of different subtasks are similar. From the statistical result, the validity of AS-SaS can be verified. AS-SaS provides search preference for the target distribution by learning the distribution of the elite samples in the source distribution, and the optimization performance can be improved. 

It can be seen obviously that the proposed strategy of DT-DoS and AS-SaS performs significantly better than incorporating KGxS into ESs by comparing CCMTO-MTES-DAKG and wo-DA. The dynamic distance threshold and gradient correction strategy ensure that the external samples can accurately fall within the distribution of their regions respectively, and improve the target fitness value. Meanwhile, adaptive elite sampling can enhance the contribution of elite samples, leading to better optimization performance. 

The statistical test results reveal that CCMTO-MTES-DAKG is superior to wo-SD, and the effectiveness of stagnant subtask detection mechanism can be verified. It can be demonstrated that the proposed stagnant subtask detection mechanism can save computational resources on stagnant subtasks and improve the efficiency of algorithm. 

### E. REAL-WORLD APPLICATION

TABLE V RESULTS OF COMPONENT ANALYSIS IN CCMTO-MTES-DAKG
<table><tr><td rowspan="2">Algorithm</td><td colspan="3">Number of sensor nodes = 500</td><td colspan="3">Number of sensor nodes = 1000</td></tr><tr><td>Average</td><td>Std</td><td>Best</td><td>Average</td><td>Std</td><td>Best</td></tr><tr><td>CCMTO-MTES-DAKG</td><td>48.10%</td><td>0.0286</td><td>48.65%</td><td>80.02%</td><td>0.0233</td><td>80.18%</td></tr><tr><td>CCFR-CMAES</td><td>45.57%</td><td>0.0942</td><td>46.69%</td><td>75.98%</td><td>0.0545</td><td>76.26%</td></tr><tr><td>CCPSO2</td><td>46.28%</td><td>0.0541</td><td>46.80%</td><td>74.86%</td><td>0.0688</td><td>75.30%</td></tr><tr><td>TPLSO</td><td>47.16%</td><td>0.0852</td><td>47.62%</td><td>77.02%</td><td>0.0259</td><td>77.69%</td></tr><tr><td>CMAES</td><td>40.67%</td><td>0.0178</td><td>41.01%</td><td>64.10%</td><td>0.0245</td><td>64.83%</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/e4396f3a-7739-44f2-a218-7a4643fa7c12/388df1e3a848063194e34719196a151bbcaa7f22a484bd38827c83a745e07bda.jpg)
(a)1000D deployment

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-02/e4396f3a-7739-44f2-a218-7a4643fa7c12/d0c1ad3e6a6a352eeadf9c18f9a0ac2d836cdc4ac65429d58879496686eb88ea.jpg)
(b) 2000D deployment

Fig. 6. The coverage curves of different algorithms for WSNs deployment problems

The real-world application is the wireless sensor networks (WSNs) deployment, its detailed descriptions and problem settings are provided in Section S-Ⅱ of the supplementary material. The initial and final WSNs deployments of different algorithms are provided in Fig. S8-Fig. S11 of the supplementary material. TABLE V gives the average and standard deviation of the final effective coverage, the best situation is also listed in the table. The coverage curves of different algorithms are shown in Fig. 6. 

From the statistical results, it can be found that CCMTO-MTES-DAKG outperformed the other algorithms in 1000D and 2000D problems, and it obtained the best average effective coverage in the 25 independent runs. As seen from the coverage curves, CCMTO-MTES-DAKG performed well in terms of the coverage and the convergence speed. CCMTO-MTES-DAKG can find a highly competitive solution with a small amount of computing resources, and it can fast converge to the optimal solution. Compared to the traditional decomposition-based and non-decomposition algorithms, the results illustrate the outstanding performance of the proposed CCMTO-MTES-DAKG for large scale WSNs deployment problems. 

### V. CONCLUSIONS AND FUTURE WORK

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
