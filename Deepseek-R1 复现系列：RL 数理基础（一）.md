# RL 数理基础（一） ✨

**系列文章**：RL 数理基础 / Deepseek-R1 复现 / 数据集 📚

>参考书目：
>1. Richard, S. Andrew G. (2019). *强化学习(第2版)*. 电子工业出版社.
>2. Zhao, S. (2025). *Mathematical Foundations of Reinforcement Learning*. Springer Nature Press and Tsinghua University Press.

>课程：
>1. Course: [https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)
>2. Code: [https://github.com/jwk1rose/RL_Learning](https://github.com/jwk1rose/RL_Learning)

说明：以下内容是我在赵世钰老师的强化学习课程（B 站）笔记。如果对你有帮助，请到 [**【强化学习的数学原理】课程：从零开始到透彻理解（完结）**](https://www.bilibili.com/video/BV1sd4y167NS/?share_source=copy_web&vd_source=a18061c5818734743376efdea7dd0438) 点赞哦！👍

# **基础概念**

## 强化学习定义 

1. 强化学习具有两个本质特征：
    1. 强化学习就是学习“做什么（即如何把当前的情境映射成动作）才能使得数值化的收益信号最大化”。学习者不会被告知应该采取什么动作，而是必须自己通过尝试去发现哪些动作会产生最丰厚的收益。在最有趣又最困难的案例中，动作往往影响的不仅仅是即时收益，也会影响下一个情境，从而影响随后的收益。这两个特征——试错和延迟收益——是强化学习两个最重要最显著的特征。
    2. 强化学习系统的核心要素：
        - 系统角度：智能体，环境 
        - 要素角度：策略模型，收益（奖励），价值函数（状态值），对环境建立的模型
    3. RLHF 所使用的方法是一种解强化学习问题的经典方法，策略梯度方法。

>参考：Richard, S. Andrew G. (2019). *强化学习(第2版)*. 电子工业出版社.

### **State (状态)**

**State** 描述了智能体在环境中的状态。在网格世界的例子中，状态对应于智能体的位置。假设网格中有 9 个单元格，那么就有 9 个状态，分别表示为 $s_1, s_2, \dots, s_9$，如图。所有状态的集合称为状态空间，记作 $S = \{ s_1, s_2, \dots, s_9 \}$。在 LLMs Based Agent 中，可以理解为 LLMs 输入的 Prompts，如果是多模态模型，这个输入还可以是 图片/音频/视频 + Prompts。

**形式化定义：**

- 状态空间： $S = \{ s_1, s_2, \dots, s_9 \}$

### **Action (动作)** 

**Action** 是智能体在某个状态下可以执行的操作。在网格世界的例子中，智能体可以采取 5 个可能的动作：向上移动、向右移动、向下移动、向左移动、保持不动。这些动作分别记作 $a_1, a_2, \dots, a_5$（如图所示）。所有动作的集合称为动作空间，记作 $A = \{ a_1, a_2, \dots, a_5 \}$。在 LLMs Based Agent 中，模型的输出就是代理在动作空间中做出的选择。在 RLHF 中，策略模型的输出，就是代理在动作空间中做出的选择。

**形式化定义：**

• 动作空间： $A = \{ a_1, a_2, \dots, a_5 \}$

• 状态 $s_1$ 下的可用动作空间： $A(s_1) = \{ a_2, a_3, a_5 \}$

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1ff923a9d57246c1a99bd7ee5e4b5d8e.png#pic_center)
>参考：Zhao, S. (2025). *Mathematical Foundations of Reinforcement Learning*. Springer Nature Press and Tsinghua University Press. Page: 3.

### **状态转移 (State Transition)** 

当智能体采取一个动作时，它可能从一个状态转换到另一个状态。这个过程被称为状态转移。例如，如果智能体处于状态 $s_1$，并选择动作 $a_2$（即向右移动），那么智能体将转移到状态 $s_2$。这个过程可以用如下公式表示：

$a_2: s_1 \rightarrow s_2$

**形式化定义：**

- 如果智能体在状态 $s_1$ 执行动作 $a_2$，状态转移的概率为：
    
    $p(s_1 | s_1, a_2) = 0, \quad p(s_2 | s_1, a_2) = 1, \quad p(s_3 | s_1, a_2) = 0, \quad \dots$
    
    这表示，当智能体在 $s_1$ 执行动作 $a_2$ 时，它一定会转移到 $s_2$，而转移到其他状态的概率为 $0$。

状态转移过程可能是随机的。智能体可能会以不同的概率转移到不同的状态。 🌪️

>*参考资料：*
>1. 应用随机过程*https://www.math.pku.edu.cn/teachers/lidf/course/stochproc/stochprocnotes/html/_book/index.html*
>2. 随机过程*https://zh.wikipedia.org/wiki/%E9%9A%8F%E6%9C%BA%E8%BF%87%E7%A8%8B*

### **奖励 (Reward)** 

**奖励 (Reward)** 是强化学习中最独特的概念之一。代理在某一状态下执行动作后，从环境中获得反馈，这个反馈就是奖励，这个奖励值用 $r$ 表示，通常用 $r(s, a)$ 来表示，它是状态 $s$ 和动作 $a$ 的函数。奖励的值可以是正数、负数或零。不同的奖励对代理最终学到的策略有不同的影响。正奖励鼓励代理执行相应的动作，而负奖励则会使代理避免该动作。RLHF 中所训练的奖励模型，就是用来拟合奖励函数的。

**形式化定义：**

- **奖励函数**：  $r(s, a)$
- **定义奖励的一个例子**：
    
    $r_{\text{boundary}} = -1, \quad r_{\text{forbidden}} = -1, \quad r_{\text{target}} = +1, \quad r_{\text{other}} = 0$

### **回报 (Return)** 

**回报 (Return)** 是指代理沿着轨迹获得的所有奖励的总和。在某一状态下执行一系列动作后，所有获得的奖励累加得到的总值被称为回报。回报常用于评估策略的优劣。

例如，给定一个策略，代理的轨迹如下：

$a_2 \rightarrow a_3 \rightarrow a_3 \rightarrow a_2 \quad s_1 \rightarrow s_2 \rightarrow s_5 \rightarrow s_8 \rightarrow s_9$

相应的回报为：

$\text{return} = 0 + 0 + 0 + 1 = 1$

回报也可以叫做总奖励 (total reward) 或累积奖励 (cumulative reward)。 

**形式化定义：**

- **回报**（轨迹总和）：
    
    $\text{return} = \sum_{i=1}^{n} r_i$
    
例如：

$\text{return} = 0 + 0 + 0 + 1 = 1$

- 另一个回报的例子：$\text{return} = 0 - 1 + 0 + 1 = 0$

**折扣回报 (Discounted Return)**：对于无限长的轨迹，我们引入折扣因子 $\gamma$ 来避免回报发散。折扣回报是对未来奖励的加权求和，较远的奖励会有较低的权重。折扣回报公式如下：

$\text{discounted return} = r_0 + \gamma r_1 + \gamma^2 r_2 + \gamma^3 r_3 + \cdots$

**回报的折扣公式**：

$\text{discounted return} = \frac{\gamma^3}{1 - \gamma}$

折扣回报和时间序列分析中的自回归模型（AR）在序列建模上有很多相似之处。它们都与动态系统相关，折扣回报需要考虑历史奖励的累积效果；自回归模型则依赖于历史数据来建模和预测序列的行为。

>*参考资料：*
>1. 金融时间序列分析讲义https://www.math.pku.edu.cn/teachers/lidf/course/fts/ftsnotes/html/_ftsnotes/index.html
>2. 何书元. (2003). 应用时间序列分析. 北京大学出版社.

### **回合 (Episode)** 

**回合 (Episode)** 是指代理与环境交互的一个完整过程，通常从初始状态开始，直到达到终止状态。每个回合的轨迹称为一个回合或试验 (trial)。如果环境或策略是随机的，从同一状态出发的回合可能会有所不同。如果一切是确定性的，从同一状态出发的回合将总是相同的。

回合通常假设为有限轨迹。没有终止状态的任务被称为持续任务 (continuing task)，而包含终止状态的任务被称为回合任务 (episodic task)。

**形式化定义：**

- **回合 (Episode)**：轨迹  $s_0 \rightarrow s_1 \rightarrow \dots \rightarrow s_n$  中的每个状态和动作对构成一个回合。

**吸收状态 (Absorbing State)**：一种特殊的状态，代理到达后将永远停留在该状态。例如：

$A(s_9) = \{ a_5 \}, \quad p(s_9 | s_9, a_i) = 1 \quad \text{for all } i = 1, \dots, 5$
这里需要注意强化学习的吸收态和动力系统的吸引子之间的区别。吸收态更侧重于在强化学习环境中描述某个特定状态的终结性和任务的完成，强调状态转移停止；吸引子则是动力系统中的一种稳定性概念，描述系统状态在长期演化中的最终行为模式，可以是多种类型的稳定结构。


### **Policy (策略)** 

**Policy** 描述了智能体在每个状态下应该采取的动作。智能体按照策略执行时，可以从初始状态开始生成一个轨迹，例如：$s_0 \rightarrow s_1 \rightarrow \dots \rightarrow s_n$.

## **形式化定义：**

- 策略 $\pi(a | s)$ 是一个条件概率分布函数，定义在每个状态上。例如，在状态 $s_1$ 下，策略 $\pi$ 可以表示为：

  $$ \pi(a_1 | s_1) = 0, \quad \pi(a_2 | s_1) = 1, \quad \pi(a_3 | s_1) = 0, \quad \pi(a_4 | s_1) = 0, \quad \pi(a_5 | s_1) = 0 $$

这表示，在状态 $s_1$ 下，采取动作 $a_2$ 的概率为 $1$，而采取其他动作的概率为 $0$。

这个策略是确定性的，也就是说，在给定状态下，智能体总是采取相同的动作。然而，策略通常是随机的，即在同一个状态下，智能体有可能采取不同的动作，且这些动作的选择具有不同的概率。例如，下图中的策略就是一个随机策略：在状态 $s_1$ 下，智能体可能选择向右移动或向下移动，且这两种动作的概率不同。 

### **随机策略示例：**

- 对于随机策略，条件概率分布可能是：

  $$ \pi(a_1 | s_1) = 0.5, \quad \pi(a_3 | s_1) = 0.5 $$

这表示，在状态 $s_1$ 下，智能体有 50% 的概率向右移动（采取 $a_2$），有 50% 的概率向下移动（采取 $a_3$）。 

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7ac67846469242d58c64dac91fc813ff.png#pic_center)


>参考：Zhao, S. (2025). *Mathematical Foundations of Reinforcement Learning*. Springer Nature Press and Tsinghua University Press. Page: 6.



策略可以通过条件概率分布的表格来表示。例如，下表表示了图 1.5 中的随机策略。表格中的第 $i$ 行第 $j$ 列的条目表示在第 $i$ 状态下采取第 $j$ 动作的概率。这种表示方法称为表格表示法.

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c1c2844c6c81443d9f95530cf075d848.png#pic_center)

>参考：Zhao, S. (2025). *Mathematical Foundations of Reinforcement Learning*. Springer Nature Press and Tsinghua University Press. Page: 6.


在这个表格中，第 $i$ 行表示状态 $s_i$，第 $j$ 列表示动作 $a_j$，每个单元格的值表示在该状态下采取该动作的概率。 📊

策略的存储方式可以有多种，例如表格表示法或参数化函数表示法。


## **参数化函数：**

连续的参数化函数的一个典型例子是 **RLHF**。在 RLHF 中使用，我们用 $\pi_\theta$ 表示策略模型，其中 $\theta$ 参数就是 LLM 需要被训练的参数。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b78590270de141de82c0608929ddc0ee.png#pic_center)

>参考：Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y.K., Wu, Y., Guo, D., 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. [https://doi.org/10.48550/arXiv.2402.03300](https://doi.org/10.48550/arXiv.2402.03300)


## **Markov Decision Process (MDP) 马尔可夫决策过程** 

**Markov Decision Process (MDP)** 是描述随机动态系统的通用框架。MDP 是强化学习中建模决策过程的基础，特别适用于具有随机性的环境。在 MDP 中，主要有以下几个组成部分：

### **1. Sets（集合）:**
- **State space（状态空间）**: 所有可能状态的集合，记为 $S$。
- **Action space（动作空间）**: 与每个状态 $s \in S$ 相关联的动作集合，记为 $A(s)$。
- **Reward set（奖励集）**: 与每个状态-动作对 $(s, a)$ 相关联的奖励集合，记为 $R(s, a)$。

### **2. Model（模型）:**
- **State transition probability（状态转移概率）**: 在状态 $s$ 下，采取动作 $a$ 时，转移到状态 $s'$ 的概率为 $p(s' | s, a)$，满足条件：
  
  $$ \sum_{s' \in S} p(s' | s, a) = 1 \quad \text{对于任何} (s, a) $$

- **Reward probability（奖励概率）**: 在状态 $s$ 下，采取动作 $a$ 时，获得奖励 $r$ 的概率为 $p(r | s, a)$，满足条件：

  $$ \sum_{r \in R(s,a)} p(r | s, a) = 1 \quad \text{对于任何} (s, a) $$

### **3. Policy（策略）:**
- **Policy** 是在每个状态下选择动作的概率分布，记为 $\pi(a | s)$，满足条件：

  $$ \sum_{a \in A(s)} \pi(a | s) = 1 \quad \text{对于任何} \ s \in S $$

### **4. Markov Property（马尔可夫性质）:**
- 马尔可夫性质指的是随机过程的“无记忆”特性。数学上，它表示：

  $$ p(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, \dots, s_0, a_0) = p(s_{t+1} | s_t, a_t) $$

  其中 $t$ 表示当前时间步，$t+1$ 表示下一个时间步。这表明，下一个状态或奖励仅依赖于当前的状态和动作，而与过去的状态和动作无关。 

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bf5d2c82deb74e2a9ded3c3d20e02267.png#pic_center)

## **5. Markov Process (MP) vs. MDP:**
- 当在 MDP 中策略已经固定时，MDP 会退化为 **Markov Process（马尔可夫过程）**。例如，上图中的网格世界就可以抽象为一个 Markov Process。
  
- 在随机过程的文献中，Markov Process 也被称为 **Markov Chain（马尔可夫链）**，如果它是一个离散时间过程并且状态的数量是有限或可数的。

## **6. 形式化定义：**
- **状态转移概率**: $p(s' | s, a)$
- **奖励概率**: $p(r | s, a)$
- **策略**: $\pi(a | s)$

>参考书籍：
>1. 应用随机过程[https://www.math.pku.edu.cn/teachers/lidf/course/stochproc/stochprocnotes/html/_book/index.html](https://www.math.pku.edu.cn/teachers/lidf/course/stochproc/stochprocnotes/html/_book/index.html)
>2. 随机过程[https://zh.wikipedia.org/wiki/%E9%9A%8F%E6%9C%BA%E8%BF%87%E7%A8%8B](https://zh.wikipedia.org/wiki/%E9%9A%8F%E6%9C%BA%E8%BF%87%E7%A8%8B)



# **强化学习的学习过程 1** 

深度学习的训练思想是让模型正向推理后，和 ground truth 比较进行 Loss 计算。而强化学习的训练思想也是类似的，通过求解贝尔曼公式，进而得到状态值 $v^{\pi}(s)$ 的数值解。这个过程称为策略评价。通过不断调整策略，得到更优的策略评价，最终得到一个较好的策略。

## **概念：状态值 (State Value)** 

状态价值是强化学习中评估策略的一个重要概念。它被定义为：若代理从某一状态出发并遵循一个给定策略，代理能够获得的期望回报。数学上，状态价值是一个期望值，表示从某个状态开始，代理能够获得的折扣回报的平均值。该概念可以用来评价一个策略的好坏，状态价值越大，表明策略越好。

**形式化定义为：**

$$
v^\pi(s) = \mathbb{E}[G_t | S_t = s]
$$

其中，$v^\pi(s)$ 是从状态 $s$ 开始并遵循策略 $\pi$ 所获得的状态价值，$G_t$ 是从时间 $t$ 开始的折扣回报。

## **工具：贝尔曼方程 (Bellman Equation)** 

贝尔曼方程是分析状态价值的重要工具。它通过描述**状态和价值**之间的关系，提供了**计算状态价值的数学方法**。贝尔曼方程是一组线性方程，定义了所有状态价值之间的相互依赖关系。

贝尔曼方程形式化的定义为：

$$
v^\pi(s) = \sum_{a \in A} \pi(a|s) \left[ \sum_{r \in R} p(r|s, a) r + \gamma \sum_{s' \in S} p(s'|s, a) v^\pi(s') \right]
$$

其中，$\pi(a|s)$ 是在状态 $s$ 下采取动作 $a$ 的策略概率，$p(r|s, a)$ 是给定状态和动作的奖励概率分布，$p(s'|s, a)$ 是状态转移概率，$\gamma$ 是折扣因子，$v^\pi(s')$ 是从下一个状态 $s'$ 开始的状态价值。简单的表示为：

$$ v_\pi = r_\pi + \gamma P_\pi v_\pi $$

**贝尔曼方程为求解状态价值提供了一个重要的框架，通过求解这个方程，我们能够评估一个策略的效果。**



# **强化学习的最终目标** 

强化学习的最终目标，是求最优策略（模型）的参数。这个策略参数使强化学习模型在状态值（价值函数）中达到最大。贝尔曼最优性方程为我们提供了一个强大的工具，通过它可以系统地分析和求解最优策略和状态值。在强化学习中，优化策略和优化状态值是核心概念。我们通过贝尔曼最优性方程（Bellman Optimality Equation, BOE）来分析并求解这些优化策略和状态值。

## **两个概念：**

1. **最优策略** $\pi^*$：一种策略，它在每个状态下的选择能使得状态值最大化。
2. **最优状态值** $v^*$：在最优策略下，每个状态的价值。

## **一个工具：**

**贝尔曼最优性方程（BOE）**：它是一个核心工具，通过解这个方程，我们能够找到最优的状态值和策略。

## **数学公式**

贝尔曼最优性方程的表达式为：

$$
v(s) = \max_{\pi(s) \in \Pi(s)} \sum_{a \in A} \pi(a|s) \left( \sum_{r \in R} p(r|s, a) r + \gamma \sum_{s' \in S} p(s'|s, a) v(s') \right)
$$

其中：
- $v(s)$ 是状态 $s$ 的价值。
- $\pi(s)$ 是在状态 $s$ 下的策略。
- $r$ 是即时奖励，$\gamma$ 是折扣因子。
- $p(r|s,a)$ 和 $p(s'|s,a)$ 分别表示从状态 $s$ 采取动作 $a$ 后得到奖励 $r$ 和转移到状态 $s'$ 的概率。

贝尔曼最优性方程通过对每个状态求解最优的策略和状态值，帮助我们得到最优的决策过程。 

## **相关数学证明：**

1. **不动点定理（Fixed-point theorem）**：该定理表明，贝尔曼最优性方程的解是一个不动点，即通过迭代过程，可以收敛到一个唯一的最优状态值。

2. **这里证明了几个基本问题**：
   - 存在性：最优策略是否存在？是 ✅
   - 唯一性：最优策略是否唯一？否 ❌
   - 随机性：最优策略是随机的还是确定性的？一定存在确定性最优解 🎯

3. **求解方程的算法**：
   - 通过值迭代法（Value Iteration）来迭代求解最优状态值，并进一步获得最优策略. 

