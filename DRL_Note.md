# DRL复建笔记

## 0 基础常识

### 0.1 一些定义

* 马尔可夫性质:

$$P({S_{t + 1}}|{S_t},{A_t}) = P({S_{t + 1}}|{S_1},{A_1},{S_2},{A_2},...,{S_t},{A_t})$$

* 状态转移函数(实际环境决定)：

$${p_t}(s'|s,a) = P(S{'_{t + 1}} = s'|{S_t} = s,{A_t} = a)$$

* 策略函数(需要智能体学习)：

$$\pi (a|s) = P(A = a|S = s)$$

* 奖励：

$${r_t} = r({s_t},{a_t},{s_{t+1}})$$
* 回报：

$${U_t} = \sum\limits_{i = t}^n {{\gamma ^{i - t}}{r_i}}$$

* 动作价值函数：

$$Q(s,a) = E[{U_t}|{s_t} = s,{a_t} = a]$$

* 最优动作价值函数：

$${Q_*}(s,a) = \mathop {\max }\limits_\pi  Q(s,a)$$

* 状态价值函数：

$$V(s) = E[{U_t}|{s_t} = s]$$

### 0.2 贝尔曼方程

#### 0.2.1 动作值贝尔曼方程

考虑动作价值函数:

$${Q_\pi }\left( {{s_t},{a_t}} \right) = {E_{{S_{t + 1:}},{A_{t + 1:}}}}\left( {{U_t}|{s_t},{a_t}} \right)$$

根据回报的定义有：

$${U_t} = {r_t} + \gamma {U_{t + 1}}$$

由于 ${r_t}$ 只依赖于 ${s_{t+1}}$ , 结合两式可以推导出**一式**：

$${Q_\pi }\left( {{s_t},{a_t}} \right) = {E_{{s_{t + 1}},{a_{t + 1}}}}\left[ {{r_t} + \gamma {Q_\pi }\left( {{s_{t+1}},{a_{t+1}}} \right)} \right]$$

将期望展开可以获得**二式**：

$$ {Q_\pi }\left( {{s_t},{a_t}} \right) = \sum\limits_{{s_{t + 1}}} {p\left( {{s_{t + 1}}|{s_t},{a_t}} \right)\left[ {{r_t} + \gamma \sum\limits_{{a_{t + 1}}} {\pi \left( {{a_{t + 1}}|{s_{t + 1}}} \right){Q_\pi }\left( {{s_{t + 1}},{a_{t + 1}}} \right)} } \right]} $$

![Bellman_3](https://raw.githubusercontent.com/Flower-Melon/image/main/img/2025/Bellman_3.png)

#### 0.2.2 状态值贝尔曼方程

由于状态价值可以写为：

$${V_\pi }\left( {{s_t}} \right) = {E_{{a_t}}}\left[ {{Q_\pi }\left( {{s_t},{a_t}} \right)} \right] = \sum\limits_{{a_t}} {\pi \left( {{a_t}|{s_t}} \right)} {Q_\pi }\left( {{s_t},{a_t}} \right)$$

根据上面**一式**的推导，展开一部分期望，动作价值函数可以写为：

$${Q_\pi }\left( {{s_t},{a_t}} \right) = {E_{{s_{t + 1}}}}\left[ {{r_t} + \gamma {V_\pi }\left( {{s_{t + 1}}} \right)} \right]$$

结合上面的推导可以写出**三式**：

$${V_\pi }\left( {{s_t}} \right) = \sum\limits_{{a_t}} {\pi \left( {{a_t}|{s_t}} \right)} \sum\limits_{{s_{t + 1}}} {p\left( {{s_{t + 1}}|{s_t},{a_t}} \right)\left[ {{r_t} + \gamma {V_\pi }\left( {{s_{t + 1}}} \right)} \right]} $$

![Bellman_2](https://raw.githubusercontent.com/Flower-Melon/image/main/img/2025/Bellman_2.png)

#### 0.2.3 最优贝尔曼方程

考虑最优策略，即智能体的每一个动作都使当前回报最大；

$${a_{t + 1}} = \mathop {argmax }\limits_{{a_{t + 1}}} {Q_ * }\left( {{s_{t + 1}},{a_{t + 1}}} \right)$$

与**一式**结合，可以化简期望获得**四式**：

$$ {Q_ * }\left( {{s_t},{a_t}} \right) = {E_{{s_{t + 1}}}}\left[ {{r_t} + \gamma \mathop {\max }\limits_{{a_{t + 1}}} {Q_ * }\left( {{s_{t + 1}},{a_{t + 1}}} \right)} \right] $$

### 0.3 蒙特卡洛方法

蒙特卡洛方法是一大类随机算法的总称，它们通过随机样本来估算真实值，在强化学习中，蒙特卡洛方法可以应用于近似最优贝尔曼方程：

$${Q_ * }\left( {{s_t},{a_t}} \right) \approx {r_t} + \gamma \mathop {\max }\limits_{{a_{t + 1}}} {Q_ * }\left( {{s_{t + 1}},{a_{t + 1}}} \right)$$

即以一个确定的 $s_{t + 1}$ 来代替最优动作价值的期望（这里真的合理吗）

***

## 1 价值学习

### 1.1 `DQN`

`DQN`（深度Q学习网络）的目标在于学习一个最优动作价值函数，从而可以制导智能体在每一步动作时都能执行最优的动作。即以 $Q\left( {s,a;\omega } \right)$ 来近似 ${Q_ * }\left( {s,a} \right)$

#### 1.1.1 收集训练数据

以一定的策略（一般是很儿戏的策略）控制智能体与环境进行交互，获得多个四元组 $\left( {{s_t},{a_t},{r_t},{s_{t + 1}}} \right)$ 存入缓存，这个缓存被称为经验回放缓存

存储的信息主要是环境的状态转移函数信息，即在一定的动作和状态下，下一个状态和奖励是什么

#### 1.1.2 `TD`误差（目标函数设计）

考虑设计`DQN`的目标函数，考虑输入的四元组，获得当前时刻价值的估计值有两种方式，一种是直接使用DQN网络进行计算：

$${q_t} = Q\left( {{s_t},{a_t};\omega } \right)$$

另一种是使用 ${r_t}$ 间接计算（TD目标），这里使用了蒙特卡洛方法进行估计：

$${y_t} = {r_t} + \mathop {\max }\limits_{{a_{t+1}}} Q\left( {{s_{t+1}},{a_{t+1}};\omega } \right)$$

显然第二种方法比第一种方法更为接近目标的真实值，因此可以构建目标函数如下所示：

$$L\left( \omega  \right) = \frac{1}{2}{\left( {Q\left( {{s_t},{a_t};\omega } \right) - {y_t}} \right)^2}$$

求导可以获得梯度为：

$$\left( {{q_t} - {y_t}} \right){\nabla _\omega }Q\left( {{s_t},{a_t};\omega } \right)$$

其中 ${{q_t} - {y_t}}$ 被称为`TD`误差

#### 1.1.3 更新参数 $\omega$

依据上述推导:

$$\omega  = \omega  - lr \cdot \left( {{q_t} - {y_t}} \right){\nabla _\omega }Q\left( {{s_t},{a_t};\omega } \right)$$

这里注意，要清楚为什么不直接使用目标函数作反向传播，而是要绕这么一大圈计算`TD`误差和梯度相乘

因为目标函数中包含 $\mathop {\max }\limits_{{a_{t+1}}} Q\left( {{s_{t+1}},{a_{t+1}};\omega } \right)$ ，出于某种我现在没法确定的原因，不希望这一部分关于 $\omega$ 的梯度用于更新参数，所以将其以常数的形式写入`TD`误差中，绕了一圈来计算梯度

### 1.2 `SARSA`

上面关于DQN的训练方法其实比较容易地感觉出其中的雷点，即抽样过程中以完全不相干的策略与环境交互获取四元组信息，这种提取环境信息的方法感觉是很低效的，因此需要引入策略学习，这里先介绍策略学习的前身`SARSA`算法

#### 1.2.1 同策略与异策略

采集数据信息的策略称为行为策略，最终要学习的策略称为目标策略，两者相同即为同策略，不同即为异策略

上述`DQN`使用的即为异策略，`DQN`要学习的策略为：

$${a_{t + 1}} = \mathop {argmax }\limits_{{a_{t + 1}}} {Q_ * }\left( {{s_{t + 1}},{a_{t + 1}}} \right)$$

#### 1.2.2 `SARSA`的改进

`SARSA`使用同策略，在获取新一轮数据时，假设已知当前状态 ${s_t}$ ,`SARSA`使用当前学习到的策略抽样获取 ${a_t}$ 并进行执行

当观察到新状态 ${s_{t+1}}$ 后，继续根据当前状态抽样获得 ${a_{t+1}}$ 但不进行执行，仅用于计算`TD`目标：

$${y_t} = {r_t} + Q\left( {{s_{t + 1}},{a_{t + 1}};\omega } \right)$$

其余参数更新过程不做赘述

可以想到，这个当前策略究竟是什么还是有一点讲究的，基于简单原理的策略都不能起到有效作用，这里值得再使用一个神经网络来近似策略选择，以此可以引出AC框架。

AC框架中价值网络的训练使用`SARSA`算法，策略网络的训练方法后面再做讨论

### 1.3 一些技巧

#### 1.3.1 经验回放

![经验回放数组](https://raw.githubusercontent.com/Flower-Melon/image/main/img/2025/经验回放数组.png)

经验回放是强化学习中一个重要的技巧，通过将收集到的数据维护在缓存中进行随机抽取。随机抽取消除了相邻两个四元组之间的相关性，而从缓存中抽取可以多次利用已有的数据

一般的经验回放无法应用于使用同策略训练的强化学习网络，要求经验必须是当前的目标策略收集到的情况下无法使用经验回放

#### 1.3.2 目标网络

* 自举导致的偏差传播

在上述`DQN`的训练过程中，实际上希望网络的预测值逼近`TD`目标，但是考虑`TD`目标：

$${y_t} = {r_t} + \mathop {\max }\limits_{{a_{t+1}}} Q\left( {{s_{t+1}},{a_{t+1}};\omega } \right)$$

$Q\left( {{s_{t+1}},{a_{t+1}};\omega } \right)$ 本身就使用了`DQN`的估计，这种使用网络估计来更新网络参数的行为被称为自举

自举很容易导致高估的问题：

![自举的危害](https://raw.githubusercontent.com/Flower-Melon/image/main/img/2025/自举的危害.png)

容易想到自举会引起高估的原因，因为 $Q\left( {{s_{t+1}},{a_{t+1}};\omega } \right)$ 每次取最大值，在考虑均值为0的噪声影响时，相当于每次会取噪声最大的价值作为目标逼近值，由此会带来高估的问题，即`DQN`会对最优价值函数进行高估

* 使用目标网络

想要切断自举，需要使用另一个神经网络来计算`TD`目标，而不是`DQN`来计算，我们将这个网络称为目标网络，记作 $Q\left( {s,a;{w^ - }} \right)$

训练过程如下所示：

![使用目标网络的DQN训练](https://raw.githubusercontent.com/Flower-Melon/image/main/img/2025/使用目标网络的DQN训练.png)

***

## 2 策略学习

### 2.1 `AC`框架

前面的价值学习中已经给出了价值网络的目标函数和训练方法，这里要完成AC框架需要补齐价值网络的相关知识，定义 $\pi \left( {a|s;\theta } \right)$ 来表示智能体做策略的依据，其中 $\theta$ 为神经网络的参数

#### 2.1.1 策略网络目标函数

定义网络的目标函数最为关键，前面的价值网络使用与TD目标的均方根误差作为损失函数，逼近的最优动作价值函数。与之对应的，策略网络应该使当前状态下的状态价值函数最大，即目标函数为：

$$J\left( \theta  \right) = {E_s}\left[ {{V_\pi }\left( s \right)} \right]$$

不依赖于当前状态 $s_t$ ,只依赖于策略 $\pi$

我们希望最大化这个目标函数，因此这里应该使用梯度上升法：

$$\theta  = \theta  + lr \cdot {\nabla _\theta }J\left( \theta  \right)$$

#### 2.1.2 策略梯度定理(`REINFORCE`方法)

显然上面提到的目标函数没法显式用网络的输出值表示，这样求梯度是不可能的，需要想办法

这里通过策略梯度定理的使用（证明过程过于复杂，后续有需要再研究吧）：

![策略梯度定理](https://raw.githubusercontent.com/Flower-Melon/image/main/img/2025/策略梯度定理.png)

很显然这里期望想求出来也是做不到的，考虑到价值网络在估计`TD`目标时使用了蒙特卡罗方法，这里继续使用，以当前时刻的状态和动作作为上述梯度的估测，则可以顺利更新参数：

$$\theta  = \theta  + lr \cdot {Q_\pi }\left( {{s_t},{a_t}} \right) \cdot {\nabla _\theta }\ln \pi \left( {{a_t}|{s_t};\theta } \right)$$

上述训练策略网络的方法即为`REINFORCE`方法

#### 2.1.3 `actor-critic`

结合`REINFORCE`和`SARSA`方法，我们可以成功完成`AC`框架的训练

* 注意这里`AC`框架中的价值网络和`DQN`略有不同，`DQN`采用异策略训练，逼近的最优价值函数，而此处的价值网络逼近的是价值函数

![AC框架](https://raw.githubusercontent.com/Flower-Melon/image/main/img/2025/AC框架.png)

这里直接给出带目标网络的`AC`框架训练方法：

![AC训练流程](https://raw.githubusercontent.com/Flower-Melon/image/main/img/2025/AC训练流程.png)

### 2.2 从`DDPG`到`TD3`

#### 2.2.1 `DDPG`（深度确定性策略梯度）

前面所有的训练都假设动作空间是离散的，而实际应用中大多数动作空间都为连续的，因此需要确定性网络。

确定性是指策略网络输出的动作不再是概率分布，而是一个确定的动作值（不然动作空间太大了，没办法输出概率分布），前文中离散动作空间的策略网络 $\pi \left( {a|s;\theta } \right)$ 输出一个动作的概率分布，而确定性策略梯度的策略网络 $\mu \left( {s;\theta } \right)$ 直接输出确定的动作

* 异策略训练

`DDPG`采用异策略进行训练，使用行为策略 $a = \mu \left( {s;{\theta _{old}}} \right) + \varepsilon$ 进行经验的收集。这里有点稍微不能理解，明明可以使用同策略进行训练，为什么要使用异策略，猜测异策略和同策略可能没有好坏之分，仅仅是训练的两种策略而已

这里附上`o4-mini`给出的同策略和异策略的选择方法：

总结选择建议：
> 若能持续在线采集数据、内存和计算资源有限，且希望算法简单稳定，可优先考虑同策略方法。
> 若数据昂贵、希望最大化样本利用率、或需离线/批量训练，则异策略方法通常更合适，但要注意收敛稳定性。

**最新更新**

> 实际上`DDPG`和接下来的`TD3`使用`off-policy`进行训练是必然的，这是由其算法的适用环境决定的，不可以修改为 `on-policy`训练。
> 因为两种算法都为连续动作空间设计，动作的可能取值无限多，同时，两种算法的策略网络都输出确定性策略；如果要对动作空间有足够的探索，必须在选取目标策略时加入一定的噪声，而策略网络的输出又不能带噪声，这天生决定了必须选取异策略进行训练。
> 这里也可以理解异策略一定是预先收集数据，进行离线训练，行为策略和目标策略一定是不同的，就没有必要使用成本和要求更高的在线训练方式，所以异策略一定对应离线训练。

* 目标函数设计

这里与前文形成一定的对比,注意前文中`REINFOCE`方法中的目标函数为状态价值函数的期望，这里直接使用价值网络输出的期望作为目标函数：

$$J\left( \theta  \right) = {E_s}\left[ {q\left( {s,\mu \left( {s;\theta } \right);\omega } \right)} \right]$$

这里目标函数实际上就是`REINFOCE`的目标函数，因为动作是连续的形式上得以化简

我们需要最大化这个目标函数，很显然这里可以使用蒙特卡洛估计和链式法则获得梯度上升法的求解方案：

$$\theta  = \theta  + lr \cdot {\nabla _\theta }\mu \left( {{s_t};\theta } \right) \cdot {\nabla _a}q\left( {{s_t},{a_t};\omega } \right)$$

具体的训练流程在下面介绍`TD3`时再给出

#### 2.2.2 `TD3`（双延迟深度确定性策略梯度）

`DDPG`存在这比较严重的高估问题，这里使用`Twin Delayed Deep Dertermin Policy Gradienrt`方法提升算法的表现

* 三种改进

1. 截断双Q学习

使用两个价值网络和一个策略网络，加上对应的目标网络，共计六个网络。由于目标网络就一定程度上可以自举消除高估的影响，使用两套价值网络分开更新训练取更小的价值估计值又进一步可以消除高估的影响

![TD3架构](https://raw.githubusercontent.com/Flower-Melon/image/main/img/2025/TD3架构.png)

2. 在动作中加入噪声

TD3在更新目标策略时，添加了噪声以使得目标策略更加平滑。这种做法可以减少策略的剧烈变化，从而提高学习的稳定性。具体而言，在计算目标Q值时，TD3会对目标动作添加小的随机噪声

3. 减小策略网络和目标网络的更新频率

TD3对策略网络和目标网络的更新进行了延迟处理。具体来说，策略网络和目标网络的更新频率低于Q网络。这种延迟更新可以使得Q网络在更新策略之前有更多的时间来收敛，从而提高学习的稳定性。由于训练之初的价值网络并不可靠，此时使用价值网络的打分更新策略网络是不合适的，因此每一轮更新一次价值网络，而每隔`k`轮更新一次策略网络和三个目标网络

* 训练流程

![TD3训练流程](https://raw.githubusercontent.com/Flower-Melon/image/main/img/2025/TD3训练流程.png)

### 2.3 从`A2C`到`PPO`

#### 2.3.1 `A2C`

上述的`TD3`专门为连续动作空间设计，实际上在实际的应用中，下面要提到的`PPO`算法更为有效，且可以在离散和连续动作空间上都能使用，是当前强化学习领域最强大的算法。

* `A2C`的主要思想

我们先来看A2C算法。在其中，演员不参考评论家预测的收益的大小来更新参数，而是根据实际收益超出评论家预期收益的程度来更新参数。这样比较合理，也训练过程也更加稳定。例如，你平时考90分，期末考96分，超出预期的程度是6；而你朋友平时考60分，期末考95分，超出预期的程度就是35。因此A2C算法也觉得你朋友的期末复习策略更值得强化。

* 带基线的策略梯度方法

`A2C`的价值网络不再是对动作价值函数的近似，而是对状态价值函数的近似，只需要输入当前状态，就可以得出一个估计值。该状态的估计值称为**基线**，策略网络采取一个新动作后，重新获得的新状态输入价值网络，得到的新价值与基线的比较值就可以作为评判该动作好坏的标准。

具体来说，

此时的`TD`误差为：

$${r_t} + v\left( {{s_{t + 1}};\omega } \right) - v\left( {{s_t};\omega } \right)$$

此时的策略梯度为:

$$\left[ {{Q_\pi }\left( {{s_t},{a_t}} \right) - {V_\pi }\left( {{s_t}} \right)} \right] \cdot {\nabla _\theta }\ln \pi \left( {{a_t}|{s_t};\theta } \right)$$

策略梯度中的优势值 $\left[ {{Q_\pi }\left( {{s_t},{a_t}} \right) - {V_\pi }\left( {{s_t}} \right)} \right]$ 实际上由TD误差进行代替。

* 训练过程
  
具体的训练流程如下所示：

![A2C训练过程](https://raw.githubusercontent.com/Flower-Melon/image/main/img/2025/A2C训练过程.png)

#### 2.3.2 `PPO`



`PPO`与`A2C`最大的不同是：`A2C`每走一步就更新一次，而`PPO`会先收集**一批 (Batch)** 数据，然后用这批数据**重复训练 K 次 (Epochs)**。

* PPO 核心训练过程

    **阶段一：收集数据 (Batch)**

    1.  **收集经验**:
        使用**旧的**策略网络 $\pi_{\theta_{\text{old}}}$ 与环境交互 $N$ 步，收集一个批次 (batch) 的数据 $\mathcal{D}$：
        $\mathcal{D} = \{s_t, a_t, r_t, s_{t+1}\}$
        并保存这 $N$ 步的**旧动作概率** $\pi_{\theta_{\text{old}}}(a_t|s_t)$。

    2.  **计算优势和回报**:
        遍历这 $N$ 步数据，计算每一步的**优势 $\hat{A}_t$** 和**回报 $\hat{y}_t$**。
        * $r_t + \gamma \cdot v(s_{t+1}; \mathbf{w})$ (目标)
        * $\hat{A}_t = \hat{y}_t - v(s_t; \mathbf{w})$ ("优势")

    **阶段二：循环更新 (K Epochs)**

    1.  **开始 K-Epoch 循环**:
        在整个数据集 $\mathcal{D}$ 上，重复以下步骤 $K$ 次：

    2.  **(核心) 计算概率比**:
        使用**当前**策略网络 $\pi_{\theta}$ (正在更新的网络) 和**旧**策略网络 $\pi_{\theta_{\text{old}}}$ (收集数据时的网络) 计算比率：
        $$r_t(\theta) = \frac{\pi(a_t | s_t; \theta)}{\pi(a_t | s_t; \theta_{\text{old}})}$$

    3.  **(核心) 计算 PPO 策略损失**:
        $\epsilon$ 是一个超参数 (例如 $0.2$)。
        $$L^{\text{CLIP}}(\theta) = \min \left( r_t(\theta) \hat{A}_t, \quad \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right)$$

    4.  **计算价值损失**:
        这和 A2C 类似，使用步骤 2 中算好的回报 $\hat{y}_t$ 作为目标：
        $$L^V(\mathbf{w}) = \left( v(s_t; \mathbf{w}) - \hat{y}_t \right)^2$$

    5.  **更新网络**:
        * **更新价值网络 (梯度下降)**：最小化价值损失 $L^V$
            $$\mathbf{w} \leftarrow \mathbf{w} - \alpha \cdot \nabla_{\mathbf{w}} L^V(\mathbf{w})$$

        * **更新策略网络 (梯度上升)**：最大化策略目标 $L^{\text{CLIP}}$
            $$\theta \leftarrow \theta + \beta \cdot \nabla_{\theta} L^{\text{CLIP}}(\theta)$$

* `GAE`广义优势估计

`TD-Advantage`优势 $A_t^{(1)} = (r_t + \gamma V(s_{t+1})) - V(s_t)$

具有高偏差, 低方差的特点，优势信号的质量**完全**依赖于 Critic 对 $V(s_{t+1})$ 估算的准确性，如果 Critic的预测不准（这在复杂任务中很常见），优势信号就是错误的，导致策略更新错误。

`GAE` (广义优势估计)  $A_t^{GAE} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \cdot \delta_{t+l}$ (其中 $\delta$ 是 1-Step TD 误差)， 通过参数 $\lambda$ 在偏差和方差之间取得平衡。

`GAE` 综合了未来**多步**的真实奖励 $r_t, r_{t+1}, \dots$，而不是只依赖于 $r_t$ 和 $V(s_{t+1})$ ,这使得 `GAE`对 `Critic` 预测的错误**不那么敏感**

GAE 通过使用 $\lambda$ 参数，通过融合多步真实奖励，显著降低了优势估计的偏差，为 `Actor` 提供了更稳定、更准确的更新信号。

