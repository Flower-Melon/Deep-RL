# Deep RL 实践仓库（PPO 与 TD3）

本仓库包含两个在连续动作空间上常用的深度强化学习算法实现：
- PPO（Proximal Policy Optimization）
- TD3（Twin Delayed Deep Deterministic Policy Gradient）

基于 PyTorch 与 Gymnasium，支持在 Pendulum-v1 与 BipedalWalker-v3 等经典环境上训练与评测，并提供若干已训练权重与详细笔记。

## 环境与依赖

- Python 3.11（脚本在 3.11.13 上验证）
- PyTorch 2.1.0
- NumPy 1.24.4
- Gymnasium 1.2.0 及其经典控制套件
- Box2D（用于 BipedalWalker 系列）

Windows PowerShell 下示例安装（可选命令，仅供参考）：

```powershell
# 基础依赖
pip install torch==2.1.0 numpy==1.24.4 gymnasium==1.2.0
pip install "gymnasium[classic-control]"

# 安装 Box2D（推荐使用 conda 提供的包以避免编译问题）
conda install -c conda-forge box2d-py -y
```

## 快速开始

PPO 与 TD3 的使用方式一致：进入对应子目录后运行 `train.py`。训练默认不渲染屏幕；测试模式会启用 `render_mode="human"`。

### 1) 训练 PPO

```powershell
cd .\PPO
python .\train.py
```

常用参数：
- `--env_name` 训练环境，默认 `BipedalWalker-v3`（也支持 `Pendulum-v1`）
- `--iteration` 训练回合数（默认 2000）
- `--max_step` 每回合最大交互步数（默认 1600）
- `--learning_rate` 学习率（PPO 中 Critic 使用此值；Actor 在实现中采用 3e-5）
- `--batch_size` mini-batch 大小（PPO 更新用，默认 256）
- `--buffer_size` 经验收集步数阈值，达到后触发一次 PPO 更新（默认 2048）
- `--seed` 随机种子（默认 42）
- `--load_model` 是否从 `--model_path` 加载已有权重（默认 False）
- `--model_path` 模型保存/加载目录（默认 `./models/{env_name}/`）

示例：
```powershell
# 指定环境与随机种子
python .\train.py --env_name "Pendulum-v1" --seed 10

# 继续训练（从默认路径加载）
python .\train.py --load_model True
```

测试：
```powershell
python .\train.py --mode test
```

模型会定期按 `--saveStep` 进行保存，默认每 100 回合一次。


### 2) 训练 TD3

```powershell
cd .\TD3
python .\train.py
```

常用参数：
- `--env_name` 训练环境，默认 `BipedalWalker-v3`（也支持 `Pendulum-v1`）
- `--iteration` 训练回合数（默认 500）
- `--max_step` 每回合最大交互步数（默认 2000）
- `--learning_rate` 学习率（用于优化器默认设置）
- `--batch_size` 批大小（默认 256）
- `--gamma` 折扣因子（默认 0.99）
- `--tau` 目标网络软更新系数（默认 0.005）
- `--policy_noise` 目标动作噪声强度（默认 0.2）
- `--noise_clip` 目标动作噪声裁剪范围（默认 0.5）
- `--policy_freq` 策略（Actor 与目标网）延迟更新频率（默认 2）
- `--load_model` 是否加载已有权重（默认 False）
- `--model_path` 模型保存/加载目录（默认 `./models/{env_name}/`，如有异常可手动指定）

示例：
```powershell
# 指定环境与随机种子
python .\train.py --env_name "Pendulum-v1" --seed 10

# 继续训练
python .\train.py --load_model True

# 测试
python .\train.py --mode test
```

训练阶段会在探索时对选出的动作加入高斯噪声，促进样本多样性；当经验池大小超过 `batch_size` 时开始每步更新。


## 算法与实现要点

### PPO（PPO.py）
- Actor 输出高斯分布参数（均值/标准差），采样动作并裁剪到合法范围。
- 使用 GAE（Generalized Advantage Estimation）计算优势，降低偏差与方差的折中（`gae_lambda`）。
- PPO-Clip 策略损失，约束更新幅度（`clip_param`）。
- Critic 使用 Smooth L1 损失拟合目标价值。
- On-policy：每次更新后清空记忆缓存（`self.memory`）。

训练逻辑（PPO/train.py）：
- 收集交互数据至 `buffer_size` 后触发多轮更新（`ppo_epochs`），mini-batch 采样训练。
- 训练时不渲染；测试模式使用 Actor 均值作为确定性动作并渲染。

### TD3（TD3.py）
- 双 Q 网络（`critic_1`/`critic_2`）与各自目标网络，目标取最小值缓解高估。
- 目标动作加入噪声并裁剪（Target Policy Smoothing）。
- 延迟更新策略与目标网络（`policy_freq`）。
- Off-policy：使用循环数组实现的经验回放（`utils.Replay_buffer`）。

训练逻辑（TD3/train.py）：
- 每步与环境交互并存入经验；当缓存量超过 `batch_size` 后进行更新。
- 每 `policy_freq` 次 Critic 更新后，执行一次 Actor 与三组目标网络的软更新。


## 预训练权重与复现

仓库已包含以下环境的样例权重：
- PPO: `PPO/models/BipedalWalker-v3/` 与 `PPO/models/Pendulum-v1/`
- TD3: `TD3/models/BipedalWalker-v3/` 与 `TD3/models/Pendulum-v1/`

直接进入对应子目录并使用测试模式即可渲染评测：
```powershell
# 例如 PPO BipedalWalker 评测
cd .\PPO
python .\train.py --mode test --env_name "BipedalWalker-v3"
```

## 测试效果

在`BipedalWalker-v3`环境中进行测试

![TD3效果](https://raw.githubusercontent.com/Flower-Melon/image/main/img/2025/TD3效果.gif)
![PPO效果](https://raw.githubusercontent.com/Flower-Melon/image/main/img/2025/PPO效果.gif)

## 参考

- 算法参考：
  - PPO: Proximal Policy Optimization Algorithms (Schulman et al., 2017)
  - TD3: Addressing Function Approximation Error in Actor-Critic Methods (Fujimoto et al., 2018)